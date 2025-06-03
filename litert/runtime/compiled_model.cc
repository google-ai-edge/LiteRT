// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "litert/runtime/compiled_model.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if defined(__ANDROID__)
#include <android/hardware_buffer.h>
#endif

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_accelerator.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_logging.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_event.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_handle.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_opaque_options.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_requirements.h"
#include "litert/compiler/plugin/compiler_plugin.h"
#include "litert/core/build_stamp.h"
#include "litert/core/model/model.h"
#include "litert/core/model/model_serialize.h"
#include "litert/core/options.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "litert/runtime/accelerator.h"
#include "litert/runtime/custom_op_dispatcher.h"
#include "litert/runtime/dispatch/dispatch_opaque_options.h"
#include "litert/runtime/external_litert_buffer_context.h"
#include "litert/runtime/litert_cpu_options.h"
#include "litert/runtime/litert_runtime_options.h"
#include "litert/runtime/metrics.h"
#include "litert/runtime/tensor_buffer.h"
#include "tensorflow/compiler/mlir/lite/allocation.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/common.h"
#include "tflite/core/interpreter_builder.h"
#include "tflite/delegates/utils/simple_opaque_delegate.h"
#include "tflite/interpreter.h"
#include "tflite/interpreter_options.h"
#include "tflite/kernels/register.h"
#include "tflite/model_builder.h"

using ::litert::Error;
using ::litert::Expected;
using ::litert::TensorBuffer;
using ::litert::Unexpected;
using ::litert::internal::DispatchDelegateOptions;
using ::litert::internal::ExternalLiteRtBufferContext;
using ::litert::internal::GetTensorBufferTypeName;
using ::litert::internal::SerializeModel;

Expected<void> LiteRtCompiledModelT::InitializeRuntime(
    LiteRtEnvironmentT* env, LiteRtOptions jit_compilation_options) {
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;

  // Apply custom ops.
  if (jit_compilation_options) {
    for (auto& option : jit_compilation_options->custom_op_options) {
      custom_op_dispatchers_.push_back(
          std::make_unique<litert::internal::CustomOpDispatcher>(option));
      auto* tflite_registration =
          custom_op_dispatchers_.back()->GetTfLiteRegistration();
      resolver.AddCustom(option.op_name.c_str(), tflite_registration);
    }
  }

  tflite::InterpreterOptions interpreter_options;
  interpreter_options.SetUseSignatureTensorNames(true);
  int num_threads = 1;
  if (jit_compilation_options) {
    litert::Options cc_options(jit_compilation_options, litert::OwnHandle::kNo);
    LITERT_ASSIGN_OR_RETURN(Expected<litert::OpaqueOptions> opaque_options,
                            cc_options.GetOpaqueOptions());

    auto runtime_options_status = litert::FindOpaqueData<LiteRtRuntimeOptionsT>(
        *opaque_options, LiteRtRuntimeOptionsT::Identifier());
    if (runtime_options_status) {
      auto runtime_opaque_options = *runtime_options_status;
      interpreter_options.SetShloCompositeInlining(
          runtime_opaque_options->shlo_composite_inlining);
    }

    auto cpu_options_status = litert::FindOpaqueData<LiteRtCpuOptionsT>(
        *opaque_options, LiteRtCpuOptionsT::Identifier());
    if (cpu_options_status) {
      auto cpu_opaque_options = *cpu_options_status;
      num_threads = cpu_opaque_options->xnn.num_threads;
    }
  }

  tflite::InterpreterBuilder builder(*fb_model_, resolver,
                             &interpreter_options);
  builder(&interp_);
  interp_->SetNumThreads(num_threads);
  if (interp_ == nullptr) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to build TFL interpreter");
  }

  signature_keys_ = interp_->signature_keys();
  if (signature_keys_.empty()) {
    static auto* default_signature_key =
        new std::string(LiteRtSignatureT::kDefaultSignatureKey);
    signature_keys_.push_back(default_signature_key);
  }
  // Register the ExternalLiteRtBufferContext for TensorBuffer handshaking.
  buffer_context_ =
      std::make_unique<litert::internal::ExternalLiteRtBufferContext>(env);
  interp_->SetExternalContext(kTfLiteLiteRtBufferContext,
                              buffer_context_.get());

  return {};
}

Expected<void> LiteRtCompiledModelT::InitializeModel(
    LiteRtModelT& model, LiteRtHwAcceleratorSet hw_accelerators,
    LiteRtEnvironmentT& env) {
  bool need_reserialization = false;

  if (hw_accelerators != kLiteRtHwAcceleratorNone) {
    LITERT_LOG(LITERT_INFO, "Applying compiler plugins...");
    // TODO: b/409819691 - Pass user provided `LiteRtOptions` down to the
    // vendor code (nullptr are safe for now).
    auto jit_result =
        litert::internal::ApplyPlugins(&env, /*options=*/nullptr, &model,
                                       hw_accelerators, &need_reserialization);
    if (!jit_result) {
      LITERT_LOG(LITERT_WARNING, "Failed to apply compiler plugins: %s",
                 jit_result.Error().Message().c_str());
    } else {
      LITERT_LOG(
          LITERT_INFO, "%d compiler plugins were applied successfully: %s",
          jit_result->num_applied_plugins, jit_result->success_message.c_str());
      LITERT_LOG(LITERT_WARNING, "Plugin errs: %s",
                 jit_result->error_message.c_str());
    }
  }

  const auto& tfl_wrapper = litert::internal::GetTflFlatbuffer(model);
  // Currently, in all situations where litert model was import from a
  // flatbuffer, the litert model will own said flatbuffer and stored it in the
  // OwningBufferRef.
  auto tfl_buf = tfl_wrapper.Buf();

  if (!need_reserialization && tfl_buf.Data() != nullptr) {
    LITERT_LOG(
        LITERT_INFO,
        "Flatbuffer model initialized directly from incoming litert model.");
    fb_model_ = tflite::FlatBufferModel::BuildFromBuffer(tfl_buf.StrData(),
                                                         tfl_buf.Size());
    return {};
  }

  LITERT_LOG(LITERT_INFO, "JIT compilation changed model, reserializing...");

  auto serialized = SerializeModel(std::move(model));
  if (!serialized) {
    return serialized.Error();
  }

  model_buf_ = std::move(*serialized);
  fb_model_ = tflite::FlatBufferModel::BuildFromBuffer(
      reinterpret_cast<const char*>(model_buf_.Data()), model_buf_.Size());
  if (fb_model_ == nullptr) {
    return Unexpected(kLiteRtStatusErrorFileIO,
                      "Failed to build flatbuffer from buffer");
  }

  return {};
}

namespace {

// A utility class that allows appending additional compilation options, but
// only for the duration of a scope.
class ScopedCompilationOptionsModifier {
 public:
  explicit ScopedCompilationOptionsModifier(LiteRtOptions compilation_options)
      : accelerator_options_(compilation_options->options) {}

  ~ScopedCompilationOptionsModifier() {
    // Remove any option that was appended during the lifetime of this object.
    while (--num_appended_options_ >= 0) {
      accelerator_options_.Pop();
    }
  }

  Expected<void> Append(litert::OpaqueOptions&& accelerator_options) {
    auto status = accelerator_options_.Append(std::move(accelerator_options));
    if (status) {
      ++num_appended_options_;
    }
    return status;
  }

 private:
  litert::OpaqueOptions& accelerator_options_;
  int num_appended_options_ = 0;
};

int GetAllocationFd(const tflite::Allocation* allocation) {
  if (allocation != nullptr &&
      allocation->type() == tflite::Allocation::Type::kMMap) {
    auto& mmap_allocation =
        static_cast<const tflite::MMAPAllocation&>(*allocation);
    return mmap_allocation.fd();
  }
  return -1;
}

}  // namespace

Expected<LiteRtCompiledModelT::Ptr> LiteRtCompiledModelT::Create(
    LiteRtEnvironmentT* env, LiteRtModel model,
    LiteRtOptions jit_compilation_options) {
  if (!jit_compilation_options) {
    return litert::ErrorStatusBuilder::InvalidArgument()
           << "No compilation options passed.";
  }

  auto compiled_model = std::make_unique<LiteRtCompiledModelT>(env);

  LiteRtHwAcceleratorSet hardware_accelerators = kLiteRtHwAcceleratorNone;
  LITERT_RETURN_IF_ERROR(LiteRtGetOptionsHardwareAccelerators(
      jit_compilation_options, &hardware_accelerators));

  if (hardware_accelerators == kLiteRtHwAcceleratorNone) {
    return litert::ErrorStatusBuilder::InvalidArgument()
           << "No acceleration provided.";
  }

  LITERT_RETURN_IF_ERROR(
      compiled_model->InitializeModel(*model, hardware_accelerators, *env));

  LITERT_RETURN_IF_ERROR(
      compiled_model->InitializeRuntime(env, jit_compilation_options));
  if (compiled_model->GetModelBase() == nullptr) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to initialize model memory.");
  }

  ScopedCompilationOptionsModifier scoped_modifier(jit_compilation_options);

  {
    // Add information about the model allocation to the opaque chain.
    LITERT_ASSIGN_OR_RETURN(auto dispatch_options,
                            DispatchDelegateOptions::Create());
    LITERT_RETURN_IF_ERROR(
        dispatch_options.SetAllocBase(compiled_model->GetModelBase()));
    LITERT_RETURN_IF_ERROR(dispatch_options.SetAllocBaseFd(
        GetAllocationFd(compiled_model->fb_model_->allocation())));
    LITERT_RETURN_IF_ERROR(scoped_modifier.Append(std::move(dispatch_options)));
  }

  // Apply accelerators matching the requested hardware support to the
  // model in the order they were registered.
  for (auto& accelerator : env->GetAcceleratorRegistry()) {
    LITERT_DEBUG_CODE({
      const char* accelerator_name = nullptr;
      if (accelerator->GetName(accelerator.get(), &accelerator_name) !=
              kLiteRtStatusOk ||
          !accelerator_name) {
        LITERT_LOG(LITERT_WARNING, "Failed to get name for accelerator");
      } else {
        LITERT_LOG(LITERT_DEBUG, "Apply accelerator %s", accelerator_name);
      }
    });

    bool delegate_responsible_for_jit = false;
    LITERT_RETURN_IF_ERROR(
        LiteRtIsAcceleratorDelegateResponsibleForJitCompilation(
            accelerator.get(), &delegate_responsible_for_jit));

    LiteRtHwAcceleratorSet accelerator_supported_hardware;
    LITERT_RETURN_IF_ERROR(accelerator->GetHardwareSupport(
        accelerator.get(), &accelerator_supported_hardware));

    // We don't apply the delegate if:
    //   - the delegate is responsible for JIT compilation
    //   - and JIT has not been requested for the hardware it supports.
    if (delegate_responsible_for_jit &&
        !(hardware_accelerators & accelerator_supported_hardware)) {
      continue;
    }

    TfLiteOpaqueDelegate* delegate_ptr = nullptr;
    LITERT_RETURN_IF_ERROR(
        accelerator->CreateDelegate(accelerator.get(), jit_compilation_options,
                                    reinterpret_cast<void**>(&delegate_ptr)));

    auto delegate = tflite::TfLiteOpaqueDelegateUniquePtr(
        delegate_ptr, reinterpret_cast<void (*)(TfLiteOpaqueDelegate*)>(
                          accelerator->DestroyDelegate));

    if (compiled_model->interp_->ModifyGraphWithDelegate(delegate_ptr) !=
        kTfLiteOk) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Failed to modify graph with delegate");
    }

    compiled_model->RegisterDelegate({std::move(delegate),
                                      accelerator->StartMetricsCollection,
                                      accelerator->StopMetricsCollection});
  }

  LITERT_ASSIGN_OR_RETURN(bool has_non_delegated_ops,
                          compiled_model->HasNonDelegatedOps());
  if (!(hardware_accelerators & kLiteRtHwAcceleratorCpu) &&
      has_non_delegated_ops) {
    return litert::Error(
        kLiteRtStatusErrorCompilation,
        "Some ops are not accelerated. Add kLiteRtHwAcceleratorCpu to the "
        "compilation accelerator set to allow using the CPU to run those.");
  }
  compiled_model->CheckCpuTensors();
  return compiled_model;
}

Expected<bool> LiteRtCompiledModelT::HasNonDelegatedOps() {
  for (int subgraph_no = 0; subgraph_no < interp_->subgraphs_size();
       ++subgraph_no) {
    const auto* const subgraph = interp_->subgraph(subgraph_no);
    if (subgraph->IsDelegationSkippable()) {
      continue;
    }
    const auto& execution_plan = subgraph->execution_plan();
    const auto& nodes_and_registration = subgraph->nodes_and_registration();
    for (int execution_plan_index = 0;
         execution_plan_index < execution_plan.size(); execution_plan_index++) {
      const int node_index = execution_plan[execution_plan_index];
      const TfLiteRegistration& registration =
          nodes_and_registration[node_index].second;
      if (registration.builtin_code != kTfLiteBuiltinDelegate &&
          (registration.builtin_code != kTfLiteBuiltinCustom ||
           litert::internal::kLiteRtDispatchOpCustomName !=
               registration.custom_name)) {
        return true;
      }
    }
  }
  return false;
}

void LiteRtCompiledModelT::CheckCpuTensors() {
  cpu_tensors_.clear();
  for (int subgraph_no = 0; subgraph_no < interp_->subgraphs_size();
       ++subgraph_no) {
    auto* subgraph = interp_->subgraph(subgraph_no);
    auto& execution_plan = subgraph->execution_plan();
    auto& nodes_and_registration = subgraph->nodes_and_registration();
    for (int execution_plan_index = 0;
         execution_plan_index < execution_plan.size(); execution_plan_index++) {
      int node_index = execution_plan[execution_plan_index];
      const TfLiteNode& node = nodes_and_registration[node_index].first;
      const TfLiteRegistration& registration =
          nodes_and_registration[node_index].second;
      // Don't mark delegate nodes as CPU nodes except for XNNPack ones.
      if (registration.builtin_code == kTfLiteBuiltinDelegate &&
          !(registration.custom_name &&
            registration.custom_name ==
                absl::string_view("TfLiteXNNPackDelegate"))) {
        continue;
      }
      // Don't mark AOT compiled NPU custom ops as CPU nodes.
      if (registration.builtin_code == kTfLiteBuiltinCustom &&
          registration.custom_name &&
          absl::StrContains(registration.custom_name,
                            litert::internal::kLiteRtDispatchOpCustomName)) {
        continue;
      }
      // Mark input of node as CPU tensors.
      for (int i = 0; i < node.inputs->size; ++i) {
        int input_tensor_index = node.inputs->data[i];
        if (input_tensor_index == kTfLiteOptionalTensor) continue;
        cpu_tensors_.insert(subgraph->tensor(input_tensor_index));
      }
    }
  }
}

litert::Expected<LiteRtTensorBufferRequirements>
LiteRtCompiledModelT::GetTensorBufferRequirements(const TfLiteTensor* tensor) {
  // Use the buffer context to get the buffer requirements only if the tensor
  // is not a CPU tensor.
  if (cpu_tensors_.find(tensor) == cpu_tensors_.end()) {
    auto requirements = buffer_context_->GetBufferRequirements(tensor);
    if (requirements) {
      return (*requirements)->Get();
    }
  } else {
    LITERT_LOG(LITERT_VERBOSE, "Tensor %s is shared with CPU.\n", tensor->name);
  }
  LiteRtTensorBufferRequirements litert_cpu_buffer_requirements;
  LiteRtTensorBufferType cpu_buffer_type[] = {
      kLiteRtTensorBufferTypeHostMemory};
  uint32_t cpu_buffer_strides[] = {0};
  auto res = LiteRtCreateTensorBufferRequirements(
      /*num_supported_tensor_buffer_types=*/1, cpu_buffer_type, tensor->bytes,
      /*num_strides=*/1, cpu_buffer_strides, &litert_cpu_buffer_requirements);
  if (res != kLiteRtStatusOk) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to create CPU buffer requirements");
  }
  cpu_buffer_requirements_[tensor] = litert::TensorBufferRequirements(
      litert_cpu_buffer_requirements, litert::OwnHandle::kYes);
  return litert_cpu_buffer_requirements;
}

Expected<LiteRtTensorBufferRequirements>
LiteRtCompiledModelT::GetInputBufferRequirements(
    absl::string_view signature_key, size_t input_index) {
  auto runner = GetSignatureRunner(signature_key);
  if (runner == nullptr) {
    return Unexpected(kLiteRtStatusErrorNotFound,
                      "Failed to get signature runner");
  }
  auto input_names = runner->subgraph_input_names();
  if (input_index >= input_names.size()) {
    return Unexpected(kLiteRtStatusErrorIndexOOB, "Input index out of range");
  }
  auto input_name = input_names[input_index];
  auto* input_tensor = runner->input_tensor(input_name);
  if (input_tensor == nullptr) {
    return Unexpected(kLiteRtStatusErrorNotFound, "Failed to get input tensor");
  }

  return GetTensorBufferRequirements(input_tensor);
}

Expected<LiteRtTensorBufferRequirements>
LiteRtCompiledModelT::GetOutputBufferRequirements(
    absl::string_view signature_key, size_t output_index) {
  auto runner = GetSignatureRunner(signature_key);
  if (runner == nullptr) {
    return Unexpected(kLiteRtStatusErrorNotFound,
                      "Failed to get signature runner");
  }
  auto output_names = runner->subgraph_output_names();
  if (output_index >= output_names.size()) {
    return Unexpected(kLiteRtStatusErrorIndexOOB, "Output index out of range");
  }
  auto output_name = output_names[output_index];
  auto* output_tensor = runner->output_tensor(output_name);
  if (output_tensor == nullptr) {
    return Unexpected(kLiteRtStatusErrorNotFound,
                      "Failed to get output tensor");
  }

  return GetTensorBufferRequirements(output_tensor);
}

tflite::SignatureRunner* LiteRtCompiledModelT::GetSignatureRunner(
    absl::string_view signature_key) {
  if (signature_runners_.contains(signature_key)) {
    return signature_runners_[signature_key];
  }
  auto runner = interp_->GetSignatureRunner(
      signature_key == LiteRtSignatureT::kDefaultSignatureKey
          ? nullptr
          : std::string(signature_key).c_str());
  signature_runners_[signature_key] = runner;
  return runner;
}

Expected<void> LiteRtCompiledModelT::RegisterBuffer(
    tflite::SignatureRunner* runner, TfLiteTensor* tensor,
    const char* tensor_name, LiteRtTensorBuffer buffer, bool is_input,
    std::vector<LiteRtTensorBuffer>& locked_buffers) {
  LITERT_DEBUG_CODE({
    absl::string_view io = is_input ? "input" : "output";
    absl::string_view name = tensor_name ? tensor_name : "<unnamed>";
    auto buffer_type = GetTensorBufferTypeName(*buffer);
    LITERT_LOG(LITERT_DEBUG,
               "Registering %s tensor from TfliteTensor %p to "
               "LiteRtTensorBuffer %p of type %s",
               io.data(), tensor, buffer, buffer_type.data());
  });

  bool backend_requires_cpu_buffer = false;

  auto requirements = buffer_context_->GetBufferRequirements(tensor);
  if (requirements) {
    auto supported_types = (*requirements)->SupportedTypes();
    if (!supported_types) {
      return supported_types.Error();
    }

    for (auto& type : *supported_types) {
      if (type == buffer->buffer_type()) {
        // Register tensor buffer if it can be used by the backend.
        buffer->Duplicate();
        TensorBuffer duplicated_buffer(buffer, litert::OwnHandle::kYes);
        if (auto status = buffer_context_->RegisterTensorBuffer(
                tensor, std::move(duplicated_buffer));
            status != kLiteRtStatusOk) {
          return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                            "Failed to register tensor buffer");
        }
        // Mark the tensor as non-CPU to avoid TFLite from allocating it.
        tensor->allocation_type = kTfLiteNonCpu;
        tensor->data.data = nullptr;
        return {};
      }
      if (type == kLiteRtTensorBufferTypeHostMemory) {
        backend_requires_cpu_buffer = true;
      }
    }
  } else {
    // If the BufferRequirement is not registered, assumes the backend requires
    // CPU buffer.
    backend_requires_cpu_buffer = true;
  }

  if (backend_requires_cpu_buffer) {
    // When backend requires CPU buffer.
    bool buffer_is_cpu_compatible =
        buffer->buffer_type() == kLiteRtTensorBufferTypeHostMemory ||
        buffer->is_opencl_memory();
#if defined(__ANDROID__)
    if (buffer->buffer_type() == kLiteRtTensorBufferTypeAhwb) {
      if (__builtin_available(android 26, *)) {
        auto ahwb = buffer->GetAhwbBuffer();
        if (ahwb) {
          // TODO: b/382330322 - Update logic to check if the AHWB (stride) is
          // CPU compatible.
          AHardwareBuffer_Desc desc;
          AHardwareBuffer_describe(*ahwb, &desc);
          buffer_is_cpu_compatible = true;
        }
      }
    } else if (buffer->buffer_type() == kLiteRtTensorBufferTypeFastRpc) {
      buffer_is_cpu_compatible = true;
    }
#endif
    if (buffer_is_cpu_compatible) {
      void* host_mem_addr;
      LiteRtTensorBufferLockMode lock_mode =
          is_input ? kLiteRtTensorBufferLockModeRead
                   : kLiteRtTensorBufferLockModeWrite;
      if (auto status =
              LiteRtLockTensorBuffer(buffer, &host_mem_addr, lock_mode);
          status != kLiteRtStatusOk) {
        return Unexpected(status, "Failed to lock the tensor buffer");
      }
      TfLiteCustomAllocation custom_allocation{host_mem_addr, tensor->bytes};
      if (is_input) {
        runner->SetCustomAllocationForInputTensor(tensor_name,
                                                  custom_allocation,
                                                  /*flags=*/0);
        // TODO: b/419350199 - Ad-hoc solution to unlock input buffers.
        LITERT_RETURN_IF_ERROR(LiteRtUnlockTensorBuffer(buffer));
      } else {
        locked_buffers.push_back(buffer);
        runner->SetCustomAllocationForOutputTensor(tensor_name,
                                                   custom_allocation,
                                                   /*flags=*/0);
      }
      return {};
    }
  }

  // If the tensor is shared with CPU, register tensor buffer as is and let
  // accelerator handle the conversion.
  if (cpu_tensors_.find(tensor) != cpu_tensors_.end()) {
    void* host_mem_addr;
    if (auto status = LiteRtLockTensorBuffer(
            buffer, &host_mem_addr, kLiteRtTensorBufferLockModeReadWrite);
        status != kLiteRtStatusOk) {
      return Unexpected(status, "Failed to lock the tensor buffer");
    }
    locked_buffers.push_back(buffer);
    TfLiteCustomAllocation custom_allocation{host_mem_addr, tensor->bytes};
    if (is_input) {
      runner->SetCustomAllocationForInputTensor(tensor_name, custom_allocation,
                                                /*flags=*/0);
    } else {
      runner->SetCustomAllocationForOutputTensor(tensor_name, custom_allocation,
                                                 /*flags=*/0);
    }
    return {};
  }
  // TODO: b/382330322 - Add buffer conversion logic instead of returning error.
  return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                    "The given buffer type is not supported.");
}

Expected<void> LiteRtCompiledModelT::Run(
    absl::string_view signature_key,
    const std::vector<LiteRtTensorBuffer>& input_buffers,
    const std::vector<LiteRtTensorBuffer>& output_buffers, bool& async) {
  auto runner = GetSignatureRunner(signature_key);
  if (runner == nullptr) {
    return Unexpected(kLiteRtStatusErrorNotFound,
                      "Failed to get signature runner");
  }
  size_t num_inputs = input_buffers.size();
  if (num_inputs != runner->subgraph_input_names().size()) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Input buffer size mismatch");
  }
  size_t num_outputs = output_buffers.size();
  if (num_outputs != runner->subgraph_output_names().size()) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Output buffer size mismatch");
  }

  // In general output buffer events are assigned by the runtime and not the
  // caller; here we check for any violation of that condition.
  for (auto litert_output_buffer : output_buffers) {
    if (litert_output_buffer->HasEvent()) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Output buffers cannot have events attached");
    }
  }

  // The collection of locked buffers. It is used to unlock the buffers after
  // the inference is done.
  std::vector<LiteRtTensorBuffer> locked_buffers;
  locked_buffers.reserve(num_inputs + num_outputs);
  auto unlock_buffers = absl::MakeCleanup([&locked_buffers]() {
    for (auto locked_buffer : locked_buffers) {
      if (LiteRtUnlockTensorBuffer(locked_buffer) != kLiteRtStatusOk) {
        LITERT_LOG(LITERT_ERROR, "Failed to unlock buffer %p", locked_buffer);
        ABSL_DCHECK(false);
      }
    }
  });
  for (int i = 0; i < num_inputs; ++i) {
    const auto& input_name = runner->subgraph_input_names()[i];
    auto* input_tensor = runner->input_tensor(input_name);
    auto res =
        RegisterBuffer(runner, input_tensor, input_name, input_buffers[i],
                       /*is_input=*/true, locked_buffers);
    if (!res) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        absl::StrCat("Failed to register input tensor buffer: ",
                                     res.Error().Message()));
    }
  }

  for (int i = 0; i < runner->subgraph_output_names().size(); ++i) {
    const auto& output_name = runner->subgraph_output_names()[i];
    auto* output_tensor = runner->output_tensor(output_name);
    auto res = RegisterBuffer(runner, const_cast<TfLiteTensor*>(output_tensor),
                              output_name, output_buffers[i],
                              /*is_input=*/false, locked_buffers);
    if (!res) {
      return Unexpected(
          kLiteRtStatusErrorRuntimeFailure,
          absl::StrCat("Failed to register output tensor buffer: ",
                       res.Error().Message()));
    }
  }

  if (auto res = runner->AllocateTensors(); res != kTfLiteOk) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to allocate tensors");
  }

  // Relay the intended async execution mode to DelegateKernel of Accelerator.
  buffer_context_->SetAsyncExecutionMode(async);

  if (auto res = runner->Invoke(); res != kTfLiteOk) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure, "Failed to invoke");
  }

  if (async) {
    // If the caller requested async execution, then set async to true if any of
    // the output buffers have been assigned a synchronization event.
    async = false;
    for (auto& tb : output_buffers) {
      async |= tb->HasEvent();
    }
  } else {
    // If the caller has not requested async execution, then wait on
    // synchronization events that have been attached to the outputs.
    for (auto& tb : output_buffers) {
      if (tb->HasEvent()) {
        auto event = tb->GetEvent();
        if (auto status = litert::Event(*event, litert::OwnHandle::kNo)
                              .Wait(/*timeout_in_ms=*/-1);
            !status) {
          return status;
        }
      }
    }
  }

  return {};
}

litert::Expected<void> LiteRtCompiledModelT::RunCApi(
    size_t signature_index, size_t num_input_buffers,
    LiteRtTensorBuffer* input_buffers, size_t num_output_buffers,
    LiteRtTensorBuffer* output_buffers, bool* async) {
  if (signature_index >= signature_keys_.size()) {
    return litert::Unexpected(
        kLiteRtStatusErrorIndexOOB,
        "Signature index is out of range of signature keys");
  }
  std::vector<LiteRtTensorBuffer> input_buffers_vec;
  input_buffers_vec.reserve(num_input_buffers);
  for (int i = 0; i < num_input_buffers; ++i) {
    input_buffers_vec.push_back(std::move(input_buffers[i]));
  }
  std::vector<LiteRtTensorBuffer> output_buffers_vec;
  output_buffers_vec.reserve(num_output_buffers);
  for (int i = 0; i < num_output_buffers; ++i) {
    output_buffers_vec.push_back(std::move(output_buffers[i]));
  }
  bool async_ = async ? *async : false;
  auto result = Run(*signature_keys_[signature_index], input_buffers_vec,
                    output_buffers_vec, async_);
  if (async) {
    *async = async_;
  }
  return result;
}

litert::Expected<void> LiteRtCompiledModelT::StartMetricsCollection(
    int detail_level) {
  if (detail_level < 0) {
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                              "Detail level must be >= 0");
  }
  for (auto& delegate : delegates_) {
    if (delegate.StartMetricsCollection) {
      LITERT_RETURN_IF_ERROR(delegate.StartMetricsCollection(
          delegate.delegate.get(), detail_level));
    }
  }
  return {};
}

litert::Expected<LiteRtMetricsT> LiteRtCompiledModelT::StopMetricsCollection() {
  std::vector<LiteRtMetricsT::Metric> metrics;
  for (auto& delegate : delegates_) {
    if (delegate.StopMetricsCollection) {
      LiteRtMetricsT accelerator_metrics;
      LITERT_RETURN_IF_ERROR(delegate.StopMetricsCollection(
          delegate.delegate.get(), &accelerator_metrics));
      metrics.insert(
          metrics.end(),
          std::make_move_iterator(accelerator_metrics.metrics.begin()),
          std::make_move_iterator(accelerator_metrics.metrics.end()));
    }
  }
  return LiteRtMetricsT{.metrics = std::move(metrics)};
}
