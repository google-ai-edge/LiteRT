// Copyright 2026 Google LLC.
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

#include "litert/runtime/compiled_model_dispatch.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <limits>
#include <utility>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "litert/c/internal/litert_logging.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/build_stamp.h"
#include "litert/core/dispatch_op_schema.h"
#include "litert/core/model/model.h"
#include "litert/core/options.h"
#include "litert/core/util/tensor_type_util.h"
#include "litert/runtime/tensor_buffer.h"
#include "litert/runtime/tensor_buffer_requirements.h"
#include "litert/vendors/c/litert_dispatch.h"

using litert::Expected;
using litert::Unexpected;

struct LiteRtCompiledModelT::Signature {
  struct Port {
    LiteRtRankedTensorType type{};
    int dispatch_index = 0;
    std::unique_ptr<LiteRtTensorBufferRequirementsT> requirements;
    LiteRtTensorBuffer buffer = nullptr;
    LiteRtTensorBufferHandle handle = 0;
    bool registered = false;
    bool attached = false;
  };

  // Qualcomm associates buffer registrations with a device's current context.
  // Give each signature its own device so different context binaries cannot
  // accidentally register memory against another signature's QNN context.
  LiteRtDispatchDeviceContext device = nullptr;
  LiteRtDispatchInvocationContext invocation = nullptr;
  LiteRtMemBuffer bytecode{};
  std::string function_name;
  std::vector<Port> inputs;
  std::vector<Port> outputs;

  Expected<void> Release(Port& port, bool input) {
    if (port.attached) {
      LITERT_RETURN_IF_ERROR(
          input ? LiteRtDispatchDetachInput(invocation, port.dispatch_index,
                                            port.handle)
                : LiteRtDispatchDetachOutput(invocation, port.dispatch_index,
                                             port.handle));
      port.attached = false;
    }
    if (port.registered) {
      LITERT_RETURN_IF_ERROR(
          LiteRtDispatchUnregisterTensorBuffer(device, port.handle));
      port.registered = false;
    }
    if (port.buffer) LiteRtDestroyTensorBuffer(port.buffer);
    port.buffer = nullptr;
    return {};
  }

  ~Signature() {
    for (bool input : {true, false}) {
      for (auto& port : input ? inputs : outputs) {
        if (auto result = Release(port, input); !result) {
          LITERT_LOG(LITERT_ERROR, "Failed to release dispatch buffer: %s",
                     result.Error().Message().c_str());
        }
      }
    }
    if (invocation) LiteRtDispatchInvocationContextDestroy(invocation);
    if (device) LiteRtDispatchDeviceContextDestroy(device);
    // On a detach/unregister failure retain the backing allocation until the
    // vendor contexts are gone. A replacement must never free registered
    // memory.
    for (auto& port : inputs)
      if (port.buffer) LiteRtDestroyTensorBuffer(port.buffer);
    for (auto& port : outputs)
      if (port.buffer) LiteRtDestroyTensorBuffer(port.buffer);
  }

  Expected<void> Bind(Port& port, LiteRtTensorBuffer buffer, bool input) {
    if (port.buffer == buffer && port.attached) return {};
    LITERT_RETURN_IF_ERROR(Release(port, input));
    LITERT_RETURN_IF_ERROR(LiteRtDuplicateTensorBuffer(buffer));
    port.buffer = buffer;
    LITERT_RETURN_IF_ERROR(
        LiteRtDispatchRegisterTensorBuffer(device, buffer, &port.handle));
    port.registered = true;
    LITERT_RETURN_IF_ERROR(
        input ? LiteRtDispatchAttachInput(invocation, port.dispatch_index,
                                          port.handle)
              : LiteRtDispatchAttachOutput(invocation, port.dispatch_index,
                                           port.handle));
    port.attached = true;
    return {};
  }

  static Expected<void> ValidateBuffer(const Port& port,
                                       LiteRtTensorBuffer buffer) {
    if (!buffer || buffer->buffer_type() != kLiteRtTensorBufferTypeFastRpc) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "Qualcomm AOT requires FastRPC tensor buffers");
    }
    // FastRPC Lock currently returns the allocation base, so the minimal
    // profile accepts whole buffers only, not views with a nonzero offset.
    if (buffer->buffer_offset() != 0) return Unsupported();
    const auto type = buffer->tensor_type();
    bool same_layout = false;
    LITERT_RETURN_IF_ERROR(
        LiteRtIsSameLayout(&type.layout, &port.type.layout, &same_layout));
    if (type.element_type != port.type.element_type || !same_layout) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "Tensor buffer type or shape does not match the model");
    }
    if (buffer->buffer_offset() > buffer->buffer_size() ||
        port.requirements->BufferSize() >
            buffer->buffer_size() - buffer->buffer_offset() ||
        buffer->buffer_offset() % port.requirements->Alignment() != 0) {
      return Unexpected(
          kLiteRtStatusErrorInvalidArgument,
          "Tensor buffer is too small or its offset is unaligned");
    }
    return {};
  }

  Expected<void> Initialize(LiteRtOptions options) {
    auto* runtime = LrtGetRuntimeContext();
    LITERT_RETURN_IF_ERROR(
        LiteRtDispatchDeviceContextCreate(runtime, options, &device));
    LITERT_RETURN_IF_ERROR(LiteRtDispatchInvocationContextCreate(
        runtime, device, kLiteRtDispatchExecutableTypeMlModel, &bytecode,
        function_name.c_str(), inputs.size(), outputs.size(), &invocation));
    for (bool input : {true, false}) {
      for (auto& port : input ? inputs : outputs) {
        LiteRtTensorBufferRequirements requirements = nullptr;
        LITERT_RETURN_IF_ERROR(input ? LiteRtDispatchGetInputRequirements(
                                           invocation, port.dispatch_index,
                                           &port.type, &requirements)
                                     : LiteRtDispatchGetOutputRequirements(
                                           invocation, port.dispatch_index,
                                           &port.type, &requirements));
        std::unique_ptr<LiteRtTensorBufferRequirementsT> owned(requirements);
        if (!owned) return Unexpected(kLiteRtStatusErrorRuntimeFailure);
        const auto& types = owned->SupportedBufferTypes();
        if (std::find(types.begin(), types.end(),
                      kLiteRtTensorBufferTypeFastRpc) == types.end() ||
            !owned->Strides().empty() || owned->Alignment() == 0) {
          return Unexpected(kLiteRtStatusErrorUnsupported,
                            "Dispatch does not support contiguous FastRPC I/O");
        }
        LITERT_ASSIGN_OR_RETURN(auto packed_size,
                                litert::internal::GetNumPackedBytes(port.type));
        if (owned->BufferSize() < packed_size) {
          return Unexpected(
              kLiteRtStatusErrorRuntimeFailure,
              "Dispatch returned an undersized buffer requirement");
        }
        const LiteRtTensorBufferType fast_rpc = kLiteRtTensorBufferTypeFastRpc;
        port.requirements = std::make_unique<LiteRtTensorBufferRequirementsT>(
            1, &fast_rpc, owned->BufferSize(), std::vector<uint32_t>{},
            owned->Alignment());
      }
    }
    return {};
  }
};

LiteRtCompiledModelT::LiteRtCompiledModelT(LiteRtEnvironment env) : env_(env) {}
LiteRtCompiledModelT::~LiteRtCompiledModelT() = default;

Expected<LiteRtCompiledModelT::Ptr> LiteRtCompiledModelT::Create(
    LiteRtEnvironment env, LiteRtModel model, LiteRtOptions options) {
  if (!env || !model || !options) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument);
  }
  if (options->hardware_accelerators != kLiteRtHwAcceleratorNpu ||
      !options->custom_op_options.empty() ||
      !options->custom_tflite_op_registrations.empty() ||
      !options->custom_tflite_op_operators.empty() ||
      !options->external_tensor_bindings.empty() || options->weight_loader ||
      options->scoped_weight_source || options->weight_in_memory_map ||
      !options->selected_signature_keys.empty()) {
    return Unexpected(
        kLiteRtStatusErrorUnsupported,
        "Qualcomm AOT requires NPU-only options and embedded context binaries");
  }
  const auto allocation = litert::internal::GetTflFlatbuffer(*model).Buf();
  if (!allocation.Data() || model->Signatures().empty()) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Expected a serialized model with a callable signature");
  }
  Ptr result(new LiteRtCompiledModelT(env));

  // Validate every signature before initializing the vendor runtime. A build
  // stamp alone is insufficient: AOT models can still contain CPU partitions.
  for (const auto* model_signature : model->Signatures()) {
    const auto& graph = model_signature->GetSubgraph();
    if (graph.Ops().size() != 1) {
      return Unexpected(kLiteRtStatusErrorUnsupported,
                        "Qualcomm AOT requires one dispatch op per signature");
    }
    const auto& op = *graph.Ops()[0];
    auto custom_code = op.CustomCode();
    if (!custom_code ||
        *custom_code != litert::internal::kLiteRtDispatchOpCustomName) {
      return Unexpected(kLiteRtStatusErrorUnsupported,
                        "Model is not fully compiled: expected DISPATCH_OP");
    }
    const auto custom = op.CustomOptions();
    if (!custom.Data() ||
        !flexbuffers::VerifyBuffer(custom.Data(), custom.Size())) {
      return Unexpected(kLiteRtStatusErrorInvalidFlatbuffer,
                        "Invalid dispatch custom options");
    }
    const auto root = flexbuffers::GetRoot(custom.Data(), custom.Size());
    const auto fields = root.AsMap();
    if (!root.IsMap() || !fields["bytecode_offset"].IsUInt() ||
        !fields["bytecode_size"].IsUInt() || !fields["name"].IsString() ||
        std::strlen(fields["name"].AsString().c_str()) !=
            fields["name"].AsString().size()) {
      return Unexpected(kLiteRtStatusErrorInvalidFlatbuffer,
                        "Invalid dispatch custom option fields");
    }
    const auto dispatch = litert::internal::GetDispatchOpOptions(custom);
    if (dispatch.bytecode_offset == 0 || dispatch.bytecode_size == 0 ||
        dispatch.bytecode_offset > allocation.Size() ||
        dispatch.bytecode_size > allocation.Size() - dispatch.bytecode_offset) {
      return Unexpected(
          kLiteRtStatusErrorInvalidFlatbuffer,
          "Dispatch context binary is outside the model allocation");
    }
    auto signature = std::make_unique<Signature>();
    signature->bytecode = {-1, allocation.Data(), dispatch.bytecode_offset,
                           dispatch.bytecode_size};
    signature->function_name = dispatch.name;
    for (bool input : {true, false}) {
      const auto& op_tensors = input ? op.Inputs() : op.Outputs();
      const auto& graph_tensors = input ? graph.Inputs() : graph.Outputs();
      const size_t count = input ? model_signature->InputNames().size()
                                 : model_signature->OutputNames().size();
      if (count != op_tensors.size() || count != graph_tensors.size() ||
          count > std::numeric_limits<int>::max()) {
        return Unexpected(kLiteRtStatusErrorUnsupported,
                          "Dispatch ports must exactly cover signature I/O");
      }
      std::vector<bool> seen(count, false);
      auto& ports = input ? signature->inputs : signature->outputs;
      for (size_t i = 0; i < count; ++i) {
        const auto* tensor = input ? model_signature->GetInputTensor(i)
                                   : model_signature->GetOutputTensor(i);
        const auto it = std::find(op_tensors.begin(), op_tensors.end(), tensor);
        if (it == op_tensors.end() ||
            std::find(graph_tensors.begin(), graph_tensors.end(), tensor) ==
                graph_tensors.end() ||
            seen[it - op_tensors.begin()]) {
          return Unexpected(
              kLiteRtStatusErrorUnsupported,
              "Signature I/O must map bijectively to dispatch ports");
        }
        const int index = it - op_tensors.begin();
        seen[index] = true;
        LITERT_ASSIGN_OR_RETURN(auto type, tensor->Ranked());
        if (type.layout.has_strides ||
            type.element_type == kLiteRtElementTypeTfString) {
          return Unsupported();
        }
        for (size_t d = 0; d < type.layout.rank; ++d) {
          if (type.layout.dimensions[d] <= 0) {
            return Unexpected(
                kLiteRtStatusErrorUnsupported,
                "Qualcomm AOT requires fixed, nonempty tensor shapes");
          }
        }
        ports.push_back(Signature::Port{type, index});
      }
    }
    result->signatures_.push_back(std::move(signature));
  }
  LITERT_RETURN_IF_ERROR(
      LiteRtDispatchInitialize(LrtGetRuntimeContext(), env, options));
  const char* vendor = nullptr;
  LITERT_RETURN_IF_ERROR(LiteRtDispatchGetVendorId(&vendor));
  if (!vendor || std::string(vendor) != "Qualcomm") {
    return Unexpected(kLiteRtStatusErrorUnsupported,
                      "Qualcomm AOT requires the Qualcomm Dispatch API");
  }
  for (auto& signature : result->signatures_) {
    LITERT_RETURN_IF_ERROR(signature->Initialize(options));
  }
  return result;
}

Expected<LiteRtCompiledModelT::Signature*> LiteRtCompiledModelT::GetSignature(
    size_t index) {
  if (index >= signatures_.size())
    return Unexpected(kLiteRtStatusErrorIndexOOB);
  return signatures_[index].get();
}

Expected<const LiteRtTensorBufferRequirementsT*>
LiteRtCompiledModelT::GetInputBufferRequirements(size_t signature_index,
                                                 size_t input_index) {
  LITERT_ASSIGN_OR_RETURN(auto* signature, GetSignature(signature_index));
  if (input_index >= signature->inputs.size())
    return Unexpected(kLiteRtStatusErrorIndexOOB);
  return signature->inputs[input_index].requirements.get();
}

Expected<LiteRtTensorBufferRequirements>
LiteRtCompiledModelT::GetOutputBufferRequirementsCApi(size_t signature_index,
                                                      size_t output_index) {
  LITERT_ASSIGN_OR_RETURN(auto* signature, GetSignature(signature_index));
  if (output_index >= signature->outputs.size())
    return Unexpected(kLiteRtStatusErrorIndexOOB);
  return signature->outputs[output_index].requirements.get();
}

Expected<LiteRtLayout> LiteRtCompiledModelT::GetInputTensorLayout(
    size_t signature_index, size_t input_index) {
  LITERT_ASSIGN_OR_RETURN(auto* signature, GetSignature(signature_index));
  if (input_index >= signature->inputs.size())
    return Unexpected(kLiteRtStatusErrorIndexOOB);
  return signature->inputs[input_index].type.layout;
}

Expected<void> LiteRtCompiledModelT::GetOutputTensorShapes(
    size_t signature_index, absl::Span<LiteRtLayout>& layouts, bool) {
  LITERT_ASSIGN_OR_RETURN(auto* signature, GetSignature(signature_index));
  if (layouts.size() != signature->outputs.size())
    return Unexpected(kLiteRtStatusErrorInvalidArgument);
  for (size_t i = 0; i < layouts.size(); ++i)
    layouts[i] = signature->outputs[i].type.layout;
  return {};
}

Expected<void> LiteRtCompiledModelT::RunCApi(
    size_t signature_index, size_t num_inputs, const LiteRtTensorBuffer* inputs,
    size_t num_outputs, const LiteRtTensorBuffer* outputs, bool* async,
    LiteRtOptions options, const LiteRtSchedulingInfo* scheduling_info) {
  if (async) *async = false;  // Qualcomm's Dispatch API is synchronous.
  if (scheduling_info) return Unsupported();
  LITERT_ASSIGN_OR_RETURN(auto* signature, GetSignature(signature_index));
  if (num_inputs != signature->inputs.size() ||
      num_outputs != signature->outputs.size() || (num_inputs && !inputs) ||
      (num_outputs && !outputs)) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Incorrect tensor buffer count");
  }
  // Validate all buffers before changing any registrations or waiting on
  // events.
  bool bindings_changed = false;
  for (size_t i = 0; i < num_inputs; ++i) {
    LITERT_RETURN_IF_ERROR(
        Signature::ValidateBuffer(signature->inputs[i], inputs[i]));
    bindings_changed |= signature->inputs[i].buffer != inputs[i];
  }
  for (size_t i = 0; i < num_outputs; ++i) {
    LITERT_RETURN_IF_ERROR(
        Signature::ValidateBuffer(signature->outputs[i], outputs[i]));
    bindings_changed |= signature->outputs[i].buffer != outputs[i];
    if (outputs[i]->HasEvent()) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "Output buffers cannot have events attached");
    }
  }
  // Qualcomm's registry deduplicates handles without reference counting. Keep
  // each port's registration independent, including when replacing one port.
  // The cached set is already checked, keeping steady-state validation linear.
  if (bindings_changed) {
    for (size_t i = 0; i < num_inputs + num_outputs; ++i) {
      const auto buffer = i < num_inputs ? inputs[i] : outputs[i - num_inputs];
      for (size_t j = 0; j < i; ++j) {
        const auto previous =
            j < num_inputs ? inputs[j] : outputs[j - num_inputs];
        if (buffer == previous) return Unsupported();
      }
    }
  }
  for (size_t i = 0; i < num_inputs; ++i) {
    if (inputs[i]->HasEvent()) {
      LITERT_ASSIGN_OR_RETURN(auto* event, inputs[i]->GetEvent());
      LITERT_RETURN_IF_ERROR(event->Wait(-1));
    }
  }
  // Release every changed port before registering replacements. This also
  // handles swapping two buffers without temporarily sharing a vendor handle.
  for (size_t i = 0; i < num_inputs; ++i) {
    if (signature->inputs[i].buffer != inputs[i]) {
      LITERT_RETURN_IF_ERROR(signature->Release(signature->inputs[i], true));
    }
  }
  for (size_t i = 0; i < num_outputs; ++i) {
    if (signature->outputs[i].buffer != outputs[i]) {
      LITERT_RETURN_IF_ERROR(signature->Release(signature->outputs[i], false));
    }
  }
  for (size_t i = 0; i < num_inputs; ++i) {
    LITERT_RETURN_IF_ERROR(
        signature->Bind(signature->inputs[i], inputs[i], true));
  }
  for (size_t i = 0; i < num_outputs; ++i) {
    LITERT_RETURN_IF_ERROR(
        signature->Bind(signature->outputs[i], outputs[i], false));
  }
  if (options) {
    LITERT_RETURN_IF_ERROR(LiteRtDispatchInvocationContextSetOptions(
        signature->invocation, options));
  }
  const auto invoke_status = LiteRtDispatchInvoke(signature->invocation);
  if (options) {
    LITERT_RETURN_IF_ERROR(LiteRtDispatchInvocationContextSetOptions(
        signature->invocation, nullptr));
  }
  LITERT_RETURN_IF_ERROR(invoke_status);
  return {};
}

void LiteRtCompiledModelT::ReportError(const char* format, ...) const {
  va_list args;
  va_start(args, format);
  std::vfprintf(stderr, format, args);
  va_end(args);
}

