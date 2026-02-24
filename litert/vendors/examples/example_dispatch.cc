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

#include <cstdint>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_scheduling_info.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/c/litert_dispatch_api.h"
#include "litert/vendors/examples/example_common.h"

namespace {
using Buffer = ::litert::example::Data;
using BufferHandle = Buffer*;

std::optional<LiteRtSchedulingInfo> LastSchedulingInfo;
}  // namespace

class LiteRtDispatchDeviceContextT {
 public:
  LiteRtDispatchDeviceContextT() = default;
  ~LiteRtDispatchDeviceContextT() = default;

  ::litert::Expected<BufferHandle> RegisterBuffer(LiteRtTensorBuffer b) {
    auto* handle = &buffers_.emplace_back();
    registered_buffers_[handle] = b;
    return handle;
  }

  ::litert::Expected<void> UnregisterBuffer(BufferHandle handle) {
    registered_buffers_.erase(handle);
    for (auto it = buffers_.begin(); it != buffers_.end(); ++it) {
      if (&*it == handle) {
        buffers_.erase(it);
        break;
      }
    }
    return {};
  }

  ::litert::TensorBuffer Lookup(BufferHandle handle) {
    return ::litert::TensorBuffer::WrapCObject(registered_buffers_[handle],
                                               ::litert::OwnHandle::kNo);
  }

 private:
  using RegistredBuffers =
      absl::flat_hash_map<BufferHandle, LiteRtTensorBuffer>;
  std::list<Buffer> buffers_;
  RegistredBuffers registered_buffers_;
};

class LiteRtDispatchInvocationContextT {
 public:
  using Ptr = std::unique_ptr<LiteRtDispatchInvocationContextT>;
  static ::litert::Expected<LiteRtDispatchInvocationContextT::Ptr> Create(
      LiteRtDispatchDeviceContext device_context,
      LiteRtDispatchExecutableType exec_type,
      const LiteRtMemBuffer* exec_bytecode_buffer, const char* function_name,
      int num_inputs, int num_outputs) {
    if (device_context == nullptr || exec_bytecode_buffer == nullptr ||
        function_name == nullptr) {
      return ::litert::Error(kLiteRtStatusErrorInvalidArgument,
                             "Inputs are null");
    }
    LITERT_ASSIGN_OR_RETURN(
        auto global_graph,
        ::litert::example::ExampleGlobalGraph::Parse(
            ::litert::BufferRef<uint8_t>(
                exec_bytecode_buffer->base_addr,
                exec_bytecode_buffer->offset + exec_bytecode_buffer->size,
                exec_bytecode_buffer->offset)));

    // Find the subgraph.
    if (global_graph.subgraphs_.find(function_name) ==
        global_graph.subgraphs_.end()) {
      return litert::Error(kLiteRtStatusErrorNotFound, "Subgraph not found");
    }
    auto& example_graph = global_graph.subgraphs_.at(function_name);

    if (example_graph.version() != "1") {
      return litert::Error(kLiteRtStatusErrorUnsupportedCompilerVersion,
                           "Bytecode version is not compatible");
    }

    auto context = Ptr(new LiteRtDispatchInvocationContextT(
        device_context, exec_type, absl::string_view(function_name),
        std::move(global_graph)));
    context->Initialize();
    return context;
  }

  void Initialize() {
    auto& example_graph =
        global_graph_.subgraphs_.at(std::string(function_name_));
    example_graph_ptr_ = &example_graph;
    // Build map from Tensor Index to Input Index
    absl::flat_hash_map<int, int> tensor_to_input_idx;
    const auto& graph_inputs = example_graph_ptr_->Inputs();
    for (int i = 0; i < graph_inputs.size(); ++i) {
      tensor_to_input_idx[graph_inputs[i]] = i;
    }

    // Pre-load constants.
    for (const auto& [tensor_idx, buf_id] : example_graph_ptr_->ConstMap()) {
      if (global_graph_.buffers_.count(buf_id)) {
        const auto& buffer_tensor = global_graph_.buffers_[buf_id];
        if (tensor_to_input_idx.contains(tensor_idx)) {
          int input_idx = tensor_to_input_idx[tensor_idx];
          constant_buffers_.push_back(buffer_tensor.data);
          inputs_[input_idx] = &constant_buffers_.back();
        } else {
          // Internal constant, populate data in ExampleGraph tensor
          if (tensor_idx < example_graph.MutableTensors().size()) {
            example_graph.MutableTensors()[tensor_idx].data =
                buffer_tensor.data;
          }
        }
      }
    }
  }

  absl::string_view FunctionName() const { return function_name_; }

  LiteRtDispatchDeviceContextT& DeviceContext() const {
    return *device_context_;
  }

  LiteRtDispatchExecutableType ExecType() const { return exec_type_; }

  void AttachInput(int graph_input_index, BufferHandle handle) {
    inputs_[graph_input_index] = handle;
  }

  void AttachOutput(int graph_output_index, BufferHandle handle) {
    outputs_[graph_output_index] = handle;
  }

  const Buffer& GetInput(int graph_input_index) const {
    return *inputs_[graph_input_index];
  }

  Buffer& GetOutput(int graph_output_index) const {
    return *outputs_[graph_output_index];
  }

  void Setup() {
    const auto& const_map = example_graph_ptr_->ConstMap();
    const auto& graph_inputs = example_graph_ptr_->Inputs();
    for (int i = 0; i < inputs_.size(); ++i) {
      if (const_map.count(graph_inputs[i])) {
        continue;
      }
      auto* input = inputs_[i];
      ::litert::TensorBuffer buffer = device_context_->Lookup(input);
      std::vector<float> input_data(4);
      buffer.Read(absl::MakeSpan(input_data));
      const auto packed_size = buffer.PackedSize();
      input->resize(*packed_size / sizeof(float));
      buffer.Read(absl::MakeSpan(input->data(), input->size()));
    }
  }

  void Finish() {
    for (auto* output : outputs_) {
      ::litert::TensorBuffer buffer = device_context_->Lookup(output);
      buffer.Write(absl::MakeConstSpan(output->data(), output->size()));
    }
  }

  // Stores per-invocation options to be used for the next invocation.
  //
  // For this example implementation, we uses the "Hardware Accelerators"
  // option as an example to demonstrate how to pass per rn option.
  // If the option is provided, it is stored and used to control the behavior
  // of the next call to `Invoke`.
  // This is only a reference implementation to demonstrate the API use case.
  void SetRunOptions(LiteRtOptions options) {
    LiteRtHwAcceleratorSet accelerators = kLiteRtHwAcceleratorNone;
    if (options) {
      // Best-effort: if the option isn't present or is invalid, fall back to
      // "no per-run override" semantics.
      if (LiteRtGetOptionsHardwareAccelerators(options, &accelerators) !=
          kLiteRtStatusOk) {
        accelerators = kLiteRtHwAcceleratorNone;
      }
    }
    run_accelerators_ = accelerators;
  }

  LiteRtHwAcceleratorSet GetRunAccelerators() const {
    return run_accelerators_;
  }

  const ::litert::example::ExampleGraph& ExampleGraph() const {
    return *example_graph_ptr_;
  }

  void SetSchedulingInfo(const LiteRtSchedulingInfo* scheduling_info) {
    if (scheduling_info == nullptr) {
      has_scheduling_info_ = false;
      scheduling_info_ = LiteRtSchedulingInfo{};
      return;
    }
    has_scheduling_info_ = true;
    scheduling_info_ = *scheduling_info;
  }

  const LiteRtSchedulingInfo* GetSchedulingInfo() const {
    return has_scheduling_info_ ? &scheduling_info_ : nullptr;
  }

  ~LiteRtDispatchInvocationContextT() = default;

 private:
  LiteRtDispatchInvocationContextT(
      LiteRtDispatchDeviceContext device_context,
      LiteRtDispatchExecutableType exec_type, absl::string_view function_name,
      ::litert::example::ExampleGlobalGraph global_graph)
      : device_context_(device_context),
        exec_type_(exec_type),
        function_name_(function_name),
        global_graph_(std::move(global_graph)) {
    // We cannot access example_graph_ptr_ yet as it points into global_graph_.
    // Initialize() sets it up.
    // Use the subgraph from global_graph_ to size inputs/outputs.
    const auto& example_graph =
        global_graph_.subgraphs_.at(std::string(function_name_));
    inputs_.resize(example_graph.Inputs().size());
    outputs_.resize(example_graph.Outputs().size());
  }

  LiteRtDispatchDeviceContext device_context_;
  LiteRtDispatchExecutableType exec_type_;
  absl::string_view function_name_;
  std::vector<BufferHandle> inputs_;
  std::vector<BufferHandle> outputs_;
  ::litert::example::ExampleGlobalGraph global_graph_;
  const ::litert::example::ExampleGraph* example_graph_ptr_ = nullptr;
  std::list<Buffer> constant_buffers_;

  bool has_scheduling_info_ = false;
  LiteRtSchedulingInfo scheduling_info_{};
  LiteRtHwAcceleratorSet run_accelerators_ = kLiteRtHwAcceleratorNone;
};

namespace litert::example {
namespace {

LiteRtEnvironment the_environment = nullptr;
LiteRtOptions the_options = nullptr;

LiteRtStatus GetVendorId(const char** vendor_id) {
  *vendor_id = "Example";
  return kLiteRtStatusOk;
}

LiteRtStatus GetBuildId(const char** build_id) {
  *build_id = "ExampleBuild";
  return kLiteRtStatusOk;
}

LiteRtStatus GetCapabilities(int* capabilities) {
  *capabilities = kLiteRtDispatchCapabilitiesBasic;
  return kLiteRtStatusOk;
}

LiteRtStatus Initialize(LiteRtEnvironment env, LiteRtOptions options) {
  the_environment = env;
  the_options = options;
  return kLiteRtStatusOk;
}

LiteRtStatus DeviceContextCreate(LiteRtDispatchDeviceContext* device_context) {
  *device_context = new LiteRtDispatchDeviceContextT();
  return kLiteRtStatusOk;
}

LiteRtStatus DeviceContextDestroy(LiteRtDispatchDeviceContext device_context) {
  delete device_context;
  return kLiteRtStatusOk;
}

Expected<LiteRtTensorBufferRequirements> GetTensorBufferRequirements(
    const LiteRtRankedTensorType& tensor_type) {
  RankedTensorType t(tensor_type);
  if (t.Layout().HasStrides()) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Tensor strides are not supported by QNN");
  }
  std::vector<LiteRtTensorBufferType> buffer_types_c = {
      kLiteRtTensorBufferTypeHostMemory};
  LITERT_ASSIGN_OR_RETURN(const auto size, t.Bytes());

  LiteRtTensorBufferRequirements requirements;
  LITERT_RETURN_IF_ERROR(LiteRtCreateTensorBufferRequirements(
      buffer_types_c.size(), buffer_types_c.data(), size, 0, nullptr,
      &requirements));
  return requirements;
}

LiteRtStatus GetInputRequirements(
    LiteRtDispatchInvocationContext invocation_context, int input_index,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements* tensor_buffer_requirements) {
  LITERT_ASSIGN_OR_RETURN(auto requirements,
                          GetTensorBufferRequirements(*tensor_type));
  *tensor_buffer_requirements = requirements;
  return kLiteRtStatusOk;
}

LiteRtStatus GetOutputRequirements(
    LiteRtDispatchInvocationContext invocation_context, int output_index,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements* tensor_buffer_requirements) {
  LITERT_ASSIGN_OR_RETURN(auto requirements,
                          GetTensorBufferRequirements(*tensor_type));
  *tensor_buffer_requirements = requirements;
  return kLiteRtStatusOk;
}

LiteRtStatus RegisterTensorBuffer(
    LiteRtDispatchDeviceContext device_context, LiteRtTensorBuffer buffer,
    LiteRtTensorBufferHandle* tensor_buffer_handle) {
  LITERT_ASSIGN_OR_RETURN(auto handle, device_context->RegisterBuffer(buffer));
  *tensor_buffer_handle = reinterpret_cast<LiteRtTensorBufferHandle>(handle);
  return kLiteRtStatusOk;
}

LiteRtStatus UnregisterTensorBuffer(LiteRtDispatchDeviceContext device_context,
                                    LiteRtTensorBufferHandle handle) {
  LITERT_RETURN_IF_ERROR((device_context->UnregisterBuffer(
      reinterpret_cast<BufferHandle>(handle))));
  return kLiteRtStatusOk;
}

LiteRtStatus InvocationContextCreate(
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableType exec_type,
    const LiteRtMemBuffer* exec_bytecode_buffer, const char* function_name,
    int num_inputs, int num_outputs,
    LiteRtDispatchInvocationContext* invocation_context) {
  LITERT_ASSIGN_OR_RETURN(auto invocation_context_ptr,
                          LiteRtDispatchInvocationContextT::Create(
                              device_context, exec_type, exec_bytecode_buffer,
                              function_name, num_inputs, num_outputs));
  *invocation_context = invocation_context_ptr.release();
  return kLiteRtStatusOk;
}

LiteRtStatus InvocationContextDestroy(
    LiteRtDispatchInvocationContext invocation_context) {
  delete invocation_context;
  return kLiteRtStatusOk;
}

LiteRtStatus InvocationContextSetSchedulingInfo(
    LiteRtDispatchInvocationContext invocation_context,
    const LiteRtSchedulingInfo* scheduling_info) {
  if (invocation_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  invocation_context->SetSchedulingInfo(scheduling_info);

  if (scheduling_info != nullptr) {
    LastSchedulingInfo = *scheduling_info;
  } else {
    LastSchedulingInfo.reset();
  }

  return kLiteRtStatusOk;
}

LiteRtStatus AttachInput(LiteRtDispatchInvocationContext invocation_context,
                         int graph_input_index,
                         LiteRtTensorBufferHandle tensor_buffer_handle) {
  invocation_context->AttachInput(
      graph_input_index, reinterpret_cast<BufferHandle>(tensor_buffer_handle));
  return kLiteRtStatusOk;
}

LiteRtStatus AttachOutput(LiteRtDispatchInvocationContext invocation_context,
                          int graph_output_index,
                          LiteRtTensorBufferHandle tensor_buffer_handle) {
  invocation_context->AttachOutput(
      graph_output_index, reinterpret_cast<BufferHandle>(tensor_buffer_handle));
  return kLiteRtStatusOk;
}

LiteRtStatus DetachInput(LiteRtDispatchInvocationContext invocation_context,
                         int graph_input_index,
                         LiteRtTensorBufferHandle tensor_buffer_handle) {
  // Don't really care about the efficiency bonus of earlier de-allocation
  // since this is an example, do nothing.
  return kLiteRtStatusOk;
}

LiteRtStatus DetachOutput(LiteRtDispatchInvocationContext invocation_context,
                          int graph_output_index,
                          LiteRtTensorBufferHandle tensor_buffer_handle) {
  // Don't really care about the efficiency bonus of earlier de-allocation
  // since this is an example, do nothing.
  return kLiteRtStatusOk;
}

// Implements the `invocation_context_set_options` hook of the Dispatch API.
//
// This function serves as the bridge between the C-based Dispatch API and the
// C++ implementation class `LiteRtDispatchInvocationContextT`. Its primary
// purpose is to receive and apply per-invocation options to a specific
// invocation context.
//
// When the LiteRT runtime needs to set options for a single execution (invoke)
// call, it uses this function from the dispatch table. The provided `options`
// are passed down to the underlying `LiteRtDispatchInvocationContextT`
// instance, allowing the dispatch plugin to modify its behavior for the next
// `Invoke` call.
//
// Parameters:
//   invocation_context: A handle to the specific invocation context object
//                       (an instance of LiteRtDispatchInvocationContextT) to
//                       which the options should be applied.
//   options:            A handle to the LiteRtOptions object containing the
//                       per-invocation settings. The dispatch plugin can query
//                       specific options from this object.
//
// Returns:
//   kLiteRtStatusOk on success, or an error code if applying the options fails.
LiteRtStatus InvocationContextSetOptions(
    LiteRtDispatchInvocationContext invocation_context, LiteRtOptions options) {
  invocation_context->SetRunOptions(options);
  return kLiteRtStatusOk;
}

LiteRtStatus Invoke(LiteRtDispatchInvocationContext invocation_context) {
  invocation_context->Setup();
  const auto num_inputs = invocation_context->ExampleGraph().Inputs().size();
  std::vector<Buffer> inputs(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    inputs[i] = invocation_context->GetInput(i);
  }
  LITERT_ASSIGN_OR_RETURN(
      auto results,
      ::litert::example::Execute(invocation_context->ExampleGraph(), inputs));

  // Apply a simple per-run behavior tweak based on the provided options:
  // if the per-run accelerator set contains CPU, scale outputs by 2.
  //
  // IMPORTANT: This section serves as a reference implementation to demonstrate
  // the usage of the per-run options API. In a real-world scenario, the
  // dispatch plugin would use the options to alter the execution flow,
  // for example, by selecting a different hardware accelerator or adjusting
  // runtime parameters.
  //
  // To facilitate testing and verification of the options propagation mechanism
  // within the LiteRT framework, we introduce an artificial behavior change
  // here. Specifically, if the `kLiteRtHwAcceleratorCpu` flag is set in the
  // per-run options, we scale all output values by a factor of 2. This allows
  // integration tests to easily check if the options were received and
  // processed correctly by this dispatch plugin.
  //
  // This scaling is purely for testing purposes and does not represent a
  // realistic use case for hardware accelerator selection. In a production
  // environment, the choice of accelerator would influence the underlying
  // computation kernels used, rather than arbitrarily changing the output
  // values.
  if (invocation_context->GetRunAccelerators() & kLiteRtHwAcceleratorCpu) {
    for (auto& output : results) {
      for (auto& v : output) {
        v *= 2.0f;
      }
    }
  }

  for (int i = 0; i < results.size(); ++i) {
    invocation_context->GetOutput(i) = std::move(results[i]);
  }

  invocation_context->Finish();
  return kLiteRtStatusOk;
}

LiteRtStatus CheckRuntimeCompatibility(LiteRtApiVersion api_version,
                                       LiteRtEnvironmentOptions env,
                                       LiteRtOptions options) {
  // Example dispatch does not depend on any runtime library, return OK.
  return kLiteRtStatusOk;
}

// /////////////////////////////////////////////////////////////////////////////

LiteRtDispatchInterface ExampleInterface = {
    /*.initialize=*/Initialize,
    /*.get_vendor_id=*/GetVendorId,
    /*.get_build_id=*/GetBuildId,
    /*.get_capabilities=*/GetCapabilities,
    /*.device_context_create=*/DeviceContextCreate,
    /*.device_context_destroy=*/DeviceContextDestroy,
    /*.get_input_requirements=*/GetInputRequirements,
    /*.get_output_requirements=*/GetOutputRequirements,
    /*.register_tensor_buffer=*/RegisterTensorBuffer,
    /*.unregister_tensor_buffer=*/UnregisterTensorBuffer,
    /*.invocation_context_create=*/InvocationContextCreate,
    /*.invocation_context_destroy=*/InvocationContextDestroy,
    /*.invocation_context_set_scheduling_info=*/
    InvocationContextSetSchedulingInfo,
    /*.attach_input=*/AttachInput,
    /*.attach_output=*/AttachOutput,
    /*.detach_input=*/DetachInput,
    /*.detach_output=*/DetachOutput,
    /*.invoke=*/Invoke,
    /*.start_metrics_collection=*/nullptr,
    /*.stop_metrics_collection=*/nullptr,
    /*.get_num_metrics=*/nullptr,
    /*.get_metric=*/nullptr,
    /*.destroy_metrics=*/nullptr,
    /*.check_runtime_compatibility=*/CheckRuntimeCompatibility,
    /*.invocation_context_set_options=*/InvocationContextSetOptions,
};

LiteRtDispatchApi ExampleApi = {
    /*.version=*/{/*.major=*/LITERT_API_VERSION_MAJOR,
                  /*.minor=*/LITERT_API_VERSION_MINOR,
                  /*.patch=*/LITERT_API_VERSION_PATCH},
    /*.interface=*/&ExampleInterface,
    /*.async_interface=*/nullptr,
    /*.graph_interface=*/nullptr,
};

}  // namespace
}  // namespace litert::example

LiteRtStatus LiteRtDispatchGetApi(LiteRtDispatchApi* api) {
  *api = ::litert::example::ExampleApi;
  return kLiteRtStatusOk;
}

extern "C" LiteRtStatus LiteRtDispatchExampleClearLastSchedulingInfo() {
  LastSchedulingInfo.reset();
  return kLiteRtStatusOk;
}

extern "C" LiteRtStatus LiteRtDispatchExampleGetLastSchedulingInfo(
    LiteRtSchedulingInfo* out) {
  if (!out) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!LastSchedulingInfo.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *out = *LastSchedulingInfo;
  return kLiteRtStatusOk;
}
