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

#include <array>
#include <cstdint>
#include <list>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_requirements.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/c/litert_dispatch_api.h"
#include "litert/vendors/examples/example_common.h"

namespace {
using Buffer = ::litert::example::Data;
using BufferHandle = Buffer*;
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
        auto example_graph,
        ::litert::example::ExampleGraph::Parse(::litert::BufferRef<uint8_t>(
            exec_bytecode_buffer->base_addr,
            exec_bytecode_buffer->offset + exec_bytecode_buffer->size,
            exec_bytecode_buffer->offset)));

    return Ptr(new LiteRtDispatchInvocationContextT(
        device_context, exec_type, absl::string_view(function_name),
        std::move(example_graph)));
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
    for (auto* input : inputs_) {
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

  const ::litert::example::ExampleGraph& ExampleGraph() const {
    return example_graph_;
  }

  ~LiteRtDispatchInvocationContextT() = default;

 private:
  LiteRtDispatchInvocationContextT(
      LiteRtDispatchDeviceContext device_context,
      LiteRtDispatchExecutableType exec_type, absl::string_view function_name,
      ::litert::example::ExampleGraph example_graph)
      : device_context_(device_context),
        exec_type_(exec_type),
        function_name_(function_name),
        inputs_(example_graph.Inputs().size()),
        outputs_(example_graph.Outputs().size()),
        example_graph_(std::move(example_graph)) {}
  LiteRtDispatchDeviceContext device_context_;
  LiteRtDispatchExecutableType exec_type_;
  absl::string_view function_name_;
  std::vector<BufferHandle> inputs_;
  std::vector<BufferHandle> outputs_;
  ::litert::example::ExampleGraph example_graph_;
};

namespace litert::example {
namespace {

LiteRtEnvironmentOptions the_environment_options = nullptr;
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

LiteRtStatus Initialize(LiteRtEnvironmentOptions environment_options,
                        LiteRtOptions options) {
  the_environment_options = environment_options;
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

Expected<TensorBufferRequirements> GetTensorBufferRequirements(
    const LiteRtRankedTensorType& tensor_type) {
  RankedTensorType t(tensor_type);
  if (t.Layout().HasStrides()) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Tensor strides are not supported by QNN");
  }
  static constexpr std::array<const LiteRtTensorBufferType, 1> types = {
      kLiteRtTensorBufferTypeHostMemory};
  LITERT_ASSIGN_OR_RETURN(const auto size, t.Bytes());
  return TensorBufferRequirements::Create(types, size, {}, OwnHandle::kNo);
}

LiteRtStatus GetInputRequirements(
    LiteRtDispatchInvocationContext invocation_context, int input_index,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements* tensor_buffer_requirements) {
  LITERT_ASSIGN_OR_RETURN(auto requirements,
                          GetTensorBufferRequirements(*tensor_type));
  *tensor_buffer_requirements = requirements.Get();
  return kLiteRtStatusOk;
}

LiteRtStatus GetOutputRequirements(
    LiteRtDispatchInvocationContext invocation_context, int output_index,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements* tensor_buffer_requirements) {
  LITERT_ASSIGN_OR_RETURN(auto requirements,
                          GetTensorBufferRequirements(*tensor_type));
  *tensor_buffer_requirements = requirements.Get();
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
  for (int i = 0; i < results.size(); ++i) {
    invocation_context->GetOutput(i) = std::move(results[i]);
  }

  invocation_context->Finish();
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
    /*.attach_input=*/AttachInput,
    /*.attach_output=*/AttachOutput,
    /*.detach_input=*/DetachInput,
    /*.detach_output=*/DetachOutput,
    /*.invoke=*/Invoke,
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
