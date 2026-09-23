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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_EXAMPLES_COMPATIBILITY_TEST_MOCK_DISPATCH_COMMON_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_EXAMPLES_COMPATIBILITY_TEST_MOCK_DISPATCH_COMMON_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/c/litert_dispatch_api.h"

namespace litert::compatibility {

class MockDispatchDeviceContextT {
 public:
  LiteRtStatus RegisterBuffer(LiteRtTensorBuffer buffer,
                              LiteRtTensorBufferHandle* handle) {
    if (!handle) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    *handle = next_handle_++;
    registered_buffers_[*handle] = buffer;
    return kLiteRtStatusOk;
  }

  LiteRtStatus UnregisterBuffer(LiteRtTensorBufferHandle handle) {
    registered_buffers_.erase(handle);
    return kLiteRtStatusOk;
  }

  LiteRtTensorBuffer Lookup(LiteRtTensorBufferHandle handle) const {
    auto it = registered_buffers_.find(handle);
    return it != registered_buffers_.end() ? it->second : nullptr;
  }

 private:
  uint64_t next_handle_ = 1;
  absl::flat_hash_map<LiteRtTensorBufferHandle, LiteRtTensorBuffer>
      registered_buffers_;
};

class MockDispatchInvocationContextT {
 public:
  MockDispatchInvocationContextT(const LiteRtRuntimeContext* runtime_context,
                                 MockDispatchDeviceContextT* device_context)
      : runtime_context_(runtime_context), device_context_(device_context) {}

  LiteRtStatus AttachInput(int index, LiteRtTensorBufferHandle handle) {
    if (index < 0) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    if (static_cast<size_t>(index) >= inputs_.size()) {
      inputs_.resize(index + 1, 0);
    }
    inputs_[index] = handle;
    return kLiteRtStatusOk;
  }

  LiteRtStatus AttachOutput(int index, LiteRtTensorBufferHandle handle) {
    if (index < 0) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    if (static_cast<size_t>(index) >= outputs_.size()) {
      outputs_.resize(index + 1, 0);
    }
    outputs_[index] = handle;
    return kLiteRtStatusOk;
  }

  LiteRtStatus DetachInput(int index) {
    if (index >= 0 && static_cast<size_t>(index) < inputs_.size()) {
      inputs_[index] = 0;
    }
    return kLiteRtStatusOk;
  }

  LiteRtStatus DetachOutput(int index) {
    if (index >= 0 && static_cast<size_t>(index) < outputs_.size()) {
      outputs_[index] = 0;
    }
    return kLiteRtStatusOk;
  }

  LiteRtStatus SetOptions(LiteRtOptions options) {
    options_set_ = true;
    return kLiteRtStatusOk;
  }

  bool OptionsSet() const { return options_set_; }

  LiteRtStatus Invoke() {
    if (!device_context_ || !runtime_context_) {
      return kLiteRtStatusErrorRuntimeFailure;
    }
    // Realistic multiplication of input float buffers into output buffer.
    if (inputs_.size() >= 2 && !outputs_.empty()) {
      LiteRtTensorBuffer in0 = device_context_->Lookup(inputs_[0]);
      LiteRtTensorBuffer in1 = device_context_->Lookup(inputs_[1]);
      LiteRtTensorBuffer out0 = device_context_->Lookup(outputs_[0]);
      if (in0 && in1 && out0 &&
          runtime_context_->get_tensor_buffer_packed_size &&
          runtime_context_->lock_tensor_buffer &&
          runtime_context_->unlock_tensor_buffer) {
        size_t size0 = 0, size1 = 0, size_out = 0;
        runtime_context_->get_tensor_buffer_packed_size(in0, &size0);
        runtime_context_->get_tensor_buffer_packed_size(in1, &size1);
        runtime_context_->get_tensor_buffer_packed_size(out0, &size_out);
        if (size0 > 0 && size0 == size1 && size0 == size_out) {
          void* mem0 = nullptr;
          void* mem1 = nullptr;
          void* mem_out = nullptr;
          runtime_context_->lock_tensor_buffer(in0, &mem0,
                                               kLiteRtTensorBufferLockModeRead);
          runtime_context_->lock_tensor_buffer(in1, &mem1,
                                               kLiteRtTensorBufferLockModeRead);
          runtime_context_->lock_tensor_buffer(
              out0, &mem_out, kLiteRtTensorBufferLockModeWrite);
          const float* f0 = static_cast<const float*>(mem0);
          const float* f1 = static_cast<const float*>(mem1);
          float* f_out = static_cast<float*>(mem_out);
          if (f0 != nullptr && f1 != nullptr && f_out != nullptr) {
            size_t count = size0 / sizeof(float);
            for (size_t i = 0; i < count; ++i) {
              f_out[i] = f0[i] * f1[i];
            }
          }
          runtime_context_->unlock_tensor_buffer(in0);
          runtime_context_->unlock_tensor_buffer(in1);
          runtime_context_->unlock_tensor_buffer(out0);
        }
      }
    }
    invoked_ = true;
    return kLiteRtStatusOk;
  }

  bool Invoked() const { return invoked_; }

 private:
  const LiteRtRuntimeContext* runtime_context_;
  MockDispatchDeviceContextT* device_context_;
  std::vector<LiteRtTensorBufferHandle> inputs_;
  std::vector<LiteRtTensorBufferHandle> outputs_;
  bool options_set_ = false;
  bool invoked_ = false;
};

inline LiteRtStatus MockDispatchInitialize(
    const LiteRtRuntimeContext* runtime_context, LiteRtEnvironment environment,
    LiteRtOptions options) {
  return kLiteRtStatusOk;
}

inline LiteRtStatus MockDispatchGetBuildId(const char** build_id) {
  if (!build_id) return kLiteRtStatusErrorInvalidArgument;
  *build_id = "mock_dispatch_build_1";
  return kLiteRtStatusOk;
}

inline LiteRtStatus MockDispatchGetCapabilities(int* capabilities) {
  if (!capabilities) return kLiteRtStatusErrorInvalidArgument;
  *capabilities = kLiteRtDispatchCapabilitiesBasic;
  return kLiteRtStatusOk;
}

inline LiteRtStatus MockDispatchDeviceContextCreate(
    const LiteRtRuntimeContext* runtime_context, LiteRtOptions options,
    LiteRtDispatchDeviceContext* device_context) {
  if (!device_context) return kLiteRtStatusErrorInvalidArgument;
  *device_context = reinterpret_cast<LiteRtDispatchDeviceContext>(
      new MockDispatchDeviceContextT());
  return kLiteRtStatusOk;
}

inline LiteRtStatus MockDispatchDeviceContextDestroy(
    LiteRtDispatchDeviceContext device_context) {
  delete reinterpret_cast<MockDispatchDeviceContextT*>(device_context);
  return kLiteRtStatusOk;
}

inline LiteRtStatus MockDispatchGetInputRequirements(
    LiteRtDispatchInvocationContext invocation_context, int input_index,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements* tensor_buffer_requirements) {
  return kLiteRtStatusOk;
}

inline LiteRtStatus MockDispatchGetOutputRequirements(
    LiteRtDispatchInvocationContext invocation_context, int output_index,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements* tensor_buffer_requirements) {
  return kLiteRtStatusOk;
}

inline LiteRtStatus MockDispatchRegisterTensorBuffer(
    LiteRtDispatchDeviceContext device_context,
    LiteRtTensorBuffer tensor_buffer,
    LiteRtTensorBufferHandle* tensor_buffer_handle) {
  auto* ctx = reinterpret_cast<MockDispatchDeviceContextT*>(device_context);
  if (!ctx) return kLiteRtStatusErrorInvalidArgument;
  return ctx->RegisterBuffer(tensor_buffer, tensor_buffer_handle);
}

inline LiteRtStatus MockDispatchUnregisterTensorBuffer(
    LiteRtDispatchDeviceContext device_context,
    LiteRtTensorBufferHandle handle) {
  auto* ctx = reinterpret_cast<MockDispatchDeviceContextT*>(device_context);
  if (!ctx) return kLiteRtStatusErrorInvalidArgument;
  return ctx->UnregisterBuffer(handle);
}

inline LiteRtStatus MockDispatchInvocationContextCreate(
    const LiteRtRuntimeContext* runtime_context,
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableType exec_type,
    const LiteRtMemBuffer* exec_bytecode_buffer, const char* function_name,
    int num_inputs, int num_outputs,
    LiteRtDispatchInvocationContext* invocation_context) {
  if (!invocation_context) return kLiteRtStatusErrorInvalidArgument;
  auto* dev_ctx = reinterpret_cast<MockDispatchDeviceContextT*>(device_context);
  *invocation_context = reinterpret_cast<LiteRtDispatchInvocationContext>(
      new MockDispatchInvocationContextT(runtime_context, dev_ctx));
  return kLiteRtStatusOk;
}

inline LiteRtStatus MockDispatchInvocationContextDestroy(
    LiteRtDispatchInvocationContext invocation_context) {
  delete reinterpret_cast<MockDispatchInvocationContextT*>(invocation_context);
  return kLiteRtStatusOk;
}

inline LiteRtStatus MockDispatchAttachInput(
    LiteRtDispatchInvocationContext invocation_context, int graph_input_index,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  auto* ctx =
      reinterpret_cast<MockDispatchInvocationContextT*>(invocation_context);
  if (!ctx) return kLiteRtStatusErrorInvalidArgument;
  return ctx->AttachInput(graph_input_index, tensor_buffer_handle);
}

inline LiteRtStatus MockDispatchAttachOutput(
    LiteRtDispatchInvocationContext invocation_context, int graph_output_index,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  auto* ctx =
      reinterpret_cast<MockDispatchInvocationContextT*>(invocation_context);
  if (!ctx) return kLiteRtStatusErrorInvalidArgument;
  return ctx->AttachOutput(graph_output_index, tensor_buffer_handle);
}

inline LiteRtStatus MockDispatchDetachInput(
    LiteRtDispatchInvocationContext invocation_context, int graph_input_index,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  auto* ctx =
      reinterpret_cast<MockDispatchInvocationContextT*>(invocation_context);
  if (!ctx) return kLiteRtStatusErrorInvalidArgument;
  return ctx->DetachInput(graph_input_index);
}

inline LiteRtStatus MockDispatchDetachOutput(
    LiteRtDispatchInvocationContext invocation_context, int graph_output_index,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  auto* ctx =
      reinterpret_cast<MockDispatchInvocationContextT*>(invocation_context);
  if (!ctx) return kLiteRtStatusErrorInvalidArgument;
  return ctx->DetachOutput(graph_output_index);
}

inline LiteRtStatus MockDispatchInvoke(
    LiteRtDispatchInvocationContext invocation_context) {
  auto* ctx =
      reinterpret_cast<MockDispatchInvocationContextT*>(invocation_context);
  if (!ctx) return kLiteRtStatusErrorInvalidArgument;
  return ctx->Invoke();
}

inline LiteRtStatus MockDispatchInvocationContextSetOptions(
    LiteRtDispatchInvocationContext invocation_context, LiteRtOptions options) {
  auto* ctx =
      reinterpret_cast<MockDispatchInvocationContextT*>(invocation_context);
  if (!ctx) return kLiteRtStatusErrorInvalidArgument;
  return ctx->SetOptions(options);
}

inline LiteRtStatus MockDispatchCheckRuntimeCompatibility(
    LiteRtApiVersion api_version, LiteRtEnvironmentOptions env,
    LiteRtOptions options) {
  return kLiteRtStatusOk;
}

// Extra function pointer signature for future V1.2 dispatch extensions.
typedef LiteRtStatus (*MockDispatchFutureExtensionT)();

inline LiteRtStatus MockDispatchFutureExtension() { return kLiteRtStatusOk; }

// Future V1.2 interface table (extended beyond V1).
typedef struct {
  LiteRtDispatchInterface_V1 v1;
  MockDispatchFutureExtensionT future_extension;
} LiteRtDispatchInterface_V1_2;

}  // namespace litert::compatibility

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_EXAMPLES_COMPATIBILITY_TEST_MOCK_DISPATCH_COMMON_H_
