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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#if defined(__APPLE__)
#include <mlx/mlx.h>
#endif

#include "litert/c/internal/litert_logging.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/vendors/apple/bytecode.h"
#include "litert/vendors/c/litert_dispatch.h"

namespace {

using ::litert::Error;
using ::litert::Expected;

#if defined(__APPLE__)
mlx::core::Dtype ConvertDataType(LiteRtElementType type) {
  switch (type) {
    case kLiteRtElementTypeFloat32:
      return mlx::core::float32;
    case kLiteRtElementTypeFloat16:
      return mlx::core::float16;
    default:
      return mlx::core::float32;
  }
}
#endif

size_t GetDataTypeBytes(LiteRtElementType type) {
  switch (type) {
    case kLiteRtElementTypeFloat32:
      return 4;
    case kLiteRtElementTypeFloat16:
      return 2;
    default:
      return 0;
  }
}

}  // namespace

struct LiteRtDispatchDeviceContextT {
  explicit LiteRtDispatchDeviceContextT(
      const LiteRtRuntimeContext* runtime_context)
      : runtime_context(runtime_context) {}

  const LiteRtRuntimeContext* runtime_context;
  LiteRtTensorBufferHandle next_handle = 1;
  std::unordered_map<LiteRtTensorBufferHandle, LiteRtTensorBuffer> records;
};

struct LiteRtDispatchInvocationContextT {
  LiteRtDispatchInvocationContextT(const LiteRtRuntimeContext* runtime_context,
                                   LiteRtDispatchDeviceContext device_context,
                                   litert::apple::MlxBytecode bytecode)
      : runtime_context(runtime_context),
        device_context(device_context),
        bytecode(std::move(bytecode)) {}

  const LiteRtRuntimeContext* runtime_context;
  LiteRtDispatchDeviceContext device_context;
  litert::apple::MlxBytecode bytecode;

  std::vector<LiteRtTensorBufferHandle> input_handles;
  std::vector<LiteRtTensorBufferHandle> output_handles;

#if defined(__APPLE__)
  mlx::core::array weights;
  mlx::core::array bias;
  std::function<std::vector<mlx::core::array>(
      const std::vector<mlx::core::array>&)>
      compiled_func;
#endif

  LiteRtStatus Initialize() {
    input_handles.resize(1, 0);   // We expect 1 dynamic input
    output_handles.resize(1, 0);  // We expect 1 dynamic output

#if defined(__APPLE__)
    try {
      std::vector<int> weights_shape;
      for (auto dim : bytecode.weights_dims) {
        weights_shape.push_back(static_cast<int>(dim));
      }
      weights = mlx::core::array(bytecode.weights_data.data(), weights_shape,
                                 ConvertDataType(bytecode.weights_type));

      if (bytecode.has_bias) {
        std::vector<int> bias_shape;
        for (auto dim : bytecode.bias_dims) {
          bias_shape.push_back(static_cast<int>(dim));
        }
        bias = mlx::core::array(bytecode.bias_data.data(), bias_shape,
                                ConvertDataType(bytecode.bias_type));
      }

      // Transpose weights once at init. TFLite FC weights are [N, K].
      // MLX matmul expects X [M, K] and W_T [K, N] to get [M, N].
      // Or we can just use mlx::core::matmul(x, transpose(weights)).
      // Transposing weights here:
      auto transposed_weights = mlx::core::transpose(weights);

      auto mlx_func = [transposed_weights, b = bias,
                       has_bias = bytecode.has_bias, act = bytecode.activation](
                          const std::vector<mlx::core::array>& inputs) {
        auto x = inputs[0];
        auto out = mlx::core::matmul(x, transposed_weights);
        if (has_bias) {
          out = mlx::core::add(out, b);
        }
        if (act == litert::kActivationFunctionTypeRelu) {
          out = mlx::core::maximum(out, mlx::core::array(0.0f, out.dtype()));
        } else if (act == litert::kActivationFunctionTypeRelu6) {
          out = mlx::core::minimum(
              mlx::core::maximum(out, mlx::core::array(0.0f, out.dtype())),
              mlx::core::array(6.0f, out.dtype()));
        }
        return std::vector<mlx::core::array>{out};
      };

      compiled_func = mlx::core::compile(mlx_func);
    } catch (const std::exception& e) {
      LITERT_LOG(LITERT_ERROR, "Failed to initialize MLX graph: %s", e.what());
      return kLiteRtStatusErrorRuntimeFailure;
    }
#endif
    return kLiteRtStatusOk;
  }
};

extern "C" {

LiteRtStatus LiteRtDispatchInitialize(
    const LiteRtRuntimeContext* runtime_context, LiteRtEnvironment environment,
    LiteRtOptions options) {
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchGetApiVersion(LiteRtApiVersion* api_version) {
  if (api_version == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  api_version->major = LITERT_API_VERSION_MAJOR;
  api_version->minor = LITERT_API_VERSION_MINOR;
  api_version->patch = LITERT_API_VERSION_PATCH;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchGetVendorId(const char** vendor_id) {
  if (vendor_id == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *vendor_id = "Apple";
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchGetBuildId(const char** build_id) {
  if (build_id == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *build_id = "Apple MLX Dispatch 1.0";
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchGetCapabilities(int* capabilities) {
  if (capabilities == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *capabilities = kLiteRtDispatchCapabilitiesBasic;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchDeviceContextCreate(
    const LiteRtRuntimeContext* runtime_context, LiteRtOptions options,
    LiteRtDispatchDeviceContext* device_context) {
  if (runtime_context == nullptr || device_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *device_context = new LiteRtDispatchDeviceContextT(runtime_context);
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchDeviceContextDestroy(
    LiteRtDispatchDeviceContext device_context) {
  delete device_context;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchGetInputRequirements(
    LiteRtDispatchInvocationContext invocation_context, int input_index,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements* tensor_buffer_requirements) {
  if (invocation_context == nullptr || tensor_type == nullptr ||
      tensor_buffer_requirements == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  // We only support host memory for now.
  LiteRtTensorBufferType supported_types[] = {
      kLiteRtTensorBufferTypeHostMemory};
  size_t size = 1;
  for (int i = 0; i < tensor_type->layout.rank; ++i) {
    size *= tensor_type->layout.dimensions[i];
  }
  size *= GetDataTypeBytes(
      static_cast<LiteRtElementType>(tensor_type->element_type));

  return invocation_context->runtime_context->create_tensor_buffer_requirements(
      1, supported_types, size, 0, nullptr, tensor_buffer_requirements);
}

LiteRtStatus LiteRtDispatchGetOutputRequirements(
    LiteRtDispatchInvocationContext invocation_context, int output_index,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements* tensor_buffer_requirements) {
  return LiteRtDispatchGetInputRequirements(invocation_context, output_index,
                                            tensor_type,
                                            tensor_buffer_requirements);
}

LiteRtStatus LiteRtDispatchRegisterTensorBuffer(
    LiteRtDispatchDeviceContext device_context,
    LiteRtTensorBuffer tensor_buffer,
    LiteRtTensorBufferHandle* tensor_buffer_handle) {
  if (device_context == nullptr || tensor_buffer == nullptr ||
      tensor_buffer_handle == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LiteRtTensorBufferType type;
  LITERT_RETURN_IF_ERROR(LiteRtGetTensorBufferType(tensor_buffer, &type));
  if (type != kLiteRtTensorBufferTypeHostMemory) {
    return kLiteRtStatusErrorUnsupported;
  }

  LiteRtTensorBufferHandle handle = device_context->next_handle++;
  device_context->records[handle] = tensor_buffer;
  *tensor_buffer_handle = handle;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchUnregisterTensorBuffer(
    LiteRtDispatchDeviceContext device_context,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (device_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  device_context->records.erase(tensor_buffer_handle);
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchInvocationContextCreate(
    const LiteRtRuntimeContext* runtime_context,
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableType exec_type,
    const LiteRtMemBuffer* exec_bytecode_buffer, const char* function_name,
    int num_inputs, int num_outputs,
    LiteRtDispatchInvocationContext* invocation_context) {
  if (runtime_context == nullptr || device_context == nullptr ||
      exec_bytecode_buffer == nullptr || invocation_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (exec_type != kLiteRtDispatchExecutableTypeMlModel) {
    return kLiteRtStatusErrorUnsupported;
  }

  const uint8_t* base =
      static_cast<const uint8_t*>(exec_bytecode_buffer->base_addr);
  auto bytecode_or = litert::apple::ParseMlxBytecode(
      base + exec_bytecode_buffer->offset, exec_bytecode_buffer->size);
  if (!bytecode_or) {
    LITERT_LOG(LITERT_ERROR, "Failed to parse MLX bytecode: %s",
               bytecode_or.Error().Message().c_str());
    return bytecode_or.Error().Status();
  }

  auto context = std::make_unique<LiteRtDispatchInvocationContextT>(
      runtime_context, device_context, std::move(*bytecode_or));
  LITERT_RETURN_IF_ERROR(context->Initialize());

  *invocation_context = context.release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchInvocationContextDestroy(
    LiteRtDispatchInvocationContext invocation_context) {
  delete invocation_context;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchAttachInput(
    LiteRtDispatchInvocationContext invocation_context, int graph_input_index,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (invocation_context == nullptr || graph_input_index < 0 ||
      static_cast<size_t>(graph_input_index) >=
          invocation_context->input_handles.size()) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  invocation_context->input_handles[graph_input_index] = tensor_buffer_handle;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchAttachOutput(
    LiteRtDispatchInvocationContext invocation_context, int graph_output_index,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (invocation_context == nullptr || graph_output_index < 0 ||
      static_cast<size_t>(graph_output_index) >=
          invocation_context->output_handles.size()) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  invocation_context->output_handles[graph_output_index] = tensor_buffer_handle;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchDetachInput(
    LiteRtDispatchInvocationContext invocation_context, int graph_input_index,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchDetachOutput(
    LiteRtDispatchInvocationContext invocation_context, int graph_output_index,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchInvoke(
    LiteRtDispatchInvocationContext invocation_context) {
  if (invocation_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

#if defined(__APPLE__)
  try {
    // 1. Get input buffer
    LiteRtTensorBufferHandle input_handle =
        invocation_context->input_handles[0];
    auto input_it =
        invocation_context->device_context->records.find(input_handle);
    if (input_it == invocation_context->device_context->records.end()) {
      return kLiteRtStatusErrorRuntimeFailure;
    }
    LiteRtTensorBuffer input_buffer = input_it->second;

    LiteRtRankedTensorType input_type;
    LITERT_RETURN_IF_ERROR(
        LiteRtGetTensorBufferTensorType(input_buffer, &input_type));

    void* input_host_ptr = nullptr;
    LITERT_RETURN_IF_ERROR(LiteRtLockTensorBuffer(
        input_buffer, &input_host_ptr, kLiteRtTensorBufferLockModeRead));

    std::vector<int> input_shape;
    for (int i = 0; i < input_type.layout.rank; ++i) {
      input_shape.push_back(static_cast<int>(input_type.layout.dimensions[i]));
    }

    // Wrap input (copies data for simplicity in this prototype)
    auto x = mlx::core::array(input_host_ptr, input_shape,
                              ConvertDataType(static_cast<LiteRtElementType>(
                                  input_type.element_type)));
    LITERT_RETURN_IF_ERROR(LiteRtUnlockTensorBuffer(input_buffer));

    // 2. Invoke MLX graph
    auto outputs = invocation_context->compiled_func({x});
    mlx::core::eval(outputs);

    // 3. Copy output back
    LiteRtTensorBufferHandle output_handle =
        invocation_context->output_handles[0];
    auto output_it =
        invocation_context->device_context->records.find(output_handle);
    if (output_it == invocation_context->device_context->records.end()) {
      return kLiteRtStatusErrorRuntimeFailure;
    }
    LiteRtTensorBuffer output_buffer = output_it->second;

    void* output_host_ptr = nullptr;
    LITERT_RETURN_IF_ERROR(LiteRtLockTensorBuffer(
        output_buffer, &output_host_ptr, kLiteRtTensorBufferLockModeWrite));

    auto y = outputs[0];
    size_t output_size_bytes = y.nbytes();
    std::memcpy(output_host_ptr, y.data<void>(), output_size_bytes);

    LITERT_RETURN_IF_ERROR(LiteRtUnlockTensorBuffer(output_buffer));

  } catch (const std::exception& e) {
    LITERT_LOG(LITERT_ERROR, "MLX invocation failed: %s", e.what());
    return kLiteRtStatusErrorRuntimeFailure;
  }
#else
  LITERT_LOG(LITERT_WARNING,
             "MLX dispatch invoke stub called (non-Apple platform)");
#endif

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchInvocationContextSetOptions(
    LiteRtDispatchInvocationContext invocation_context, LiteRtOptions options) {
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchInvocationContextSetSchedulingInfo(
    LiteRtDispatchInvocationContext invocation_context,
    const LiteRtSchedulingInfo* scheduling_info) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchStartMetricsCollection(
    LiteRtDispatchInvocationContext invocation_context, int detail_level) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchStopMetricsCollection(
    LiteRtDispatchInvocationContext invocation_context,
    LiteRtDispatchMetrics* metrics) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchGetNumMetrics(LiteRtDispatchMetrics metrics,
                                         int* num_metrics) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchGetMetric(LiteRtDispatchMetrics metrics,
                                     int metric_index, LiteRtMetric* metric) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchDestroyMetrics(LiteRtDispatchMetrics metrics) {
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchCheckRuntimeCompatibility(
    LiteRtApiVersion api_version, LiteRtEnvironmentOptions env,
    LiteRtOptions options) {
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchAttachInputEvent(
    LiteRtDispatchInvocationContext invocation_context, int graph_input_index,
    LiteRtEvent input_event) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchInvokeAsync(
    LiteRtDispatchInvocationContext invocation_context, int num_output_events,
    LiteRtEvent* output_events) {
  return kLiteRtStatusErrorUnsupported;
}

}  // extern "C"
