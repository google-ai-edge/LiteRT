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

#include "litert/vendors/google_tensor/dispatch/litert_dispatch_device_context.h"

#include <inttypes.h>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

#if __ANDROID__
#include <android/hardware_buffer.h>
#endif  // __ANDROID__

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/options/litert_darwinn_options.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/google_tensor/dispatch/dispatch_api_config.h"
#include "litert/vendors/google_tensor/dispatch/dispatch_api_macros.h"
#include "litert/vendors/google_tensor/dispatch/sb_api.h"

namespace gt = litert::google_tensor;

LiteRtStatus LiteRtDispatchDeviceContextT::Create(
    LiteRtDispatchDeviceContext& device_context) {
  ThrContext* thr_context = thrContextCreate();
  if (thr_context == nullptr) {
    LITERT_LOG(LITERT_ERROR, "Failed to create SB context");
    return kLiteRtStatusErrorRuntimeFailure;
  }
  absl::Cleanup thr_context_cleanup = [thr_context] {
    thrContextDelete(thr_context);
  };

  GT_LOG_RETURN_IF_SB_ERROR(thrVendorSetSystemAttributeStr(
                                thr_context, "edgetpu_use_tpu_tachyon", "1"),
                            "Failed to enable Tachyon SB");

  // If provided, store DarwiNN options to be applied to graphs.
  std::optional<DarwinnOptionsData> options_data;
  if (litert::DarwinnRuntimeOptions* absl_nullable darwinn_options =
          gt::GetTheDarwinnOptions();
      darwinn_options != nullptr) {
    options_data.emplace();

    if (litert::Expected<uint32_t> inference_power_state =
            darwinn_options->GetInferencePowerState();
        inference_power_state.HasValue()) {
      options_data->inference_power_state = *inference_power_state;
    }

    if (litert::Expected<uint32_t> mem_power_state =
            darwinn_options->GetInferenceMemoryPowerState();
        mem_power_state.HasValue()) {
      options_data->inference_memory_power_state = *mem_power_state;
    }

    if (litert::Expected<int8_t> priority =
            darwinn_options->GetInferencePriority(); priority.HasValue()) {
      options_data->inference_priority = *priority;
    }

    if (litert::Expected<bool> atomic = darwinn_options->GetAtomicInference();
        atomic.HasValue()) {
      options_data->atomic_inference = *atomic;
    }

    if (litert::Expected<bool> prefer_coherent =
            darwinn_options->GetPreferCoherent(); prefer_coherent.HasValue()) {
      options_data->prefer_coherent = *prefer_coherent;
    }

    LITERT_LOG(LITERT_INFO,
               "DarwiNN runtime options will be applied to graphs");
  }

  // The returned instance must be allocated with `new`, as it will be
  // deallocated via `delete` in `Destroy`.
  device_context =
      new LiteRtDispatchDeviceContextT(thr_context, std::move(options_data));

  std::move(thr_context_cleanup).Cancel();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchDeviceContextT::Destroy() {
  if (!registered_graphs_.empty()) {
    LITERT_LOG(LITERT_ERROR,
               "Cannot destroy device context with %zu graphs registered",
               registered_graphs_.size());
    return kLiteRtStatusErrorRuntimeFailure;
  }

  GT_LOG_RETURN_IF_SB_ERROR(thrContextDelete(thr_context_),
                            "Failed to delete SB context");

  delete this;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchDeviceContextT::RegisterTensorBuffer(
    LiteRtTensorBuffer tensor_buffer,
    LiteRtTensorBufferHandle& tensor_buffer_handle) {
  LiteRtRankedTensorType tensor_type;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetTensorBufferTensorType(tensor_buffer, &tensor_type));
  if (tensor_type.layout.has_strides) {
    LITERT_LOG(LITERT_ERROR, "Tensor strides are not supported");
    return kLiteRtStatusErrorUnsupported;
  }

  LiteRtTensorBufferType tensor_buffer_type;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetTensorBufferType(tensor_buffer, &tensor_buffer_type));
  if (!gt::IsTensorBufferTypeSupported(tensor_buffer_type)) {
    LITERT_LOG(LITERT_ERROR, "Unsupported tensor buffer type %d",
               tensor_buffer_type);
    return kLiteRtStatusErrorUnsupported;
  }

  size_t tensor_buffer_size;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetTensorBufferSize(tensor_buffer, &tensor_buffer_size));

  size_t tensor_buffer_offset;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetTensorBufferOffset(tensor_buffer, &tensor_buffer_offset));

  switch (tensor_buffer_type) {
#if LITERT_HAS_AHWB_SUPPORT
    case kLiteRtTensorBufferTypeAhwb: {
      AHardwareBuffer* ahwb;
      LITERT_RETURN_IF_ERROR(LiteRtGetTensorBufferAhwb(tensor_buffer, &ahwb));

      GT_LOG_RETURN_IF_SB_ERROR(
          thrRegisterBufferWithOffset(
              thr_context_, kThrBufferTypeAHardwareBuffer, ahwb,
              tensor_buffer_offset, tensor_buffer_size, &tensor_buffer_handle),
          "Failed to register AHardwareBuffer with SB");
      break;
    }
#endif
#if LITERT_HAS_DMABUF_SUPPORT
    case kLiteRtTensorBufferTypeDmaBuf: {
      void* dmabuf_buffer_addr;
      int dmabuf_buffer_fd;
      LITERT_RETURN_IF_ERROR(LiteRtGetTensorBufferDmaBufBuffer(
          tensor_buffer, &dmabuf_buffer_addr, &dmabuf_buffer_fd));

      GT_LOG_RETURN_IF_SB_ERROR(
          thrRegisterBufferDmaBufWithOffset(
              thr_context_, dmabuf_buffer_fd, tensor_buffer_offset,
              tensor_buffer_size, &tensor_buffer_handle),
          "Failed to register dma-buf with SB");
      break;
    }
#endif
    case kLiteRtTensorBufferTypeHostMemory: {
      void* host_memory_addr;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetTensorBufferHostMemory(tensor_buffer, &host_memory_addr));

      GT_LOG_RETURN_IF_SB_ERROR(
          thrRegisterBufferWithOffset(
              thr_context_, kThrBufferTypeHostMemory, host_memory_addr,
              tensor_buffer_offset, tensor_buffer_size, &tensor_buffer_handle),
          "Failed to register host memory with SB");
      break;
    }
    default:
      LITERT_FATAL("Unsupported tensor buffer type %d", tensor_buffer_type);
  }

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchDeviceContextT::UnregisterTensorBuffer(
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  GT_LOG_RETURN_IF_SB_ERROR(
      thrUnregisterBuffer(thr_context_, tensor_buffer_handle),
      "Failed to unregister buffer %" PRIu64 " from SB", tensor_buffer_handle);

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchDeviceContextT::LoadExecutable(
    LiteRtDispatchExecutableType type, const LiteRtMemBuffer& bytecode_buffer,
    LiteRtDispatchExecutableHandle& exec_handle) {
  ThrSqContainerType thr_type;
  switch (type) {
    case kLiteRtDispatchExecutableTypeDspLibrary:
      thr_type = kThrSqContainerTypeFunctionLibrary;
      break;
    case kLiteRtDispatchExecutableTypeMlModel:
      thr_type = kThrSqContainerTypeMlModel;
      break;
    default:
      LITERT_LOG(LITERT_ERROR, "Invalid executable type %d", type);
      return kLiteRtStatusErrorInvalidArgument;
  }

  if (bytecode_buffer.fd >= 0) {
    GT_LOG_RETURN_IF_SB_ERROR(
        thrLoadSqContainerFdWithOffset(thr_context_, thr_type,
                                       bytecode_buffer.fd, bytecode_buffer.size,
                                       bytecode_buffer.offset,
                                       /*lazy_loading=*/false, &exec_handle),
        "Failed to load SQ container from fd %d with size %zu and offset %zu",
        bytecode_buffer.fd, bytecode_buffer.size, bytecode_buffer.offset);
  } else {
    const auto* sq_bytecode =
        static_cast<const std::byte*>(bytecode_buffer.base_addr) +
        bytecode_buffer.offset;

    GT_LOG_RETURN_IF_SB_ERROR(
        thrLoadSqContainer(thr_context_, thr_type, sq_bytecode,
                           bytecode_buffer.size, &exec_handle),
        "Failed to load SQ container from buffer with base 0x%p and size %zu",
        sq_bytecode, bytecode_buffer.size);
  }

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchDeviceContextT::UnloadExecutable(
    LiteRtDispatchExecutableHandle exec_handle) {
  GT_LOG_RETURN_IF_SB_ERROR(thrUnloadSqContainer(thr_context_, exec_handle),
                            "Failed to unload SQ container %" PRIu64,
                            exec_handle);

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchDeviceContextT::RegisterGraph(
    LiteRtDispatchGraph graph) {
  if (auto [_, inserted] = registered_graphs_.insert(graph); !inserted) {
    LITERT_LOG(LITERT_ERROR, "Graph 0x%p is already registered", graph);
    return kLiteRtStatusErrorInvalidArgument;
  }

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchDeviceContextT::UnregisterGraph(
    LiteRtDispatchGraph graph) {
  if (registered_graphs_.erase(graph) == 0) {
    LITERT_LOG(LITERT_ERROR, "Graph 0x%p was not previously registered", graph);
    return kLiteRtStatusErrorInvalidArgument;
  }

  return kLiteRtStatusOk;
}
