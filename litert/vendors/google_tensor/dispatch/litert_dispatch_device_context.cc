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
#include <memory>
#include <optional>
#include <string>
#include <utility>

#if __ANDROID__
#include <android/hardware_buffer.h>
#endif  // __ANDROID__

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
#include "litert/vendors/google_tensor/dispatch/litert_dispatch_graph.h"
#include "litert/vendors/google_tensor/dispatch/sb_api.h"
#include "litert/vendors/google_tensor/dispatch/sb_dispatch_annotations.h"

namespace gt = litert::google_tensor;

void LiteRtDispatchDeviceContextT::ThrContextDeleter(ThrContext* thr_context) {
  GT_LOG_IF_SB_ERROR(thrContextDelete(thr_context),
                     "Failed to delete SB context");
}

LiteRtStatus LiteRtDispatchDeviceContextT::Create(
    LiteRtDispatchDeviceContext& device_context) {
  ThrContext* thr_context_raw = thrContextCreate();
  if (thr_context_raw == nullptr) {
    LITERT_LOG(LITERT_ERROR, "Failed to create SB context");
    return kLiteRtStatusErrorRuntimeFailure;
  }
  ThrContextPtr thr_context(thr_context_raw, ThrContextDeleter);

  GT_LOG_RETURN_IF_SB_ERROR(
      thrVendorSetSystemAttributeStr(
          thr_context.get(), "edgetpu_use_tpu_tachyon", "1"),
      "Failed to enable Tachyon SB");

  // If provided, store DarwiNN options to be applied to graphs.
  std::optional<DarwinnOptionsData> options_data;

  if (litert::DarwinnRuntimeOptions* darwinn_options =
          gt::GetTheDarwinnOptions(); darwinn_options != nullptr) {
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

  device_context =
      new LiteRtDispatchDeviceContextT(std::move(thr_context),
                                       std::move(options_data));
  return kLiteRtStatusOk;
}

LiteRtDispatchDeviceContextT::~LiteRtDispatchDeviceContextT() {
  for (ThrGraph* graph : thr_graphs_) {
    GT_LOG_IF_SB_ERROR(thrGraphDelete(graph), "Failed to delete SB graph");
  }
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
              thr_context_.get(), kThrBufferTypeAHardwareBuffer, ahwb,
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
              thr_context_.get(), dmabuf_buffer_fd, tensor_buffer_offset,
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
              thr_context_.get(), kThrBufferTypeHostMemory, host_memory_addr,
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
      thrUnregisterBuffer(thr_context_.get(), tensor_buffer_handle),
      "Failed to unregister buffer %" PRIu64 " from SB", tensor_buffer_handle);

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchDeviceContextT::CreateGraph(
    LiteRtDispatchGraph& graph) {
  std::unique_ptr<LiteRtDispatchGraphT> raii_graph;
  LITERT_RETURN_IF_ERROR(LiteRtDispatchGraphT::Create(this, raii_graph));

  if (darwinn_options_.has_value()) {
    if (std::optional<uint32_t> device_power_state =
            darwinn_options_->inference_power_state;
        device_power_state.has_value()) {
      LITERT_RETURN_IF_ERROR(raii_graph->AnnotateGraph(
          gt::DispatchDirectiveAnnotations::kEdgetpuDevicePowerState.data(),
          std::to_string(*device_power_state).c_str()));
    }

    if (std::optional<uint32_t> memory_power_state =
            darwinn_options_->inference_memory_power_state;
        memory_power_state.has_value()) {
      LITERT_RETURN_IF_ERROR(raii_graph->AnnotateGraph(
          gt::DispatchDirectiveAnnotations::kEdgetpuMemoryPowerState.data(),
          std::to_string(*memory_power_state).c_str()));
    }

    if (std::optional<int8_t> inference_priority =
            darwinn_options_->inference_priority;
        inference_priority.has_value()) {
      LITERT_RETURN_IF_ERROR(raii_graph->AnnotateGraph(
          gt::DispatchDirectiveAnnotations::kPriority.data(),
          std::to_string(*inference_priority).c_str()));
    }

    if (darwinn_options_->atomic_inference) {
      LITERT_RETURN_IF_ERROR(raii_graph->AnnotateGraph(
          gt::DispatchDirectiveAnnotations::kEdgetpuAtomicInference.data(),
          "1"));
    }

    if (darwinn_options_->prefer_coherent) {
      LITERT_RETURN_IF_ERROR(raii_graph->AnnotateGraph(
          gt::GraphDirectiveAnnotations::kPreferCoherent.data(), "1"));
    }

    LITERT_LOG(LITERT_INFO,
               "Successfully applied Darwinn options as graph annotations");
  }

  graph = raii_graph.release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchDeviceContextT::DestroyGraph(
  LiteRtDispatchGraph graph) {
  GT_LOG_RETURN_IF_NULL(graph);

  thr_graphs_.erase(graph->thr_graph());

  GT_LOG_RETURN_IF_SB_ERROR(thrGraphDelete(graph->thr_graph()),
                            "Failed to delete SB graph");

  delete graph;
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
        thrLoadSqContainerFdWithOffset(
            thr_context_.get(), thr_type, bytecode_buffer.fd,
            bytecode_buffer.size, bytecode_buffer.offset,
            /*lazy_loading=*/false, &exec_handle),
        "Failed to load SQ container from fd %d with size %zu and offset %zu",
        bytecode_buffer.fd, bytecode_buffer.size, bytecode_buffer.offset);
  } else {
    const auto* sq_bytecode =
        static_cast<const std::byte*>(bytecode_buffer.base_addr) +
        bytecode_buffer.offset;

    GT_LOG_RETURN_IF_SB_ERROR(
        thrLoadSqContainer(
            thr_context_.get(), thr_type, sq_bytecode, bytecode_buffer.size,
            &exec_handle),
        "Failed to load SQ container from buffer with base 0x%p and size %zu",
        sq_bytecode, bytecode_buffer.size);
  }

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchDeviceContextT::UnloadExecutable(
    LiteRtDispatchExecutableHandle exec_handle) {
  GT_LOG_RETURN_IF_SB_ERROR(
      thrUnloadSqContainer(thr_context_.get(), exec_handle),
      "Failed to unload SQ container %" PRIu64, exec_handle);

  return kLiteRtStatusOk;
}
