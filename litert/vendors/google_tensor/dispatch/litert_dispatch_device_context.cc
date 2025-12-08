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

#include <cstddef>
#include <cstdint>
#include <memory>
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
#include "litert/cc/options/litert_darwinn_options.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/google_tensor/dispatch/litert_dispatch_graph.h"
#include "litert/vendors/google_tensor/dispatch/sb_api.h"
#include "litert/vendors/google_tensor/dispatch/sb_dispatch_annotations.h"

using litert::Error;
using litert::Expected;

LiteRtDispatchDeviceContextT::~LiteRtDispatchDeviceContextT() {
  for (auto* thr_graph : thr_graphs_) {
    thrGraphDelete(thr_graph);
  }

  if (thr_context_) {
    thrContextDelete(thr_context_);
  }
}

Expected<LiteRtDispatchDeviceContextT::Ptr>
LiteRtDispatchDeviceContextT::Create(
    const litert::DarwinnRuntimeOptions* darwinn_options) {
  Ptr device_context(new LiteRtDispatchDeviceContextT());

  device_context->thr_context_ = thrContextCreate();

  if (auto status = thrVendorSetSystemAttributeStr(
          device_context->thr_context_, "edgetpu_use_tpu_tachyon", "1");
      status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR, "Failed to enable Tachyon");
    return Error(kLiteRtStatusErrorRuntimeFailure, "Failed to enable Tachyon");
  }

  // Store Darwinn options to be applied to graphs later
  if (darwinn_options) {
    DarwinnOptionsData options_data;

    // Extract inference power state if available
    if (auto power_state = darwinn_options->GetInferencePowerState();
        power_state) {
      options_data.inference_power_state = *power_state;
    }

    // Extract inference memory power state if available
    if (auto mem_power_state = darwinn_options->GetInferenceMemoryPowerState();
        mem_power_state) {
      options_data.inference_memory_power_state = *mem_power_state;
    }

    // Extract inference priority if available
    if (auto priority = darwinn_options->GetInferencePriority(); priority) {
      options_data.inference_priority = *priority;
    }

    // Extract atomic inference if available
    if (auto atomic = darwinn_options->GetAtomicInference(); atomic) {
      options_data.atomic_inference = *atomic;
    }

    // Extract prefer coherent if available
    if (auto prefer_coherent = darwinn_options->GetPreferCoherent();
        prefer_coherent) {
      options_data.prefer_coherent = *prefer_coherent;
    }

    device_context->darwinn_options_ = std::move(options_data);
    LITERT_LOG(LITERT_INFO,
               "Darwinn runtime options will be applied to graphs");
  }
  return device_context;
}

Expected<LiteRtTensorBufferHandle>
LiteRtDispatchDeviceContextT::RegisterTensorBuffer(
    LiteRtTensorBuffer tensor_buffer) {
  LiteRtTensorBufferType tensor_buffer_type;
  if (auto status =
          LiteRtGetTensorBufferType(tensor_buffer, &tensor_buffer_type);
      status != kLiteRtStatusOk) {
    return Error(status, "Failed to get buffer type");
  }

  size_t tensor_buffer_size;
  if (auto status =
          LiteRtGetTensorBufferSize(tensor_buffer, &tensor_buffer_size);
      status != kLiteRtStatusOk) {
    return Error(status, "Failed to get buffer size");
  }

  size_t tensor_buffer_offset;
  if (auto status =
          LiteRtGetTensorBufferOffset(tensor_buffer, &tensor_buffer_offset);
      status != kLiteRtStatusOk) {
    if (status == kLiteRtStatusErrorNotFound) {
      tensor_buffer_offset = 0;
    } else {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "Failed to get buffer offset");
    }
  }

  LiteRtRankedTensorType tensor_type;
  if (auto status =
          LiteRtGetTensorBufferTensorType(tensor_buffer, &tensor_type);
      status != kLiteRtStatusOk) {
    return Error(status, "Failed to get tensor buffer type");
  }

  if (tensor_type.layout.has_strides) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Tensor strides are not supported");
  }

  ThrBufferType thr_buffer_type;
  void* buffer;
#if LITERT_HAS_AHWB_SUPPORT
  if (tensor_buffer_type == kLiteRtTensorBufferTypeAhwb) {
    thr_buffer_type = kThrBufferTypeAHardwareBuffer;

    if (auto status = LiteRtGetTensorBufferAhwb(
            tensor_buffer, reinterpret_cast<AHardwareBuffer**>(&buffer));
        status != kLiteRtStatusOk) {
      return Error(status, "Failed to get AHWB");
    }
  } else {
#else
  if (tensor_buffer_type == kLiteRtTensorBufferTypeHostMemory) {
    thr_buffer_type = kThrBufferTypeHostMemory;

    if (auto status = LiteRtGetTensorBufferHostMemory(tensor_buffer, &buffer);
        status != kLiteRtStatusOk) {
      return Error(status, "Failed to get host memory");
    }
  } else {
#endif
    return Error(kLiteRtStatusErrorUnsupported, "Unsupported buffer type");
  }

  ThrBufferHandle thr_buffer_handle;
  if (auto status = thrRegisterBufferWithOffset(
            thr_context_, thr_buffer_type, buffer, tensor_buffer_offset,
            tensor_buffer_size, &thr_buffer_handle);
        status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR, "thr_register_buffer_with_offset failed: %d",
               status);
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_register_buffer_with_offset failed");
  }

  return thr_buffer_handle;
}

litert::Expected<void> LiteRtDispatchDeviceContextT::UnregisterTensorBuffer(
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  ThrBufferHandle thr_buffer_handle = tensor_buffer_handle;
  if (auto status = thrUnregisterBuffer(thr_context_, thr_buffer_handle);
      status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR, "thr_unregister_buffer failed: %d", status);
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_unregister_buffer failed");
  }

  return {};
}

litert::Expected<LiteRtDispatchGraph>
LiteRtDispatchDeviceContextT::CreateGraph() {
  ThrGraph* thr_graph = thrGraphCreate(thr_context_);
  if (!thr_graph) {
    return Error(kLiteRtStatusErrorRuntimeFailure, "thr_graph_create failed");
  }

  auto* graph = new LiteRtDispatchGraphT(thr_graph, this);

  // Apply Darwinn options as graph annotations
  if (darwinn_options_.has_value()) {
    const auto& options = darwinn_options_.value();

    // Apply inference power state annotation
    if (options.inference_power_state.has_value()) {
      auto status = graph->AnnotateGraph(
          litert::google_tensor::DispatchDirectiveAnnotations::
              kEdgetpuDevicePowerState.data(),
          std::to_string(options.inference_power_state.value()).c_str());
      if (!status) {
        LITERT_LOG(LITERT_WARNING,
                   "Failed to apply inference_power_state annotation: %s",
                   status.Error().Message().c_str());
      }
    }

    // Apply inference memory power state annotation
    if (options.inference_memory_power_state.has_value()) {
      auto status = graph->AnnotateGraph(
          litert::google_tensor::DispatchDirectiveAnnotations::
              kEdgetpuMemoryPowerState.data(),
          std::to_string(options.inference_memory_power_state.value()).c_str());
      if (!status) {
        LITERT_LOG(
            LITERT_WARNING,
            "Failed to apply inference_memory_power_state annotation: %s",
            status.Error().Message().c_str());
      }
    }
    // Apply inference priority annotation
    if (options.inference_priority.has_value()) {
      auto status = graph->AnnotateGraph(
          litert::google_tensor::DispatchDirectiveAnnotations::kPriority.data(),
          std::to_string(options.inference_priority.value()).c_str());
      if (!status) {
        LITERT_LOG(LITERT_WARNING,
                   "Failed to apply inference_priority annotation: %s",
                   status.Error().Message().c_str());
      }
    }

    // Apply atomic inference annotation
    if (options.atomic_inference) {
      auto status = graph->AnnotateGraph(
          litert::google_tensor::DispatchDirectiveAnnotations::
              kEdgetpuAtomicInference.data(),
          "1");
      if (!status) {
        LITERT_LOG(LITERT_WARNING,
                   "Failed to apply atomic_inference annotation: %s",
                   status.Error().Message().c_str());
      }
    }

    // Apply prefer coherent annotation
    if (options.prefer_coherent) {
      auto status = graph->AnnotateGraph(
          litert::google_tensor::GraphDirectiveAnnotations::kPreferCoherent
              .data(),
          "1");
      if (!status) {
        LITERT_LOG(LITERT_WARNING,
                   "Failed to apply prefer_coherent annotation: %s",
                   status.Error().Message().c_str());
      }
    }

    LITERT_LOG(LITERT_INFO,
               "Applied Darwinn runtime options as graph annotations");
  }

  return graph;
}

litert::Expected<void> LiteRtDispatchDeviceContextT::DestroyGraph(
    LiteRtDispatchGraph graph) {
  thr_graphs_.erase(graph->thr_graph());

  ThrGraph* thr_graph = graph->thr_graph();
  if (auto status = thrGraphDelete(thr_graph); status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR, "thr_graph_destroy failed: %d", status);
    return Error(kLiteRtStatusErrorRuntimeFailure, "thr_graph_destroy failed");
  }

  delete graph;
  return {};
}

litert::Expected<LiteRtDispatchExecutableHandle>
LiteRtDispatchDeviceContextT::LoadExecutable(
    LiteRtDispatchExecutableType type, const LiteRtMemBuffer* bytecode_buffer) {
  ThrSqContainerType thr_type;
  switch (type) {
    case kLiteRtDispatchExecutableTypeDspLibrary:
      thr_type = kThrSqContainerTypeFunctionLibrary;
      break;
    case kLiteRtDispatchExecutableTypeMlModel:
      thr_type = kThrSqContainerTypeMlModel;
      break;
    default:
      LITERT_LOG(LITERT_ERROR, "Unexpected executable type: %d", type);
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "Unexpected executable type");
  }

  ThrSqContainerHandle sq_handle;
  ThrStatus status;
  if (bytecode_buffer->fd >= 0 &&
      // Unfortunately thrLoadSqContainerFd doesn't support passing an
      // offset. So if the offset is non-zero, we fallback to passing a CPU
      // memory address right below.
      (bytecode_buffer->offset == 0)) {
    bool lazy_loading = false;
    status = thrLoadSqContainerFd(
        thr_context_, thr_type, bytecode_buffer->fd, bytecode_buffer->size,
        lazy_loading, &sq_handle);
  } else {
    auto bytecode_ptr =
        static_cast<const uint8_t*>(bytecode_buffer->base_addr) +
        bytecode_buffer->offset;
    status = thrLoadSqContainer(
        thr_context_, thr_type, bytecode_ptr, bytecode_buffer->size,
        &sq_handle);
  }
  if (status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR, "thr_load_sq_container failed: %d", status);
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_load_sq_container failed");
  }

  return sq_handle;
}

litert::Expected<void> LiteRtDispatchDeviceContextT::UnloadExecutable(
    LiteRtDispatchExecutableHandle exec_handle) {
  ThrSqContainerHandle sq_handle = exec_handle;
  if (auto status = thrUnloadSqContainer(thr_context_, sq_handle);
      status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR, "thr_unload_sq_container failed: %d", status);
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_unload_sq_container failed");
  }

  return {};
}
