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
#include <memory>
#include <string>
#include <utility>

#if __ANDROID__
#include <android/hardware_buffer.h>
#endif  // __ANDROID__

#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/options/darwinn_options.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/google_tensor/dispatch/litert_dispatch_graph.h"
#include "litert/vendors/google_tensor/dispatch/sb_api.h"
#include "litert/vendors/google_tensor/dispatch/sb_dispatch_annotations.h"

using litert::Error;
using litert::Expected;
using litert::Unexpected;

LiteRtDispatchDeviceContextT::~LiteRtDispatchDeviceContextT() {
  if (!thr_graphs_.empty()) {
    auto thr_graph_delete = southbound_.api().thr_graph_delete;
    if (!thr_graph_delete) {
      LITERT_LOG(LITERT_ERROR, "thr_graph_delete not found");
    } else {
      for (auto* thr_graph : thr_graphs_) {
        thr_graph_delete(thr_graph);
      }
    }
  }

  if (thr_context_) {
    auto thr_context_delete = southbound_.api().thr_context_delete;
    if (!thr_context_delete) {
      LITERT_LOG(LITERT_ERROR, "thr_context_delete not found");
    } else {
      thr_context_delete(thr_context_);
    }
  }
}

Expected<LiteRtDispatchDeviceContextT::Ptr>
LiteRtDispatchDeviceContextT::Create(
    const litert::google_tensor::Southbound& southbound,
    const litert::DarwinnRuntimeOptions* darwinn_options) {
  Ptr device_context(new LiteRtDispatchDeviceContextT(southbound));

  auto thr_context_create = southbound.api().thr_context_create;
  if (!thr_context_create) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "thr_context_create not found");
  }

  device_context->thr_context_ = thr_context_create();
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

  if (tensor_buffer_type != kLiteRtTensorBufferTypeAhwb) {
    return Error(kLiteRtStatusErrorUnsupported, "Unsupported buffer type");
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

  AHardwareBuffer* ahwb;
#if LITERT_HAS_AHWB_SUPPORT
  if (auto status = LiteRtGetTensorBufferAhwb(tensor_buffer, &ahwb);
      status != kLiteRtStatusOk) {
    return Error(status, "Failed to get AHWB");
  }
#else
  return Error(kLiteRtStatusErrorRuntimeFailure,
               "AHardwareBuffer is not supported on this platform");
#endif

  ThrBufferHandle thr_buffer_handle;

  if (tensor_buffer_offset == 0) {
    auto thr_register_buffer = southbound_.api().thr_register_buffer;
    if (!thr_register_buffer) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "thr_register_buffer not found");
    }

    if (auto status = thr_register_buffer(
            thr_context_, ThrBufferType::kThrBufferTypeAHardwareBuffer, ahwb,
            tensor_buffer_size, &thr_buffer_handle);
        status != kThrStatusSuccess) {
      LITERT_LOG(LITERT_ERROR, "thr_register_buffer failed: %d", status);
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "thr_register_buffer failed");
    }

  } else {
    auto thr_register_buffer_with_offset =
        southbound_.api().thr_register_buffer_with_offset;
    if (!thr_register_buffer_with_offset) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "thr_register_buffer_with_offset not found");
    }

    if (auto status = thr_register_buffer_with_offset(
            thr_context_, ThrBufferType::kThrBufferTypeAHardwareBuffer, ahwb,
            tensor_buffer_offset, tensor_buffer_size, &thr_buffer_handle);
        status != kThrStatusSuccess) {
      LITERT_LOG(LITERT_ERROR, "thr_register_buffer_with_offset failed: %d",
                 status);
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "thr_register_buffer_with_offset failed");
    }
  }

  return thr_buffer_handle;
}

litert::Expected<void> LiteRtDispatchDeviceContextT::UnregisterTensorBuffer(
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  auto thr_unregister_buffer = southbound_.api().thr_unregister_buffer;
  if (!thr_unregister_buffer) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_unregister_buffer not found");
  }

  ThrBufferHandle thr_buffer_handle = tensor_buffer_handle;
  if (auto status = thr_unregister_buffer(thr_context_, thr_buffer_handle);
      status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR, "thr_unregister_buffer failed: %d", status);
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_unregister_buffer failed");
  }

  return {};
}

litert::Expected<LiteRtDispatchGraph>
LiteRtDispatchDeviceContextT::CreateGraph() {
  auto thr_graph_create = southbound_.api().thr_graph_create;
  if (!thr_graph_create) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_graph_create not found");
  }

  ThrGraph* thr_graph = thr_graph_create(thr_context_);
  if (!thr_graph) {
    return Error(kLiteRtStatusErrorRuntimeFailure, "thr_graph_create failed");
  }

  auto* graph = new LiteRtDispatchGraphT(southbound_, thr_graph, this);

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
  auto thr_graph_delete = southbound_.api().thr_graph_delete;
  if (!thr_graph_delete) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_graph_delete not found");
  }

  thr_graphs_.erase(graph->thr_graph());

  ThrGraph* thr_graph = graph->thr_graph();
  if (auto status = thr_graph_delete(thr_graph); status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR, "thr_graph_destroy failed: %d", status);
    return Error(kLiteRtStatusErrorRuntimeFailure, "thr_graph_destroy failed");
  }

  delete graph;
  return {};
}

litert::Expected<LiteRtDispatchExecutableHandle>
LiteRtDispatchDeviceContextT::LoadExecutable(
    LiteRtDispatchExecutableType type, const LiteRtMemBuffer* bytecode_buffer) {
  auto thr_load_sq_container = southbound_.api().thr_load_sq_container;
  if (!thr_load_sq_container) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_load_sq_container not found");
  }

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
    status = southbound_.api().thr_load_sq_container_fd(
        thr_context_, thr_type, bytecode_buffer->fd, bytecode_buffer->size,
        lazy_loading, &sq_handle);
  } else {
    auto bytecode_ptr =
        static_cast<const uint8_t*>(bytecode_buffer->base_addr) +
        bytecode_buffer->offset;
    status = southbound_.api().thr_load_sq_container(
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
  auto thr_unload_sq_container = southbound_.api().thr_unload_sq_container;
  if (!thr_unload_sq_container) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_unload_sq_container not found");
  }

  ThrSqContainerHandle sq_handle = exec_handle;
  if (auto status = thr_unload_sq_container(thr_context_, sq_handle);
      status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR, "thr_unload_sq_container failed: %d", status);
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_unload_sq_container failed");
  }

  return {};
}
