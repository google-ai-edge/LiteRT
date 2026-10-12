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

#include "litert/vendors/mediatek/dispatch/litert_dispatch_device_context.h"

#include <sys/mman.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>

#include "neuron/api/NeuronAdapter.h"
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/vendors/mediatek/dispatch/file_backed_pages.h"

using litert::Error;

namespace {

// Weights are copied in chunks of this size so that, for weights in an mmapped
// model file, only one chunk of source pages is resident at a time.
constexpr size_t kWeightCopyChunkSize = 4 * 1024 * 1024;

}  // namespace

LiteRtDispatchDeviceContextT::~LiteRtDispatchDeviceContextT() {
  for (const auto& [key, shared_weight] : shared_weights_) {
    // The NeuronMemory must be freed before the buffer that backs it.
    (void)neuron_memory_registry_.Unregister(shared_weight.handle);
    runtime_context_->destroy_tensor_buffer(shared_weight.tensor_buffer);
  }
}

litert::Expected<LiteRtDispatchDeviceContextT::Ptr>
LiteRtDispatchDeviceContextT::Create(
    const LiteRtRuntimeContext* runtime_context,
    const litert::mediatek::NeuronAdapterApi& neuron_adapter_api) {
  return std::unique_ptr<LiteRtDispatchDeviceContextT>(
      new LiteRtDispatchDeviceContextT(runtime_context, neuron_adapter_api));
}

litert::Expected<LiteRtTensorBufferHandle>
LiteRtDispatchDeviceContextT::RegisterTensorBuffer(
    LiteRtTensorBuffer tensor_buffer) {
  LiteRtTensorBufferType tensor_buffer_type;
  LITERT_RETURN_IF_ERROR(runtime_context_->get_tensor_buffer_type(
      tensor_buffer, &tensor_buffer_type));

  if (tensor_buffer_type != kLiteRtTensorBufferTypeAhwb &&
      tensor_buffer_type != kLiteRtTensorBufferTypeDmaBuf) {
    return Error(
        kLiteRtStatusErrorUnsupported,
        absl::StrFormat("Unsupported buffer type %d", tensor_buffer_type));
  }

  size_t tensor_buffer_size;
  LITERT_RETURN_IF_ERROR(runtime_context_->get_tensor_buffer_size(
      tensor_buffer, &tensor_buffer_size));

  size_t tensor_buffer_offset;
  if (auto status = runtime_context_->get_tensor_buffer_offset(
          tensor_buffer, &tensor_buffer_offset);
      status != kLiteRtStatusOk) {
    if (status == kLiteRtStatusErrorNotFound) {
      tensor_buffer_offset = 0;
    } else {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "Failed to get buffer offset");
    }
  }

  LiteRtRankedTensorType tensor_type;
  LITERT_RETURN_IF_ERROR(runtime_context_->get_tensor_buffer_tensor_type(
      tensor_buffer, &tensor_type));

  // Strides are allowed as they are used for padding.
  if (tensor_type.layout.has_strides) {
    LITERT_LOG(LITERT_DEBUG, "Registering tensor buffer with strides");
  }

  switch (tensor_buffer_type) {
    case kLiteRtTensorBufferTypeAhwb:
#if LITERT_HAS_AHWB_SUPPORT
      AHardwareBuffer* ahwb;
      if (auto status =
              runtime_context_->get_tensor_buffer_ahwb(tensor_buffer, &ahwb);
          status != kLiteRtStatusOk) {
        return Error(status, "Failed to get AHWB");
      }
#else
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "AHardwareBuffer is not supported on this platform");
#endif  // LITERT_HAS_AHWB_SUPPORT
      NeuronMemory* neuron_memory;
#if LITERT_HAS_AHWB_SUPPORT
      if (neuron_adapter_api_.api().memory_create_from_ahwb(
              ahwb, &neuron_memory) != NEURON_NO_ERROR) {
        return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                  "Failed to create NeuronMemory from AHWB");
      }
      return neuron_memory_registry_.Register(neuron_memory, tensor_buffer_size,
                                              tensor_buffer_offset);
#else
      (void)neuron_adapter_api_;
      return litert::Unexpected(
          kLiteRtStatusErrorRuntimeFailure,
          "AHardwareBuffer is not supported on this platform");
#endif  // LITERT_HAS_AHWB_SUPPORT
      break;

    case kLiteRtTensorBufferTypeDmaBuf:

      int fd;
#if LITERT_HAS_DMABUF_SUPPORT
      void* addr;
      if (auto status = runtime_context_->get_tensor_buffer_dma_buf_buffer(
              tensor_buffer, &addr, &fd);
          status != kLiteRtStatusOk) {
        return Error(status, "Failed to get DMA-BUF");
      }
#else
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "DMA-BUF is not supported on this platform");
#endif  // LITERT_HAS_DMABUF_SUPPORT
      if (neuron_adapter_api_.api().memory_create_from_fd(
              tensor_buffer_size, /*protect*/ PROT_READ | PROT_WRITE, fd,
              tensor_buffer_offset, &neuron_memory) != NEURON_NO_ERROR) {
        return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                  "Failed to create NeuronMemory from DMA-BUF");
      }
      return neuron_memory_registry_.Register(neuron_memory, tensor_buffer_size,
                                              tensor_buffer_offset);
      break;

    default:
      LITERT_LOG(LITERT_ERROR, "Unsupported buffer type: %d",
                 tensor_buffer_type);
      return litert::Unexpected(kLiteRtStatusErrorUnsupported);
  }
}

litert::Expected<LiteRtDispatchDeviceContextT::NeuronMemoryInfo>
LiteRtDispatchDeviceContextT::GetOrCreateSharedWeightMemory(
    int fd, const void* weight_data, size_t weight_size, size_t padded_size) {
  if (weight_data == nullptr || weight_size == 0) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Invalid shared weight buffer");
  }
  const size_t alloc_size = std::max(weight_size, padded_size);
  const auto key = std::make_pair(weight_data, alloc_size);
  if (auto it = shared_weights_.find(key); it != shared_weights_.end()) {
    return GetNeuronMemoryInfo(it->second.handle);
  }

  LiteRtRankedTensorType weight_tensor_type = {};
  weight_tensor_type.element_type = kLiteRtElementTypeUInt8;
  weight_tensor_type.layout.rank = 1;
  weight_tensor_type.layout.dimensions[0] = static_cast<int32_t>(alloc_size);

  LiteRtTensorBuffer tensor_buffer = nullptr;
  LiteRtStatus alloc_status = runtime_context_->create_managed_tensor_buffer(
      /*env=*/nullptr, kLiteRtTensorBufferTypeAhwb, &weight_tensor_type,
      alloc_size, &tensor_buffer);
  if (alloc_status != kLiteRtStatusOk || tensor_buffer == nullptr) {
    alloc_status = runtime_context_->create_managed_tensor_buffer(
        /*env=*/nullptr, kLiteRtTensorBufferTypeDmaBuf, &weight_tensor_type,
        alloc_size, &tensor_buffer);
  }
  if (alloc_status != kLiteRtStatusOk || tensor_buffer == nullptr) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to allocate DMA-BUF/AHWB tensor buffer for shared "
                 "weights");
  }

  void* host_mem_addr = nullptr;
  if (auto lock_status = runtime_context_->lock_tensor_buffer(
          tensor_buffer, &host_mem_addr, kLiteRtTensorBufferLockModeWrite);
      lock_status != kLiteRtStatusOk || host_mem_addr == nullptr) {
    runtime_context_->destroy_tensor_buffer(tensor_buffer);
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to lock shared weight tensor buffer");
  }

  const size_t released = litert::mediatek::CopyAndReleaseFileBackedPages(
      fd, weight_data, host_mem_addr, weight_size, kWeightCopyChunkSize);
  if (alloc_size > weight_size) {
    std::memset(static_cast<uint8_t*>(host_mem_addr) + weight_size, 0,
                alloc_size - weight_size);
  }

  if (auto unlock_status =
          runtime_context_->unlock_tensor_buffer(tensor_buffer);
      unlock_status != kLiteRtStatusOk) {
    runtime_context_->destroy_tensor_buffer(tensor_buffer);
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to unlock shared weight tensor buffer");
  }

  auto handle = RegisterTensorBuffer(tensor_buffer);
  if (!handle) {
    runtime_context_->destroy_tensor_buffer(tensor_buffer);
    return handle.Error();
  }
  LITERT_LOG(LITERT_INFO,
             "Copied %zu bytes of MediaTek weights to device memory; released "
             "%zu bytes of file-backed pages",
             weight_size, released);

  shared_weights_.emplace(key, SharedWeight{tensor_buffer, *handle});
  return GetNeuronMemoryInfo(*handle);
}

LiteRtDispatchDeviceContextT::NeuronMemoryRegistry::~NeuronMemoryRegistry() {
  for (auto i = 0; i < records_.size(); ++i) {
    auto& record = records_[i];
    if (record.neuron_memory != nullptr) {
      neuron_adapter_api_.api().memory_free(record.neuron_memory);
    }
  }
}

LiteRtTensorBufferHandle
LiteRtDispatchDeviceContextT::NeuronMemoryRegistry::Register(
    NeuronMemory* neuron_memory, size_t size, size_t offset) {
  int dest_index = -1;
  for (auto i = 0; i < records_.size(); ++i) {
    if (!records_[i].neuron_memory) {
      dest_index = i;
      break;
    }
  }
  if (dest_index < 0) {
    dest_index = records_.size();
    records_.push_back({});
  }
  auto& dest = records_[dest_index];
  dest = {neuron_memory, size, offset};
  return dest_index;
}

litert::Expected<void>
LiteRtDispatchDeviceContextT::NeuronMemoryRegistry::Unregister(
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  auto record = Find(tensor_buffer_handle);
  if (!record) {
    return record.Error();
  } else {
    auto& mem = (*record)->neuron_memory;
    neuron_adapter_api_.api().memory_free(mem);
    mem = nullptr;
    return {};
  }
}

litert::Expected<LiteRtDispatchDeviceContextT::NeuronMemoryInfo*>
LiteRtDispatchDeviceContextT::NeuronMemoryRegistry::Find(
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (tensor_buffer_handle < 0 || tensor_buffer_handle >= records_.size()) {
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                              "Invalid tensor buffer handle");
  }
  return &records_[tensor_buffer_handle];
}
