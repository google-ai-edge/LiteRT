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

#include "litert/vendors/qualcomm/dispatch/litert_dispatch_device_context.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "HTP/QnnHtpMem.h"  // from @qairt
#include "QnnCommon.h"  // from @qairt
#include "QnnMem.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/qualcomm/common.h"
#include "litert/vendors/qualcomm/core/backends/qnn_backend.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/qnn_manager.h"

using litert::Expected;
using litert::Unexpected;
using litert::qnn::QnnManager;

Expected<LiteRtDispatchDeviceContextT::Ptr>
LiteRtDispatchDeviceContextT::Create(
    const LiteRtRuntimeContext* runtime_context, QnnManager& qnn,
    ::qnn::QnnBackend& qnn_backend) {
  return Ptr(
      new LiteRtDispatchDeviceContextT(runtime_context, qnn, qnn_backend));
}

LiteRtDispatchDeviceContextT::~LiteRtDispatchDeviceContextT() {
  // The QNN contexts in context_cache_ are freed right after this body runs.
  // Deregister the memory handles still registered against them first. Memory
  // handles of contexts that this device context does not own (JIT contexts)
  // are left alone.
  tensor_buffer_registry_.ForEach([this](TensorBufferRegistryEntry& entry) {
    std::vector<Qnn_ContextHandle_t> owned_contexts;
    for (const auto& handle : entry.qnn_mem_handles) {
      if (handle.owned && OwnsContext(handle.context_handle)) {
        owned_contexts.push_back(handle.context_handle);
      }
    }
    for (auto context_handle : owned_contexts) {
      if (auto status = DeregisterMemHandles(entry, context_handle); !status) {
        LITERT_LOG(LITERT_WARNING, "%s", status.Error().Message().c_str());
      }
    }
  });
}

bool LiteRtDispatchDeviceContextT::OwnsContext(
    Qnn_ContextHandle_t context_handle) const {
  for (const auto& [key, handle] : context_cache_) {
    if (handle && handle->Get() == context_handle) {
      return true;
    }
  }
  return false;
}

Expected<LiteRtTensorBuffer> LiteRtDispatchDeviceContextT::GetTensorBuffer(
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  auto registry_entry = tensor_buffer_registry_.Get(tensor_buffer_handle);
  if (!registry_entry) {
    return Unexpected(registry_entry.Error());
  }

  return (*registry_entry)->tensor_buffer;
}

Expected<Qnn_MemHandle_t> LiteRtDispatchDeviceContextT::GetMemHandle(
    LiteRtTensorBufferHandle tensor_buffer_handle, const Qnn_Tensor_t& tensor,
    Qnn_ContextHandle_t context_handle) {
  auto registry_entry = tensor_buffer_registry_.Get(tensor_buffer_handle);
  if (!registry_entry) {
    return Unexpected(registry_entry.Error());
  }
  TensorBufferRegistryEntry& entry = **registry_entry;

  for (const auto& handle : entry.qnn_mem_handles) {
    if (handle.context_handle == context_handle) {
      return handle.mem_handle;
    }
  }

  auto qnn_mem_handle =
      RegisterTensorBuffer(entry.tensor_buffer, tensor, context_handle);
  if (qnn_mem_handle) {
    entry.qnn_mem_handles.push_back(
        {context_handle, *qnn_mem_handle, /*owned=*/true});
    return *qnn_mem_handle;
  }

  // Registering with this context failed. If the buffer is already registered
  // with another context, fall back to that memory handle. This only works
  // when both contexts live in the same HTP PD.
  for (const auto& handle : entry.qnn_mem_handles) {
    if (!handle.owned) {
      continue;
    }
    const QnnMemHandleEntry reused{context_handle, handle.mem_handle,
                                   /*owned=*/false};
    LITERT_LOG(LITERT_WARNING,
               "Failed to register tensor buffer %p with QNN context %p (%s); "
               "reusing its memory handle %p of QNN context %p",
               entry.tensor_buffer, context_handle,
               qnn_mem_handle.Error().Message().c_str(), handle.mem_handle,
               handle.context_handle);
    entry.qnn_mem_handles.push_back(reused);
    return reused.mem_handle;
  }

  return Unexpected(qnn_mem_handle.Error());
}

Expected<void> LiteRtDispatchDeviceContextT::DeregisterMemHandles(
    TensorBufferRegistryEntry& entry, Qnn_ContextHandle_t context_handle) {
  Qnn_ErrorHandle_t first_error = QNN_SUCCESS;
  std::vector<Qnn_MemHandle_t> deregistered;
  auto& handles = entry.qnn_mem_handles;
  for (auto it = handles.begin(); it != handles.end();) {
    if (context_handle != nullptr && it->context_handle != context_handle) {
      ++it;
      continue;
    }
    if (it->owned) {
      Qnn_MemHandle_t mem_handle = it->mem_handle;
      LITERT_LOG(LITERT_DEBUG,
                 "Deregistering tensor buffer %p from QNN context %p",
                 entry.tensor_buffer, it->context_handle);
      if (auto status = qnn_manager_.Api()->memDeRegister(&mem_handle, 1UL);
          status != QNN_SUCCESS && first_error == QNN_SUCCESS) {
        first_error = status;
      }
      deregistered.push_back(it->mem_handle);
    }
    it = handles.erase(it);
  }

  // Drop the entries that reuse a memory handle which is gone now.
  handles.erase(std::remove_if(
                    handles.begin(), handles.end(),
                    [&deregistered](const QnnMemHandleEntry& h) {
                      return !h.owned &&
                             std::find(deregistered.begin(), deregistered.end(),
                                       h.mem_handle) != deregistered.end();
                    }),
                handles.end());

  if (first_error != QNN_SUCCESS) {
    return Unexpected(
        kLiteRtStatusErrorRuntimeFailure,
        absl::StrFormat(
            "Failed to unregister tensor buffer, QNN error code: %d",
            first_error));
  }
  return {};
}

Expected<Qnn_MemHandle_t> LiteRtDispatchDeviceContextT::RegisterTensorBuffer(
    LiteRtTensorBuffer tensor_buffer, const Qnn_Tensor_t& tensor,
    Qnn_ContextHandle_t context_handle) {
  LITERT_LOG(LITERT_DEBUG, "Registering tensor buffer %p with QNN context %p",
             tensor_buffer, context_handle);
  LiteRtTensorBufferType tensor_buffer_type;
  if (auto status = runtime_context_->get_tensor_buffer_type(
          tensor_buffer, &tensor_buffer_type);
      status != kLiteRtStatusOk) {
    return Unexpected(status, "Failed to get tensor buffer type");
  }

  LiteRtRankedTensorType tensor_type;
  if (auto status = runtime_context_->get_tensor_buffer_tensor_type(
          tensor_buffer, &tensor_type);
      status != kLiteRtStatusOk) {
    return Unexpected(status, "Failed to get tensor buffer's type");
  }

  auto element_type =
      static_cast<enum litert::ElementType>(tensor_type.element_type);
  Qnn_DataType_t tensor_data_type;
  if (auto status = LegalizeElementType(element_type, &tensor_data_type);
      status != kLiteRtStatusOk) {
    return Unexpected(status, "Failed to legalize datatype");
  }

  uint32_t tensor_rank = tensor_type.layout.rank;
  uint32_t* tensor_dimensions = reinterpret_cast<uint32_t*>(
      const_cast<int32_t*>(tensor_type.layout.dimensions));
  if (tensor_type.layout.has_strides) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Tensor strides are not supported by QNN");
  }

  void* buffer_host_addr;
  int buffer_fd;

  switch (tensor_buffer_type) {
    case kLiteRtTensorBufferTypeFastRpc:
#if LITERT_HAS_FASTRPC_SUPPORT
      if (auto status = runtime_context_->get_tensor_buffer_fast_rpc_buffer(
              tensor_buffer, &buffer_host_addr, &buffer_fd);
          status != kLiteRtStatusOk) {
        return Unexpected(status, "Failed to get FastRPC buffer");
      }
#else
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "FastRPC support is missing on this platform");
#endif  // LRT_HAS_FASTRPC_SUPPORT
      break;

    case kLiteRtTensorBufferTypeDmaBuf:
#if LITERT_HAS_DMABUF_SUPPORT
      if (auto status = runtime_context_->get_tensor_buffer_dma_buf_buffer(
              tensor_buffer, &buffer_host_addr, &buffer_fd);
          status != kLiteRtStatusOk) {
        return Unexpected(status, "Failed to get DMA-BUF buffer");
      }
#else
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "DmaBuf support is missing on this platform");
#endif  // LRT_HAS_DMABUF_SUPPORT
      break;

    default:
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Unsupported tensor buffer type");
  }

  Qnn_MemDescriptor_t mem_descriptor = {};
  // QNN does not support 0-dimensional tensors.
  std::array<uint32_t, 1> dim{1};
  if (tensor_rank == 0) {
    mem_descriptor.memShape = {1, dim.data(), nullptr};
  } else {
    mem_descriptor.memShape = {tensor_rank, tensor_dimensions, nullptr};
  }
  mem_descriptor.dataType = tensor_data_type;

  QnnMemHtp_Descriptor_t mem_htp_descriptor = {};
  switch (qnn_manager_.GetOptions().GetBackendType()) {
    case ::qnn::BackendType::kGpuBackend:
      // QnnGpu imports the DMA-BUF fd as OpenCL memory; it does not accept
      // QNN_MEM_TYPE_ION (returns QNN_MEM_ERROR_UNSUPPORTED_MEMTYPE, 8005).
      if (tensor_buffer_type != kLiteRtTensorBufferTypeDmaBuf) {
        return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                          "GPU backend only supports DMA-BUF tensor buffers");
      }
      mem_descriptor.memType = QNN_MEM_TYPE_DMA_BUF;
      mem_descriptor.dmaBufInfo =
          Qnn_MemDmaBufInfo_t{buffer_fd, buffer_host_addr};
      break;
    // DSP Backend only supports QNN_MEM_TYPE_ION.
    case ::qnn::BackendType::kDspBackend:
      mem_descriptor.memType = QNN_MEM_TYPE_ION;
      mem_descriptor.ionInfo.fd = buffer_fd;

      break;
    case ::qnn::BackendType::kHtpBackend:
      [[fallthrough]];
    default:
      size_t tensor_buffer_size;
      if (auto status = runtime_context_->get_tensor_buffer_size(
              tensor_buffer, &tensor_buffer_size);
          status != kLiteRtStatusOk) {
        return Unexpected(status, "Failed to get tensor buffer size");
      }

      size_t tensor_buffer_offset;
      if (auto status = runtime_context_->get_tensor_buffer_offset(
              tensor_buffer, &tensor_buffer_offset);
          status != kLiteRtStatusOk) {
        return Unexpected(status, "Failed to get tensor buffer offset");
      }

      mem_htp_descriptor.type = QNN_HTP_MEM_SHARED_BUFFER;
      mem_htp_descriptor.size = tensor_buffer_size;
      mem_htp_descriptor.sharedBufferConfig =
          QnnHtpMem_SharedBufferConfig_t{buffer_fd, tensor_buffer_offset};
      mem_descriptor.memType = QNN_MEM_TYPE_CUSTOM;
      mem_descriptor.customInfo = &mem_htp_descriptor;

      break;
  }

  if (context_handle == nullptr) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Missing QNN context handle");
  }

  Qnn_MemHandle_t mem_handle = nullptr;
  if (auto status = qnn_manager_.Api()->memRegister(
          context_handle, &mem_descriptor, 1UL, &mem_handle);
      status != QNN_SUCCESS) {
    return Unexpected(
        kLiteRtStatusErrorRuntimeFailure,
        absl::StrFormat("Failed to register tensor buffer, QNN error code: %d",
                        status));
  }

  if (!mem_handle) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to register buffer: null mem_handle");
  }

  return mem_handle;
}

Expected<const litert::qnn::QnnManager::ContextHandle&>
LiteRtDispatchDeviceContextT::GetOrCreateContext(
    const void* bytecode_ptr, size_t bytecode_size,
    Qnn_ProfileHandle_t profile_handle) {
  ContextCacheKey key{bytecode_ptr, bytecode_size};
  auto it = context_cache_.find(key);
  if (it != context_cache_.end()) {
    LITERT_LOG(LITERT_INFO,
               "Reusing cached QNN context for bytecode %p (size %zu)",
               bytecode_ptr, bytecode_size);
    return *(it->second);
  }

  LITERT_LOG(LITERT_INFO, "Creating new QNN context for bytecode %p (size %zu)",
             bytecode_ptr, bytecode_size);
  auto context_handle_expected = qnn_manager_.CreateContextHandle(
      qnn_backend_, QnnManager::DefaultContextConfigs(),
      absl::MakeSpan(static_cast<const uint8_t*>(bytecode_ptr), bytecode_size),
      profile_handle);

  if (!context_handle_expected) {
    return Unexpected(context_handle_expected.Error());
  }

  context_cache_[key] = std::make_unique<QnnManager::ContextHandle>(
      std::move(*context_handle_expected));
  return *(context_cache_[key]);
}

litert::Expected<void> LiteRtDispatchDeviceContextT::UnregisterTensorBuffer(
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  LITERT_ASSIGN_OR_RETURN(auto* registry_entry,
                          tensor_buffer_registry_.Get(tensor_buffer_handle));
  LITERT_LOG(LITERT_DEBUG, "Unregistering tensor buffer %p",
             registry_entry->tensor_buffer);
  // The memory handles are removed from the entry, so that unregistering the
  // buffer again releases nothing.
  auto deregistered =
      DeregisterMemHandles(*registry_entry, /*context_handle=*/nullptr);
  LITERT_RETURN_IF_ERROR(
      tensor_buffer_registry_.Unregister(tensor_buffer_handle));
  return deregistered;
}
