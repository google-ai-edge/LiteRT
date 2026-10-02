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

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "HTP/QnnHtpMem.h"  // from @qairt
#include "QnnCommon.h"  // from @qairt
#include "QnnContext.h"  // from @qairt
#include "QnnMem.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
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
#include "litert/vendors/qualcomm/dispatch/active_functions.h"
#include "litert/vendors/qualcomm/dispatch/litert_dispatch_invocation_context.h"
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

Expected<void> LiteRtDispatchDeviceContextT::SetActiveFunctions(
    absl::flat_hash_set<std::string> active_functions) {
  if (!context_cache_.empty()) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Active functions must be set before any QNN context "
                      "is created");
  }
  active_functions_ = std::move(active_functions);
  return {};
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
    LiteRtTensorBufferHandle tensor_buffer_handle, const Qnn_Tensor_t& tensor) {
  auto registry_entry = tensor_buffer_registry_.Get(tensor_buffer_handle);
  if (!registry_entry) {
    return Unexpected(registry_entry.Error());
  }

  if (!(*registry_entry)->qnn_mem_handle) {
    auto qnn_mem_handle =
        RegisterTensorBuffer((*registry_entry)->tensor_buffer, tensor);
    if (!qnn_mem_handle) {
      return Unexpected(qnn_mem_handle.Error());
    }
    (*registry_entry)->qnn_mem_handle = *qnn_mem_handle;
  }

  return (*registry_entry)->qnn_mem_handle;
}

Expected<Qnn_MemHandle_t> LiteRtDispatchDeviceContextT::RegisterTensorBuffer(
    LiteRtTensorBuffer tensor_buffer, const Qnn_Tensor_t& tensor) {
  LITERT_LOG(LITERT_DEBUG, "Registering tensor buffer %p", tensor_buffer);
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

  if (invocation_context_ == nullptr) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Missing invocation context");
  }

  Qnn_ContextHandle_t context_handle = invocation_context_->GetContextHandle();

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
    absl::Span<const std::string> binary_graphs,
    absl::string_view function_name, Qnn_ProfileHandle_t profile_handle) {
  ContextCacheKey key{bytecode_ptr, bytecode_size};
  std::vector<CachedContext>& contexts = context_cache_[key];
  for (const CachedContext& context : contexts) {
    if (context.enabled_graphs.empty() ||
        context.enabled_graphs.contains(function_name)) {
      LITERT_LOG(LITERT_INFO,
                 "Reusing cached QNN context for bytecode %p (size %zu)",
                 bytecode_ptr, bytecode_size);
      return *context.handle;
    }
  }

  std::vector<std::string> graphs_to_enable;
  if (contexts.empty()) {
    graphs_to_enable = litert::qnn::SelectGraphsToEnable(
        binary_graphs, active_functions_, function_name);
  } else {
    // The existing context does not include `function_name`, which means that
    // the function was not reported as active. Load it in its own context
    // rather than failing.
    LITERT_LOG(LITERT_WARNING,
               "Function %s was not reported as active, creating a separate "
               "QNN context for it",
               std::string(function_name).c_str());
    graphs_to_enable = {std::string(function_name)};
  }

  const auto bytecode =
      absl::MakeSpan(static_cast<const uint8_t*>(bytecode_ptr), bytecode_size);
  std::unique_ptr<QnnManager::ContextHandle> context_handle;
  if (!graphs_to_enable.empty()) {
    LITERT_LOG(LITERT_INFO,
               "Creating new QNN context for bytecode %p (size %zu) with %zu "
               "of %zu graphs enabled: %s",
               bytecode_ptr, bytecode_size, graphs_to_enable.size(),
               binary_graphs.size(),
               absl::StrJoin(graphs_to_enable, ", ").c_str());
    std::vector<const char*> graph_names;
    graph_names.reserve(graphs_to_enable.size() + 1);
    for (const std::string& name : graphs_to_enable) {
      graph_names.push_back(name.c_str());
    }
    graph_names.push_back(nullptr);
    QnnContext_Config_t enable_graphs_config = QNN_CONTEXT_CONFIG_INIT;
    enable_graphs_config.option = QNN_CONTEXT_CONFIG_ENABLE_GRAPHS;
    enable_graphs_config.enableGraphs = graph_names.data();
    const QnnContext_Config_t* configs[] = {&enable_graphs_config, nullptr};
    if (auto handle = qnn_manager_.CreateContextHandle(
            qnn_backend_, absl::MakeSpan(configs), bytecode, profile_handle);
        handle) {
      context_handle =
          std::make_unique<QnnManager::ContextHandle>(std::move(*handle));
    } else {
      LITERT_LOG(LITERT_WARNING,
                 "Failed to create QNN context with selected graphs, "
                 "falling back to enabling all graphs: %s",
                 handle.Error().Message().c_str());
      graphs_to_enable.clear();
    }
  }

  if (!context_handle) {
    LITERT_LOG(LITERT_INFO,
               "Creating new QNN context for bytecode %p (size %zu)",
               bytecode_ptr, bytecode_size);
    LITERT_ASSIGN_OR_RETURN(
        auto handle, qnn_manager_.CreateContextHandle(
                         qnn_backend_, QnnManager::DefaultContextConfigs(),
                         bytecode, profile_handle));
    context_handle =
        std::make_unique<QnnManager::ContextHandle>(std::move(handle));
  }

  contexts.push_back(
      CachedContext{std::move(context_handle),
                    absl::flat_hash_set<std::string>(graphs_to_enable.begin(),
                                                     graphs_to_enable.end())});
  return *contexts.back().handle;
}

litert::Expected<void> LiteRtDispatchDeviceContextT::UnregisterTensorBuffer(
    LiteRtTensorBufferHandle tensor_buffer_handle, const Qnn_Tensor_t& tensor) {
  LITERT_ASSIGN_OR_RETURN(auto tensor_buffer,
                          GetTensorBuffer(tensor_buffer_handle));
  LITERT_LOG(LITERT_DEBUG, "Unregistering tensor buffer %p", tensor_buffer);
  LITERT_RETURN_IF_ERROR(
      tensor_buffer_registry_.Unregister(tensor_buffer_handle));
  LITERT_ASSIGN_OR_RETURN(auto mem_handle,
                          GetMemHandle(tensor_buffer_handle, tensor));
  if (auto status = qnn_manager_.Api()->memDeRegister(&mem_handle, 1UL);
      status != QNN_SUCCESS) {
    return Unexpected(
        kLiteRtStatusErrorRuntimeFailure,
        absl::StrFormat(
            "Failed to unregister tensor buffer, QNN error code: %d", status));
  }
  return {};
}
