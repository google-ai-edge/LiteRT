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
#include <sys/mman.h>
#include <unistd.h>

#include <cstddef>
#include <optional>
#include <utility>

#include "litert/c/options/litert_google_tensor_options.h"

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
#include "litert/c/options/litert_google_tensor_options_type.h"
#include "litert/cc/litert_macros.h"
#include "litert/vendors/c/litert_dispatch.h"
#if LITERT_HAS_DARWINN_OPTIONS_SUPPORT
#include "litert/vendors/google_tensor/dispatch/google/darwinn_options_utils.h"
#include "litert/vendors/google_tensor/dispatch/google/litert_darwinn_runtime_options.h"
#endif  // LITERT_HAS_DARWINN_OPTIONS_SUPPORT
#include "litert/vendors/google_tensor/dispatch/dispatch_api_config.h"
#include "litert/vendors/google_tensor/dispatch/dispatch_api_macros.h"
#include "litert/vendors/google_tensor/dispatch/sb_api.h"
#include "litert/vendors/google_tensor/dispatch/sb_api_features.h"

namespace gt = litert::google_tensor;

namespace {
std::optional<LiteRtDispatchDeviceContextT::GoogleTensorOptionsData>
GetGoogleTensorOptions(const LiteRtRuntimeContext* runtime_context,
                       LiteRtOptions options) {
  if (options == nullptr) return std::nullopt;
  LiteRtOpaqueOptions opaque_opts = nullptr;
  runtime_context->get_opaque_options(options, &opaque_opts);
  if (opaque_opts == nullptr) return std::nullopt;

  void* payload = nullptr;
  if (runtime_context->find_opaque_options_data(
          opaque_opts, LrtGoogleTensorOptionsGetIdentifier(), &payload) !=
      kLiteRtStatusOk) {
    return std::nullopt;
  }

  // We assume the payload is a const char* pointing to a TOML string.
  LrtGoogleTensorOptions google_tensor_options;
  if (LrtCreateGoogleTensorOptionsFromToml(
          reinterpret_cast<const char*>(payload), &google_tensor_options) !=
      kLiteRtStatusOk) {
    return std::nullopt;
  }
  absl::Cleanup google_tensor_options_deleter = [google_tensor_options] {
    LrtDestroyGoogleTensorOptions(google_tensor_options);
  };
  LITERT_LOG(LITERT_INFO, "Found GoogleTensorOptions");

  // Parse the performance mode.
  LiteRtDispatchDeviceContextT::GoogleTensorOptionsData
      google_tensor_options_data;
  LiteRtGoogleTensorOptionsPerformanceMode performance_mode;
  if (LrtGoogleTensorOptionsGetPerformanceMode(
          google_tensor_options, &performance_mode) == kLiteRtStatusOk) {
    google_tensor_options_data.performance_mode = performance_mode;
  }

  return google_tensor_options_data;
}
}  // namespace

LiteRtStatus LiteRtDispatchDeviceContextT::Create(
    const LiteRtRuntimeContext* runtime_context, LiteRtOptions options,
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

  GT_LOG_RETURN_IF_SB_ERROR(
      thrVendorSetSystemAttributeStr(thr_context, "sq_container_load_method",
                                     "dmabuf"),
      "Failed to enable SB dmabuf SQ loading");

  // The returned instance must be allocated with `new`, as it will be
  // deallocated via `delete` in `Destroy`.
  device_context =
      new LiteRtDispatchDeviceContextT(runtime_context, thr_context);

#if LITERT_HAS_DARWINN_OPTIONS_SUPPORT
  std::optional<litert::LiteRtDarwinnRuntimeOptionsT> options_data =
      gt::GetDarwinnOptionsData(runtime_context, options);
  LITERT_RETURN_IF_ERROR(
      gt::ApplyDarwinnOptionsToDeviceContext(device_context, options_data));
  device_context->darwinn_options() = std::move(options_data);
#endif  // LITERT_HAS_DARWINN_OPTIONS_SUPPORT

  device_context->google_tensor_options() =
      GetGoogleTensorOptions(runtime_context, options);

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

  if (!mmap_regions_.empty()) {
    LITERT_LOG(LITERT_ERROR,
               "Cannot destroy device context with %zu mmap'd executable(s) "
               "still loaded",
               mmap_regions_.size());
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
  LITERT_RETURN_IF_ERROR(runtime_context_->get_tensor_buffer_tensor_type(
      tensor_buffer, &tensor_type));
  if (tensor_type.layout.has_strides) {
    LITERT_LOG(LITERT_ERROR, "Tensor strides are not supported");
    return kLiteRtStatusErrorUnsupported;
  }

  LiteRtTensorBufferType tensor_buffer_type;
  LITERT_RETURN_IF_ERROR(runtime_context_->get_tensor_buffer_type(
      tensor_buffer, &tensor_buffer_type));
  if (!gt::IsTensorBufferTypeSupported(tensor_buffer_type)) {
    LITERT_LOG(LITERT_ERROR, "Unsupported tensor buffer type %d",
               tensor_buffer_type);
    return kLiteRtStatusErrorUnsupported;
  }

  size_t tensor_buffer_size;
  LITERT_RETURN_IF_ERROR(runtime_context_->get_tensor_buffer_size(
      tensor_buffer, &tensor_buffer_size));

  size_t tensor_buffer_offset;
  LITERT_RETURN_IF_ERROR(runtime_context_->get_tensor_buffer_offset(
      tensor_buffer, &tensor_buffer_offset));

  switch (tensor_buffer_type) {
#if LITERT_HAS_AHWB_SUPPORT
    case kLiteRtTensorBufferTypeAhwb: {
      AHardwareBuffer* ahwb;
      LITERT_RETURN_IF_ERROR(
          runtime_context_->get_tensor_buffer_ahwb(tensor_buffer, &ahwb));

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
      LITERT_RETURN_IF_ERROR(runtime_context_->get_tensor_buffer_dma_buf_buffer(
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
      LITERT_RETURN_IF_ERROR(runtime_context_->get_tensor_buffer_host_memory(
          tensor_buffer, &host_memory_addr));

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
    if (GoogleTensorSouthBoundFeatureSupported(
            GoogleTensorSouthBoundFeature::kSqContainerFdWithOffset)) {
      GT_LOG_RETURN_IF_SB_ERROR(
          thrLoadSqContainerFdWithOffset(
              thr_context_, thr_type, bytecode_buffer.fd, bytecode_buffer.size,
              bytecode_buffer.offset, /*lazy_loading=*/false, &exec_handle),
          "Failed to load SQ container from FD with offset");
      return kLiteRtStatusOk;
    }

    // Old SouthBound, offset == 0: legacy FD entry point handles it directly.
    if (bytecode_buffer.offset == 0) {
      GT_LOG_RETURN_IF_SB_ERROR(
          thrLoadSqContainerFd(thr_context_, thr_type, bytecode_buffer.fd,
                               bytecode_buffer.size, /*lazy_loading=*/false,
                               &exec_handle),
          "Failed to load SQ container from FD");
      return kLiteRtStatusOk;
    }

    // Old SouthBound, offset > 0 (e.g. AOT .tflite with embedded DISPATCH_OP):
    // mmap a page-aligned region and load via the pointer-based API.
    const size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const size_t aligned_offset = bytecode_buffer.offset & ~(page_size - 1);
    const size_t offset_delta = bytecode_buffer.offset - aligned_offset;
    const size_t map_length = bytecode_buffer.size + offset_delta;
    void* mapped = mmap(nullptr, map_length, PROT_READ, MAP_PRIVATE,
                        bytecode_buffer.fd, static_cast<off_t>(aligned_offset));
    if (mapped == MAP_FAILED) {
      LITERT_LOG(LITERT_ERROR,
                 "Failed to mmap SQ bytecode (fd=%d, size=%zu, offset=%zu)",
                 bytecode_buffer.fd, bytecode_buffer.size,
                 bytecode_buffer.offset);
      return kLiteRtStatusErrorRuntimeFailure;
    }
    absl::Cleanup mmap_cleanup = [mapped, map_length] {
      munmap(mapped, map_length);
    };

    const void* bytecode_ptr =
        static_cast<const std::byte*>(mapped) + offset_delta;
    GT_LOG_RETURN_IF_SB_ERROR(
        thrLoadSqContainer(thr_context_, thr_type, bytecode_ptr,
                           bytecode_buffer.size, &exec_handle),
        "Failed to load SQ container from mmap'd buffer");

    mmap_regions_.push_back({exec_handle, mapped, map_length});
    std::move(mmap_cleanup).Cancel();
    return kLiteRtStatusOk;
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

  // Release any mmap'd region associated with this executable.
  for (auto it = mmap_regions_.begin(); it != mmap_regions_.end(); ++it) {
    if (it->exec_handle == exec_handle) {
      munmap(it->addr, it->length);
      mmap_regions_.erase(it);
      break;
    }
  }

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

LiteRtStatus LiteRtDispatchDeviceContextT::AnnotateSystemAttribute(
    const char* absl_nonnull key, const char* absl_nonnull value) {
  GT_LOG_RETURN_IF_SB_ERROR(
      thrVendorSetSystemAttributeStr(thr_context_, key, value),
      "Failed to set system attribute %s to %s", key, value);
  return kLiteRtStatusOk;
}
