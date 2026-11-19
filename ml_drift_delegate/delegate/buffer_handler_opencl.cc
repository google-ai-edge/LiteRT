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

#include "ml_drift_delegate/delegate/buffer_handler_opencl.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "ml_drift/cl/cl_command_queue.h"  // from @ml_drift
#include "ml_drift/cl/cl_context.h"  // from @ml_drift
#include "ml_drift/cl/cl_memory.h"  // from @ml_drift
#if LITERT_HAS_OPENGL_SUPPORT
#include "ml_drift/cl/gl_interop.h"  // from @ml_drift
#include "ml_drift/common/access_type.h"  // from @ml_drift
#endif
#include "ml_drift/cl/opencl_wrapper.h"  // from @ml_drift
#include "ml_drift/cl/tensor.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_tensor_buffer.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/internal/litert_tensor_buffer_utils.h"
#include "litert/cc/litert_macros.h"
#include "litert/runtime/tensor_buffer_lockstate.h"
#include "ml_drift_delegate/delegate/buffer_handler_utils.h"
#include <CL/cl.h>

using ::litert::internal::LockState;

namespace {

std::atomic<bool> force_host_alloc_for_testing{false};

std::string GetClDeviceStringInfo(cl_device_id device, cl_device_info info) {
  constexpr size_t kMaxDeviceInfoBytes = 4096;
  if (device == nullptr || ::ml_drift::cl::clGetDeviceInfo == nullptr) {
    return "";
  }
  size_t value_size = 0;
  if (::ml_drift::cl::clGetDeviceInfo(device, info, 0, nullptr, &value_size) !=
          CL_SUCCESS ||
      value_size == 0 || value_size > kMaxDeviceInfoBytes) {
    return "";
  }
  std::string result(value_size, '\0');
  if (::ml_drift::cl::clGetDeviceInfo(device, info, result.size(),
                                      result.data(), nullptr) != CL_SUCCESS) {
    return "";
  }
  while (!result.empty() && result.back() == '\0') {
    result.pop_back();
  }
  return result;
}

bool IsPowerVrContext(cl_context context) {
  if (force_host_alloc_for_testing.load(std::memory_order_relaxed)) {
    return true;
  }
  if (context == nullptr || ::ml_drift::cl::clGetContextInfo == nullptr ||
      ::ml_drift::cl::clGetDeviceInfo == nullptr) {
    return false;
  }
  cl_device_id device = nullptr;
  size_t returned_size = 0;
  if (::ml_drift::cl::clGetContextInfo(context, CL_CONTEXT_DEVICES,
                                       sizeof(device), &device,
                                       &returned_size) != CL_SUCCESS ||
      returned_size < sizeof(device) || device == nullptr) {
    return false;
  }
  cl_device_id parent_device = nullptr;
  if (::ml_drift::cl::clGetDeviceInfo(device, CL_DEVICE_PARENT_DEVICE,
                                      sizeof(parent_device), &parent_device,
                                      nullptr) == CL_SUCCESS &&
      parent_device != nullptr) {
    device = parent_device;
  }
  struct CachedDeviceEntry {
    cl_device_id device = nullptr;
    bool is_powervr = false;
  };
  static absl::NoDestructor<absl::Mutex> cache_mu;
  static std::array<CachedDeviceEntry, 8> cache;
  static size_t next_cache_slot = 0;
  {
    absl::MutexLock lock(*cache_mu);
    for (const CachedDeviceEntry& entry : cache) {
      if (entry.device == device) {
        return entry.is_powervr;
      }
    }
  }
  const std::string name = GetClDeviceStringInfo(device, CL_DEVICE_NAME);
  const std::string vendor = GetClDeviceStringInfo(device, CL_DEVICE_VENDOR);
  ::ml_drift::GpuInfo gpu_info;
  ::ml_drift::GetGpuInfoFromDeviceDescription(
      absl::StrCat(name, " ", vendor), ::ml_drift::GpuApi::kOpenCl, &gpu_info);
  const bool is_powervr = gpu_info.IsPowerVR();
  {
    absl::MutexLock lock(*cache_mu);
    cache[next_cache_slot % cache.size()] = {device, is_powervr};
    ++next_cache_slot;
  }
  return is_powervr;
}

bool TryCreateHostAllocPackedTensor(
    cl_context cl_ctx_handle, LiteRtTensorBufferType buffer_type,
    const ::ml_drift::TensorDescriptor& tensor_desc, size_t desc_bytes,
    size_t bytes, size_t packed_bytes, ::ml_drift::cl::Tensor* out_tensor) {
  // On PowerVR GPUs, standard `clCreateBuffer(CL_MEM_READ_WRITE)` maps with
  // uncached Write-Combined CPU page attributes on `clEnqueueMapBuffer`, making
  // CPU readback expensive. `CL_MEM_ALLOC_HOST_PTR` allocates zero-copy
  // Write-Back CPU-cached pages on PowerVR OpenCL drivers while avoiding
  // regressions on vendors (e.g. Adreno) where `CL_MEM_ALLOC_HOST_PTR` uses
  // Write-Combined host memory.
  if (buffer_type != kLiteRtTensorBufferTypeOpenClBufferPacked ||
      !IsPowerVrContext(cl_ctx_handle) ||
      ::ml_drift::cl::clCreateBuffer == nullptr) {
    return false;
  }
  const size_t alloc_bytes =
      std::max({bytes, packed_bytes, desc_bytes, size_t{1}});
  cl_int err = CL_SUCCESS;
  cl_mem mem = ::ml_drift::cl::clCreateBuffer(
      cl_ctx_handle, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, alloc_bytes,
      nullptr, &err);
  if (err == CL_SUCCESS && mem != nullptr) {
    *out_tensor =
        ::ml_drift::cl::Tensor(mem, /*memory_owner=*/true, tensor_desc);
    return true;
  }
  if (mem != nullptr && ::ml_drift::cl::clReleaseMemObject != nullptr) {
    ::ml_drift::cl::clReleaseMemObject(mem);
  }
  ABSL_LOG(WARNING) << "clCreateBuffer(CL_MEM_ALLOC_HOST_PTR) failed (" << err
                    << "); falling back to standard allocation.";
  return false;
}

// Move-only RAII wrapper around a retained `cl_command_queue` that guards
// against null OpenCL dispatch function pointers during teardown.
class RetainedCommandQueue {
 public:
  RetainedCommandQueue() = default;

  static LiteRtStatus Retain(cl_command_queue queue,
                             RetainedCommandQueue* out_queue) {
    if (out_queue == nullptr) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    bool owns_ref = false;
    if (queue != nullptr && ::ml_drift::cl::clRetainCommandQueue != nullptr) {
      const cl_int err = ::ml_drift::cl::clRetainCommandQueue(queue);
      if (err != CL_SUCCESS) {
        ABSL_LOG(ERROR) << "clRetainCommandQueue failed with: " << err;
        return kLiteRtStatusErrorRuntimeFailure;
      }
      owns_ref = true;
    }
    *out_queue = RetainedCommandQueue(queue, owns_ref);
    return kLiteRtStatusOk;
  }

  RetainedCommandQueue(const RetainedCommandQueue&) = delete;
  RetainedCommandQueue& operator=(const RetainedCommandQueue&) = delete;

  RetainedCommandQueue(RetainedCommandQueue&& other) noexcept
      : queue_(other.queue_), owns_ref_(other.owns_ref_) {
    other.queue_ = nullptr;
    other.owns_ref_ = false;
  }

  RetainedCommandQueue& operator=(RetainedCommandQueue&& other) noexcept {
    if (this != &other) {
      Reset();
      queue_ = other.queue_;
      owns_ref_ = other.owns_ref_;
      other.queue_ = nullptr;
      other.owns_ref_ = false;
    }
    return *this;
  }

  ~RetainedCommandQueue() { Reset(); }

  cl_command_queue queue() const { return queue_; }

 private:
  RetainedCommandQueue(cl_command_queue queue, bool owns_ref)
      : queue_(queue), owns_ref_(owns_ref) {}

  void Reset() {
    if (owns_ref_ && queue_ != nullptr &&
        ::ml_drift::cl::clReleaseCommandQueue != nullptr) {
      ::ml_drift::cl::clReleaseCommandQueue(queue_);
    }
    queue_ = nullptr;
    owns_ref_ = false;
  }

  cl_command_queue queue_ = nullptr;
  bool owns_ref_ = false;
};

// A `HwMemoryInfo` implementation for OpenCL Custom Buffer integration with
// atomic construction and deterministic RAII teardown.
class OpenClMemoryInfo : public HwMemoryInfo {
 public:
  static LiteRtStatus Create(const LiteRtRankedTensorType& tensor_type,
                             LiteRtTensorBufferType buffer_type,
                             size_t packed_bytes, cl_context context,
                             cl_command_queue command_queue,
                             ::ml_drift::cl::Tensor cl_tensor,
                             ::ml_drift::cl::CLMemory imported_cl_mem,
                             std::unique_ptr<OpenClMemoryInfo>* out_info) {
    if (out_info == nullptr) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    RetainedCommandQueue retained_queue;
    LITERT_RETURN_IF_ERROR(
        RetainedCommandQueue::Retain(command_queue, &retained_queue));
    out_info->reset(new OpenClMemoryInfo(tensor_type, buffer_type, packed_bytes,
                                         context, std::move(retained_queue),
                                         std::move(imported_cl_mem),
                                         std::move(cl_tensor)));
    return kLiteRtStatusOk;
  }

  OpenClMemoryInfo(const OpenClMemoryInfo&) = delete;
  OpenClMemoryInfo& operator=(const OpenClMemoryInfo&) = delete;
  OpenClMemoryInfo(OpenClMemoryInfo&&) = delete;
  OpenClMemoryInfo& operator=(OpenClMemoryInfo&&) = delete;

  ~OpenClMemoryInfo() {
    if (host_memory_ != nullptr) {
      if (buffer_type_ == kLiteRtTensorBufferTypeOpenClBufferPacked) {
        if (retained_queue_.queue() != nullptr &&
            ::ml_drift::cl::clEnqueueUnmapMemObject != nullptr) {
          ::ml_drift::cl::clEnqueueUnmapMemObject(
              retained_queue_.queue(), cl_tensor_.GetMemoryPtr(), host_memory_,
              0, nullptr, nullptr);
        }
      } else {
        litert_aligned_free(host_memory_);
      }
      host_memory_ = nullptr;
    }
  }

  LiteRtStatus Lock(LiteRtTensorBufferLockMode mode, void** host_memory_ptr) {
    LITERT_RETURN_IF_ERROR(
        lock_state_ == LockState::kUnlocked,
        litert::ErrorStatusBuilder(kLiteRtStatusErrorRuntimeFailure)
            << "The OpenCL memory is already locked.");

    const LockState new_lock_state = litert::internal::ToLockState(mode);
    const cl_command_queue command_queue = retained_queue_.queue();

    if (buffer_type_ == kLiteRtTensorBufferTypeOpenClBufferPacked) {
      if (host_memory_ == nullptr) {
        if (command_queue == nullptr ||
            ::ml_drift::cl::clEnqueueMapBuffer == nullptr) {
          ABSL_LOG(ERROR)
              << "OpenCL command queue is null; cannot map OpenClBufferPacked.";
          return kLiteRtStatusErrorRuntimeFailure;
        }
        cl_map_flags map_flags = CL_MAP_READ | CL_MAP_WRITE;
        if (mode == kLiteRtTensorBufferLockModeRead) {
          map_flags = CL_MAP_READ;
        } else if (mode == kLiteRtTensorBufferLockModeWrite) {
          // Use CL_MAP_WRITE (not CL_MAP_WRITE_INVALIDATE_REGION) so OpenCL
          // drivers that gate unmap CPU cache clean on
          // `map_flags & CL_MAP_WRITE` flush dirty CPU cache lines to DRAM/GPU.
          map_flags = CL_MAP_WRITE;
        }
        const size_t map_bytes = std::max<size_t>(packed_bytes_, 1);
        cl_int error_code = CL_SUCCESS;
        void* mapped_ptr = ::ml_drift::cl::clEnqueueMapBuffer(
            command_queue, cl_tensor_.GetMemoryPtr(), CL_TRUE, map_flags, 0,
            map_bytes, 0, nullptr, nullptr, &error_code);
        if (error_code != CL_SUCCESS || mapped_ptr == nullptr) {
          ABSL_LOG(ERROR) << "clEnqueueMapBuffer failed with: " << error_code;
          return kLiteRtStatusErrorRuntimeFailure;
        }
        host_memory_ = mapped_ptr;
      }
      *host_memory_ptr = host_memory_;
      lock_state_ = new_lock_state;
      return kLiteRtStatusOk;
    }

    if (host_memory_ == nullptr) {
      // Ensure the data is aligned.
      if (auto rc =
              posix_memalign(&host_memory_, LITERT_HOST_MEMORY_BUFFER_ALIGNMENT,
                             packed_bytes_);
          rc) {
        ABSL_LOG(ERROR) << "Failed to allocate aligned memory";
        return kLiteRtStatusErrorRuntimeFailure;
      }
    }
    *host_memory_ptr = host_memory_;

    // Don't download data if write only.
    if (new_lock_state == LockState::kWriteLocked) {
      lock_state_ = new_lock_state;
      return kLiteRtStatusOk;
    }

    ::ml_drift::cl::CLCommandQueue queue(command_queue,
                                         /*has_ownership=*/false);
    auto descriptor_with_data = cl_tensor_.GetDescriptor();
    LITERT_RETURN_IF_ERROR(
        cl_tensor_.ToDescriptor(&descriptor_with_data, &queue));

    ::litert::ml_drift::ConvertDataFromDescriptor(
        descriptor_with_data, host_memory_, tensor_type_.element_type);

    lock_state_ = new_lock_state;
    return kLiteRtStatusOk;
  }

  LiteRtStatus Unlock() {
    LITERT_RETURN_IF_ERROR(
        lock_state_ != LockState::kUnlocked,
        litert::ErrorStatusBuilder(kLiteRtStatusErrorRuntimeFailure)
            << "The OpenCL memory is already unlocked.");
    absl::Cleanup unlock = [this] { lock_state_ = LockState::kUnlocked; };

    const cl_command_queue command_queue = retained_queue_.queue();
    if (buffer_type_ == kLiteRtTensorBufferTypeOpenClBufferPacked) {
      if (command_queue == nullptr ||
          ::ml_drift::cl::clEnqueueUnmapMemObject == nullptr) {
        ABSL_LOG(ERROR)
            << "OpenCL command queue is null; cannot unmap OpenClBufferPacked.";
        return kLiteRtStatusErrorRuntimeFailure;
      }
      void* mapped_ptr = host_memory_;
      host_memory_ = nullptr;
      const cl_int error_code = ::ml_drift::cl::clEnqueueUnmapMemObject(
          command_queue, cl_tensor_.GetMemoryPtr(), mapped_ptr, 0, nullptr,
          nullptr);
      if (error_code != CL_SUCCESS) {
        ABSL_LOG(ERROR) << "clEnqueueUnmapMemObject failed with: "
                        << error_code;
        return kLiteRtStatusErrorRuntimeFailure;
      }
      return kLiteRtStatusOk;
    }

    // Don't upload data if read only.
    if (lock_state_ == LockState::kReadLocked) {
      return kLiteRtStatusOk;
    }

    ::ml_drift::cl::CLCommandQueue queue(command_queue,
                                         /*has_ownership=*/false);
    ::ml_drift::cl::CLContext context(context_, /*has_ownership=*/false);

    auto descriptor_with_data = cl_tensor_.GetDescriptor();
    ::litert::ml_drift::ConvertDataToDescriptor(
        host_memory_, descriptor_with_data, tensor_type_.element_type);
    auto status = cl_tensor_.UploadDescriptorData(descriptor_with_data, &queue);
    if (!status.ok()) {
      ABSL_LOG(ERROR) << "Failed to upload data to tensor: " << status;
      return kLiteRtStatusErrorRuntimeFailure;
    }
    return kLiteRtStatusOk;
  }

  LiteRtStatus Clear() {
    LITERT_RETURN_IF_ERROR(
        lock_state_ == LockState::kUnlocked,
        litert::ErrorStatusBuilder(kLiteRtStatusErrorRuntimeFailure)
            << "Cannot clear OpenCL memory while it is locked.");

    if (buffer_type_ == kLiteRtTensorBufferTypeOpenClBufferPacked) {
      if (packed_bytes_ == 0) {
        return kLiteRtStatusOk;
      }
      void* mapped_ptr = nullptr;
      LITERT_RETURN_IF_ERROR(
          Lock(kLiteRtTensorBufferLockModeWrite, &mapped_ptr));
      std::memset(mapped_ptr, 0, packed_bytes_);
      LITERT_RETURN_IF_ERROR(Unlock());
      if (::ml_drift::cl::clFinish != nullptr) {
        const cl_int error_code =
            ::ml_drift::cl::clFinish(retained_queue_.queue());
        if (error_code != CL_SUCCESS) {
          ABSL_LOG(ERROR) << "clFinish for Clear failed with: " << error_code;
          return kLiteRtStatusErrorRuntimeFailure;
        }
      }
      return kLiteRtStatusOk;
    }

    const cl_command_queue command_queue = retained_queue_.queue();
    if (command_queue == nullptr) {
      ABSL_LOG(ERROR) << "OpenCL command queue is null; cannot clear memory.";
      return kLiteRtStatusErrorRuntimeFailure;
    }
    ::ml_drift::cl::CLCommandQueue queue(command_queue,
                                         /*has_ownership=*/false);
    auto desc_with_data = cl_tensor_.GetDescriptor();
    std::vector<uint8_t> zeros(desc_with_data.GetMemorySizeInBytes(), 0);
    desc_with_data.SetData(std::move(zeros));
    LITERT_RETURN_IF_ERROR(
        cl_tensor_.UploadDescriptorData(desc_with_data, &queue));
    return kLiteRtStatusOk;
  }

 private:
  OpenClMemoryInfo(const LiteRtRankedTensorType& tensor_type,
                   LiteRtTensorBufferType buffer_type, size_t packed_bytes,
                   cl_context context, RetainedCommandQueue retained_queue,
                   ::ml_drift::cl::CLMemory imported_cl_mem,
                   ::ml_drift::cl::Tensor cl_tensor)
      : retained_queue_(std::move(retained_queue)),
        imported_cl_mem_(std::move(imported_cl_mem)),
        cl_tensor_(std::move(cl_tensor)),
        tensor_type_(tensor_type),
        buffer_type_(buffer_type),
        packed_bytes_(packed_bytes),
        context_(context) {
    memory_handle = &cl_tensor_;
    raw_handle = cl_tensor_.GetMemoryPtrForWriting();
  }

  // Declared in reverse destruction order: after `~OpenClMemoryInfo()` unmaps
  // any mapped host pointer, C++ destroys `cl_tensor_`, then
  // `imported_cl_mem_`, and finally `retained_queue_`.
  RetainedCommandQueue retained_queue_;
  ::ml_drift::cl::CLMemory imported_cl_mem_;
  ::ml_drift::cl::Tensor cl_tensor_;
  LiteRtRankedTensorType tensor_type_{};
  LiteRtTensorBufferType buffer_type_ = kLiteRtTensorBufferTypeUnknown;
  size_t packed_bytes_ = 0;
  void* host_memory_ = nullptr;
  LockState lock_state_ = LockState::kUnlocked;
  cl_context context_ = nullptr;
};

}  // namespace

namespace ml_drift::delegate::testing {

ScopedOpenClPackedHostAllocForTesting::ScopedOpenClPackedHostAllocForTesting()
    : previous_(force_host_alloc_for_testing.exchange(
          true, std::memory_order_relaxed)) {}

ScopedOpenClPackedHostAllocForTesting::
    ~ScopedOpenClPackedHostAllocForTesting() {
  force_host_alloc_for_testing.store(previous_, std::memory_order_relaxed);
}

}  // namespace ml_drift::delegate::testing

LiteRtStatus LiteRtDestroyOpenClMemory(HwMemoryInfoPtr hw_memory_info) {
  if (hw_memory_info == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  delete static_cast<OpenClMemoryInfo*>(hw_memory_info);
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtLockOpenClMemory(HwMemoryInfoPtr hw_memory_info,
                                    LiteRtTensorBufferLockMode mode,
                                    void** host_memory_ptr) {
  if (hw_memory_info == nullptr || host_memory_ptr == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  return static_cast<OpenClMemoryInfo*>(hw_memory_info)
      ->Lock(mode, host_memory_ptr);
}

LiteRtStatus LiteRtUnlockOpenClMemory(HwMemoryInfoPtr hw_memory_info) {
  if (hw_memory_info == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  return static_cast<OpenClMemoryInfo*>(hw_memory_info)->Unlock();
}

LiteRtStatus LiteRtClearOpenClMemory(HwMemoryInfoPtr hw_memory_info) {
  if (hw_memory_info == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  return static_cast<OpenClMemoryInfo*>(hw_memory_info)->Clear();
}

LiteRtStatus LiteRtCreateOpenClMemory(LiteRtGpuDeviceId device_id,
                                      LiteRtGpuQueueId queue_id,
                                      const LiteRtRankedTensorType* tensor_type,
                                      LiteRtTensorBufferType buffer_type,
                                      size_t bytes, size_t packed_bytes,
                                      HwMemoryInfoPtr* opencl_memory_info) {
  if (tensor_type == nullptr || opencl_memory_info == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const uint64_t fp16_scale = IsGpuFloat16Memory(buffer_type) ? 2 : 1;
  auto tensor_desc =
      ::litert::ml_drift::CreateTensorDescriptor(*tensor_type, buffer_type);
  if (!tensor_desc.ok()) {
    ABSL_LOG(ERROR) << "Failed to create tensor descriptor: "
                    << tensor_desc.status();
    return kLiteRtStatusErrorUnsupported;
  }
  const uint64_t desc_bytes = tensor_desc->GetMemorySizeInBytes();
  if (desc_bytes > std::numeric_limits<size_t>::max() ||
      desc_bytes > std::numeric_limits<uint64_t>::max() / fp16_scale) {
    ABSL_LOG(ERROR) << "Tensor descriptor memory size overflows size_t: "
                    << desc_bytes;
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (desc_bytes * fp16_scale < static_cast<uint64_t>(packed_bytes)) {
    ABSL_LOG(ERROR) << "Too big memory requested: max_size=" << desc_bytes
                    << " vs requested=" << packed_bytes;
    return kLiteRtStatusErrorInvalidArgument;
  }

  if (device_id == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  cl_context cl_ctx_handle = static_cast<cl_context>(device_id);
  ::ml_drift::cl::CLContext context(cl_ctx_handle, /*has_ownership=*/false);

  ::ml_drift::cl::Tensor cl_tensor;
  if (!TryCreateHostAllocPackedTensor(cl_ctx_handle, buffer_type, *tensor_desc,
                                      static_cast<size_t>(desc_bytes), bytes,
                                      packed_bytes, &cl_tensor)) {
    LITERT_RETURN_IF_ERROR(
        ::ml_drift::cl::CreateTensor(context, *tensor_desc, &cl_tensor));
  }

  std::unique_ptr<OpenClMemoryInfo> memory_info;
  LITERT_RETURN_IF_ERROR(OpenClMemoryInfo::Create(
      *tensor_type, buffer_type, packed_bytes, cl_ctx_handle,
      static_cast<cl_command_queue>(queue_id), std::move(cl_tensor),
      ::ml_drift::cl::CLMemory(), &memory_info));

  *opencl_memory_info = memory_info.release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtImportOpenClMemory(LiteRtGpuDeviceId device_id,
                                      LiteRtGpuQueueId queue_id,
                                      const LiteRtRankedTensorType* tensor_type,
                                      LiteRtTensorBufferType buffer_type,
                                      HwMemoryHandle hw_buffer_handle,
                                      size_t bytes, size_t packed_bytes,
                                      HwMemoryInfoPtr* hw_memory_info) {
  if (tensor_type == nullptr || hw_buffer_handle == nullptr ||
      hw_memory_info == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  auto tensor_desc =
      ::litert::ml_drift::CreateTensorDescriptor(*tensor_type, buffer_type);
  if (!tensor_desc.ok()) {
    ABSL_LOG(ERROR) << "Failed to create tensor descriptor for import: "
                    << tensor_desc.status();
    return kLiteRtStatusErrorUnsupported;
  }

  if (device_id == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  cl_context cl_ctx_handle = static_cast<cl_context>(device_id);
  ::ml_drift::cl::CLContext context(cl_ctx_handle, /*has_ownership=*/false);

  // TODO(487801415): Add support for AHWB and OpenGL memory.
  cl_mem cl_memory = nullptr;
  ::ml_drift::cl::CLMemory imported_cl_mem;

  // Try to infer the source buffer type.
  cl_mem candidate_cl_mem = static_cast<cl_mem>(hw_buffer_handle);
  cl_mem_object_type cl_mem_type;
  if (ml_drift::cl::clGetMemObjectInfo(candidate_cl_mem, CL_MEM_TYPE,
                                       sizeof(cl_mem_type), &cl_mem_type,
                                       nullptr) == CL_SUCCESS) {
    cl_memory = candidate_cl_mem;
#if LITERT_HAS_AHWB_SUPPORT
  } else if (::ml_drift::cl::clImportMemoryARM != nullptr &&
             hw_buffer_handle != nullptr) {
    const cl_import_properties_arm properties[] = {
        CL_IMPORT_TYPE_ARM,
        CL_IMPORT_TYPE_ANDROID_HARDWARE_BUFFER_ARM,
        0,
    };

    cl_int error = CL_SUCCESS;
    cl_mem candidate_cl_mem = ::ml_drift::cl::clImportMemoryARM(
        cl_ctx_handle, CL_MEM_READ_WRITE, properties, hw_buffer_handle, bytes,
        &error);
    if (error == CL_SUCCESS) {
      cl_memory = candidate_cl_mem;
    }
#endif
  } else {
#if LITERT_HAS_OPENGL_SUPPORT
    // Assume it is an OpenGL buffer ID.
    cl_GLuint gl_buffer_id =
        static_cast<cl_GLuint>(reinterpret_cast<uintptr_t>(hw_buffer_handle));
    auto status = ::ml_drift::cl::CreateClMemoryFromGlBuffer(
        gl_buffer_id, ::ml_drift::AccessType::kReadWrite, &context,
        &imported_cl_mem);
    if (status.ok()) {
      cl_memory = imported_cl_mem.memory();
    }
#else
    ABSL_LOG(ERROR)
        << "OpenGL support is disabled, cannot import OpenGL buffer.";
    return kLiteRtStatusErrorUnsupported;
#endif
  }
  // Re-use `buffer_handler_webgpu.cc`'s logic for parsing
  // `kLiteRtTensorBufferTypeWebGpuBuffer` /
  // `kLiteRtTensorBufferTypeWebGpuTexture`
  ::ml_drift::cl::Tensor cl_tensor;
  if (buffer_type == kLiteRtTensorBufferTypeOpenClBuffer ||
      buffer_type == kLiteRtTensorBufferTypeOpenClBufferFp16 ||
      buffer_type == kLiteRtTensorBufferTypeOpenClBufferPacked) {
    LITERT_RETURN_IF_ERROR(::ml_drift::cl::CreateTensorShared(
        context, cl_memory, *tensor_desc, &cl_tensor));
  } else if (buffer_type == kLiteRtTensorBufferTypeOpenClTexture ||
             buffer_type == kLiteRtTensorBufferTypeOpenClTextureFp16 ||
             buffer_type == kLiteRtTensorBufferTypeOpenClImageBuffer ||
             buffer_type == kLiteRtTensorBufferTypeOpenClImageBufferFp16) {
    // For Image buffer, use `CreateTensorSharedImage2DBuffer`
    // However, it is not general enough, so we re-use `CreateTensorShared`.
    LITERT_RETURN_IF_ERROR(::ml_drift::cl::CreateTensorShared(
        context, cl_memory, *tensor_desc, &cl_tensor));
  } else {
    ABSL_LOG(ERROR) << "Unsupported buffer type for import: "
                    << litert::BufferTypeToString(buffer_type);
    return kLiteRtStatusErrorUnsupported;
  }

  std::unique_ptr<OpenClMemoryInfo> memory_info;
  LITERT_RETURN_IF_ERROR(OpenClMemoryInfo::Create(
      *tensor_type, buffer_type, packed_bytes, cl_ctx_handle,
      static_cast<cl_command_queue>(queue_id), std::move(cl_tensor),
      std::move(imported_cl_mem), &memory_info));

  *hw_memory_info = memory_info.release();
  return kLiteRtStatusOk;
}
