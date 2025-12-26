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

#ifndef ODML_LITERT_LITERT_CC_LITERT_TENSOR_BUFFER_H_
#define ODML_LITERT_LITERT_CC_LITERT_TENSOR_BUFFER_H_

#include <cstddef>
#include <cstring>
#include <utility>

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_tensor_buffer.h"
#include "litert/c/litert_gl_types.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_event.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer_types.h"

#if LITERT_HAS_OPENCL_SUPPORT
#include <CL/cl.h>
#else
typedef struct _cl_mem* cl_mem;
#endif
#if LITERT_HAS_WEBGPU_SUPPORT
typedef struct WGPUBufferImpl* WGPUBuffer;
#endif  // LITERT_HAS_WEBGPU_SUPPORT

/// @file
/// @brief Defines C++ wrappers for LiteRT tensor buffers and related utilities.

namespace litert {

/// @brief A C++ wrapper for `LiteRtTensorBuffer`, representing a tensor and
/// its associated backing buffer.
class TensorBuffer
    : public internal::Handle<LiteRtTensorBuffer, LiteRtDestroyTensorBuffer> {
 public:
  TensorBuffer() = default;

  /// @brief Creates a managed `TensorBuffer` of a given buffer type and size.
  ///
  /// The returned object is owned by the caller. For host memory, this
  /// allocator guarantees `LITERT_HOST_MEMORY_BUFFER_ALIGNMENT` alignment and
  /// reserves any delegate-specific padding (e.g., XNNPACK extra bytes), so
  /// callers do not need to over-allocate manually.
  static Expected<TensorBuffer> CreateManaged(
      const Environment& env, TensorBufferType buffer_type,
      const RankedTensorType& tensor_type, size_t buffer_size);

  /// @brief Creates a managed host memory `TensorBuffer` using the default
  /// environment (if applicable).
  ///
  /// The returned object is owned by the caller.
  static Expected<TensorBuffer> CreateManagedHostMemory(
      const RankedTensorType& tensor_type, size_t buffer_size);

  /// @brief Creates a `TensorBuffer` that wraps the provided host memory.
  ///
  /// The provided host memory is not owned by the `TensorBuffer` object and
  /// must outlive it. Callers are responsible for ensuring that the pointer is
  /// aligned to at least `LITERT_HOST_MEMORY_BUFFER_ALIGNMENT` bytes and that
  /// any required padding for delegates like XNNPACK is included and
  /// initialized.
  static Expected<TensorBuffer> CreateFromHostMemory(
      const Environment& env, const RankedTensorType& tensor_type,
      void* host_mem_addr, size_t buffer_size);

  static Expected<TensorBuffer> CreateFromHostMemory(
      const RankedTensorType& tensor_type, void* host_mem_addr,
      size_t buffer_size);

  /// @brief Creates a `TensorBuffer` that wraps an Android Hardware Buffer.
  ///
  /// The provided `AHardwareBuffer` is not owned by the `TensorBuffer` and must
  /// outlive it.
  /// @param ahwb_offset The offset in bytes from the start of the
  /// `AHardwareBuffer` where the tensor data begins.
  static Expected<TensorBuffer> CreateFromAhwb(
      const Environment& env, const RankedTensorType& tensor_type,
      AHardwareBuffer* ahwb, size_t ahwb_offset);

  static Expected<TensorBuffer> CreateFromClBuffer(
      const Environment& env, const RankedTensorType& tensor_type,
      TensorBufferType buffer_type, cl_mem cl_memory, size_t size_bytes);

  static Expected<TensorBuffer> CreateFromGlBuffer(
      const Environment& env, const RankedTensorType& tensor_type,
      LiteRtGLenum target, LiteRtGLuint id, size_t size_bytes, size_t offset);

  static Expected<TensorBuffer> CreateFromGlTexture(
      const Environment& env, const RankedTensorType& tensor_type,
      LiteRtGLenum target, LiteRtGLuint id, LiteRtGLenum format,
      size_t size_bytes, LiteRtGLint layer);

#if LITERT_HAS_WEBGPU_SUPPORT
  static Expected<TensorBuffer> CreateFromWebGpuBuffer(
      const Environment& env, const RankedTensorType& tensor_type,
      TensorBufferType buffer_type, WGPUBuffer buffer, size_t size_bytes);
#endif  // LITERT_HAS_WEBGPU_SUPPORT

#if LITERT_HAS_METAL_SUPPORT
  static Expected<TensorBuffer> CreateFromMetalBuffer(
      const Environment& env, const RankedTensorType& tensor_type,
      TensorBufferType buffer_type, void* buffer, size_t size_bytes);
#endif  // LITERT_HAS_METAL_SUPPORT

  /// @brief Creates a duplicate of the current `TensorBuffer` object.
  ///
  /// The returned object is reference-counted, so the underlying
  /// `LiteRtTensorBuffer` handle is not released until the last reference is
  /// removed.
  Expected<TensorBuffer> Duplicate() const;

  litert::Expected<AHardwareBuffer*> GetAhwb() const {
#if LITERT_HAS_AHWB_SUPPORT
    AHardwareBuffer* ahwb;
    LITERT_RETURN_IF_ERROR(LiteRtGetTensorBufferAhwb(Get(), &ahwb));
    return ahwb;
#else
    return litert::Unexpected(
        kLiteRtStatusErrorRuntimeFailure,
        "AHardwareBuffer is not supported on this platform");
#endif
  }

  struct DmaBuf {
    void* addr;
    int fd;
  };

  litert::Expected<DmaBuf> GetDmaBuf() const {
#if LITERT_HAS_DMABUF_SUPPORT
    DmaBuf dma_buf;
    LITERT_RETURN_IF_ERROR(
        LiteRtGetTensorBufferDmaBufBuffer(Get(), &dma_buf.addr, &dma_buf.fd));
    return dma_buf;
#else
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "DMA-BUF is not supported on this platform");
#endif
  }

  Expected<cl_mem> GetOpenClMemory() const {
#if LITERT_HAS_OPENCL_SUPPORT
    cl_mem cl_mem;
    LITERT_RETURN_IF_ERROR(LiteRtGetTensorBufferOpenClMemory(Get(), &cl_mem));
    return cl_mem;
#else
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "OpenCL is not supported on this platform");
#endif
  }

  Expected<HwMemoryHandle> GetWebGpuBuffer() const {
#if LITERT_HAS_WEBGPU_SUPPORT
    HwMemoryHandle hw_memory_handle;
    LITERT_RETURN_IF_ERROR(
        LiteRtGetTensorBufferWebGpuBuffer(Get(), &hw_memory_handle));
    return hw_memory_handle;
#else
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "WebGPU is not supported on this platform");
#endif
  }

  Expected<void*> GetMetalBuffer() const {
#if LITERT_HAS_METAL_SUPPORT
    HwMemoryHandle hw_memory_handle;
    LITERT_RETURN_IF_ERROR(
        LiteRtGetTensorBufferMetalMemory(Get(), &hw_memory_handle));
    return hw_memory_handle;
#else
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "Metal is not supported on this platform");
#endif  // LITERT_HAS_METAL_SUPPORT
  }

  Expected<HwMemoryHandle> GetVulkanMemory() const {
#if LITERT_HAS_VULKAN_SUPPORT
    HwMemoryHandle hw_memory_handle;
    LITERT_RETURN_IF_ERROR(
        LiteRtGetTensorBufferVulkanMemory(Get(), &hw_memory_handle));
    return hw_memory_handle;
#else
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "Vulkan is not supported on this platform");
#endif
  }

  struct GlBuffer {
    LiteRtGLenum target;
    LiteRtGLuint id;
    size_t size_bytes;
    size_t offset;
  };

  Expected<GlBuffer> GetGlBuffer() const {
    GlBuffer gl_buffer;
    LITERT_RETURN_IF_ERROR(LiteRtGetTensorBufferGlBuffer(
        Get(), &gl_buffer.target, &gl_buffer.id, &gl_buffer.size_bytes,
        &gl_buffer.offset));
    return gl_buffer;
  }

  struct GlTexture {
    LiteRtGLenum target;
    LiteRtGLuint id;
    LiteRtGLenum format;
    size_t size_bytes;
    LiteRtGLint layer;
  };

  Expected<GlTexture> GetGlTexture() const {
    GlTexture gl_texture;
    LITERT_RETURN_IF_ERROR(LiteRtGetTensorBufferGlTexture(
        Get(), &gl_texture.target, &gl_texture.id, &gl_texture.format,
        &gl_texture.size_bytes, &gl_texture.layer));
    return gl_texture;
  }

  Expected<TensorBufferType> BufferType() const {
    LiteRtTensorBufferType tensor_buffer_type;
    LITERT_RETURN_IF_ERROR(
        LiteRtGetTensorBufferType(Get(), &tensor_buffer_type));
    return static_cast<enum TensorBufferType>(tensor_buffer_type);
  }

  /// @brief Returns `true` if the tensor buffer is an OpenCL memory object.
  /// @note This function does not return an `Expected<bool>` to prevent
  /// potential misuse.
  bool IsOpenClMemory() const;

  /// @brief Returns `true` if the tensor buffer is a WebGPU memory object.
  /// @note This function does not return an `Expected<bool>` to prevent
  /// potential misuse.
  bool IsWebGpuMemory() const;

  /// @brief Returns `true` if the tensor buffer is a Metal memory object.
  /// @note This function does not return an `Expected<bool>` to prevent
  /// potential misuse.
  bool IsMetalMemory() const;

  /// @brief Returns `true` if the tensor buffer is a Vulkan memory object.
  /// @note This function does not return an `Expected<bool>` to prevent
  /// potential misuse.
  bool IsVulkanMemory() const;

  Expected<RankedTensorType> TensorType() const {
    LiteRtRankedTensorType tensor_type;
    LITERT_RETURN_IF_ERROR(
        LiteRtGetTensorBufferTensorType(Get(), &tensor_type));
    return RankedTensorType(tensor_type);
  }

  bool HasType(const RankedTensorType& type) const {
    auto t = TensorType();
    return t && *t == type;
  }

  bool HasType(const LiteRtRankedTensorType& type) const {
    auto t = TensorType();
    return t && *t == ::litert::RankedTensorType(type);
  }
  /// @brief Returns the size of the underlying hardware tensor buffer.
  ///
  /// This size can differ from `PackedSize()` if striding and padding exist.
  Expected<size_t> Size() const {
    size_t size;
    LITERT_RETURN_IF_ERROR(LiteRtGetTensorBufferSize(Get(), &size));
    return size;
  }

  /// @brief Returns the size of the tensor buffer in packed bytes.
  ///
  /// This size is used for reading/writing data on a locked tensor buffer.
  Expected<size_t> PackedSize() const {
    size_t size;
    LITERT_RETURN_IF_ERROR(LiteRtGetTensorBufferPackedSize(Get(), &size));
    return size;
  }

  Expected<size_t> Offset() const {
    size_t offset;
    LITERT_RETURN_IF_ERROR(LiteRtGetTensorBufferOffset(Get(), &offset));
    return offset;
  }

  bool HasEvent() const {
    bool has_event;
    internal::AssertOk(LiteRtHasTensorBufferEvent, Get(), &has_event);
    return has_event;
  }

  Expected<Event> GetEvent() const {
    LiteRtEvent event;
    LITERT_RETURN_IF_ERROR(LiteRtGetTensorBufferEvent(Get(), &event));
    return Event::WrapCObject(event, OwnHandle::kNo);
  }

  /// @brief Sets the C++ `Event` object for the tensor buffer.
  ///
  /// This function takes ownership of the provided `Event` object.
  Expected<void> SetEvent(Event&& event) {
    if (!event.IsOwned()) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Expected an owned event");
    }
    LITERT_RETURN_IF_ERROR(LiteRtSetTensorBufferEvent(Get(), event.Release()));
    return {};
  }

  Expected<void> ClearEvent() {
    LITERT_RETURN_IF_ERROR(LiteRtClearTensorBufferEvent(Get()));
    return {};
  }

  enum class LockMode {
    kRead = kLiteRtTensorBufferLockModeRead,
    kWrite = kLiteRtTensorBufferLockModeWrite,
    kReadWrite = kLiteRtTensorBufferLockModeReadWrite,
  };

  static LiteRtTensorBufferLockMode ToLiteRtLockMode(LockMode mode) {
    switch (mode) {
      case LockMode::kRead:
        return kLiteRtTensorBufferLockModeRead;
      case LockMode::kWrite:
        return kLiteRtTensorBufferLockModeWrite;
      case LockMode::kReadWrite:
        return kLiteRtTensorBufferLockModeReadWrite;
    }
  }

  Expected<void*> Lock(LockMode mode = LockMode::kWrite) {
    void* host_mem_addr;
    LITERT_RETURN_IF_ERROR(
        LiteRtLockTensorBuffer(Get(), &host_mem_addr, ToLiteRtLockMode(mode)));
    return host_mem_addr;
  }

  Expected<void> Unlock() {
    LITERT_RETURN_IF_ERROR(LiteRtUnlockTensorBuffer(Get()));
    return {};
  }

  /// @brief Writes data from a user-provided `absl::Span<const T>` to the
  /// tensor buffer.
  ///
  /// Returns an error if the provided buffer is larger than the tensor
  /// buffer's size.
  template <typename T>
  Expected<void> Write(absl::Span<const T> data) {
    LITERT_ASSIGN_OR_RETURN(void* host_mem_addr, Lock(LockMode::kWrite));
    absl::Cleanup unlock = [this] { Unlock(); };
    LITERT_ASSIGN_OR_RETURN(size_t size, PackedSize());
    if (size < data.size() * sizeof(T)) {
      return Unexpected(
          kLiteRtStatusErrorRuntimeFailure,
          absl::StrFormat(
              "TensorBuffer host memory buffer size is smaller than the "
              "given data size, %zu vs %zu",
              size, data.size() * sizeof(T)));
    }
    std::memcpy(host_mem_addr, data.data(), data.size() * sizeof(T));
    return {};
  }

  /// @brief Reads data from the tensor buffer into a user-provided
  /// `absl::Span<T>`.
  ///
  /// If the provided buffer is smaller than the tensor buffer, data will be
  /// read up to the size of the provided buffer. Returns an error if the
  /// provided buffer is larger than the tensor buffer.
  template <typename T>
  Expected<void> Read(absl::Span<T> data) {
    LITERT_ASSIGN_OR_RETURN(void* host_mem_addr, Lock(LockMode::kRead));
    absl::Cleanup unlock = [this] { Unlock(); };
    LITERT_ASSIGN_OR_RETURN(size_t size, PackedSize());
    size_t total_read_size = data.size() * sizeof(T);
    if (size < total_read_size) {
      return Unexpected(
          kLiteRtStatusErrorRuntimeFailure,
          absl::StrFormat(
              "TensorBuffer host memory buffer size is smaller than the "
              "given data size, %zu vs %zu",
              size, total_read_size));
    }
    std::memcpy(data.data(), host_mem_addr, total_read_size);
    return {};
  }

  /// @internal
  /// @brief Wraps a `LiteRtTensorBuffer` C object in a `TensorBuffer` C++
  /// object.
  /// @warning This is for internal use only.
  static TensorBuffer WrapCObject(LiteRtTensorBuffer tensor_buffer,
                                  OwnHandle owned) {
    return TensorBuffer(tensor_buffer, owned);
  }

 private:
  /// @param owned Indicates if the created `TensorBuffer` object should take
  /// ownership of the provided `tensor_buffer` handle.
  explicit TensorBuffer(LiteRtTensorBuffer tensor_buffer, OwnHandle owned)
      : Handle(tensor_buffer, owned) {}
};

/// @brief A scoped lock for a `TensorBuffer`.
class TensorBufferScopedLock {
 public:
  TensorBufferScopedLock(const TensorBufferScopedLock& arg) = delete;
  TensorBufferScopedLock(TensorBufferScopedLock&& arg) noexcept
      : tensor_buffer_(arg.tensor_buffer_) {
    arg.tensor_buffer_ = nullptr;
  };

  TensorBufferScopedLock& operator=(TensorBufferScopedLock&& other) noexcept {
    if (this != &other) {
      tensor_buffer_ = other.tensor_buffer_;
      other.tensor_buffer_ = nullptr;
    }
    return *this;
  }

  ~TensorBufferScopedLock() {
    if (tensor_buffer_ != nullptr) {
      (void)LiteRtUnlockTensorBuffer(tensor_buffer_);
    }
  }

  template <typename T = void>
  static Expected<std::pair<TensorBufferScopedLock, T*>> Create(
      TensorBuffer& tensor_buffer, TensorBuffer::LockMode mode) {
    return Create<T>(tensor_buffer.Get(), mode);
  }

  template <typename T = void>
  static Expected<std::pair<TensorBufferScopedLock, const T*>> Create(
      const TensorBuffer& tensor_buffer, TensorBuffer::LockMode mode) {
    return Create<const T>(tensor_buffer.Get(), mode);
  }

  template <typename T = void>
  static Expected<std::pair<TensorBufferScopedLock, T*>> Create(
      LiteRtTensorBuffer tensor_buffer, TensorBuffer::LockMode mode) {
    void* host_mem_addr;
    LITERT_RETURN_IF_ERROR(LiteRtLockTensorBuffer(
        tensor_buffer, &host_mem_addr, TensorBuffer::ToLiteRtLockMode(mode)));
    return std::make_pair(TensorBufferScopedLock(tensor_buffer),
                          static_cast<T*>(host_mem_addr));
  }

 private:
  explicit TensorBufferScopedLock(LiteRtTensorBuffer& tensor_buffer)
      : tensor_buffer_(tensor_buffer) {}

  LiteRtTensorBuffer tensor_buffer_;
};

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_LITERT_TENSOR_BUFFER_H_
