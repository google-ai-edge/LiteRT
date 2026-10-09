// Copyright 2025 Google LLC.
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

#include "litert/runtime/gl_buffer.h"

#include <stdlib.h>

#include <cstddef>
#include <cstring>
#include <memory>
#include <utility>

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_gl_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/runtime/ahwb_buffer.h"
#include "litert/runtime/gpu_environment.h"

#if LITERT_HAS_OPENGL_SUPPORT
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <GLES3/gl32.h>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/gl/egl_environment.h"  // from @ml_drift
#include "ml_drift/gl/gl_call.h"  // from @ml_drift
#include "ml_drift/gl/gl_errors.h"  // from @ml_drift
#include "ml_drift/gl/portable_gl31.h"  // from @ml_drift
#endif  // LITERT_HAS_OPENGL_SUPPORT

namespace litert {
namespace internal {

#if LITERT_HAS_AHWB_SUPPORT && LITERT_HAS_OPENGL_SUPPORT

PFNGLBUFFERSTORAGEEXTERNALEXTPROC glBufferStorageExternalEXT;
PFNEGLGETNATIVECLIENTBUFFERANDROIDPROC eglGetNativeClientBufferANDROID;

bool IsAhwbToGlBufferInteropSupported() {
  static const bool extensions_allowed = [] {
    eglGetNativeClientBufferANDROID =
        reinterpret_cast<PFNEGLGETNATIVECLIENTBUFFERANDROIDPROC>(
            eglGetProcAddress("eglGetNativeClientBufferANDROID"));
    glBufferStorageExternalEXT =
        reinterpret_cast<PFNGLBUFFERSTORAGEEXTERNALEXTPROC>(
            eglGetProcAddress("glBufferStorageExternalEXT"));
    return eglGetNativeClientBufferANDROID && glBufferStorageExternalEXT;
  }();
  return extensions_allowed;
}

#endif  // LITERT_HAS_AHWB_SUPPORT && LITERT_HAS_OPENGL_SUPPORT

Expected<GlBuffer> GlBuffer::AllocFromAhwbBuffer(GpuEnvironment* gpu_env,
                                                 AhwbBuffer& ahwb_buffer) {
#if LITERT_HAS_AHWB_SUPPORT && LITERT_HAS_OPENGL_SUPPORT
  LITERT_RETURN_IF_ERROR(gpu_env->GetEglDisplay() != EGL_NO_DISPLAY,
                         litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                            "Failed to get EGL display"));
  LITERT_RETURN_IF_ERROR(
      IsAhwbToGlBufferInteropSupported(),
      Unexpected(kLiteRtStatusErrorRuntimeFailure,
                 "AHardwareBuffer to GL interop is not supported"));
  LITERT_RETURN_IF_ERROR(
      ahwb_buffer.ahwb != nullptr,
      Unexpected(kLiteRtStatusErrorRuntimeFailure, "AHardwareBuffer is null"));

  // Create GL buffer id.
  GLuint gl_id;
  glGenBuffers(1, &gl_id);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, gl_id);

  // Create EGLClientBuffer from AHardwareBuffer.
  EGLClientBuffer native_buffer =
      eglGetNativeClientBufferANDROID(ahwb_buffer.ahwb);
  LITERT_RETURN_IF_ERROR(
      native_buffer != nullptr,
      Unexpected(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to create EGLClientBuffer from AHardwareBuffer"));

  LITERT_ASSIGN_OR_RETURN(
      size_t size_bytes,
      litert::internal::AhwbBuffer::GetSize(ahwb_buffer.ahwb));
  LITERT_RETURN_IF_ERROR(size_bytes != 0,
                         Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                    "AHardwareBuffer size is 0"));

  // Create OpenGl buffer object backed by the AHardwareBuffer.
  glBufferStorageExternalEXT(
      GL_SHADER_STORAGE_BUFFER, 0, size_bytes, native_buffer,
      GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_MAP_COHERENT_BIT_EXT |
          GL_MAP_PERSISTENT_BIT_EXT);
  // Check for OpenGL errors.
  absl::Status status = ::ml_drift::gl::GetOpenGlErrors();
  if (!status.ok()) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      absl::StrCat("glBufferStorageExternalEXT: Failed to "
                                   "create GL buffer from AHardwareBuffer: ",
                                   status.message()));
  }
  // Unbind the buffer.
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

  // Create GL buffer object. We assume ownership of the GL buffer id so that it
  // will be automatically deallocated when the internal::GlBuffer is destroyed.
  return GlBuffer(gpu_env, GL_SHADER_STORAGE_BUFFER, gl_id, size_bytes,
                  /*offset=*/0, /*has_ownership=*/true, ahwb_buffer.ahwb);
#else
  return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                    "AHardwareBuffer to GL interop is not supported.");
#endif  // LITERT_HAS_AHWB_SUPPORT && LITERT_HAS_OPENGL_SUPPORT
}

GlBuffer::GlBuffer(GpuEnvironment* gpu_env, LiteRtGLenum target,
                   LiteRtGLuint id, size_t size_bytes, size_t offset,
                   LiteRtGlBufferDeallocator deallocator) {
  gpu_env_ = gpu_env;
#if LITERT_HAS_OPENGL_SUPPORT
  target_ = target;
  id_ = id;
  size_bytes_ = size_bytes;
  offset_ = offset;
  // has_ownership_ is set to false since buffer deletion is determined by the
  // deallocator in this case.
  has_ownership_ = false;
  deallocator_ = deallocator;
#else
  LITERT_LOG(LITERT_ERROR, "GlBuffer::GlBuffer() is not supported");
#endif  // LITERT_HAS_OPENGL_SUPPORT
}

GlBuffer::GlBuffer(GlBuffer&& other) {
  gpu_env_ = other.gpu_env_;
#if LITERT_HAS_OPENGL_SUPPORT
  target_ = other.target_;
  id_ = other.id_;
  size_bytes_ = other.size_bytes_;
  offset_ = other.offset_;
  has_ownership_ = other.has_ownership_;
  deallocator_ = std::move(other.deallocator_);
  data_ = other.data_;
#if LITERT_HAS_AHWB_SUPPORT
  ahwb_ = other.ahwb_;
#endif  // LITERT_HAS_AHWB_SUPPORT
  // Reset the other GlBuffer to a default state.
  other.target_ = GL_INVALID_ENUM;
  other.id_ = GL_INVALID_INDEX;
  other.size_bytes_ = 0;
  other.offset_ = 0;
  other.has_ownership_ = false;
  other.deallocator_ = nullptr;
  other.data_ = nullptr;
#if LITERT_HAS_AHWB_SUPPORT
  other.ahwb_ = nullptr;
#endif  // LITERT_HAS_AHWB_SUPPORT
#else
  LITERT_LOG(LITERT_ERROR, "GlBuffer::GlBuffer() is not supported");
#endif  // LITERT_HAS_OPENGL_SUPPORT
}

GlBuffer::~GlBuffer() {
#if LITERT_HAS_OPENGL_SUPPORT
  if (id_ != GL_INVALID_INDEX) {
    if (deallocator_ != nullptr) {
      deallocator_(reinterpret_cast<void*>(id_));
    } else if (has_ownership_) {
      ML_DRIFT_CALL_GL(glDeleteBuffers, 1, &id_).IgnoreError();
    }
  }
  if (data_ != nullptr) {
    litert_aligned_free(data_);
  }
#else
  LITERT_LOG(LITERT_ERROR, "GlBuffer::~GlBuffer() is not supported");
#endif  // LITERT_HAS_OPENGL_SUPPORT
}

LiteRtGLenum GlBuffer::target() const {
#if LITERT_HAS_OPENGL_SUPPORT
  return target_;
#else
  LITERT_LOG(LITERT_ERROR, "GlBuffer::target() is not supported");
  return 0;
#endif  // LITERT_HAS_OPENGL_SUPPORT
}
LiteRtGLuint GlBuffer::id() const {
#if LITERT_HAS_OPENGL_SUPPORT
  return id_;
#else
  LITERT_LOG(LITERT_ERROR, "GlBuffer::id() is not supported");
  return 0;
#endif  // LITERT_HAS_OPENGL_SUPPORT
}
size_t GlBuffer::size_bytes() const {
#if LITERT_HAS_OPENGL_SUPPORT
  return size_bytes_;
#else
  LITERT_LOG(LITERT_ERROR, "GlBuffer::size_bytes() is not supported");
  return 0;
#endif  // LITERT_HAS_OPENGL_SUPPORT
}
size_t GlBuffer::offset() const {
#if LITERT_HAS_OPENGL_SUPPORT
  return offset_;
#else
  LITERT_LOG(LITERT_ERROR, "GlBuffer::offset() is not supported");
  return 0;
#endif
}

Expected<GlBuffer> GlBuffer::Alloc(GpuEnvironment* gpu_env, size_t size_bytes) {
#if LITERT_HAS_OPENGL_SUPPORT
  LITERT_RETURN_IF_ERROR(gpu_env->GetEglDisplay() != EGL_NO_DISPLAY,
                         litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                            "Failed to get EGL display"));
  GLuint id = GL_INVALID_INDEX;
  if (!ML_DRIFT_CALL_GL(glGenBuffers, 1, &id).ok()) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to allocate GL buffer");
  }
  ML_DRIFT_CALL_GL(glBindBuffer, GL_SHADER_STORAGE_BUFFER, id).IgnoreError();
  std::vector<std::byte> zeros(size_bytes);
  absl::Status status =
      ML_DRIFT_CALL_GL(glBufferData, GL_SHADER_STORAGE_BUFFER, size_bytes,
                       zeros.data(), GL_STREAM_COPY);
  ML_DRIFT_CALL_GL(glBindBuffer, GL_SHADER_STORAGE_BUFFER, 0).IgnoreError();
  if (!status.ok()) {
    ML_DRIFT_CALL_GL(glDeleteBuffers, 1, &id).IgnoreError();
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to allocate GL buffer");
  }

  return GlBuffer(gpu_env, GL_SHADER_STORAGE_BUFFER, id, size_bytes,
                  /*offset=*/0, /*has_ownership=*/true);
#else
  return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                    "OpenGL buffers are not supported");
#endif  // LITERT_HAS_OPENGL_SUPPORT
}

template Expected<float*> GlBuffer::Lock<float>(
    LiteRtTensorBufferLockMode mode);
template Expected<char*> GlBuffer::Lock<char>(LiteRtTensorBufferLockMode mode);
template Expected<void> GlBuffer::Unlock<float>();
template Expected<void> GlBuffer::Unlock<char>();

template <typename T>
Expected<T*> GlBuffer::Lock(LiteRtTensorBufferLockMode mode) {
#if LITERT_HAS_OPENGL_SUPPORT
  absl::MutexLock lock(&mutex_);
  lock_mode_ = mode;
#if LITERT_HAS_AHWB_SUPPORT
  if (ahwb_ != nullptr) {
    LITERT_ASSIGN_OR_RETURN(void* data,
                            litert::internal::AhwbBuffer::Lock(ahwb_));
    return static_cast<T*>(data);
  }
#endif  // LITERT_HAS_AHWB_SUPPORT
  if (data_ == nullptr) {
    // Ensure the data is aligned.
    if (auto rc = posix_memalign(&data_, LITERT_HOST_MEMORY_BUFFER_ALIGNMENT,
                                 size_bytes_);
        rc) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Failed to allocate aligned memory");
    }
  }
  if (mode != kLiteRtTensorBufferLockModeWrite) {
    if (size_bytes_ % sizeof(T) != 0) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Failed to read GL buffer: Buffer is not aligned");
    }
    ML_DRIFT_CALL_GL(glBindBuffer, target_, id_).IgnoreError();
    void* mapped =
        glMapBufferRange(target_, offset_, size_bytes_, GL_MAP_READ_BIT);
    if (mapped == nullptr) {
      absl::Status status = ::ml_drift::gl::GetOpenGlErrors();
      ML_DRIFT_CALL_GL(glBindBuffer, target_, 0).IgnoreError();
      return Unexpected(
          kLiteRtStatusErrorRuntimeFailure,
          absl::StrCat("Failed to read GL buffer: ", status.message()));
    }
    std::memcpy(data_, mapped, size_bytes_);
    ML_DRIFT_CALL_GL(glUnmapBuffer, target_).IgnoreError();
    ML_DRIFT_CALL_GL(glBindBuffer, target_, 0).IgnoreError();
  }
  return Expected<T*>(static_cast<T*>(data_));
#else
  return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                    "GlBuffer::Lock() is not supported");
#endif  // LITERT_HAS_OPENGL_SUPPORT
}

template <typename T>
Expected<void> GlBuffer::Unlock() {
#if LITERT_HAS_OPENGL_SUPPORT
  absl::MutexLock lock(&mutex_);
#if LITERT_HAS_AHWB_SUPPORT
  if (ahwb_ != nullptr) {
    return litert::internal::AhwbBuffer::Unlock(ahwb_);
  }
#endif  // LITERT_HAS_AHWB_SUPPORT
  if (data_ == nullptr) {
    return Error(
        kLiteRtStatusErrorRuntimeFailure,
        "Cannot unlock a buffer that wasn't locked in the first place");
  }
  if (lock_mode_ != kLiteRtTensorBufferLockModeRead) {
    ML_DRIFT_CALL_GL(glBindBuffer, target_, id_).IgnoreError();
    absl::Status status =
        ML_DRIFT_CALL_GL(glBufferSubData, target_, offset_, size_bytes_, data_);
    ML_DRIFT_CALL_GL(glBindBuffer, target_, 0).IgnoreError();
    if (!status.ok()) {
      return Unexpected(
          kLiteRtStatusErrorRuntimeFailure,
          absl::StrCat("Failed to write GL buffer: ", status.message()));
    }
  }
  return Expected<void>();
#else
  return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                    "GlBuffer::Unlock() is not supported");
#endif  // LITERT_HAS_OPENGL_SUPPORT
}

}  // namespace internal
}  // namespace litert
