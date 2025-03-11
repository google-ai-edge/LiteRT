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

#include "litert/c/litert_common.h"

#if LITERT_HAS_OPENGL_SUPPORT

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <GLES3/gl32.h>
#include <stdlib.h>

#include <cstddef>
#include <memory>
#include <utility>

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_tensor_buffer.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/runtime/gl_buffer.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_buffer.h"  // from @org_tensorflow

namespace litert {
namespace internal {

#if LITERT_HAS_AHWB_SUPPORT

PFNGLBUFFERSTORAGEEXTERNALEXTPROC glBufferStorageExternalEXT;
PFNEGLGETNATIVECLIENTBUFFERANDROIDPROC eglGetNativeClientBufferANDROID;
PFNEGLDUPNATIVEFENCEFDANDROIDPROC eglDupNativeFenceFDANDROID;
PFNEGLCREATESYNCKHRPROC eglCreateSyncKHR;
PFNEGLWAITSYNCKHRPROC eglWaitSyncKHR;
PFNEGLCLIENTWAITSYNCKHRPROC eglClientWaitSyncKHR;
PFNEGLDESTROYSYNCKHRPROC eglDestroySyncKHR;

bool IsAhwbToGlInteropSupported() {
  static const bool extensions_allowed = [] {
    eglGetNativeClientBufferANDROID =
        reinterpret_cast<PFNEGLGETNATIVECLIENTBUFFERANDROIDPROC>(
            eglGetProcAddress("eglGetNativeClientBufferANDROID"));
    glBufferStorageExternalEXT =
        reinterpret_cast<PFNGLBUFFERSTORAGEEXTERNALEXTPROC>(
            eglGetProcAddress("glBufferStorageExternalEXT"));
    eglDupNativeFenceFDANDROID =
        reinterpret_cast<PFNEGLDUPNATIVEFENCEFDANDROIDPROC>(
            eglGetProcAddress("eglDupNativeFenceFDANDROID"));
    eglCreateSyncKHR = reinterpret_cast<PFNEGLCREATESYNCKHRPROC>(
        eglGetProcAddress("eglCreateSyncKHR"));
    eglWaitSyncKHR = reinterpret_cast<PFNEGLWAITSYNCKHRPROC>(
        eglGetProcAddress("eglWaitSyncKHR"));
    eglClientWaitSyncKHR = reinterpret_cast<PFNEGLCLIENTWAITSYNCKHRPROC>(
        eglGetProcAddress("eglClientWaitSyncKHR"));
    eglDestroySyncKHR = reinterpret_cast<PFNEGLDESTROYSYNCKHRPROC>(
        eglGetProcAddress("eglDestroySyncKHR"));
    return eglClientWaitSyncKHR && eglWaitSyncKHR &&
           eglGetNativeClientBufferANDROID && glBufferStorageExternalEXT &&
           eglCreateSyncKHR && eglDupNativeFenceFDANDROID && eglDestroySyncKHR;
  }();
  return extensions_allowed;
}

Expected<GlBuffer> GlBuffer::AllocFromAhwbBuffer(AhwbBuffer& ahwb_buffer) {
  LITERT_RETURN_IF_ERROR(
      IsAhwbToGlInteropSupported(),
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
  absl::Status status = tflite::gpu::gl::GetOpenGlErrors();
  if (!status.ok()) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      absl::StrCat("glBufferStorageExternalEXT: Failed to "
                                   "create GL buffer from AHardwareBuffer: ",
                                   status.message()));
  }
  // Unbind the buffer.
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

  // Create GL buffer object.
  tflite::gpu::gl::GlBuffer tflite_gl_buffer(GL_SHADER_STORAGE_BUFFER, /*id=*/0,
                                             size_bytes, /*offset=*/0,
                                             /*has_ownership=*/false);
  return GlBuffer(std::move(tflite_gl_buffer), ahwb_buffer.ahwb);
}
#endif  // LITERT_HAS_AHWB_SUPPORT

Expected<GlBuffer> GlBuffer::Alloc(size_t bytes_size) {
  tflite::gpu::gl::GlBuffer tflite_gl_buffer;

  if (!tflite::gpu::gl::CreateReadWriteShaderStorageBuffer<std::byte>(
           bytes_size, &tflite_gl_buffer)
           .ok()) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to allocate GL buffer");
  };

  return GlBuffer(std::move(tflite_gl_buffer));
}

template Expected<float*> GlBuffer::Lock<float>();
template Expected<char*> GlBuffer::Lock<char>();
template Expected<void> GlBuffer::Unlock<float>();
template Expected<void> GlBuffer::Unlock<char>();

template <typename T>
Expected<T*> GlBuffer::Lock() {
  absl::MutexLock lock(&mutex_);
  if (data_ == nullptr) {
    // Ensure the data is aligned.
    if (auto rc =
            posix_memalign(&data_, LITERT_HOST_MEMORY_BUFFER_ALIGNMENT, size_);
        rc) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Failed to allocate aligned memory");
    }
    if (auto status = tflite_gl_buffer_.Read(
            absl::MakeSpan(static_cast<T*>(data_), size_ / sizeof(T)));
        !status.ok()) {
      return Unexpected(
          kLiteRtStatusErrorRuntimeFailure,
          absl::StrCat("Failed to read GL buffer: ", status.message()));
    }
  }
  return Expected<T*>(static_cast<T*>(data_));
}

template <typename T>
Expected<void> GlBuffer::Unlock() {
  absl::MutexLock lock(&mutex_);
  if (data_ == nullptr) {
    return Error(
        kLiteRtStatusErrorRuntimeFailure,
        "Cannot unlock a buffer that wasn't locked in the first place");
  }
  if (auto status = tflite_gl_buffer_.Write(
          absl::MakeSpan(static_cast<const T*>(data_), size_ / sizeof(T)));
      !status.ok()) {
    return Unexpected(
        kLiteRtStatusErrorRuntimeFailure,
        absl::StrCat("Failed to write GL buffer: ", status.message()));
  }
  return Expected<void>();
}

}  // namespace internal
}  // namespace litert

#endif  // LITERT_HAS_OPENGL_SUPPORT
