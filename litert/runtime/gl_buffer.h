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

#ifndef ODML_LITERT_LITERT_RUNTIME_GL_BUFFER_H_
#define ODML_LITERT_LITERT_RUNTIME_GL_BUFFER_H_

#include <cstddef>
#include <cstdlib>

#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_gl_types.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/runtime/ahwb_buffer.h"
#include "litert/runtime/gpu_environment.h"

#if LITERT_HAS_OPENGL_SUPPORT
#include "tflite/delegates/gpu/gl/gl_buffer.h"
#endif  // LITERT_HAS_OPENGL_SUPPORT

namespace litert::internal {

class GlBuffer {
 public:
  // NULL deallocator means that the buffer id is not owned by the GlBuffer
  // and therefore must be released separately by the caller.
  GlBuffer(GpuEnvironment* gpu_env, LiteRtGLenum target, LiteRtGLuint id,
           size_t size_bytes, size_t offset,
           LiteRtGlBufferDeallocator deallocator);

  GlBuffer(GlBuffer&& other);

  ~GlBuffer();

  static bool IsSupported() { return true; }
  // Allocates an owned GL buffer.
  static Expected<GlBuffer> Alloc(GpuEnvironment* gpu_env, size_t size_bytes);

  // Allocates an owned GL buffer from an AHardwareBuffer.
  static Expected<GlBuffer> AllocFromAhwbBuffer(GpuEnvironment* gpu_env,
                                                AhwbBuffer& ahwb_buffer);

  template <typename T>
  Expected<T*> Lock();

  template <typename T>
  Expected<void> Unlock();

  LiteRtGLenum target() const;
  LiteRtGLuint id() const;
  size_t size_bytes() const;
  size_t offset() const;

 private:
#if LITERT_HAS_OPENGL_SUPPORT
  // Used to create an owned GlBuffer from a tflite::gpu::gl::GlBuffer.
  // tflite_gl_buffer is expected to be owned.
  explicit GlBuffer(tflite::gpu::gl::GlBuffer&& tflite_gl_buffer
#if LITERT_HAS_AHWB_SUPPORT
                    ,
                    AHardwareBuffer* ahwb = nullptr
#endif  // LITERT_HAS_AHWB_SUPPORT
                    )
      : tflite_gl_buffer_(std::move(tflite_gl_buffer)),
        deallocator_(nullptr),  // deallocator is not needed since
                                // tflite_gl_buffer is owned.
        size_bytes_(tflite_gl_buffer.bytes_size())
#if LITERT_HAS_AHWB_SUPPORT
        ,
        ahwb_(ahwb)
#endif  // LITERT_HAS_AHWB_SUPPORT
  {
  }
#endif  // LITERT_HAS_OPENGL_SUPPORT
  absl::Mutex mutex_;
#if LITERT_HAS_OPENGL_SUPPORT
  tflite::gpu::gl::GlBuffer tflite_gl_buffer_;
  // NULL deallocator_ means that buffer id is not owned by the GlBuffer, UNLESS
  // tflite_gl_buffer_ is owned.
  LiteRtGlBufferDeallocator deallocator_;
  // The cpu memory buffer pointer.
  void* data_ = nullptr;
  // The size of the buffer in bytes.
  size_t size_bytes_ = 0;
#endif  // LITERT_HAS_OPENGL_SUPPORT
#if LITERT_HAS_AHWB_SUPPORT
  AHardwareBuffer* ahwb_ = nullptr;
#endif  // LITERT_HAS_AHWB_SUPPORT
  GpuEnvironment* gpu_env_;
};

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_RUNTIME_GL_BUFFER_H_
