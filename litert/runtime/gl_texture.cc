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

#include "litert/runtime/gl_texture.h"

#include <cstddef>

#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_gl_types.h"
#include "litert/c/litert_tensor_buffer_types.h"

#if LITERT_HAS_OPENGL_SUPPORT
#include "ml_drift/gl/gl_call.h"  // from @ml_drift
#include "ml_drift/gl/portable_gl31.h"  // from @ml_drift
#endif  // LITERT_HAS_OPENGL_SUPPORT

namespace litert::internal {

LiteRtGLenum GlTexture::target() const {
#if LITERT_HAS_OPENGL_SUPPORT
  return target_;
#endif
  LITERT_LOG(LITERT_ERROR, "GlTexture::target() is not supported");
  return 0;
}

LiteRtGLuint GlTexture::id() const {
#if LITERT_HAS_OPENGL_SUPPORT
  return id_;
#endif
  LITERT_LOG(LITERT_ERROR, "GlTexture::id() is not supported");
  return 0;
}

LiteRtGLenum GlTexture::format() const {
#if LITERT_HAS_OPENGL_SUPPORT
  return format_;
#endif
  LITERT_LOG(LITERT_ERROR, "GlTexture::format() is not supported");
  return 0;
}

size_t GlTexture::size_bytes() const {
#if LITERT_HAS_OPENGL_SUPPORT
  return size_bytes_;
#endif
  LITERT_LOG(LITERT_ERROR, "GlTexture::size_bytes() is not supported");
  return 0;
}

LiteRtGLint GlTexture::layer() const {
#if LITERT_HAS_OPENGL_SUPPORT
  return layer_;
#else
  LITERT_LOG(LITERT_ERROR, "GlTexture::layer() is not supported");
  return 0;
#endif
}

GlTexture::GlTexture(LiteRtGLenum target, LiteRtGLuint id, LiteRtGLenum format,
                     size_t size_bytes, LiteRtGLint layer,
                     LiteRtGlTextureDeallocator deallocator) {
#if LITERT_HAS_OPENGL_SUPPORT
  target_ = target;
  id_ = id;
  format_ = format;
  size_bytes_ = size_bytes;
  layer_ = layer;
  deallocator_ = deallocator;
#else
  LITERT_LOG(LITERT_ERROR, "GlTexture::GlTexture() is not supported");
#endif  // LITERT_HAS_OPENGL_SUPPORT
}

GlTexture::GlTexture(GlTexture&& other) {
#if LITERT_HAS_OPENGL_SUPPORT
  target_ = other.target_;
  id_ = other.id_;
  format_ = other.format_;
  size_bytes_ = other.size_bytes_;
  layer_ = other.layer_;
  deallocator_ = other.deallocator_;
  other.id_ = GL_INVALID_INDEX;
  other.deallocator_ = nullptr;
#else
  LITERT_LOG(LITERT_ERROR, "GlTexture::GlTexture() is not supported");
#endif  // LITERT_HAS_OPENGL_SUPPORT
}

GlTexture::~GlTexture() {
#if LITERT_HAS_OPENGL_SUPPORT
  if (id_ != GL_INVALID_INDEX) {
    if (deallocator_ != nullptr) {
      deallocator_(reinterpret_cast<void*>(id_));
    } else {
      ML_DRIFT_CALL_GL(glDeleteTextures, 1, &id_).IgnoreError();
    }
  }
#else
  LITERT_LOG(LITERT_ERROR, "GlTexture::~GlTexture() is not supported");
#endif  // LITERT_HAS_OPENGL_SUPPORT
}

}  // namespace litert::internal
