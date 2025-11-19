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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_C_LITERT_PLATFORM_SUPPORT_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_C_LITERT_PLATFORM_SUPPORT_H_

#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

inline bool LiteRtHasOpenClSupport() {
#if LITERT_HAS_OPENCL_SUPPORT
  return true;
#else
  return false;
#endif
}

inline bool LiteRtHasOpenGlSupport() {
#if LITERT_HAS_OPENGL_SUPPORT
  return true;
#else
  return false;
#endif
}

inline bool LiteRtHasAhwbSupport() {
#if LITERT_HAS_AHWB_SUPPORT
  return true;
#else
  return false;
#endif
}

inline bool LiteRtHasIonSupport() {
#if LITERT_HAS_ION_SUPPORT
  return true;
#else
  return false;
#endif
}

inline bool LiteRtHasDmaBufSupport() {
#if LITERT_HAS_DMABUF_SUPPORT
  return true;
#else
  return false;
#endif
}

inline bool LiteRtHasFastRpcSupport() {
#if LITERT_HAS_FASTRPC_SUPPORT
  return true;
#else
  return false;
#endif
}

inline bool LiteRtHasSyncFenceSupport() {
#if LITERT_HAS_SYNC_FENCE_SUPPORT
  return true;
#else
  return false;
#endif
}

inline bool LiteRtHasMetalSupport() {
#if LITERT_HAS_METAL_SUPPORT
  return true;
#else
  return false;
#endif
}

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_LITERT_PLATFORM_SUPPORT_H_
