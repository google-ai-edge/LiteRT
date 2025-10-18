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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_LITERT_PLATFORM_SUPPORT_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_LITERT_PLATFORM_SUPPORT_H_

#include "litert/c/litert_platform_support.h"

namespace litert {

// Check whether LiteRT has been compiled with OpenCL support.
inline bool HasOpenClSupport() { return LiteRtHasOpenClSupport(); }

// Check whether LiteRT has been compiled with OpenGl support.
inline bool HasOpenGlSupport() { return LiteRtHasOpenGlSupport(); }

// Check whether LiteRT has been compiled with AHWB support.
inline bool HasAhwbSupport() { return LiteRtHasAhwbSupport(); }

// Check whether LiteRT has been compiled with Ion support.
inline bool HasIonSupport() { return LiteRtHasIonSupport(); }

// Check whether LiteRT has been compiled with DMA-BUF support.
inline bool HasDmaBufSupport() { return LiteRtHasDmaBufSupport(); }

// Check whether LiteRT has been compiled with FastRPC support.
inline bool HasFastRpcSupport() { return LiteRtHasFastRpcSupport(); }

// Check whether LiteRT has been compiled with SyncFence support.
inline bool HasSyncFenceSupport() { return LiteRtHasSyncFenceSupport(); }

// Check whether LiteRT has been compiled with Metal support.
inline bool HasMetalSupport() { return LiteRtHasMetalSupport(); }

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_LITERT_PLATFORM_SUPPORT_H_
