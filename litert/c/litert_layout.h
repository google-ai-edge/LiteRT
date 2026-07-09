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

#ifndef ODML_LITERT_LITERT_C_LITERT_LAYOUT_H_
#define ODML_LITERT_LITERT_C_LITERT_LAYOUT_H_

#include <stdbool.h>  // NOLINT: To use bool type in C
#include <stddef.h>
#include <stdint.h>

#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Max number of dimensions in any ranked tensor type.
#define LITERT_TENSOR_MAX_RANK 8

/// The shape information for tensor types of fixed rank.
///
/// @note This concrete type is part of the public API and is ABI stable.
typedef struct {
  unsigned int rank : 7;  // The number of dimensions.
  // Whether the layout has strides.
  //
  // NOTE: This bit-field uses `unsigned int` (matching `rank` above) rather
  // than `bool` on purpose. MSVC does not coalesce adjacent bit-fields with
  // different underlying types into the same storage unit, so a `bool` field
  // here would open a fresh 4-byte unit and shift `dimensions`/`strides`,
  // making this public struct binary-incompatible between MSVC and GCC/Clang
  // builds. Keeping the underlying type identical packs both fields together
  // on every compiler. See https://github.com/google-ai-edge/LiteRT/issues/7459
  unsigned int has_strides : 1;

  // Dimension sizes, array of length `rank`. Dynamic dimensions are anything
  // less than 0. Everything from [rank, LITERT_MAX_RANK) is undefined.
  int32_t dimensions[LITERT_TENSOR_MAX_RANK];

  // Strides. Used only if has_strides is true.
  uint32_t strides[LITERT_TENSOR_MAX_RANK];
} LiteRtLayout;

// The layout below is now identical across MSVC and GCC/Clang because `rank`
// and `has_strides` share an underlying type and pack into a single storage
// unit on every compiler. These asserts intentionally have no `_MSC_VER`
// branch so any future change that reintroduces cross-compiler divergence
// fails the build.
#if defined(__cplusplus) && defined(__SIZEOF_POINTER__) && \
    __SIZEOF_POINTER__ == 8
static_assert(sizeof(LiteRtLayout) == 68, "LiteRtLayout size mismatch");
static_assert(offsetof(LiteRtLayout, dimensions) == 4,
              "LiteRtLayout dimensions offset mismatch");
static_assert(offsetof(LiteRtLayout, strides) == 36,
              "LiteRtLayout strides offset mismatch");
#endif  // __cplusplus

// Return the number of scalar elements in the provided tensor layout. Return an
// error if the layout includes dynamic dimensions.
//
// Note: LiteRtLayout is a non-opaque type (struct is defined in this header).
// Therefore, its access methods do not need to and should NOT be exported
// from the dynamic C API library (e.g. libLiteRt.so).
LiteRtStatus LiteRtGetNumLayoutElements(const LiteRtLayout* layout,
                                        size_t* num_elements);

LiteRtStatus LiteRtIsSameLayout(const LiteRtLayout* layout1,
                                const LiteRtLayout* layout2, bool* result);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // ODML_LITERT_LITERT_C_LITERT_LAYOUT_H_
