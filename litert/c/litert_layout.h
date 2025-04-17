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

// The shape information for tensor types of fixed rank.
typedef struct {
  unsigned int rank : 7;  // The number of dimensions.
  bool has_strides : 1;   // Whether the layout has strides.

  // Dimension sizes, array of length `rank`. Dynamic dimensions are anything
  // less than 0. Everything from [rank, LITERT_MAX_RANK) is undefined.
  int32_t dimensions[LITERT_TENSOR_MAX_RANK];

  // Strides. Used only if has_strides is true.
  uint32_t strides[LITERT_TENSOR_MAX_RANK];
} LiteRtLayout;

// Return the number of scalar elements in the provided tensor layout. Return an
// error if the layout includes dynamic dimensions.
LiteRtStatus LiteRtGetNumLayoutElements(const LiteRtLayout* layout,
                                        size_t* num_elements);

LiteRtStatus LiteRtIsSameLayout(const LiteRtLayout* layout1,
                                const LiteRtLayout* layout2, bool* result);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // ODML_LITERT_LITERT_C_LITERT_LAYOUT_H_
