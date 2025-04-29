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

#ifndef ODML_LITERT_LITERT_C_LITERT_CUSTOM_OP_KERNEL_H_
#define ODML_LITERT_LITERT_C_LITERT_CUSTOM_OP_KERNEL_H_

#include <stddef.h>
#include <stdint.h>

#include "litert/c/litert_common.h"
#include "litert/c/litert_layout.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct {
  // NOLINTBEGIN(*-readability-class-member-naming)
  LiteRtStatus (*Init)(void* user_data, const void* init_data,
                       size_t init_data_size);
  // Called when the input size has changed.
  LiteRtStatus (*GetOutputLayouts)(void* user_data, size_t num_inputs,
                                   const LiteRtLayout* input_layouts,
                                   size_t num_outputs,
                                   LiteRtLayout* output_layouts);
  LiteRtStatus (*Run)(void* user_data, size_t num_inputs,
                      const LiteRtTensorBuffer* inputs, size_t num_outputs,
                      LiteRtTensorBuffer* outputs);
  LiteRtStatus (*Destroy)(void* user_data);
  // NOLINTEND(*-readability-class-member-naming)
} LiteRtCustomOpKernel;

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // ODML_LITERT_LITERT_C_LITERT_CUSTOM_OP_KERNEL_H_
