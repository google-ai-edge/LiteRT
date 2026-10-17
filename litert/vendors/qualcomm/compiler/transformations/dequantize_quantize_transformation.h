// Copyright 2026 Google LLC.
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
//
// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_TRANSFORMATIONS_DEQUANTIZE_QUANTIZE_TRANSFORMATION_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_TRANSFORMATIONS_DEQUANTIZE_QUANTIZE_TRANSFORMATION_H_

#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Rewrites a quantized INT8/INT16 -> Dequantize(F32) -> Quantize(INT16/INT8)
// sequence into direct requantization between INT8 and INT16.
LiteRtStatus DequantizeQuantizeTransformation(
    const LiteRtCompilerContext* context, LiteRtBuilder builder_ptr,
    LiteRtOp op);

#ifdef __cplusplus
}
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_TRANSFORMATIONS_DEQUANTIZE_QUANTIZE_TRANSFORMATION_H_
