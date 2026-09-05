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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_MEDIATEK_COMPILER_TRANSFORMATIONS_RMS_NORM_QUANT_TRANSFORMATION_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_MEDIATEK_COMPILER_TRANSFORMATIONS_RMS_NORM_QUANT_TRANSFORMATION_H_

#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Transformation that decomposes quantized inputs and outputs of odml.rms_norm:
// 1. If input or gamma is quantized, insert tfl.dequantize to float32.
// 2. Clone odml.rms_norm with float32 inputs and output.
// 3. If output is quantized, insert tfl.quantize from float32 to the original
//    quantized output.
// 4. Erase the original odml.rms_norm.
LiteRtStatus RmsNormQuantTransformation(const LiteRtCompilerContext* context,
                                        LiteRtBuilder builder_ptr, LiteRtOp op);

#ifdef __cplusplus
}
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_MEDIATEK_COMPILER_TRANSFORMATIONS_RMS_NORM_QUANT_TRANSFORMATION_H_
