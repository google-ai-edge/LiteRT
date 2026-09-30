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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_TRANSFORMATIONS_LEGALIZE_INT32_OPS_TRANSFORMATION_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_TRANSFORMATIONS_LEGALIZE_INT32_OPS_TRANSFORMATION_H_

#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Legalizes INT32 tfl.sign to tfl.cast(f32) -> tfl.sign(f32) -> tfl.cast(i32)
// to allow execution on Qualcomm HTP NPU.
//
// Before:
//        in_tensor (int32)
//               |
//        +------v------+
//        |   tfl.sign  |
//        +------┬------+
//               |
//        out_tensor (int32)
//
// After:
//        in_tensor (int32)
//               |
//        +------v------+
//        |   tfl.cast  | (int32 -> float32)
//        +------┬------+
//               | f32_in
//        +------v------+
//        |   tfl.sign  |
//        +------┬------+
//               | f32_out
//        +------v------+
//        |   tfl.cast  | (float32 -> int32)
//        +------┬------+
//               |
//        out_tensor (int32)
LiteRtStatus LegalizeInt32SignTransformation(
    const LiteRtCompilerContext* context, LiteRtBuilder builder_ptr,
    LiteRtOp op);

// Legalizes INT32 tfl.reduce_max to tfl.cast(f32) -> tfl.reduce_max(f32) ->
// tfl.cast(i32) to allow execution on Qualcomm HTP NPU.
//
// Before:
//        in_tensor (int32)    axis_tensor
//               \                  /
//                +────────┬───────+
//                         |
//              +──────────v──────────+
//              |    tfl.reduce_max   |
//              +──────────┬──────────+
//                         |
//                 out_tensor (int32)
//
// After:
//        in_tensor (int32)
//               |
//        +------v------+
//        |   tfl.cast  | (int32 -> float32)
//        +------┬------+
//               | f32_in      axis_tensor
//               \                  /
//                +────────┬───────+
//                         |
//              +──────────v──────────+
//              |    tfl.reduce_max   |
//              +──────────┬──────────+
//                         | f32_out
//              +──────────v──────────+
//              |       tfl.cast      | (float32 -> int32)
//              +──────────┬──────────+
//                         |
//                 out_tensor (int32)
LiteRtStatus LegalizeInt32ReduceMaxTransformation(
    const LiteRtCompilerContext* context, LiteRtBuilder builder_ptr,
    LiteRtOp op);

#ifdef __cplusplus
}
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_TRANSFORMATIONS_LEGALIZE_INT32_OPS_TRANSFORMATION_H_
