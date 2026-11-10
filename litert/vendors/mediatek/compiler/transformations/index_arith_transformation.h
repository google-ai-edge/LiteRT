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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_MEDIATEK_COMPILER_TRANSFORMATIONS_INDEX_ARITH_TRANSFORMATION_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_MEDIATEK_COMPILER_TRANSFORMATIONS_INDEX_ARITH_TRANSFORMATION_H_

#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"

// Bool-free rewrites of integer index math. MediaTek NPUs have no bool type,
// so bool producers/consumers (comparisons, logical ops, Select, OneHot) are
// either removed or replaced with float arithmetic the NPU can run.

#ifdef __cplusplus
extern "C" {
#endif

// Replaces the int32 floor-division emulation emitted by converters
//   q = Div(a, k)                                   (truncating)
//   Select(Sign(a) != 1 && FloorMod(a, k) != 0, q - 1, q)
// with a single FloorDiv(a, k) (k: positive int32 constant). TFLite int32 Div
// truncates toward zero, so both forms are equal for every a.
//
// Before:                                      After:
//            a                                        a     k
//   +--------+---------+----------+                   |     |
//   |        |         |          |               +---v-----v---+
//  Sign  FloorMod(k)  Div(k)      |               |  FloorDiv   |
//   |        |         |  \       |               +------+------+
//  NE(1)   NE(0)       |  Sub(1)  |                      |
//   \       /          |   |      |                      q
//   LogicalAnd         |   |      |
//        \             |   |
//         +------> Select(cond, q - 1, q) --> q
LiteRtStatus FloorDivTransformation(const LiteRtCompilerContext* context,
                                    LiteRtBuilder builder_ptr, LiteRtOp op);

// Replaces a float OneHot (on = 1, off = 0, last axis, depth <= 1024) whose
// users are all Quantize ops with exact float arithmetic runnable on the NPU:
//   one_hot[..., j] = Relu(1 - |float(idx) - j|)
// For integer idx this is exactly 1 when idx == j and 0 otherwise, and all
// intermediate values are exactly representable in fp16 (|j| <= 1024).
//
// If the (quantized, rescaled) one-hot also feeds the "non-empty bin" mask
//   LogicalNot(ReduceAll(Equal(Mul(Quantize(one_hot), c), 0), axes))
// the mask is rewritten as NotEqual(ReduceMax(one_hot, axes), 0), which is
// equal because one_hot >= 0 and the quantized rescale maps 1 to a nonzero
// value (checked from the quantization parameters). Only the final 140-wide
// NotEqual producing the bool graph output stays on the CPU.
//
// Before:                                   After:
//   idx [1, S] int32                          idx [1, S] int32
//        |                                         |
//   OneHot(depth D)  (CPU)                     Cast f32 -> Reshape [1, S, 1]
//        | [1, S, D] f32                           |
//   Quantize -> Mul(c) -----------+           Sub(iota [1, 1, D]) -> Abs
//        |                        |                |
//   BatchMatmul ...          Equal(0) (CPU)   Sub(1, .) -> Relu -> one_hot
//                                 |                |              |
//                            ReduceAll (CPU)  Quantize -> Mul  ReduceMax(axes)
//                                 |                |              |
//                            LogicalNot       BatchMatmul ...  NotEqual(0)
//                                 |                               |
//                               mask                             mask
LiteRtStatus OneHotArithTransformation(const LiteRtCompilerContext* context,
                                       LiteRtBuilder builder_ptr, LiteRtOp op);

#ifdef __cplusplus
}
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_MEDIATEK_COMPILER_TRANSFORMATIONS_INDEX_ARITH_TRANSFORMATION_H_
