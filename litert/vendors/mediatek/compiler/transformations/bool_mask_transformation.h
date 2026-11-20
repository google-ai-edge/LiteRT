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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_MEDIATEK_COMPILER_TRANSFORMATIONS_BOOL_MASK_TRANSFORMATION_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_MEDIATEK_COMPILER_TRANSFORMATIONS_BOOL_MASK_TRANSFORMATION_H_

#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Rewrites consumers reading Cast(Cast(src_f32 -> bool) -> f32) (where src_f32
// is proven to hold {0.0, 1.0} values) or Cast(bool_const -> f32) to consume
// the underlying Float32 {0.0, 1.0} tensor directly. Also lowers non-scalar
// Bool->Float32 casts backed by a Bool constant or proven {0.0, 1.0} Float32
// tensor that directly produce subgraph outputs into Add(resolved_f32, 0.0f).
LiteRtStatus BoolMaskBypassCastTransformation(
    const LiteRtCompilerContext* context, LiteRtBuilder builder_ptr,
    LiteRtOp op);

// Erases dead Cast ops involving Bool tensors whose output has no consumers and
// is not a subgraph output.
LiteRtStatus BoolMaskCleanupCastTransformation(
    const LiteRtCompilerContext* context, LiteRtBuilder builder_ptr,
    LiteRtOp op);

// Rewrites NotEqual ops on unquantized Bool or proven {0.0, 1.0} Float32
// tensors into exact Float32 {0.0, 1.0} arithmetic followed by a boundary
// Cast(f32 -> bool). Zero checks on proven {0.0, 1.0} Float32 tensors collapse
// directly to Cast(x -> out); same-shape Float32 NotEqual ops (natively
// delegable as NEURON_NOT_EQUAL) that already directly feed a subgraph output
// with no internal uses are left intact so they can run natively on NPU.
LiteRtStatus BoolMaskNotEqualTransformation(
    const LiteRtCompilerContext* context, LiteRtBuilder builder_ptr,
    LiteRtOp op);

// Rewrites Cast(Tile(Cast(bool_in -> int32), reps) -> bool_out) into a native
// Float32 Tile (converting Int64 reps constants to Int32) plus a boundary
// Cast(f32 -> bool_out).
LiteRtStatus BoolMaskTileWrapTransformation(
    const LiteRtCompilerContext* context, LiteRtBuilder builder_ptr,
    LiteRtOp op);

// Propagates Float32 {0.0, 1.0} arithmetic forward through Boolean and Select
// subgraph ops (Reshape, Squeeze, ExpandDims, Transpose, Slice, StridedSlice,
// Concatenation, Tile, LogicalNot, LogicalAnd, LogicalOr, Equal, Select,
// SelectV2), wrapping Boolean outputs in boundary Cast ops for subsequent
// consumer bypassing and dead-cast cleanup.
LiteRtStatus BoolMaskOpToFloatTransformation(
    const LiteRtCompilerContext* context, LiteRtBuilder builder_ptr,
    LiteRtOp op);

// Rewrites a subgraph-output Cast(f32 -> bool) of rank >= 1 with no internal
// consumers into a same-shape NotEqual(f32, zeros_f32) op supported directly by
// MediaTek NPU.
LiteRtStatus BoolMaskOutputNotEqualTransformation(
    const LiteRtCompilerContext* context, LiteRtBuilder builder_ptr,
    LiteRtOp op);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_MEDIATEK_COMPILER_TRANSFORMATIONS_BOOL_MASK_TRANSFORMATION_H_
