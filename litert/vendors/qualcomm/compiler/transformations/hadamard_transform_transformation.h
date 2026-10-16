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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_TRANSFORMATIONS_HADAMARD_TRANSFORM_TRANSFORMATION_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_TRANSFORMATIONS_HADAMARD_TRANSFORM_TRANSFORMATION_H_

#include <optional>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/compiler/cc/litert_builder.h"
#include "litert/compiler/cc/litert_model.h"

namespace litert::qnn {

inline constexpr absl::string_view kHadamardTransformCompositeName =
    "HadamardTransform";

bool IsHadamardTransformOp(const litert::compiler::Op& op);

std::optional<float> GetHadamardTransformScale(const litert::compiler::Op& op);

litert::Expected<litert::compiler::Op> BuildHadamardTransformOp(
    litert::compiler::Builder& builder, const litert::compiler::Tensor& input,
    const litert::compiler::Tensor& output, float scale);

}  // namespace litert::qnn

#ifdef __cplusplus
extern "C" {
#endif

// Replaces a quantized INT16 tfl.fully_connected op whose constant INT8/INT4
// weight matrix [N, N] (where N is a power of 2) is a scaled Sylvester
// Hadamard matrix H_N (H_N[i, j] = (-1)^popcount(i & j) * v) and has no bias
// with a "HadamardTransform" composite op carrying the effective scale
// attribute `scale = weight_scale * v * sqrt(N)`. During QNN graph composition,
// this composite op is lowered to QNN's native HTP HadamardTransform op,
// eliminating the O(N^2) dense matrix multiplication and weight memory reads.
//
// Before:
//     input [..., N] (sym int16)      weight [N, N] (const sym int8/int4, H_N)
//                   \                        /
//                    +──────────┬───────────+
//                               |
//                +──────────────v──────────────+
//                |     tfl.fully_connected     | (bias = None, act = NONE)
//                +──────────────┬──────────────+
//                               |
//                    output [..., N] (sym int16)
//
// After:
//                    input [..., N] (sym int16)
//                               |
//            +──────────────────v──────────────────+
//            |  shlo.composite("HadamardTransform")|
//            |  attr: { scale = s_w * v * sqrt(N) }|
//            +──────────────────┬──────────────────+
//                               |
//                    output [..., N] (sym int16)
LiteRtStatus ConvertFcToHadamardTransform(const LiteRtCompilerContext* context,
                                          LiteRtBuilder builder_ptr,
                                          LiteRtOp op);

// Rewrites a block-diagonal Hadamard rotation expressed as
// Reshape([..., K*N] -> [..., K, N]) -> HadamardTransform ->
// Reshape([..., K, N] -> [..., K*N]) into a Split(num_splits=K) ->
// K parallel HadamardTransform ops on [..., N] -> Concat(axis=-1).
// Placing the K independent Hadamard transforms under a single Split parent
// allows the Qualcomm HTP scheduler to execute them concurrently across HVX
// vector threads without intermediate 4D/5D reshape copies.
//
// Before (root = post-Reshape; K = D / N blocks of size N):
//
//                   input [..., K * N] (sym int16)
//                                 |
//                     +───────────v───────────+
//                     |      tfl.reshape      |
//                     +───────────┬───────────+
//                                 | h_input [..., K, N]
//                     +───────────v───────────+
//                     |   HadamardTransform   | (scale = s)
//                     +───────────┬───────────+
//                                 | h_output [..., K, N]
//                     +───────────v───────────+
//                     |  tfl.reshape (root)   |
//                     +───────────┬───────────+
//                                 |
//                   output [..., K * N] (sym int16)
//
// After (K parallel HadamardTransform branches on [..., N]):
//
//                   input [..., K * N] (sym int16)
//                                 |
//                     +───────────v───────────+
//                     |  tfl.split (axis=-1)  | (num_splits = K)
//                     +─┬─────────┬─────────┬─+
//      slice_0 [..., N] |   ...   |   ...   | slice_{K-1} [..., N]
//                       v         v         v
//                   Hadamard  Hadamard  Hadamard   (scale = s)
//                       |         |         |
//      h_out_0 [..., N] |   ...   |   ...   | h_out_{K-1} [..., N]
//                     +─v─────────v─────────v─+
//                     | tfl.concatenation(-1) |
//                     +───────────┬───────────+
//                                 |
//                   output [..., K * N] (sym int16)
LiteRtStatus ParallelizeHadamardTransform(const LiteRtCompilerContext* context,
                                          LiteRtBuilder builder_ptr,
                                          LiteRtOp op);

#ifdef __cplusplus
}
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_TRANSFORMATIONS_HADAMARD_TRANSFORM_TRANSFORMATION_H_
