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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_TRANSFORMATIONS_TRANSFORMATION_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_TRANSFORMATIONS_TRANSFORMATION_UTILS_H_

#include <cstdint>
#include <optional>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_expected.h"
#include "litert/compiler/cc/litert_builder.h"
#include "litert/compiler/cc/litert_model.h"

namespace litert::qnn {

bool HasDims(const compiler::Tensor& tensor,
             absl::Span<const int32_t> expected);

bool HasPerm(const compiler::Op& transpose,
             absl::Span<const int32_t> expected);

Expected<compiler::Tensor> ConstI32(compiler::Builder& builder,
                                    absl::Span<const int32_t> data);

Expected<compiler::Tensor> NewLike(
    compiler::Builder& builder, const compiler::Tensor& like,
    absl::Span<const int32_t> dimensions);

Expected<compiler::Tensor> BuildSlice(
    compiler::Builder& builder, const compiler::Tensor& input,
    absl::Span<const int32_t> begin, absl::Span<const int32_t> size,
    const compiler::Tensor& like);

Expected<void> BuildConcatInto(
    compiler::Builder& builder,
    absl::Span<const compiler::Tensor> inputs, int32_t axis,
    const compiler::Tensor& output);

Expected<compiler::Tensor> BuildConcat(
    compiler::Builder& builder,
    absl::Span<const compiler::Tensor> inputs, int32_t axis,
    const compiler::Tensor& like, absl::Span<const int32_t> dimensions);

Expected<compiler::Tensor> BuildSoftmax(
    compiler::Builder& builder, const compiler::Tensor& input, float beta,
    const compiler::Tensor& like,
    absl::Span<const int32_t> dimensions);

Expected<compiler::Tensor> BuildAdd(
    compiler::Builder& builder, const compiler::Tensor& lhs,
    const compiler::Tensor& rhs, const compiler::Tensor& like,
    absl::Span<const int32_t> dimensions);

Expected<void> BuildReshapeInto(compiler::Builder& builder,
                                const compiler::Tensor& input,
                                absl::Span<const int32_t> dimensions,
                                const compiler::Tensor& output);

Expected<compiler::Tensor> BuildReshape(
    compiler::Builder& builder, const compiler::Tensor& input,
    const compiler::Tensor& like, absl::Span<const int32_t> dimensions);

Expected<compiler::Tensor> BuildBatchMatmul(
    compiler::Builder& builder, const compiler::Tensor& lhs,
    const compiler::Tensor& rhs, const compiler::Tensor& like,
    absl::Span<const int32_t> dimensions,
    bool asymmetric_quantize_input = false,
    bool adj_y = false);

// Returns the consuming op if `tensor` has exactly one use with opcode `code`
// at `input_index`.
std::optional<compiler::Op> SingleUser(const compiler::Tensor& tensor,
                                       LiteRtOpCode code,
                                       LiteRtParamIndex input_index = 0);

std::optional<compiler::Tensor> SingleOutput(const compiler::Op& op);

}  // namespace litert::qnn

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_TRANSFORMATIONS_TRANSFORMATION_UTILS_H_
