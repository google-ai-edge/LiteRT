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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_COMPILER_CC_TRANSFORMATION_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_COMPILER_CC_TRANSFORMATION_UTILS_H_

#include <cstdint>
#include <optional>
#include <string>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_expected.h"
#include "litert/compiler/cc/litert_builder.h"
#include "litert/compiler/cc/litert_model.h"

namespace litert::compiler {

// Extracts the namespace prefix from a tensor name.
// E.g.: "model/layers.0/self_attn/q_proj/MatMul" ->
//       "model/layers.0/self_attn/q_proj"
// If semicolon-delimited, extracts from the first candidate.
// Returns empty string_view if no '/' is found.
absl::string_view GetTensorNamespace(absl::string_view tensor_name);

inline absl::string_view GetTensorNamespace(const Tensor& tensor) {
  return GetTensorNamespace(tensor.Name());
}

// Combines a namespace prefix with an op/tensor tag and optional discriminator.
// E.g.: MakeNamespacedTensorName("model/layers.0/self_attn", "logits")
//       -> "model/layers.0/self_attn/logits"
std::string MakeNamespacedTensorName(
    absl::string_view ns, absl::string_view tag,
    std::optional<int32_t> index = std::nullopt);

inline std::string MakeNamespacedTensorName(
    const Tensor& like, absl::string_view tag,
    std::optional<int32_t> index = std::nullopt) {
  return MakeNamespacedTensorName(GetTensorNamespace(like), tag, index);
}

bool HasDims(const Tensor& tensor, absl::Span<const int32_t> expected);

bool HasPerm(const Op& transpose, absl::Span<const int32_t> expected);

Expected<Tensor> ConstI32(Builder& builder, absl::Span<const int32_t> data);

Expected<Tensor> NewLike(Builder& builder, const Tensor& like,
                         absl::Span<const int32_t> dimensions,
                         absl::string_view tag = "");

Expected<Tensor> BuildSlice(Builder& builder, const Tensor& input,
                            absl::Span<const int32_t> begin,
                            absl::Span<const int32_t> size, const Tensor& like,
                            absl::string_view tag = "Slice");

Expected<void> BuildConcatInto(Builder& builder,
                               absl::Span<const Tensor> inputs, int32_t axis,
                               const Tensor& output);

Expected<Tensor> BuildConcat(Builder& builder,
                             absl::Span<const Tensor> inputs, int32_t axis,
                             const Tensor& like,
                             absl::Span<const int32_t> dimensions,
                             absl::string_view tag = "Concat");

Expected<Tensor> BuildSoftmax(Builder& builder, const Tensor& input, float beta,
                              const Tensor& like,
                              absl::Span<const int32_t> dimensions,
                              absl::string_view tag = "Softmax");

Expected<Tensor> BuildAdd(Builder& builder, const Tensor& lhs,
                          const Tensor& rhs, const Tensor& like,
                          absl::Span<const int32_t> dimensions,
                          absl::string_view tag = "Add");

Expected<void> BuildReshapeInto(Builder& builder, const Tensor& input,
                                absl::Span<const int32_t> dimensions,
                                const Tensor& output);

Expected<Tensor> BuildReshape(Builder& builder, const Tensor& input,
                              const Tensor& like,
                              absl::Span<const int32_t> dimensions,
                              absl::string_view tag = "Reshape");

Expected<Tensor> BuildBatchMatmul(Builder& builder, const Tensor& lhs,
                                  const Tensor& rhs, const Tensor& like,
                                  absl::Span<const int32_t> dimensions,
                                  bool asymmetric_quantize_input,
                                  bool adj_y = false,
                                  absl::string_view tag = "BatchMatmul");

// Returns the consuming op if `tensor` has exactly one use with opcode `code`
// at `input_index`.
std::optional<Op> SingleUser(const Tensor& tensor, LiteRtOpCode code,
                             LiteRtParamIndex input_index);

std::optional<Tensor> SingleOutput(const Op& op);

}  // namespace litert::compiler

namespace litert::qnn {

// Backwards compatibility aliases for vendor transformations during migration.
using ::litert::compiler::BuildAdd;
using ::litert::compiler::BuildBatchMatmul;
using ::litert::compiler::BuildConcat;
using ::litert::compiler::BuildConcatInto;
using ::litert::compiler::BuildReshape;
using ::litert::compiler::BuildReshapeInto;
using ::litert::compiler::BuildSlice;
using ::litert::compiler::BuildSoftmax;
using ::litert::compiler::ConstI32;
using ::litert::compiler::GetTensorNamespace;
using ::litert::compiler::HasDims;
using ::litert::compiler::HasPerm;
using ::litert::compiler::MakeNamespacedTensorName;
using ::litert::compiler::NewLike;
using ::litert::compiler::SingleOutput;
using ::litert::compiler::SingleUser;

}  // namespace litert::qnn

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_COMPILER_CC_TRANSFORMATION_UTILS_H_
