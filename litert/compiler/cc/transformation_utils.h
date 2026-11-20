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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/compiler/cc/litert_builder.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/compiler/cc/litert_op_options.h"

namespace litert::compiler {

// Extracts the namespace prefix from a tensor name.
// E.g.: "model/layers.0/self_attn/q_proj/MatMul" ->
//       "model/layers.0/self_attn/q_proj"
// If semicolon-delimited, extracts from the first candidate.
// Returns empty string_view if no '/' is found.
absl::string_view GetTensorNamespace(absl::string_view tensor_name);

inline absl::string_view GetTensorNamespace(const Tensor& tensor) {
  return tensor.Get() == nullptr ? absl::string_view()
                                 : GetTensorNamespace(tensor.Name());
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

Expected<Tensor> ConstI32(const Builder& builder,
                          absl::Span<const int32_t> data);

Expected<Tensor> NewLike(const Builder& builder, const Tensor& like,
                         absl::Span<const int32_t> dimensions,
                         absl::string_view tag = "");

Expected<Tensor> BuildSlice(const Builder& builder, const Tensor& input,
                            absl::Span<const int32_t> begin,
                            absl::Span<const int32_t> size, const Tensor& like,
                            absl::string_view tag = "Slice");

Expected<void> BuildConcatInto(const Builder& builder,
                               absl::Span<const Tensor> inputs, int32_t axis,
                               const Tensor& output);

Expected<Tensor> BuildConcat(const Builder& builder,
                             absl::Span<const Tensor> inputs, int32_t axis,
                             const Tensor& like,
                             absl::Span<const int32_t> dimensions,
                             absl::string_view tag = "Concat");

Expected<Tensor> BuildSoftmax(const Builder& builder, const Tensor& input,
                              float beta, const Tensor& like,
                              absl::Span<const int32_t> dimensions,
                              absl::string_view tag = "Softmax");

Expected<Tensor> BuildAdd(const Builder& builder, const Tensor& lhs,
                          const Tensor& rhs, const Tensor& like,
                          absl::Span<const int32_t> dimensions,
                          absl::string_view tag = "Add");

Expected<void> BuildReshapeInto(const Builder& builder, const Tensor& input,
                                absl::Span<const int32_t> dimensions,
                                const Tensor& output);

Expected<Tensor> BuildReshape(const Builder& builder, const Tensor& input,
                              const Tensor& like,
                              absl::Span<const int32_t> dimensions,
                              absl::string_view tag = "Reshape");

Expected<Tensor> BuildBatchMatmul(const Builder& builder, const Tensor& lhs,
                                  const Tensor& rhs, const Tensor& like,
                                  absl::Span<const int32_t> dimensions,
                                  bool asymmetric_quantize_input = false,
                                  bool adj_y = false,
                                  absl::string_view tag = "BatchMatmul");

// Returns the consuming op if `tensor` has exactly one use with opcode `code`
// at `input_index`.
std::optional<Op> SingleUser(const Tensor& tensor, LiteRtOpCode code,
                             LiteRtParamIndex input_index = 0);

std::optional<Tensor> SingleOutput(const Op& op);

// Returns static positive dimensions for a ranked tensor (empty vector for
// rank-0 scalars), or std::nullopt for unranked, dynamic (< 0), or zero-sized
// (== 0) tensors.
std::optional<std::vector<int32_t>> GetStaticPositiveDims(const Tensor& tensor);

// Overflow-safe element count for static positive dimensions (`1` for `{}`).
std::optional<size_t> CheckedNumElements(absl::Span<const int32_t> dims);

// NumPy/TFLite right-aligned broadcast shape on static positive dimensions.
std::optional<std::vector<int32_t>> BroadcastShapes(
    absl::Span<const int32_t> lhs, absl::Span<const int32_t> rhs);

// Reads unquantized static-positive-shape Int32 or Int64 constant weights as
// int64_t values.
std::optional<std::vector<int64_t>> ReadIntConstValues(const Tensor& tensor);

Expected<Tensor> BuildTypedTensor(const Builder& builder, ElementType type,
                                  absl::Span<const int32_t> dims,
                                  absl::string_view name = "");

template <typename T>
Expected<Tensor> BuildConst(const Builder& builder,
                            absl::Span<const int32_t> dims,
                            absl::Span<const T> values,
                            absl::string_view name = "") {
  const auto expected_elems = CheckedNumElements(dims);
  if (!expected_elems || *expected_elems != values.size()) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument);
  }
  LITERT_ASSIGN_OR_RETURN(
      Tensor tensor,
      BuildTypedTensor(builder, GetElementType<T>(), dims, name));
  LITERT_RETURN_IF_ERROR(builder.BuildWeights<T>(values, tensor));
  return tensor;
}

template <typename T>
Expected<Tensor> BuildScalarConst(const Builder& builder, T value,
                                  absl::string_view name = "") {
  return BuildConst<T>(builder, {}, absl::Span<const T>(&value, 1), name);
}

Expected<void> BuildCastInto(const Builder& builder, const Tensor& input,
                             const Tensor& output);

Expected<Tensor> BuildCast(const Builder& builder, const Tensor& input,
                           ElementType out_type,
                           absl::Span<const int32_t> out_dims,
                           absl::string_view name = "");

Expected<void> BuildBinaryOpInto(const Builder& builder, LiteRtOpCode code,
                                 const Tensor& lhs, const Tensor& rhs,
                                 const Tensor& output,
                                 uint32_t fused_activation = 0);

Expected<Tensor> BuildBinaryOp(const Builder& builder, LiteRtOpCode code,
                               const Tensor& lhs, const Tensor& rhs,
                               ElementType out_type,
                               absl::Span<const int32_t> out_dims,
                               uint32_t fused_activation = 0,
                               absl::string_view name = "");

template <typename OptionsT>
Expected<void> CopyOpOptions(const Builder& builder, const Op& src,
                             const Op& dst) {
  if (src.Get() == nullptr || dst.Get() == nullptr) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument);
  }
  LITERT_ASSIGN_OR_RETURN(OptionsT opts,
                          GetOptionsAs<OptionsT>(src.ctx(), src.Get()));
  return builder.SetOpOptions(dst, std::move(opts));
}

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
