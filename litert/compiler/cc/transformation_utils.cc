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

#include "litert/compiler/cc/transformation_utils.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/compiler/cc/litert_builder.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/compiler/cc/litert_op_options.h"

namespace litert::compiler {

absl::string_view GetTensorNamespace(absl::string_view tensor_name) {
  // If semicolon-delimited (composite name), use the first candidate.
  size_t semi = tensor_name.find(';');
  if (semi != absl::string_view::npos) {
    tensor_name = tensor_name.substr(0, semi);
  }
  size_t slash = tensor_name.rfind('/');
  if (slash == absl::string_view::npos) {
    return "";
  }
  return tensor_name.substr(0, slash);
}

std::string MakeNamespacedTensorName(absl::string_view ns,
                                     absl::string_view tag,
                                     std::optional<int32_t> index) {
  if (ns.empty()) {
    if (index.has_value()) {
      return absl::StrCat(tag, "_", *index);
    }
    return std::string(tag);
  }
  if (index.has_value()) {
    return absl::StrCat(ns, "/", tag, "_", *index);
  }
  return absl::StrCat(ns, "/", tag);
}

bool HasDims(const Tensor& tensor, absl::Span<const int32_t> expected) {
  auto type = tensor.RankedTensorType();
  return type && type->Layout().Dimensions() == expected;
}

bool HasPerm(const Op& transpose, absl::Span<const int32_t> expected) {
  if (transpose.Code() != kLiteRtOpCodeTflTranspose ||
      transpose.Inputs().size() != 2) {
    return false;
  }
  const Tensor perm = transpose.Inputs()[1];
  if (!perm.IsConstant()) return false;
  auto perm_data = perm.WeightsData<int32_t>();
  return perm_data && *perm_data == expected;
}

Expected<Tensor> ConstI32(Builder& builder, absl::Span<const int32_t> data) {
  RankedTensorType type(
      ElementType::Int32,
      Layout(BuildLayout({static_cast<int32_t>(data.size())})));
  LITERT_ASSIGN_OR_RETURN(
      auto tensor,
      builder.BuildTensor(RankedTensorSpecBuilder(type).Build()));
  LITERT_RETURN_IF_ERROR(builder.BuildWeights<int32_t>(data, tensor));
  return tensor;
}

Expected<Tensor> NewLike(Builder& builder, const Tensor& like,
                         absl::Span<const int32_t> dimensions,
                         absl::string_view tag) {
  RankedTensorType type(
      like.ElementType(),
      Layout(BuildLayout(dimensions.data(),
                         dimensions.data() + dimensions.size())));
  RankedTensorSpec spec{std::move(type), std::nullopt, std::nullopt,
                        std::nullopt, std::nullopt, std::nullopt};
  if (like.QTypeId() == kLiteRtQuantizationPerTensor) {
    spec.per_tensor_quantization = like.PerTensorQuantization();
  }
  if (!tag.empty()) {
    spec.tensor_name = MakeNamespacedTensorName(like, tag);
  }
  return builder.BuildTensor(spec);
}

Expected<Tensor> BuildSlice(Builder& builder, const Tensor& input,
                            absl::Span<const int32_t> begin,
                            absl::Span<const int32_t> size,
                            const Tensor& like,
                            absl::string_view tag) {
  LITERT_ASSIGN_OR_RETURN(auto output, NewLike(builder, like, size, tag));
  LITERT_ASSIGN_OR_RETURN(auto begin_tensor, ConstI32(builder, begin));
  LITERT_ASSIGN_OR_RETURN(auto size_tensor, ConstI32(builder, size));
  LITERT_RETURN_IF_ERROR(builder.BuildOp(
      kLiteRtOpCodeTflSlice, {input, begin_tensor, size_tensor}, {output}));
  return output;
}

Expected<void> BuildConcatInto(Builder& builder,
                               absl::Span<const Tensor> inputs, int32_t axis,
                               const Tensor& output) {
  LITERT_ASSIGN_OR_RETURN(
      auto op,
      builder.BuildOp(kLiteRtOpCodeTflConcatenation, inputs, {output}));
  ConcatenationOptions options;
  options.axis = axis;
  options.fused_activation_function = 0;
  LITERT_RETURN_IF_ERROR(builder.SetOpOptions(op, std::move(options)));
  return {};
}

Expected<Tensor> BuildConcat(Builder& builder,
                             absl::Span<const Tensor> inputs, int32_t axis,
                             const Tensor& like,
                             absl::Span<const int32_t> dimensions,
                             absl::string_view tag) {
  LITERT_ASSIGN_OR_RETURN(auto output, NewLike(builder, like, dimensions, tag));
  LITERT_RETURN_IF_ERROR(BuildConcatInto(builder, inputs, axis, output));
  return output;
}

Expected<Tensor> BuildSoftmax(Builder& builder, const Tensor& input, float beta,
                              const Tensor& like,
                              absl::Span<const int32_t> dimensions,
                              absl::string_view tag) {
  LITERT_ASSIGN_OR_RETURN(auto output, NewLike(builder, like, dimensions, tag));
  LITERT_ASSIGN_OR_RETURN(
      auto op,
      builder.BuildOp(kLiteRtOpCodeTflSoftmax, {input}, {output}));
  SoftmaxOptions options;
  options.beta = beta;
  LITERT_RETURN_IF_ERROR(builder.SetOpOptions(op, std::move(options)));
  return output;
}

Expected<Tensor> BuildAdd(Builder& builder, const Tensor& lhs,
                          const Tensor& rhs, const Tensor& like,
                          absl::Span<const int32_t> dimensions,
                          absl::string_view tag) {
  LITERT_ASSIGN_OR_RETURN(auto output, NewLike(builder, like, dimensions, tag));
  LITERT_ASSIGN_OR_RETURN(
      auto op, builder.BuildOp(kLiteRtOpCodeTflAdd, {lhs, rhs}, {output}));
  AddOptions options;
  options.fused_activation_function = 0;
  LITERT_RETURN_IF_ERROR(builder.SetOpOptions(op, std::move(options)));
  return output;
}

Expected<void> BuildReshapeInto(Builder& builder,
                                const Tensor& input,
                                absl::Span<const int32_t> dimensions,
                                const Tensor& output) {
  LITERT_ASSIGN_OR_RETURN(auto shape, ConstI32(builder, dimensions));
  LITERT_ASSIGN_OR_RETURN(
      auto op, builder.BuildOp(kLiteRtOpCodeTflReshape, {input, shape},
                               {output}));
  ReshapeOptions options;
  options.new_shape.assign(dimensions.begin(), dimensions.end());
  LITERT_RETURN_IF_ERROR(builder.SetOpOptions(op, std::move(options)));
  return {};
}

Expected<Tensor> BuildReshape(Builder& builder, const Tensor& input,
                              const Tensor& like,
                              absl::Span<const int32_t> dimensions,
                              absl::string_view tag) {
  LITERT_ASSIGN_OR_RETURN(auto output, NewLike(builder, like, dimensions, tag));
  LITERT_RETURN_IF_ERROR(
      BuildReshapeInto(builder, input, dimensions, output));
  return output;
}

Expected<Tensor> BuildBatchMatmul(Builder& builder, const Tensor& lhs,
                                  const Tensor& rhs, const Tensor& like,
                                  absl::Span<const int32_t> dimensions,
                                  bool asymmetric_quantize_input,
                                  bool adj_y,
                                  absl::string_view tag) {
  LITERT_ASSIGN_OR_RETURN(auto output, NewLike(builder, like, dimensions, tag));
  LITERT_ASSIGN_OR_RETURN(
      auto op, builder.BuildOp(kLiteRtOpCodeTflBatchMatmul, {lhs, rhs},
                               {output}));
  BatchMatmulOptions options;
  options.adj_x = false;
  options.adj_y = adj_y;
  options.asymmetric_quantize_input = asymmetric_quantize_input;
  LITERT_RETURN_IF_ERROR(builder.SetOpOptions(op, std::move(options)));
  return output;
}

std::optional<Op> SingleUser(const Tensor& tensor,
                             LiteRtOpCode code,
                             LiteRtParamIndex input_index) {
  auto uses = tensor.Uses();
  if (uses.size() != 1 || uses[0].user.Code() != code ||
      uses[0].user_arg_ind != input_index) {
    return std::nullopt;
  }
  return uses[0].user;
}

std::optional<Tensor> SingleOutput(const Op& op) {
  auto outputs = op.Outputs();
  if (outputs.size() != 1) {
    return std::nullopt;
  }
  return outputs[0];
}

}  // namespace litert::compiler
