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

#include "litert/vendors/qualcomm/compiler/transformations/transformation_utils.h"

#include <cstdint>
#include <optional>
#include <string>  // NOLINT
#include <utility>

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

namespace litert::qnn {

bool HasDims(const compiler::Tensor& tensor,
             absl::Span<const int32_t> expected) {
  auto type = tensor.RankedTensorType();
  return type && type->Layout().Dimensions() == expected;
}

bool HasPerm(const compiler::Op& transpose,
             absl::Span<const int32_t> expected) {
  if (transpose.Code() != kLiteRtOpCodeTflTranspose ||
      transpose.Inputs().size() != 2) {
    return false;
  }
  const compiler::Tensor perm = transpose.Inputs()[1];
  if (!perm.IsConstant()) return false;
  auto perm_data = perm.WeightsData<int32_t>();
  return perm_data && *perm_data == expected;
}

Expected<compiler::Tensor> ConstI32(compiler::Builder& builder,
                                    absl::Span<const int32_t> data) {
  RankedTensorType type(
      ElementType::Int32,
      Layout(BuildLayout({static_cast<int32_t>(data.size())})));
  LITERT_ASSIGN_OR_RETURN(
      auto tensor,
      builder.BuildTensor(compiler::RankedTensorSpecBuilder(type).Build()));
  LITERT_RETURN_IF_ERROR(builder.BuildWeights<int32_t>(data, tensor));
  return tensor;
}

Expected<compiler::Tensor> NewLike(
    compiler::Builder& builder, const compiler::Tensor& like,
    absl::Span<const int32_t> dimensions) {
  RankedTensorType type(
      like.ElementType(),
      Layout(BuildLayout(dimensions.data(),
                         dimensions.data() + dimensions.size())));
  compiler::RankedTensorSpec spec{std::move(type), std::nullopt, std::nullopt,
                                  std::nullopt, std::nullopt, std::nullopt};
  if (like.QTypeId() == kLiteRtQuantizationPerTensor) {
    spec.per_tensor_quantization = like.PerTensorQuantization();
  }
  // TODO(jiunkaiy): Resolve tensor name propagation and re-enable this block.
  // const auto name = like.Name();
  // if (!name.empty()) {
  //   spec.tensor_name = std::string(name);
  // }
  return builder.BuildTensor(spec);
}

Expected<compiler::Tensor> BuildSlice(
    compiler::Builder& builder, const compiler::Tensor& input,
    absl::Span<const int32_t> begin, absl::Span<const int32_t> size,
    const compiler::Tensor& like) {
  LITERT_ASSIGN_OR_RETURN(auto output, NewLike(builder, like, size));
  LITERT_ASSIGN_OR_RETURN(auto begin_tensor, ConstI32(builder, begin));
  LITERT_ASSIGN_OR_RETURN(auto size_tensor, ConstI32(builder, size));
  LITERT_RETURN_IF_ERROR(builder.BuildOp(
      kLiteRtOpCodeTflSlice, {input, begin_tensor, size_tensor}, {output}));
  return output;
}

Expected<void> BuildConcatInto(compiler::Builder& builder,
                               absl::Span<const compiler::Tensor> inputs,
                               int32_t axis,
                               const compiler::Tensor& output) {
  LITERT_ASSIGN_OR_RETURN(
      auto op,
      builder.BuildOp(kLiteRtOpCodeTflConcatenation, inputs, {output}));
  compiler::ConcatenationOptions options;
  options.axis = axis;
  options.fused_activation_function = 0;
  LITERT_RETURN_IF_ERROR(builder.SetOpOptions(op, std::move(options)));
  return {};
}

Expected<compiler::Tensor> BuildConcat(
    compiler::Builder& builder, absl::Span<const compiler::Tensor> inputs,
    int32_t axis, const compiler::Tensor& like,
    absl::Span<const int32_t> dimensions) {
  LITERT_ASSIGN_OR_RETURN(auto output, NewLike(builder, like, dimensions));
  LITERT_RETURN_IF_ERROR(BuildConcatInto(builder, inputs, axis, output));
  return output;
}

Expected<compiler::Tensor> BuildSoftmax(
    compiler::Builder& builder, const compiler::Tensor& input, float beta,
    const compiler::Tensor& like,
    absl::Span<const int32_t> dimensions) {
  LITERT_ASSIGN_OR_RETURN(auto output, NewLike(builder, like, dimensions));
  LITERT_ASSIGN_OR_RETURN(
      auto op,
      builder.BuildOp(kLiteRtOpCodeTflSoftmax, {input}, {output}));
  compiler::SoftmaxOptions options;
  options.beta = beta;
  LITERT_RETURN_IF_ERROR(builder.SetOpOptions(op, std::move(options)));
  return output;
}

Expected<compiler::Tensor> BuildAdd(
    compiler::Builder& builder, const compiler::Tensor& lhs,
    const compiler::Tensor& rhs, const compiler::Tensor& like,
    absl::Span<const int32_t> dimensions) {
  LITERT_ASSIGN_OR_RETURN(auto output, NewLike(builder, like, dimensions));
  LITERT_ASSIGN_OR_RETURN(
      auto op, builder.BuildOp(kLiteRtOpCodeTflAdd, {lhs, rhs}, {output}));
  compiler::AddOptions options;
  options.fused_activation_function = 0;
  LITERT_RETURN_IF_ERROR(builder.SetOpOptions(op, std::move(options)));
  return output;
}

Expected<void> BuildReshapeInto(compiler::Builder& builder,
                                const compiler::Tensor& input,
                                absl::Span<const int32_t> dimensions,
                                const compiler::Tensor& output) {
  LITERT_ASSIGN_OR_RETURN(auto shape, ConstI32(builder, dimensions));
  LITERT_ASSIGN_OR_RETURN(
      auto op, builder.BuildOp(kLiteRtOpCodeTflReshape, {input, shape},
                               {output}));
  compiler::ReshapeOptions options;
  options.new_shape.assign(dimensions.begin(), dimensions.end());
  LITERT_RETURN_IF_ERROR(builder.SetOpOptions(op, std::move(options)));
  return {};
}

Expected<compiler::Tensor> BuildReshape(
    compiler::Builder& builder, const compiler::Tensor& input,
    const compiler::Tensor& like, absl::Span<const int32_t> dimensions) {
  LITERT_ASSIGN_OR_RETURN(auto output, NewLike(builder, like, dimensions));
  LITERT_RETURN_IF_ERROR(
      BuildReshapeInto(builder, input, dimensions, output));
  return output;
}

Expected<compiler::Tensor> BuildBatchMatmul(
    compiler::Builder& builder, const compiler::Tensor& lhs,
    const compiler::Tensor& rhs, const compiler::Tensor& like,
    absl::Span<const int32_t> dimensions, bool asymmetric_quantize_input,
    bool adj_y) {
  LITERT_ASSIGN_OR_RETURN(auto output, NewLike(builder, like, dimensions));
  LITERT_ASSIGN_OR_RETURN(
      auto op, builder.BuildOp(kLiteRtOpCodeTflBatchMatmul, {lhs, rhs},
                               {output}));
  compiler::BatchMatmulOptions options;
  options.adj_x = false;
  options.adj_y = adj_y;
  options.asymmetric_quantize_input = asymmetric_quantize_input;
  LITERT_RETURN_IF_ERROR(builder.SetOpOptions(op, std::move(options)));
  return output;
}

std::optional<compiler::Op> SingleUser(const compiler::Tensor& tensor,
                                       LiteRtOpCode code,
                                       LiteRtParamIndex input_index) {
  auto uses = tensor.Uses();
  if (uses.size() != 1 || uses[0].user.Code() != code ||
      uses[0].user_arg_ind != input_index) {
    return std::nullopt;
  }
  return uses[0].user;
}

std::optional<compiler::Tensor> SingleOutput(const compiler::Op& op) {
  auto outputs = op.Outputs();
  if (outputs.size() != 1) {
    return std::nullopt;
  }
  return outputs[0];
}

}  // namespace litert::qnn
