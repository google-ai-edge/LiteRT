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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_layout.h"
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
  if (tensor.Get() == nullptr) {
    return false;
  }
  auto type = tensor.RankedTensorType();
  return type && type->Layout().Dimensions() == expected;
}

bool HasPerm(const Op& transpose, absl::Span<const int32_t> expected) {
  if (transpose.Get() == nullptr ||
      transpose.Code() != kLiteRtOpCodeTflTranspose ||
      transpose.Inputs().size() != 2) {
    return false;
  }
  const Tensor perm = transpose.Inputs()[1];
  if (perm.Get() == nullptr || !perm.IsConstant()) return false;
  auto perm_data = perm.WeightsData<int32_t>();
  return perm_data && *perm_data == expected;
}

Expected<Tensor> ConstI32(const Builder& builder,
                          absl::Span<const int32_t> data) {
  RankedTensorType type(
      ElementType::Int32,
      Layout(BuildLayout({static_cast<int32_t>(data.size())})));
  LITERT_ASSIGN_OR_RETURN(
      auto tensor, builder.BuildTensor(RankedTensorSpecBuilder(type).Build()));
  LITERT_RETURN_IF_ERROR(builder.BuildWeights<int32_t>(data, tensor));
  return tensor;
}

Expected<Tensor> NewLike(const Builder& builder, const Tensor& like,
                         absl::Span<const int32_t> dimensions,
                         absl::string_view tag) {
  if (like.Get() == nullptr || dimensions.size() > LITERT_TENSOR_MAX_RANK) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument);
  }
  RankedTensorType type(
      like.ElementType(),
      Layout(BuildLayout(dimensions.data(),
                         dimensions.data() + dimensions.size())));
  RankedTensorSpec spec{std::move(type), std::nullopt, std::nullopt,
                        std::nullopt,    std::nullopt, std::nullopt};
  if (like.QTypeId() == kLiteRtQuantizationPerTensor) {
    spec.per_tensor_quantization = like.PerTensorQuantization();
  }
  if (!tag.empty()) {
    spec.tensor_name = MakeNamespacedTensorName(like, tag);
  }
  return builder.BuildTensor(spec);
}

Expected<Tensor> BuildSlice(const Builder& builder, const Tensor& input,
                            absl::Span<const int32_t> begin,
                            absl::Span<const int32_t> size, const Tensor& like,
                            absl::string_view tag) {
  LITERT_ASSIGN_OR_RETURN(auto output, NewLike(builder, like, size, tag));
  LITERT_ASSIGN_OR_RETURN(auto begin_tensor, ConstI32(builder, begin));
  LITERT_ASSIGN_OR_RETURN(auto size_tensor, ConstI32(builder, size));
  LITERT_RETURN_IF_ERROR(builder.BuildOp(
      kLiteRtOpCodeTflSlice, {input, begin_tensor, size_tensor}, {output}));
  return output;
}

Expected<void> BuildConcatInto(const Builder& builder,
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

Expected<Tensor> BuildConcat(const Builder& builder,
                             absl::Span<const Tensor> inputs, int32_t axis,
                             const Tensor& like,
                             absl::Span<const int32_t> dimensions,
                             absl::string_view tag) {
  LITERT_ASSIGN_OR_RETURN(auto output, NewLike(builder, like, dimensions, tag));
  LITERT_RETURN_IF_ERROR(BuildConcatInto(builder, inputs, axis, output));
  return output;
}

Expected<Tensor> BuildSoftmax(const Builder& builder, const Tensor& input,
                              float beta, const Tensor& like,
                              absl::Span<const int32_t> dimensions,
                              absl::string_view tag) {
  LITERT_ASSIGN_OR_RETURN(auto output, NewLike(builder, like, dimensions, tag));
  LITERT_ASSIGN_OR_RETURN(
      auto op, builder.BuildOp(kLiteRtOpCodeTflSoftmax, {input}, {output}));
  SoftmaxOptions options;
  options.beta = beta;
  LITERT_RETURN_IF_ERROR(builder.SetOpOptions(op, std::move(options)));
  return output;
}

Expected<Tensor> BuildAdd(const Builder& builder, const Tensor& lhs,
                          const Tensor& rhs, const Tensor& like,
                          absl::Span<const int32_t> dimensions,
                          absl::string_view tag) {
  LITERT_ASSIGN_OR_RETURN(auto output, NewLike(builder, like, dimensions, tag));
  LITERT_RETURN_IF_ERROR(
      BuildBinaryOpInto(builder, kLiteRtOpCodeTflAdd, lhs, rhs, output));
  return output;
}

Expected<void> BuildReshapeInto(const Builder& builder, const Tensor& input,
                                absl::Span<const int32_t> dimensions,
                                const Tensor& output) {
  LITERT_ASSIGN_OR_RETURN(auto shape, ConstI32(builder, dimensions));
  LITERT_ASSIGN_OR_RETURN(auto op, builder.BuildOp(kLiteRtOpCodeTflReshape,
                                                   {input, shape}, {output}));
  ReshapeOptions options;
  options.new_shape.assign(dimensions.begin(), dimensions.end());
  LITERT_RETURN_IF_ERROR(builder.SetOpOptions(op, std::move(options)));
  return {};
}

Expected<Tensor> BuildReshape(const Builder& builder, const Tensor& input,
                              const Tensor& like,
                              absl::Span<const int32_t> dimensions,
                              absl::string_view tag) {
  LITERT_ASSIGN_OR_RETURN(auto output, NewLike(builder, like, dimensions, tag));
  LITERT_RETURN_IF_ERROR(BuildReshapeInto(builder, input, dimensions, output));
  return output;
}

Expected<Tensor> BuildBatchMatmul(const Builder& builder, const Tensor& lhs,
                                  const Tensor& rhs, const Tensor& like,
                                  absl::Span<const int32_t> dimensions,
                                  bool asymmetric_quantize_input, bool adj_y,
                                  absl::string_view tag) {
  LITERT_ASSIGN_OR_RETURN(auto output, NewLike(builder, like, dimensions, tag));
  LITERT_ASSIGN_OR_RETURN(auto op, builder.BuildOp(kLiteRtOpCodeTflBatchMatmul,
                                                   {lhs, rhs}, {output}));
  BatchMatmulOptions options;
  options.adj_x = false;
  options.adj_y = adj_y;
  options.asymmetric_quantize_input = asymmetric_quantize_input;
  LITERT_RETURN_IF_ERROR(builder.SetOpOptions(op, std::move(options)));
  return output;
}

std::optional<Op> SingleUser(const Tensor& tensor, LiteRtOpCode code,
                             LiteRtParamIndex input_index) {
  if (tensor.Get() == nullptr) {
    return std::nullopt;
  }
  auto uses = tensor.Uses();
  if (uses.size() != 1 || uses[0].user.Get() == nullptr ||
      uses[0].user.Code() != code || uses[0].user_arg_ind != input_index) {
    return std::nullopt;
  }
  return uses[0].user;
}

std::optional<Tensor> SingleOutput(const Op& op) {
  if (op.Get() == nullptr) {
    return std::nullopt;
  }
  auto outputs = op.Outputs();
  if (outputs.size() != 1) {
    return std::nullopt;
  }
  return outputs[0];
}

std::optional<std::vector<int32_t>> GetStaticPositiveDims(
    const Tensor& tensor) {
  if (tensor.Get() == nullptr || tensor.TypeId() != kLiteRtRankedTensorType) {
    return std::nullopt;
  }
  auto ranked_type = tensor.RankedTensorType();
  if (!ranked_type) {
    return std::nullopt;
  }
  const auto dims = ranked_type->Layout().Dimensions();
  for (int32_t dim : dims) {
    if (dim <= 0) {
      return std::nullopt;
    }
  }
  return std::vector<int32_t>(dims.begin(), dims.end());
}

std::optional<size_t> CheckedNumElements(absl::Span<const int32_t> dims) {
  size_t count = 1;
  for (int32_t dim : dims) {
    if (dim <= 0) {
      return std::nullopt;
    }
    const size_t udim = static_cast<size_t>(dim);
    if (count > (std::numeric_limits<size_t>::max)() / udim) {
      return std::nullopt;
    }
    count *= udim;
  }
  return count;
}

std::optional<std::vector<int32_t>> BroadcastShapes(
    absl::Span<const int32_t> lhs, absl::Span<const int32_t> rhs) {
  for (int32_t d : lhs) {
    if (d <= 0) return std::nullopt;
  }
  for (int32_t d : rhs) {
    if (d <= 0) return std::nullopt;
  }
  const size_t rank = (std::max)(lhs.size(), rhs.size());
  std::vector<int32_t> out(rank, 1);
  for (size_t i = 0; i < rank; ++i) {
    const int32_t da = (i < lhs.size()) ? lhs[lhs.size() - 1 - i] : 1;
    const int32_t db = (i < rhs.size()) ? rhs[rhs.size() - 1 - i] : 1;
    if (da != db && da != 1 && db != 1) {
      return std::nullopt;
    }
    out[rank - 1 - i] = (da == 1) ? db : da;
  }
  return out;
}

std::optional<std::vector<int64_t>> ReadIntConstValues(const Tensor& tensor) {
  if (tensor.Get() == nullptr || !tensor.IsConstant() ||
      tensor.HasQuantization() || !GetStaticPositiveDims(tensor)) {
    return std::nullopt;
  }
  if (tensor.ElementType() == ElementType::Int32) {
    auto vals32 = tensor.WeightsData<int32_t>();
    if (!vals32 || vals32->empty()) {
      return std::nullopt;
    }
    return std::vector<int64_t>(vals32->begin(), vals32->end());
  }
  if (tensor.ElementType() == ElementType::Int64) {
    auto vals64 = tensor.WeightsData<int64_t>();
    if (!vals64 || vals64->empty()) {
      return std::nullopt;
    }
    return std::vector<int64_t>(vals64->begin(), vals64->end());
  }
  return std::nullopt;
}

Expected<Tensor> BuildTypedTensor(const Builder& builder, ElementType type,
                                  absl::Span<const int32_t> dims,
                                  absl::string_view name) {
  if (dims.size() > LITERT_TENSOR_MAX_RANK) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument);
  }
  RankedTensorType tensor_type(
      type, Layout(BuildLayout(dims.data(), dims.data() + dims.size())));
  RankedTensorSpec spec{
      std::move(tensor_type),
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      name.empty() ? std::nullopt : std::optional<std::string>(name)};
  return builder.BuildTensor(spec);
}

Expected<void> BuildCastInto(const Builder& builder, const Tensor& input,
                             const Tensor& output) {
  LITERT_ASSIGN_OR_RETURN(
      Op op, builder.BuildOp(kLiteRtOpCodeTflCast, {input}, {output}));
  (void)op;
  return {};
}

Expected<Tensor> BuildCast(const Builder& builder, const Tensor& input,
                           ElementType out_type,
                           absl::Span<const int32_t> out_dims,
                           absl::string_view name) {
  LITERT_ASSIGN_OR_RETURN(Tensor output,
                          BuildTypedTensor(builder, out_type, out_dims, name));
  LITERT_RETURN_IF_ERROR(BuildCastInto(builder, input, output));
  return output;
}

Expected<void> BuildBinaryOpInto(const Builder& builder, LiteRtOpCode code,
                                 const Tensor& lhs, const Tensor& rhs,
                                 const Tensor& output,
                                 uint32_t fused_activation) {
  if (fused_activation != 0 && code != kLiteRtOpCodeTflAdd &&
      code != kLiteRtOpCodeTflSub && code != kLiteRtOpCodeTflMul &&
      code != kLiteRtOpCodeTflDiv) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument);
  }
  LITERT_ASSIGN_OR_RETURN(Op op, builder.BuildOp(code, {lhs, rhs}, {output}));
  switch (code) {
    case kLiteRtOpCodeTflAdd: {
      AddOptions opts;
      opts.fused_activation_function = fused_activation;
      return builder.SetOpOptions(op, std::move(opts));
    }
    case kLiteRtOpCodeTflSub: {
      SubOptions opts;
      opts.fused_activation_function = fused_activation;
      return builder.SetOpOptions(op, std::move(opts));
    }
    case kLiteRtOpCodeTflMul: {
      MulOptions opts;
      opts.fused_activation_function = fused_activation;
      return builder.SetOpOptions(op, std::move(opts));
    }
    case kLiteRtOpCodeTflDiv: {
      DivOptions opts;
      opts.fused_activation_function = fused_activation;
      return builder.SetOpOptions(op, std::move(opts));
    }
    default:
      return {};
  }
}

Expected<Tensor> BuildBinaryOp(const Builder& builder, LiteRtOpCode code,
                               const Tensor& lhs, const Tensor& rhs,
                               ElementType out_type,
                               absl::Span<const int32_t> out_dims,
                               uint32_t fused_activation,
                               absl::string_view name) {
  LITERT_ASSIGN_OR_RETURN(Tensor output,
                          BuildTypedTensor(builder, out_type, out_dims, name));
  LITERT_RETURN_IF_ERROR(
      BuildBinaryOpInto(builder, code, lhs, rhs, output, fused_activation));
  return output;
}

}  // namespace litert::compiler
