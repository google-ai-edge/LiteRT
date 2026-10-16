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

#include "litert/vendors/qualcomm/compiler/transformations/hadamard_transform_transformation.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/numeric/bits.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "litert/c/internal/litert_abi_header.h"
#include "litert/c/internal/litert_compiler_context.h"
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
#include "litert/vendors/qualcomm/compiler/transformations/orphan_cleanup_transformation.h"

namespace litert::qnn {
namespace {

using litert::BuildLayout;
using litert::ElementType;
using litert::Expected;
using litert::Layout;
using litert::RankedTensorType;
using litert::Unexpected;
using litert::compiler::Builder;
using litert::compiler::CompositeOptions;
using litert::compiler::ConcatenationOptions;
using litert::compiler::FullyConnectedOptions;
using litert::compiler::GetOptionsAs;
using litert::compiler::Op;
using litert::compiler::RankedTensorSpecBuilder;
using litert::compiler::SplitOptions;
using litert::compiler::Tensor;

struct HadamardTransformOpts : public CompositeOptions {
  float scale = 1.0f;

  Expected<void> SetOpOptions(const LiteRtCompilerContext* ctx,
                              LiteRtBuilder builder) override {
    if (!LITERT_ABI_HAS_API(ctx, /*req_major=*/1,
                            build_shlo_composite_op_option)) {
      return Unexpected(kLiteRtStatusErrorWrongVersion);
    }
    flexbuffers::Builder fbb;
    fbb.Map([&]() { fbb.Float("scale", scale); });
    fbb.Finish();
    const auto& buffer = fbb.GetBuffer();
    std::string name_str(name.empty()
                             ? std::string(kHadamardTransformCompositeName)
                             : std::string(name));
    int subgraph_idx = -1;
    int32_t ver = 1;
    LITERT_RETURN_IF_ERROR(ctx->build_shlo_composite_op_option(
        builder, op, name_str.c_str(), &subgraph_idx, &ver, buffer.data(),
        static_cast<int32_t>(buffer.size())));
    return Expected<void>();
  }
};

std::vector<int32_t> Dims(const Tensor& t) {
  auto type = t.RankedTensorType();
  if (!type) return {};
  auto dims = type->Layout().Dimensions();
  return std::vector<int32_t>(dims.begin(), dims.end());
}

bool IsPerTensorSymInt16(const Tensor& t) {
  return t.ElementType() == ElementType::Int16 &&
         t.QTypeId() == kLiteRtQuantizationPerTensor &&
         t.PerTensorQuantization().zero_point == 0;
}

std::optional<float> GetSylvesterHadamardScale(const Tensor& weight) {
  if (!weight.IsConstant() ||
      weight.QTypeId() != kLiteRtQuantizationPerTensor) {
    return std::nullopt;
  }
  const auto qparams = weight.PerTensorQuantization();
  if (qparams.zero_point != 0) {
    return std::nullopt;
  }
  const auto dims = Dims(weight);
  if (dims.size() != 2 || dims[0] != dims[1]) {
    return std::nullopt;
  }
  const uint32_t n = static_cast<uint32_t>(dims[0]);
  if (n == 0 || (n & (n - 1)) != 0) {
    return std::nullopt;
  }

  const ElementType et = weight.ElementType();
  const size_t total_elems = static_cast<size_t>(n) * static_cast<size_t>(n);
  absl::Span<const uint8_t> raw_bytes = weight.Weights().Bytes();
  if (raw_bytes.empty()) {
    return std::nullopt;
  }

  auto get_elem = [&](size_t idx) -> int8_t {
    if (et == ElementType::Int8) {
      return static_cast<int8_t>(raw_bytes[idx]);
    }
    const uint8_t byte = raw_bytes[idx / 2];
    const uint8_t nibble =
        (idx % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
    return (nibble & 0x08) ? static_cast<int8_t>(nibble | 0xF0)
                           : static_cast<int8_t>(nibble);
  };

  if (et == ElementType::Int8) {
    if (raw_bytes.size() != total_elems) return std::nullopt;
  } else if (et == ElementType::Int4) {
    if (raw_bytes.size() * 2 != total_elems) return std::nullopt;
  } else {
    return std::nullopt;
  }

  const int8_t hadamard_value = get_elem(0);
  if (hadamard_value == 0) {
    return std::nullopt;
  }

  for (uint32_t i = 0; i < n; ++i) {
    for (uint32_t j = 0; j < n; ++j) {
      const int bits = absl::popcount(i & j);
      const int8_t expected =
          ((bits & 1) == 0) ? +hadamard_value : -hadamard_value;
      if (get_elem(static_cast<size_t>(i) * n + j) != expected) {
        return std::nullopt;
      }
    }
  }

  return qparams.scale * static_cast<float>(hadamard_value) *
         std::sqrt(static_cast<float>(n));
}

Expected<Tensor> NewLike(Builder& builder, const Tensor& like,
                         const std::vector<int32_t>& dims) {
  RankedTensorType type(
      like.ElementType(),
      Layout(BuildLayout(dims.data(), dims.data() + dims.size())));
  auto spec = RankedTensorSpecBuilder(type);
  if (like.QTypeId() == kLiteRtQuantizationPerTensor) {
    spec =
        std::move(spec).WithPerTensorQuantization(like.PerTensorQuantization());
  }
  return builder.BuildTensor(std::move(spec).Build());
}

Expected<Tensor> ConstI32(Builder& builder, const std::vector<int32_t>& data) {
  const std::vector<int32_t> dims = {static_cast<int32_t>(data.size())};
  RankedTensorType type(ElementType::Int32,
                        Layout(BuildLayout(dims.data(), dims.data() + 1)));
  LITERT_ASSIGN_OR_RETURN(
      auto t, builder.BuildTensor(RankedTensorSpecBuilder(type).Build()));
  LITERT_RETURN_IF_ERROR(
      builder.BuildWeights<int32_t>(absl::MakeConstSpan(data), t));
  return t;
}

Expected<void> BuildParallelHadamard(Builder& builder, const Op& pre_reshape,
                                     const Op& hadamard_op,
                                     const Op& post_reshape, float scale,
                                     int32_t num_h) {
  const Tensor input = pre_reshape.Inputs()[0];
  const Tensor output = post_reshape.Outputs()[0];
  const auto in_dims = Dims(input);
  const int32_t axis = static_cast<int32_t>(in_dims.size()) - 1;
  std::vector<int32_t> slice_dims = in_dims;
  slice_dims[axis] /= num_h;

  LITERT_ASSIGN_OR_RETURN(auto axis_t, ConstI32(builder, {axis}));
  std::vector<Tensor> split_outs;
  split_outs.reserve(num_h);
  for (int32_t i = 0; i < num_h; ++i) {
    LITERT_ASSIGN_OR_RETURN(auto t, NewLike(builder, input, slice_dims));
    split_outs.push_back(t);
  }
  LITERT_ASSIGN_OR_RETURN(
      auto split_op,
      builder.BuildOp(kLiteRtOpCodeTflSplit, {axis_t, input}, split_outs));
  SplitOptions split_opts;
  split_opts.num_splits = num_h;
  LITERT_RETURN_IF_ERROR(builder.SetOpOptions(split_op, std::move(split_opts)));

  std::vector<Tensor> hadamard_outs;
  hadamard_outs.reserve(num_h);
  for (int32_t i = 0; i < num_h; ++i) {
    LITERT_ASSIGN_OR_RETURN(auto h_out, NewLike(builder, output, slice_dims));
    LITERT_RETURN_IF_ERROR(
        BuildHadamardTransformOp(builder, split_outs[i], h_out, scale));
    hadamard_outs.push_back(h_out);
  }

  LITERT_ASSIGN_OR_RETURN(
      auto concat_op,
      builder.BuildOp(kLiteRtOpCodeTflConcatenation, hadamard_outs, {output}));
  ConcatenationOptions concat_opts;
  concat_opts.axis = axis;
  concat_opts.fused_activation_function = 0;
  LITERT_RETURN_IF_ERROR(
      builder.SetOpOptions(concat_op, std::move(concat_opts)));

  LITERT_RETURN_IF_ERROR(builder.EraseOp(post_reshape));
  RegisterOrphanGroup({pre_reshape.Get(), hadamard_op.Get()});
  return {};
}

}  // namespace

bool IsHadamardTransformOp(const Op& op) {
  if (op.Code() == kLiteRtOpCodeShloComposite) {
    auto info = GetOptionsAs<CompositeOptions>(op.ctx(), op.Get());
    return info.HasValue() && info->name == kHadamardTransformCompositeName;
  }
  if (op.Code() == kLiteRtOpCodeTflCustom) {
    auto code = op.CustomCode();
    return code.HasValue() && *code == kHadamardTransformCompositeName;
  }
  return false;
}

std::optional<float> GetHadamardTransformScale(const Op& op) {
  if (op.Code() == kLiteRtOpCodeShloComposite) {
    auto info = GetOptionsAs<CompositeOptions>(op.ctx(), op.Get());
    if (!info || info->name != kHadamardTransformCompositeName ||
        !info->attributes_map.has_value()) {
      return std::nullopt;
    }
    flexbuffers::Reference raw_scale = info->attributes_map.value()["scale"];
    if (raw_scale.IsNull()) return std::nullopt;
    return raw_scale.AsFloat();
  }
  if (op.Code() == kLiteRtOpCodeTflCustom) {
    auto code = op.CustomCode();
    if (!code || *code != kHadamardTransformCompositeName) return std::nullopt;
    auto custom_opts = op.CustomOptions();
    if (!custom_opts || custom_opts->empty() ||
        !flexbuffers::VerifyBuffer(custom_opts->data(), custom_opts->size())) {
      return std::nullopt;
    }
    auto root = flexbuffers::GetRoot(custom_opts->data(), custom_opts->size());
    if (!root.IsMap()) return std::nullopt;
    flexbuffers::Reference raw_scale = root.AsMap()["scale"];
    if (raw_scale.IsNull()) return std::nullopt;
    return raw_scale.AsFloat();
  }
  return std::nullopt;
}

Expected<Op> BuildHadamardTransformOp(Builder& builder, const Tensor& input,
                                      const Tensor& output, float scale) {
  LITERT_ASSIGN_OR_RETURN(
      auto op, builder.BuildOp(kLiteRtOpCodeShloComposite, {input}, {output}));
  HadamardTransformOpts opts;
  opts.name = kHadamardTransformCompositeName;
  opts.scale = scale;
  LITERT_RETURN_IF_ERROR(builder.SetOpOptions(op, std::move(opts)));
  return op;
}

}  // namespace litert::qnn

extern "C" {

LiteRtStatus ConvertFcToHadamardTransform(const LiteRtCompilerContext* context,
                                          LiteRtBuilder builder_ptr,
                                          LiteRtOp op_ptr) {
  using litert::compiler::Builder;
  using litert::compiler::FullyConnectedOptions;
  using litert::compiler::GetOptionsAs;
  using litert::compiler::Op;
  using litert::compiler::Tensor;

  Op op(context, op_ptr);
  if (op.Code() != kLiteRtOpCodeTflFullyConnected) {
    return kLiteRtStatusPatternNoMatch;
  }
  const auto inputs = op.Inputs();
  const auto outputs = op.Outputs();
  const bool has_valid_inputs =
      inputs.size() == 2 ||
      (inputs.size() == 3 &&
       inputs[2].ElementType() == litert::ElementType::None);
  if (!has_valid_inputs || outputs.size() != 1) {
    return kLiteRtStatusPatternNoMatch;
  }
  auto fc_opts = GetOptionsAs<FullyConnectedOptions>(context, op_ptr);
  if (!fc_opts || fc_opts->fused_activation_function != 0 ||
      fc_opts->weights_format != 0) {
    return kLiteRtStatusPatternNoMatch;
  }

  const Tensor& input = inputs[0];
  const Tensor& weight = inputs[1];
  const Tensor& output = outputs[0];
  if (!litert::qnn::IsPerTensorSymInt16(input) ||
      !litert::qnn::IsPerTensorSymInt16(output)) {
    return kLiteRtStatusPatternNoMatch;
  }
  const auto in_dims = litert::qnn::Dims(input);
  const auto out_dims = litert::qnn::Dims(output);
  if (in_dims.empty() || in_dims != out_dims) {
    return kLiteRtStatusPatternNoMatch;
  }

  auto scale = litert::qnn::GetSylvesterHadamardScale(weight);
  if (!scale.has_value()) {
    return kLiteRtStatusPatternNoMatch;
  }

  Builder builder(context, builder_ptr);
  auto built =
      litert::qnn::BuildHadamardTransformOp(builder, input, output, *scale);
  if (!built) {
    return built.Error().Status();
  }
  auto erase_status = builder.EraseOp(op);
  if (!erase_status) {
    return erase_status.Error().Status();
  }
  return kLiteRtStatusOk;
}

LiteRtStatus ParallelizeHadamardTransform(const LiteRtCompilerContext* context,
                                          LiteRtBuilder builder_ptr,
                                          LiteRtOp op_ptr) {
  using litert::compiler::Builder;
  using litert::compiler::Op;
  using litert::compiler::Tensor;

  Op post_reshape(context, op_ptr);
  if (post_reshape.Code() != kLiteRtOpCodeTflReshape ||
      post_reshape.Inputs().empty() || post_reshape.Outputs().size() != 1) {
    return kLiteRtStatusPatternNoMatch;
  }

  const Tensor h_output = post_reshape.Inputs()[0];
  if (h_output.Uses().size() != 1) {
    return kLiteRtStatusPatternNoMatch;
  }
  auto hadamard_op_or = h_output.GetDefiningOp();
  if (!hadamard_op_or || !litert::qnn::IsHadamardTransformOp(*hadamard_op_or)) {
    return kLiteRtStatusPatternNoMatch;
  }
  const Op hadamard_op = *hadamard_op_or;
  if (hadamard_op.Inputs().size() != 1 || hadamard_op.Outputs().size() != 1) {
    return kLiteRtStatusPatternNoMatch;
  }
  auto scale = litert::qnn::GetHadamardTransformScale(hadamard_op);
  if (!scale.has_value()) {
    return kLiteRtStatusPatternNoMatch;
  }

  const Tensor h_input = hadamard_op.Inputs()[0];
  if (h_input.Uses().size() != 1) {
    return kLiteRtStatusPatternNoMatch;
  }
  auto pre_reshape_or = h_input.GetDefiningOp();
  if (!pre_reshape_or || pre_reshape_or->Code() != kLiteRtOpCodeTflReshape ||
      pre_reshape_or->Inputs().empty()) {
    return kLiteRtStatusPatternNoMatch;
  }
  const Op pre_reshape = *pre_reshape_or;

  const Tensor input = pre_reshape.Inputs()[0];
  const Tensor output = post_reshape.Outputs()[0];
  const auto in_dims = litert::qnn::Dims(input);
  const auto out_dims = litert::qnn::Dims(output);
  const auto h_in_dims = litert::qnn::Dims(h_input);
  if (in_dims.empty() || h_in_dims.empty() || in_dims != out_dims) {
    return kLiteRtStatusPatternNoMatch;
  }

  const int32_t input_size = in_dims.back();
  const int32_t hadamard_size = h_in_dims.back();
  if (hadamard_size <= 0 || input_size <= hadamard_size ||
      input_size % hadamard_size != 0) {
    return kLiteRtStatusPatternNoMatch;
  }
  const int32_t num_h = input_size / hadamard_size;

  Builder builder(context, builder_ptr);
  auto status = litert::qnn::BuildParallelHadamard(
      builder, pre_reshape, hadamard_op, post_reshape, *scale, num_h);
  if (!status) {
    return status.Error().Status();
  }
  return kLiteRtStatusOk;
}

}  // extern "C"
