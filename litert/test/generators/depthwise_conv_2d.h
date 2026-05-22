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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_DEPTHWISE_CONV_2D_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_DEPTHWISE_CONV_2D_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/internal/litert_rng.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/model/model.h"
#include "litert/core/model/ops/convolution.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/graph_helpers.h"
#include "litert/test/simple_buffer.h"
#include "tflite/schema/schema_generated.h"

namespace litert::testing {

// Specialize OpDetails for DepthwiseConv2D to handle options.
template <>
struct OpDetails<kLiteRtOpCodeTflDepthwiseConv2d> {
  using FbTypes = FbOpTypes<kLiteRtOpCodeTflDepthwiseConv2d>;
  using OptionsT = typename FbTypes::OptionsT;

  OpDetails(tflite::Padding padding, int32_t stride_w, int32_t stride_h,
            int32_t depth_multiplier,
            tflite::ActivationFunctionType fused_activation_function,
            int32_t dilation_w_factor, int32_t dilation_h_factor) {
    options.padding = padding;
    options.stride_w = stride_w;
    options.stride_h = stride_h;
    options.depth_multiplier = depth_multiplier;
    options.fused_activation_function = fused_activation_function;
    options.dilation_w_factor = dilation_w_factor;
    options.dilation_h_factor = dilation_h_factor;
  }

  TflOptions MakeTflOptions() const {
    TflOptions res;
    res.type = FbTypes::kBuiltinOptions;
    res.Set(tflite::DepthwiseConv2DOptionsT(options));
    return res;
  }

  TflOpCodePtr MakeTflCode() const {
    auto code = std::make_unique<TflOpCode>();
    code->builtin_code = FbTypes::kBuiltinOperator;
    code->version = 1;
    return code;
  }

 private:
  OptionsT options;
};

template <typename Rank, typename T, typename OpCode,
          typename Padding =
              std::integral_constant<tflite::Padding, tflite::Padding_VALID>,
          typename StrideH = SizeC<1>, typename StrideW = SizeC<1>,
          typename DilationH = SizeC<1>, typename DilationW = SizeC<1>,
          typename Fa = FaC<tflite::ActivationFunctionType_NONE>,
          typename DepthMultiplier = SizeC<1>>
class DepthwiseConv2d : public TestGraph {
 private:
  static_assert(std::is_same_v<typename OpCode::value_type, LiteRtOpCode>);
  static constexpr LiteRtOpCode kOpCode = OpCode::value;
  static_assert(kOpCode == kLiteRtOpCodeTflDepthwiseConv2d);

  static_assert(std::is_same_v<typename Rank::value_type, size_t>);
  static constexpr size_t kRank = Rank::value;
  static_assert(kRank == 4, "DepthwiseConv2D only supports rank 4");

  static constexpr TensorNames<1> kInputNames = {"input"};
  static constexpr TensorNames<1> kOutputNames = {"output"};

  static constexpr ElementType kElementType = GetElementType<T>();

  static constexpr tflite::Padding kPadding = Padding::value;
  static constexpr int32_t kStrideH = StrideH::value;
  static constexpr int32_t kStrideW = StrideW::value;
  static constexpr int32_t kDilationH = DilationH::value;
  static constexpr int32_t kDilationW = DilationW::value;
  static constexpr tflite::ActivationFunctionType kFa = Fa::value;
  static constexpr int32_t kDepthMultiplier = DepthMultiplier::value;

  struct Params {
    std::array<Layout::Dim, 4> input_shape;
    std::array<Layout::Dim, 4> filter_shape;
    std::array<Layout::Dim, 1> bias_shape;
    std::array<Layout::Dim, 4> output_shape;
    std::vector<T> filter_data;
    std::vector<T> bias_data;
  };

 public:
  using Traits = TestLogicTraits<TypeList<T>, TypeList<T>, Params>;
  using Ptr = std::unique_ptr<DepthwiseConv2d>;

  static constexpr absl::string_view Name() { return "DepthwiseConv2d"; }

  template <typename Rng>
  static Expected<DepthwiseConv2d::Ptr> Create(Rng& rng) {
    Params params;
    params.input_shape = {1, 10, 10, 1};
    params.filter_shape = {1, 3, 3, kDepthMultiplier};  // [1, H, W, C * M]
    params.bias_shape = {kDepthMultiplier};

    // Fill filter and bias with fixed representative values
    params.filter_data.assign(3 * 3 * kDepthMultiplier, 1.0f);
    params.bias_data.assign(kDepthMultiplier, 0.0f);

    // Leverage shape inference to compute output shape
    LiteRtOpT op;
    op.SetOpCode(kLiteRtOpCodeTflDepthwiseConv2d);
    auto options = std::make_unique<tflite::DepthwiseConv2DOptionsT>();
    options->padding = kPadding;
    options->stride_w = kStrideW;
    options->stride_h = kStrideH;
    options->depth_multiplier = kDepthMultiplier;
    options->fused_activation_function = kFa;
    options->dilation_w_factor = kDilationW;
    options->dilation_h_factor = kDilationH;

    TflOptions tfl_opts;
    tfl_opts.type = tflite::BuiltinOptions_DepthwiseConv2DOptions;
    tfl_opts.value = options.release();
    litert::internal::SetTflOptions(op, std::move(tfl_opts));

    std::vector<litert::internal::Dims> input_shapes = {
        {params.input_shape.begin(), params.input_shape.end()},
        {params.filter_shape.begin(), params.filter_shape.end()},
        {params.bias_shape.begin(), params.bias_shape.end()}};
    std::vector<litert::internal::Dims> output_shapes(1);

    LITERT_RETURN_IF_ERROR(litert::internal::InferDepthwiseConv2D(
        op, absl::MakeSpan(input_shapes), output_shapes));

    std::copy(output_shapes[0].begin(), output_shapes[0].end(),
              params.output_shape.begin());

    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<DepthwiseConv2d>(std::move(params),
                                             std::move(model));
  }

  bool HasReference() const override { return true; }

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    VarBuffers inputs;
    LITERT_ASSIGN_OR_RETURN(auto input,
                            SimpleBuffer::Create<T>(params_.input_shape));
    LITERT_RETURN_IF_ERROR(
        (input.template WriteRandom<T>(data_builder, device)));
    inputs.push_back(std::move(input));
    return inputs;
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    LITERT_ASSIGN_OR_RETURN(auto ref_inputs,
                            Traits::MakeReferenceInputs(inputs));
    LITERT_ASSIGN_OR_RETURN(auto ref_outputs,
                            Traits::MakeReferenceOutputs(outputs));

    auto [input] = ref_inputs;
    auto [output] = ref_outputs;

    int batch = params_.input_shape[0];
    int in_h = params_.input_shape[1];
    int in_w = params_.input_shape[2];
    int in_c = params_.input_shape[3];

    int out_h = params_.output_shape[1];
    int out_w = params_.output_shape[2];
    int out_c = params_.output_shape[3];

    int filter_h = params_.filter_shape[1];
    int filter_w = params_.filter_shape[2];

    int pad_t = ComputePaddingBefore(in_h, filter_h, kStrideH, kDilationH,
                                     kPadding, out_h);
    int pad_l = ComputePaddingBefore(in_w, filter_w, kStrideW, kDilationW,
                                     kPadding, out_w);

    // Simple reference for C=1, M=1 (same as Conv2D with 1 channel)
    litert::internal::ReferenceDepthwiseConv2D(
        input.data.data(), params_.filter_data.data(), params_.bias_data.data(),
        output.data.data(), batch, in_h, in_w, in_c, out_h, out_w, out_c,
        filter_h, filter_w, kStrideH, kStrideW, kDilationH, kDilationW, pad_t,
        pad_l, kDepthMultiplier, kFa);
    return {};
  }

  DepthwiseConv2d(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

 private:
  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    std::vector<TensorDetails> op_inputs(3);
    std::vector<TensorDetails> op_outputs(1);

    op_inputs[0] = TensorDetails{
        std::vector<int32_t>(params.input_shape.begin(),
                             params.input_shape.end()),
        LiteRtElementType(kElementType), std::string(kInputNames[0])};

    auto filter_buf = MakeOwningBufferRef(params.filter_data);

    op_inputs[1] = TensorDetails{
        std::vector<int32_t>(params.filter_shape.begin(),
                             params.filter_shape.end()),
        LiteRtElementType(kElementType), "filter", std::move(filter_buf)};

    auto bias_buf = MakeOwningBufferRef(params.bias_data);

    op_inputs[2] = TensorDetails{std::vector<int32_t>(params.bias_shape.begin(),
                                                      params.bias_shape.end()),
                                 LiteRtElementType(kElementType), "bias",
                                 std::move(bias_buf)};

    op_outputs[0] = TensorDetails{
        std::vector<int32_t>(params.output_shape.begin(),
                             params.output_shape.end()),
        LiteRtElementType(kElementType), std::string(kOutputNames[0])};

    return SingleOpModel<kLiteRtOpCodeTflDepthwiseConv2d>(
        op_inputs, op_outputs, kPadding, kStrideW, kStrideH, kDepthMultiplier,
        kFa, kDilationW, kDilationH);
  }

  Params params_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_DEPTHWISE_CONV_2D_H_
