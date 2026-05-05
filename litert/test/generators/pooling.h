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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_POOLING_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_POOLING_H_

#include <array>
#include <cstddef>
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
#include "litert/core/model/model_serialize.h"
#include "litert/core/model/ops/pooling.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/graph_helpers.h"
#include "litert/test/simple_buffer.h"
#include "tflite/schema/schema_generated.h"

namespace litert {
namespace testing {

using ::litert::internal::AttachInput;
using ::litert::internal::AttachOutput;
using ::litert::internal::SerializeModel;
using ::litert::internal::SetTflOpCodeInd;
using ::litert::internal::SetTflOpCodes;
using ::litert::internal::SetTflOptions;
using ::litert::internal::TflOpCode;
using ::litert::internal::TflOpCodePtr;
using ::litert::internal::TflOptions;

// Helper to construct OpDetails for SingleOpModel for Pooling.
struct PoolingOpDetails {
  tflite::Pool2DOptionsT options;

  PoolingOpDetails(tflite::Padding p, int sw, int sh, int fw, int fh,
                   tflite::ActivationFunctionType fa) {
    options.padding = p;
    options.stride_w = sw;
    options.stride_h = sh;
    options.filter_width = fw;
    options.filter_height = fh;
    options.fused_activation_function = fa;
  }

  TflOptions MakeTflOptions() const {
    TflOptions tfl_opts;
    tfl_opts.type = tflite::BuiltinOptions_Pool2DOptions;
    auto copy = std::make_unique<tflite::Pool2DOptionsT>();
    *copy = options;
    tfl_opts.value = copy.release();
    return tfl_opts;
  }
};

template <typename T, typename OpCode, typename Padding, typename Stride,
          typename FilterSize, typename Fa = FaC<>>
class Pooling : public TestGraph {
  static_assert(std::is_same_v<typename OpCode::value_type, LiteRtOpCode>);
  static constexpr LiteRtOpCode kOpCode = OpCode::value;

  static_assert(std::is_same_v<typename Padding::value_type, tflite::Padding>);
  static constexpr tflite::Padding kPadding = Padding::value;

  static_assert(std::is_same_v<typename Stride::value_type, size_t>);
  static constexpr int kStride = Stride::value;

  static_assert(std::is_same_v<typename FilterSize::value_type, size_t>);
  static constexpr int kFilterSize = FilterSize::value;

  static_assert(
      std::is_same_v<typename Fa::value_type, tflite::ActivationFunctionType>);
  static constexpr tflite::ActivationFunctionType kFa = Fa::value;

  static constexpr ElementType kElementType = ElementType::Float32;
  static constexpr TensorNames<1> kInputNames = {"input"};
  static constexpr TensorNames<1> kOutputNames = {"output"};

  struct Params {
    std::array<Layout::Dim, 4> input_shape;
    std::vector<Layout::Dim> output_shape;
  };

 public:
  using Traits = TestLogicTraits<TypeList<T>, TypeList<T>, Params>;
  using Ptr = std::unique_ptr<Pooling>;

  static constexpr absl::string_view Name() { return "Pooling"; }

  template <typename Rng>
  static Expected<Pooling::Ptr> Create(Rng& rng) {
    Params params;
    params.input_shape = {1, 10, 10, 1};

    // Leverage shape inference to compute output shape
    LiteRtOpT op;
    op.SetOpCode(kOpCode);
    auto options = std::make_unique<tflite::Pool2DOptionsT>();
    options->padding = kPadding;
    options->stride_w = kStride;
    options->stride_h = kStride;
    options->filter_width = kFilterSize;
    options->filter_height = kFilterSize;
    options->fused_activation_function = kFa;

    TflOptions tfl_opts;
    tfl_opts.type = tflite::BuiltinOptions_Pool2DOptions;
    tfl_opts.value = options.release();
    litert::internal::SetTflOptions(op, std::move(tfl_opts));

    std::vector<litert::internal::Dims> input_shapes = {
        {params.input_shape.begin(), params.input_shape.end()}};
    std::vector<litert::internal::Dims> output_shapes(1);

    LITERT_RETURN_IF_ERROR(litert::internal::InferPool2D(
        op, absl::MakeSpan(input_shapes), output_shapes));

    params.output_shape = {output_shapes[0].begin(), output_shapes[0].end()};

    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<Pooling>(std::move(params), std::move(model));
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

    int filter_h = kFilterSize;
    int filter_w = kFilterSize;
    int stride_h = kStride;
    int stride_w = kStride;

    int pad_t = ComputePaddingBefore(in_h, filter_h, stride_h, /*dilation=*/1,
                                     kPadding, out_h);
    int pad_l = ComputePaddingBefore(in_w, filter_w, stride_w, /*dilation=*/1,
                                     kPadding, out_w);

    constexpr bool kIsMax = (kOpCode == kLiteRtOpCodeTflMaxPool2d);

    litert::internal::ReferencePool2D<kIsMax>(
        input.data.data(), output.data.data(), batch, in_h, in_w, in_c, out_h,
        out_w, filter_h, filter_w, stride_h, stride_w, pad_t, pad_l, kFa);

    return {};
  }

  Pooling(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

 private:
  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    LiteRtModelT model;
    std::vector<TflOpCodePtr> tfl_codes;
    auto& sg = model.EmplaceSubgraph();
    auto& op = sg.EmplaceOp();

    op.SetOpCode(kOpCode);

    PoolingOpDetails op_details(kPadding, kStride, kStride, kFilterSize,
                                kFilterSize, kFa);
    litert::internal::SetTflOptions(op, op_details.MakeTflOptions());

    // Set opcode index. For simplicity, assume it's 0.
    litert::internal::SetTflOpCodeInd(op, 0);

    tflite::BuiltinOperator builtin_op;
    if constexpr (kOpCode == kLiteRtOpCodeTflMaxPool2d) {
      builtin_op = tflite::BuiltinOperator_MAX_POOL_2D;
    } else {
      builtin_op = tflite::BuiltinOperator_AVERAGE_POOL_2D;
    }
    auto op_code = std::make_unique<tflite::OperatorCodeT>();
    op_code->builtin_code = builtin_op;
    tfl_codes.push_back(std::move(op_code));

    auto& in_tensor = sg.EmplaceTensor();
    in_tensor.SetType(::MakeRankedTensorType(LiteRtElementType(kElementType),
                                             params.input_shape));
    in_tensor.SetName(std::string(kInputNames[0]));
    sg.Inputs().push_back(&in_tensor);
    AttachInput(&in_tensor, op);

    auto& out_tensor = sg.EmplaceTensor();
    out_tensor.SetType(::MakeRankedTensorType(LiteRtElementType(kElementType),
                                              params.output_shape));
    out_tensor.SetName(std::string(kOutputNames[0]));
    sg.Outputs().push_back(&out_tensor);
    AttachOutput(&out_tensor, op);

    std::vector<std::string> input_names = {std::string(kInputNames[0])};
    std::vector<LiteRtTensor> input_tensors = {&in_tensor};
    std::vector<std::string> output_names = {std::string(kOutputNames[0])};
    std::vector<LiteRtTensor> output_tensors = {&out_tensor};

    model.EmplaceSignature(&sg, std::move(input_names),
                           std::move(input_tensors), std::move(output_names),
                           std::move(output_tensors), "default");

    SetTflOpCodes(model, std::move(tfl_codes));

    LITERT_ASSIGN_OR_RETURN(auto serialized, SerializeModel(std::move(model)));
    return LoadModelFromBuffer(std::move(serialized));
  }

  Params params_;
};

}  // namespace testing
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_POOLING_H_
