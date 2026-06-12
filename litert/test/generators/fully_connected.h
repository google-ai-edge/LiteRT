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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_FULLY_CONNECTED_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_FULLY_CONNECTED_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/internal/litert_rng.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/model/model.h"
#include "litert/core/model/ops/matmul.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/graph_helpers.h"
#include "litert/test/simple_buffer.h"
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"
#include "tflite/schema/schema_generated.h"

namespace litert::testing {

template <typename Rank, typename T_in, typename T_out, typename OpCode,
          typename KeepNumDims = std::false_type,
          typename Fa = FaC<tflite::ActivationFunctionType_NONE>,
          typename HasBias = std::true_type>
class FullyConnected : public TestGraph {
  static_assert(std::is_same_v<typename Rank::value_type, size_t>);
  static constexpr size_t kRank = Rank::value;

  static constexpr bool kKeepNumDims = KeepNumDims::value;
  static constexpr tflite::ActivationFunctionType kFa = Fa::value;
  static constexpr bool kHasBias = HasBias::value;

  static constexpr ElementType kInElementType = GetElementType<T_in>();
  static constexpr ElementType kOutElementType = GetElementType<T_out>();
  static constexpr TensorNames<3> kInputNames = {"input", "weights", "bias"};
  static constexpr TensorNames<1> kOutputNames = {"output"};

  struct Params {
    std::array<Layout::Dim, kRank> input_shape;
    std::array<Layout::Dim, 2> weights_shape;
    std::array<Layout::Dim, 1> bias_shape;
    std::vector<Layout::Dim> output_shape;
    std::vector<T_in> weights_data;
    std::vector<T_out> bias_data;
  };

 public:
  using InputTypesT = TypeList<T_in>;
  using Traits = TestLogicTraits<InputTypesT, TypeList<T_out>, Params>;
  using Ptr = std::unique_ptr<FullyConnected>;

  static constexpr absl::string_view Name() { return "FullyConnected"; }

  template <typename Rng>
  static Expected<FullyConnected::Ptr> Create(Rng& rng) {
    Params params;
    std::uniform_int_distribution<int> dim_dist(2, 6);

    int input_dim = dim_dist(rng);
    int output_dim = dim_dist(rng);

    for (size_t i = 0; i < kRank - 1; ++i) {
      params.input_shape[i] = dim_dist(rng);
    }
    params.input_shape[kRank - 1] = input_dim;

    params.weights_shape = {output_dim, input_dim};
    params.bias_shape = {output_dim};

    params.weights_data.assign(output_dim * input_dim, static_cast<T_in>(1.0f));
    params.bias_data.assign(output_dim, static_cast<T_out>(0.0f));

    LiteRtOpT op;
    op.SetOpCode(kLiteRtOpCodeTflFullyConnected);
    auto options = std::make_unique<tflite::FullyConnectedOptionsT>();
    options->keep_num_dims = kKeepNumDims;
    options->weights_format =
        tflite::FullyConnectedOptionsWeightsFormat_DEFAULT;
    options->fused_activation_function = kFa;

    TflOptions tfl_opts;
    tfl_opts.type = tflite::BuiltinOptions_FullyConnectedOptions;
    tfl_opts.value = options.release();
    litert::internal::SetTflOptions(op, std::move(tfl_opts));

    std::vector<litert::internal::Dims> input_shapes = {
        {params.input_shape.begin(), params.input_shape.end()},
        {params.weights_shape.begin(), params.weights_shape.end()}};
    if constexpr (kHasBias) {
      input_shapes.push_back(
          {params.bias_shape.begin(), params.bias_shape.end()});
    }
    std::vector<litert::internal::Dims> output_shapes(1);

    LITERT_RETURN_IF_ERROR(litert::internal::InferFullyConnected(
        op, absl::MakeSpan(input_shapes), output_shapes));

    params.output_shape.resize(output_shapes[0].size());
    std::copy(output_shapes[0].begin(), output_shapes[0].end(),
              params.output_shape.begin());

    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<FullyConnected>(std::move(params),
                                            std::move(model));
  }

  bool HasReference() const override { return true; }

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    VarBuffers inputs;
    inputs.reserve(1);
    RandomTensorDataBuilder modified_data_builder = data_builder;
    modified_data_builder.SetFloatRange(-2.0f, 2.0f);

    LITERT_ASSIGN_OR_RETURN(auto input,
                            SimpleBuffer::Create<T_in>(params_.input_shape));
    LITERT_RETURN_IF_ERROR(
        (input.template WriteRandom<T_in>(modified_data_builder, device)));
    inputs.push_back(std::move(input));

    return inputs;
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    LITERT_ASSIGN_OR_RETURN(auto ref_inputs,
                            Traits::MakeReferenceInputs(inputs));
    LITERT_ASSIGN_OR_RETURN(auto ref_outputs,
                            Traits::MakeReferenceOutputs(outputs));

    int64_t batch_size = 1;
    for (size_t i = 0; i < kRank - 1; ++i) {
      batch_size *= params_.input_shape[i];
    }
    int64_t input_dim = params_.input_shape[kRank - 1];
    int64_t output_dim = params_.weights_shape[0];

    auto [input] = ref_inputs;
    auto [output] = ref_outputs;
    std::vector<float> out_f32(output.data.size());
    std::vector<float> in_f32 = UnpackToFloat(input.data);
    std::vector<float> wt_f32 =
        UnpackToFloat(absl::MakeConstSpan(params_.weights_data));

    if constexpr (kHasBias) {
      std::vector<float> bs_f32 =
          UnpackToFloat(absl::MakeConstSpan(params_.bias_data));
      litert::internal::ReferenceFullyConnected(
          in_f32.data(), wt_f32.data(), bs_f32.data(), out_f32.data(),
          batch_size, input_dim, output_dim, kFa);
    } else {
      litert::internal::ReferenceFullyConnected(
          in_f32.data(), wt_f32.data(), nullptr, out_f32.data(), batch_size,
          input_dim, output_dim, kFa);
    }

    PackFromFloat(absl::MakeConstSpan(out_f32), output.data);
    return {};
  }

  FullyConnected(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

 private:
  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;
    std::vector<int32_t> dims_in(params.input_shape.begin(),
                                 params.input_shape.end());
    std::vector<int32_t> dims_wt(params.weights_shape.begin(),
                                 params.weights_shape.end());

    TensorTf input =
        litert::tensor::Create(std::string(kInputNames[0]),
                               litert::tensor::ApiType<T_in>::value, dims_in);

    auto weights_buf = std::make_shared<litert::tensor::SpanCpuBuffer>(
        reinterpret_cast<const std::byte*>(params.weights_data.data()),
        params.weights_data.size() * sizeof(T_in));
    TensorTf weights = litert::tensor::Create(
        std::string(kInputNames[1]), litert::tensor::ApiType<T_in>::value,
        dims_wt, std::move(weights_buf));

    std::optional<TensorTf> bias_opt = std::nullopt;
    if constexpr (kHasBias) {
      std::vector<int32_t> dims_bs(params.bias_shape.begin(),
                                   params.bias_shape.end());
      auto bias_buf = std::make_shared<litert::tensor::SpanCpuBuffer>(
          reinterpret_cast<const std::byte*>(params.bias_data.data()),
          params.bias_data.size() * sizeof(T_out));
      bias_opt = litert::tensor::Create(std::string(kInputNames[2]),
                                        litert::tensor::ApiType<T_out>::value,
                                        dims_bs, std::move(bias_buf));
    }

    litert::tensor::FusedActivation act = litert::tensor::kActNone;
    if constexpr (kFa == tflite::ActivationFunctionType_RELU) {
      act = litert::tensor::kActRelu;
    } else if constexpr (kFa == tflite::ActivationFunctionType_RELU_N1_TO_1) {
      act = litert::tensor::kActReluN1To1;
    } else if constexpr (kFa == tflite::ActivationFunctionType_RELU6) {
      act = litert::tensor::kActRelu6;
    }

    TensorTf output = litert::tensor::FullyConnected(input, weights, bias_opt,
                                                     act, kKeepNumDims);
    output.SetType(litert::tensor::ApiType<T_out>::value);
    output.SetName(std::string(kOutputNames[0]));

    return SaveTensorGraph({output});
  }

  Params params_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_FULLY_CONNECTED_H_
