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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_CONCATENATION_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_CONCATENATION_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
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
#include "litert/core/model/ops/concatenation.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/graph_helpers.h"
#include "litert/test/simple_buffer.h"
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"
#include "tflite/schema/schema_generated.h"

namespace litert::testing {

template <typename Rank, typename T, typename OpCode, typename Axis = SizeC<0>,
          typename Fa = FaC<tflite::ActivationFunctionType_NONE>>
class Concatenation : public TestGraph {
  static_assert(std::is_same_v<typename Rank::value_type, size_t>);
  static constexpr size_t kRank = Rank::value;

  static constexpr int32_t kAxis = Axis::value;
  static constexpr tflite::ActivationFunctionType kFa = Fa::value;

  static constexpr ElementType kElementType = GetElementType<T>();
  static constexpr TensorNames<2> kInputNames = {"input1", "input2"};
  static constexpr TensorNames<1> kOutputNames = {"output"};

  struct Params {
    std::array<Layout::Dim, kRank> input1_shape;
    std::array<Layout::Dim, kRank> input2_shape;
    std::array<Layout::Dim, kRank> output_shape;
  };

 public:
  using Traits = TestLogicTraits<TypeList<T, T>, TypeList<T>, Params>;
  using Ptr = std::unique_ptr<Concatenation>;

  static constexpr absl::string_view Name() { return "Concatenation"; }

  template <typename Rng>
  static Expected<Concatenation::Ptr> Create(Rng& rng) {
    Params params;
    std::uniform_int_distribution<int> dim_dist(2, 6);

    for (size_t i = 0; i < kRank; ++i) {
      int d = dim_dist(rng);
      params.input1_shape[i] = d;
      params.input2_shape[i] = d;
    }

    // Vary the concatenation dimension
    params.input1_shape[kAxis] = dim_dist(rng);
    params.input2_shape[kAxis] = dim_dist(rng);

    LiteRtOpT op;
    op.SetOpCode(kLiteRtOpCodeTflConcatenation);
    auto options = std::make_unique<tflite::ConcatenationOptionsT>();
    options->axis = kAxis;
    options->fused_activation_function = kFa;

    TflOptions tfl_opts;
    tfl_opts.type = tflite::BuiltinOptions_ConcatenationOptions;
    tfl_opts.value = options.release();
    litert::internal::SetTflOptions(op, std::move(tfl_opts));

    std::vector<litert::internal::Dims> input_shapes = {
        {params.input1_shape.begin(), params.input1_shape.end()},
        {params.input2_shape.begin(), params.input2_shape.end()}};
    std::vector<litert::internal::Dims> output_shapes(1);

    LITERT_RETURN_IF_ERROR(litert::internal::InferConcatenation(
        op, absl::MakeSpan(input_shapes), output_shapes));

    std::copy(output_shapes[0].begin(), output_shapes[0].end(),
              params.output_shape.begin());

    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<Concatenation>(std::move(params), std::move(model));
  }

  bool HasReference() const override { return true; }

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    VarBuffers inputs;
    inputs.reserve(2);
    RandomTensorDataBuilder modified_data_builder = data_builder;
    modified_data_builder.SetFloatRange(-5.0f, 5.0f);

    LITERT_ASSIGN_OR_RETURN(auto input1,
                            SimpleBuffer::Create<T>(params_.input1_shape));
    LITERT_RETURN_IF_ERROR(
        (input1.template WriteRandom<T>(modified_data_builder, device)));
    inputs.push_back(std::move(input1));

    LITERT_ASSIGN_OR_RETURN(auto input2,
                            SimpleBuffer::Create<T>(params_.input2_shape));
    LITERT_RETURN_IF_ERROR(
        (input2.template WriteRandom<T>(modified_data_builder, device)));
    inputs.push_back(std::move(input2));

    return inputs;
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    LITERT_ASSIGN_OR_RETURN(auto ref_inputs,
                            Traits::MakeReferenceInputs(inputs));
    LITERT_ASSIGN_OR_RETURN(auto ref_outputs,
                            Traits::MakeReferenceOutputs(outputs));

    auto [in1, in2] = ref_inputs;
    auto [output] = ref_outputs;

    std::array<const T*, 2> in_ptrs = {in1.data.data(), in2.data.data()};
    std::array<litert::internal::Dims, 2> in_dims = {
        litert::internal::Dims(params_.input1_shape.begin(),
                               params_.input1_shape.end()),
        litert::internal::Dims(params_.input2_shape.begin(),
                               params_.input2_shape.end())};

    litert::internal::ReferenceConcatenation<T>(absl::MakeSpan(in_ptrs),
                                                absl::MakeSpan(in_dims),
                                                output.data.data(), kAxis, kFa);

    return {};
  }

  Concatenation(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

 private:
  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;
    std::vector<int32_t> dims1(params.input1_shape.begin(),
                               params.input1_shape.end());
    std::vector<int32_t> dims2(params.input2_shape.begin(),
                               params.input2_shape.end());

    TensorTf input1 = litert::tensor::Create(
        std::string(kInputNames[0]), litert::tensor::ApiType<T>::value, dims1);
    TensorTf input2 = litert::tensor::Create(
        std::string(kInputNames[1]), litert::tensor::ApiType<T>::value, dims2);

    litert::tensor::FusedActivation act = litert::tensor::kActNone;
    if constexpr (kFa == tflite::ActivationFunctionType_RELU) {
      act = litert::tensor::kActRelu;
    } else if constexpr (kFa == tflite::ActivationFunctionType_RELU_N1_TO_1) {
      act = litert::tensor::kActReluN1To1;
    } else if constexpr (kFa == tflite::ActivationFunctionType_RELU6) {
      act = litert::tensor::kActRelu6;
    }

    TensorTf output =
        litert::tensor::Concatenation({input1, input2}, kAxis, act);
    output.SetName(std::string(kOutputNames[0]));

    return SaveTensorGraph({output});
  }

  Params params_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_CONCATENATION_H_
