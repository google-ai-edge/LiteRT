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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_SOFTMAX_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_SOFTMAX_H_

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
#include "litert/core/model/ops/simple_unary.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/graph_helpers.h"
#include "litert/test/simple_buffer.h"
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"
#include "tflite/schema/schema_generated.h"
#include "tflite/types/half.h"

namespace litert::testing {

template <typename Rank, typename T_in, typename T_out,
          typename OpCode = OpCodeC<kLiteRtOpCodeTflSoftmax>,
          typename Beta = SizeC<1>>
class Softmax : public TestGraph {
  static_assert(std::is_same_v<typename Rank::value_type, size_t>);
  static constexpr size_t kRank = Rank::value;

  static constexpr LiteRtOpCode kOpCode = OpCode::value;
  static constexpr float kBeta = static_cast<float>(Beta::value);

  static constexpr ElementType kInElementType = GetElementType<T_in>();
  static constexpr ElementType kOutElementType = GetElementType<T_out>();
  static constexpr TensorNames<1> kInputNames = {"input"};
  static constexpr TensorNames<1> kOutputNames = {"output"};

  struct Params {
    std::array<Layout::Dim, kRank> input_shape;
    std::array<Layout::Dim, kRank> output_shape;
  };

 public:
  using Traits = TestLogicTraits<TypeList<T_in>, TypeList<T_out>, Params>;
  using Ptr = std::unique_ptr<Softmax>;

  static constexpr absl::string_view Name() {
    return kOpCode == kLiteRtOpCodeTflSoftmax ? "Softmax" : "LogSoftmax";
  }

  template <typename Rng>
  static Expected<Softmax::Ptr> Create(Rng& rng) {
    Params params;
    std::uniform_int_distribution<int> dim_dist(2, 5);

    for (size_t i = 0; i < kRank; ++i) {
      params.input_shape[i] = dim_dist(rng);
    }

    LiteRtOpT op;
    op.SetOpCode(kOpCode);
    if constexpr (kOpCode == kLiteRtOpCodeTflSoftmax) {
      auto options = std::make_unique<tflite::SoftmaxOptionsT>();
      options->beta = kBeta;

      TflOptions tfl_opts;
      tfl_opts.type = tflite::BuiltinOptions_SoftmaxOptions;
      tfl_opts.value = options.release();
      litert::internal::SetTflOptions(op, std::move(tfl_opts));
    }

    std::vector<litert::internal::Dims> input_shapes = {
        {params.input_shape.begin(), params.input_shape.end()}};
    std::vector<litert::internal::Dims> output_shapes(1);

    if constexpr (kOpCode == kLiteRtOpCodeTflSoftmax) {
      LITERT_RETURN_IF_ERROR(litert::internal::InferSoftmax(
          op, absl::MakeSpan(input_shapes), output_shapes));
    } else {
      LITERT_RETURN_IF_ERROR(litert::internal::InferLogSoftmax(
          op, absl::MakeSpan(input_shapes), output_shapes));
    }

    std::copy(output_shapes[0].begin(), output_shapes[0].end(),
              params.output_shape.begin());

    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<Softmax>(std::move(params), std::move(model));
  }

  bool HasReference() const override { return true; }

  ConformanceSpec GetConformanceSpec() const override {
    ConformanceSpec spec;
    spec.comparator_kind = ConformanceComparatorKind::kFloatAccumulationAware;
    spec.accumulation_depth = params_.input_shape[kRank - 1];
    if constexpr (std::is_same_v<T_in, tflite::half> ||
                  std::is_same_v<T_out, tflite::half>) {
      spec.relative_tolerance = 5e-3;
    } else {
      spec.relative_tolerance = 1e-4;
    }
    return spec;
  }

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

    auto [in] = ref_inputs;
    auto [output] = ref_outputs;

    int64_t batch = 1;
    for (size_t i = 0; i < kRank - 1; ++i) {
      batch *= params_.input_shape[i];
    }
    int64_t depth = params_.input_shape[kRank - 1];

    std::vector<float> in_f32 = UnpackToFloat(in.data);
    std::vector<float> out_f32(output.data.size());

    if constexpr (kOpCode == kLiteRtOpCodeTflSoftmax) {
      litert::internal::ReferenceSoftmax(in_f32.data(), out_f32.data(), batch,
                                         depth, kBeta);
    } else {
      litert::internal::ReferenceLogSoftmax(in_f32.data(), out_f32.data(),
                                            batch, depth);
    }

    PackFromFloat(absl::MakeConstSpan(out_f32), output.data);
    return {};
  }

  Softmax(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

 private:
  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;
    std::vector<int32_t> dims(params.input_shape.begin(),
                              params.input_shape.end());

    TensorTf input =
        litert::tensor::Create(std::string(kInputNames[0]),
                               litert::tensor::ApiType<T_in>::value, dims);

    TensorTf output;
    if constexpr (kOpCode == kLiteRtOpCodeTflSoftmax) {
      output = litert::tensor::Softmax(input, kBeta);
    } else {
      output = litert::tensor::LogSoftmax(input);
    }
    output.SetName(std::string(kOutputNames[0]));

    return SaveTensorGraph({output});
  }

  Params params_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_SOFTMAX_H_
