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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_MEAN_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_MEAN_H_

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
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
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/model/model.h"
#include "litert/core/model/ops/reductions.h"
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

template <typename Rank, typename T,
          typename OpCode = OpCodeC<kLiteRtOpCodeTflMean>,
          typename KeepDims = std::false_type>
class Mean : public TestGraph {
  static_assert(std::is_same_v<typename Rank::value_type, size_t>);
  static constexpr size_t kRank = Rank::value;

  static constexpr LiteRtOpCode kOpCode = OpCode::value;
  static_assert(kOpCode == kLiteRtOpCodeTflMean);

  static constexpr bool kKeepDims = KeepDims::value;

  static constexpr ElementType kElementType = GetElementType<T>();
  static constexpr TensorNames<1> kInputNames = {"input"};
  static constexpr TensorNames<1> kOutputNames = {"output"};

  struct Params {
    std::array<Layout::Dim, kRank> input_shape;
    std::vector<int32_t> axes;
    std::vector<Layout::Dim> output_shape;
  };

 public:
  using Traits = TestLogicTraits<TypeList<T>, TypeList<T>, Params>;
  using Ptr = std::unique_ptr<Mean>;

  static constexpr absl::string_view Name() { return "Mean"; }

  template <typename Rng>
  static Expected<Mean::Ptr> Create(Rng& rng) {
    Params params;
    params.input_shape.fill(1);
    std::uniform_int_distribution<int> dim_dist(2, 10);
    for (size_t i = 0; i < kRank; ++i) {
      params.input_shape[i] = dim_dist(rng);
    }

    std::uniform_int_distribution<size_t> num_axes_dist(1, kRank);
    size_t num_axes = num_axes_dist(rng);

    // Select random axes to reduce.
    std::vector<int32_t> reduce_axes(kRank);
    std::iota(reduce_axes.begin(), reduce_axes.end(), 0);
    std::shuffle(reduce_axes.begin(), reduce_axes.end(), rng);
    reduce_axes.resize(num_axes);
    std::sort(reduce_axes.begin(), reduce_axes.end());
    params.axes = std::move(reduce_axes);

    // Leverage shape inference to compute output shape.
    LiteRtOpT op;
    op.SetOpCode(kOpCode);
    auto options = std::make_unique<tflite::ReducerOptionsT>();
    options->keep_dims = kKeepDims;

    TflOptions tfl_opts;
    tfl_opts.type = tflite::BuiltinOptions_ReducerOptions;
    tfl_opts.value = options.release();
    litert::internal::SetTflOptions(op, std::move(tfl_opts));

    LiteRtTensorT axis_tensor;
    std::vector<int32_t> axes_data(params.axes.begin(), params.axes.end());
    OwningBufferRef<uint8_t> axes_buf(
        reinterpret_cast<const uint8_t*>(axes_data.data()),
        axes_data.size() * sizeof(int32_t));
    SetWeightsFromOwnedBuffer(axis_tensor.Weights(), std::move(axes_buf));
    LiteRtTensorT dummy_input;
    op.Inputs().push_back(&dummy_input);
    op.Inputs().push_back(&axis_tensor);

    std::vector<litert::internal::Dims> input_shapes = {
        {params.input_shape.begin(), params.input_shape.begin() + kRank},
        {static_cast<int32_t>(axes_data.size())}};
    std::vector<litert::internal::Dims> output_shapes(1);

    LITERT_RETURN_IF_ERROR(litert::internal::InferReduce(
        op, absl::MakeSpan(input_shapes), output_shapes));

    params.output_shape = {output_shapes[0].begin(), output_shapes[0].end()};

    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<Mean>(std::move(params), std::move(model));
  }

  bool HasReference() const override { return true; }

  ConformanceSpec GetConformanceSpec() const override {
    ConformanceSpec spec;
    spec.comparator_kind = ConformanceComparatorKind::kFloatAccumulationAware;
    size_t in_elements = 1;
    for (size_t d : params_.input_shape) in_elements *= d;
    size_t out_elements = 1;
    for (size_t d : params_.output_shape) out_elements *= d;
    spec.accumulation_depth = out_elements > 0 ? in_elements / out_elements : 1;

    if constexpr (std::is_same_v<T, tflite::half>) {
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

    auto [in] = ref_inputs;
    auto [output] = ref_outputs;

    std::vector<int> input_dims_int(params_.input_shape.begin(),
                                    params_.input_shape.end());
    std::vector<int> output_dims_int(params_.output_shape.begin(),
                                     params_.output_shape.end());
    std::vector<int> axes_int(params_.axes.begin(), params_.axes.end());

    std::vector<float> in_f32 = UnpackToFloat(in.data);
    std::vector<float> out_f32(output.data.size());

    if (!litert::internal::ReferenceMean<float, float>(
            in_f32.data(), input_dims_int.data(), kRank, out_f32.data(),
            output_dims_int.data(), output_dims_int.size(), axes_int.data(),
            axes_int.size(), kKeepDims)) {
      return Error(kLiteRtStatusErrorInvalidArgument, "Reference Mean failed.");
    }

    PackFromFloat(absl::MakeConstSpan(out_f32), output.data);
    return {};
  }

  Mean(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

 private:
  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;
    std::vector<int32_t> dims(params.input_shape.begin(),
                              params.input_shape.end());
    std::vector<int> axes(params.axes.begin(), params.axes.end());

    TensorTf input = litert::tensor::Create(
        std::string(kInputNames[0]), litert::tensor::ApiType<T>::value, dims);

    TensorTf output = litert::tensor::Mean(input, axes, kKeepDims);
    output.SetName(std::string(kOutputNames[0]));

    return SaveTensorGraph({output});
  }

  Params params_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_MEAN_H_
