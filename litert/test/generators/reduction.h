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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_REDUCTION_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_REDUCTION_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
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
#include "tflite/schema/schema_generated.h"

namespace litert::testing {

// Specialize OpDetails for Reduction to handle options.
template <LiteRtOpCode OpCode>
struct ReductionOpDetails {
  using FbTypes = FbOpTypes<OpCode>;
  using OptionsT = typename FbTypes::OptionsT;

  explicit ReductionOpDetails(bool keep_dims) { options.keep_dims = keep_dims; }

  TflOptions MakeTflOptions() const {
    TflOptions res;
    res.type = FbTypes::kBuiltinOptions;
    res.Set(tflite::ReducerOptionsT(options));
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

template <>
struct OpDetails<kLiteRtOpCodeTflReduceMax>
    : public ReductionOpDetails<kLiteRtOpCodeTflReduceMax> {
  using ReductionOpDetails::ReductionOpDetails;
};
template <>
struct OpDetails<kLiteRtOpCodeTflReduceMin>
    : public ReductionOpDetails<kLiteRtOpCodeTflReduceMin> {
  using ReductionOpDetails::ReductionOpDetails;
};
template <>
struct OpDetails<kLiteRtOpCodeTflReduceProd>
    : public ReductionOpDetails<kLiteRtOpCodeTflReduceProd> {
  using ReductionOpDetails::ReductionOpDetails;
};
template <>
struct OpDetails<kLiteRtOpCodeTflReduceAny>
    : public ReductionOpDetails<kLiteRtOpCodeTflReduceAny> {
  using ReductionOpDetails::ReductionOpDetails;
};
template <>
struct OpDetails<kLiteRtOpCodeTflReduceAll>
    : public ReductionOpDetails<kLiteRtOpCodeTflReduceAll> {
  using ReductionOpDetails::ReductionOpDetails;
};

template <typename Rank, typename T, typename OpCode,
          typename KeepDims = std::false_type>
class Reduction : public TestGraph {
 private:
  static_assert(std::is_same_v<typename OpCode::value_type, LiteRtOpCode>);
  static constexpr LiteRtOpCode kOpCode = OpCode::value;

  static_assert(std::is_same_v<typename Rank::value_type, size_t>);
  static constexpr size_t kRank = Rank::value;
  static_assert(kRank == 4, "Reduction test assumes rank 4 input");

  static constexpr bool kKeepDims = KeepDims::value;

  static constexpr TensorNames<1> kInputNames = {"input"};
  static constexpr TensorNames<1> kOutputNames = {"output"};

  static constexpr ElementType kElementType = GetElementType<T>();

  struct Params {
    std::array<Layout::Dim, 4> input_shape;
    std::vector<int32_t> axes;
    std::vector<Layout::Dim> output_shape;
  };

 public:
  using Traits = TestLogicTraits<TypeList<T>, TypeList<T>, Params>;
  using Ptr = std::unique_ptr<Reduction>;

  static constexpr absl::string_view Name() { return "Reduction"; }

  template <typename Rng>
  static Expected<Reduction::Ptr> Create(Rng& rng) {
    Params params;
    params.input_shape = {1, 10, 10, 1};
    params.axes = {1, 2};  // Reduce spatial dimensions

    // Leverage shape inference to compute output shape
    LiteRtOpT op;
    op.SetOpCode(kOpCode);
    auto options = std::make_unique<tflite::ReducerOptionsT>();
    options->keep_dims = kKeepDims;

    TflOptions tfl_opts;
    tfl_opts.type = tflite::BuiltinOptions_ReducerOptions;
    tfl_opts.value = options.release();
    litert::internal::SetTflOptions(op, std::move(tfl_opts));

    LiteRtTensorT axis_tensor;
    std::vector<int32_t> axes_data = {1, 2};
    OwningBufferRef<uint8_t> axes_buf(
        reinterpret_cast<const uint8_t*>(axes_data.data()),
        axes_data.size() * sizeof(int32_t));
    SetWeightsFromOwnedBuffer(axis_tensor.Weights(), std::move(axes_buf));
    LiteRtTensorT dummy_input;
    op.Inputs().push_back(&dummy_input);
    op.Inputs().push_back(&axis_tensor);

    std::vector<litert::internal::Dims> input_shapes = {
        {params.input_shape.begin(), params.input_shape.end()},
        {axes_data.size()}};
    std::vector<litert::internal::Dims> output_shapes(1);

    LITERT_RETURN_IF_ERROR(litert::internal::InferReduce(
        op, absl::MakeSpan(input_shapes), output_shapes));
    LITERT_LOG(LITERT_INFO, "Output shape size: %lu", output_shapes[0].size());

    params.output_shape = {output_shapes[0].begin(), output_shapes[0].end()};

    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<Reduction>(std::move(params), std::move(model));
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

    T init_value = 0;

    // Function pointer type for the reducer.
    using ReducerFn = T (*)(const T, const T);
    ReducerFn reducer = nullptr;

    if constexpr (kOpCode == kLiteRtOpCodeTflReduceMax) {
      init_value = std::numeric_limits<T>::lowest();
      reducer = [](const T a, const T b) { return std::max(a, b); };
    } else if constexpr (kOpCode == kLiteRtOpCodeTflReduceMin) {
      init_value = std::numeric_limits<T>::max();
      reducer = [](const T a, const T b) { return std::min(a, b); };
    } else if constexpr (kOpCode == kLiteRtOpCodeTflReduceProd) {
      init_value = 1;
      reducer = [](const T a, const T b) { return a * b; };
    } else if constexpr (kOpCode == kLiteRtOpCodeTflReduceAny) {
      init_value = 0;
      reducer = [](const T a, const T b) { return (a || b) ? T(1) : T(0); };
    } else if constexpr (kOpCode == kLiteRtOpCodeTflReduceAll) {
      init_value = 1;
      reducer = [](const T a, const T b) { return (a && b) ? T(1) : T(0); };
    }

    litert::internal::ReferenceReduction<T>(
        input.data.data(),
        reinterpret_cast<const int*>(params_.input_shape.data()), 4,
        output.data.data(),
        reinterpret_cast<const int*>(params_.output_shape.data()),
        params_.output_shape.size(),
        reinterpret_cast<const int*>(params_.axes.data()), params_.axes.size(),
        kKeepDims, init_value, reducer);

    return {};
  }

  Reduction(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

 private:
  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    std::vector<TensorDetails> op_inputs(2);
    std::vector<TensorDetails> op_outputs(1);

    op_inputs[0] = TensorDetails{
        std::vector<int32_t>(params.input_shape.begin(),
                             params.input_shape.end()),
        LiteRtElementType(kElementType), std::string(kInputNames[0])};

    OwningBufferRef<uint8_t> axes_buf(
        reinterpret_cast<const uint8_t*>(params.axes.data()),
        params.axes.size() * sizeof(int32_t));

    op_inputs[1] = TensorDetails{{static_cast<int32_t>(params.axes.size())},
                                 kLiteRtElementTypeInt32,
                                 "axes",
                                 std::move(axes_buf)};

    op_outputs[0] = TensorDetails{
        std::vector<int32_t>(params.output_shape.begin(),
                             params.output_shape.end()),
        LiteRtElementType(kElementType), std::string(kOutputNames[0])};

    return SingleOpModel<kOpCode>(op_inputs, op_outputs, kKeepDims);
  }

  Params params_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_REDUCTION_H_
