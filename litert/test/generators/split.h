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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_SPLIT_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_SPLIT_H_

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
#include "litert/core/model/ops/split.h"
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

template <>
struct OpDetails<kLiteRtOpCodeTflSplit> {
  using FbTypes = FbOpTypes<kLiteRtOpCodeTflSplit>;
  using OptionsT = typename FbTypes::OptionsT;

  explicit OpDetails(int num_splits = 2) { options.num_splits = num_splits; }

  TflOptions MakeTflOptions() const {
    TflOptions res;
    res.type = FbTypes::kBuiltinOptions;
    res.Set(tflite::SplitOptionsT(options));
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

template <typename Rank, typename T,
          typename OpCode = OpCodeC<kLiteRtOpCodeTflSplit>,
          typename Axis = SizeC<0>, typename NumSplits = SizeC<2>>
class Split : public TestGraph {
  static_assert(std::is_same_v<typename Rank::value_type, size_t>);
  static constexpr size_t kRank = Rank::value;

  static constexpr int32_t kAxis = Axis::value;
  static constexpr int32_t kNumSplits = NumSplits::value;

  static constexpr ElementType kElementType = GetElementType<T>();
  static constexpr TensorNames<1> kInputNames = {"input"};
  static constexpr TensorNames<3> kOutputNames = {"output1", "output2",
                                                  "output3"};

  struct Params {
    std::array<Layout::Dim, kRank> input_shape;
    int32_t axis;
    int32_t num_splits;
    std::array<Layout::Dim, kRank> output_shape;
  };

 public:
  using OutputTypesT =
      SelectT<std::bool_constant<kNumSplits == 2>, TypeList<T, T>,
              std::bool_constant<kNumSplits == 3>, TypeList<T, T, T>>;
  using Traits = TestLogicTraits<TypeList<T>, OutputTypesT, Params>;
  using Ptr = std::unique_ptr<Split>;

  static constexpr absl::string_view Name() { return "Split"; }

  template <typename Rng>
  static Expected<Split::Ptr> Create(Rng& rng) {
    Params params;
    std::uniform_int_distribution<int> dim_dist(1, 5);

    for (size_t i = 0; i < kRank; ++i) {
      params.input_shape[i] = dim_dist(rng);
    }
    params.input_shape[kAxis] = dim_dist(rng) * kNumSplits;

    params.axis = kAxis;
    params.num_splits = kNumSplits;

    LiteRtOpT op;
    op.SetOpCode(kLiteRtOpCodeTflSplit);
    auto options = std::make_unique<tflite::SplitOptionsT>();
    options->num_splits = kNumSplits;

    TflOptions tfl_opts;
    tfl_opts.type = tflite::BuiltinOptions_SplitOptions;
    tfl_opts.value = options.release();
    litert::internal::SetTflOptions(op, std::move(tfl_opts));

    int32_t axis_val = kAxis;
    auto axis_buf = MakeOwningBufferRef(std::vector<int32_t>{axis_val});

    LiteRtTensorT axis_tensor;
    axis_tensor.SetType(::MakeRankedTensorType(kLiteRtElementTypeInt32, {1}));
    ::SetWeightsFromUnownedBuffer(axis_tensor.Weights(), axis_buf);
    op.Inputs().push_back(&axis_tensor);
    op.Inputs().push_back(nullptr);

    std::vector<litert::internal::Dims> input_shapes = {
        {1}, {params.input_shape.begin(), params.input_shape.end()}};
    std::vector<litert::internal::Dims> output_shapes(kNumSplits);

    LITERT_RETURN_IF_ERROR(litert::internal::InferSplit(
        op, absl::MakeSpan(input_shapes), output_shapes));

    std::copy(output_shapes[0].begin(), output_shapes[0].end(),
              params.output_shape.begin());

    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<Split>(std::move(params), std::move(model));
  }

  bool HasReference() const override { return true; }

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    VarBuffers inputs;
    inputs.reserve(1);
    RandomTensorDataBuilder modified_data_builder = data_builder;
    modified_data_builder.SetFloatRange(-5.0f, 5.0f);

    LITERT_ASSIGN_OR_RETURN(auto input,
                            SimpleBuffer::Create<T>(params_.input_shape));
    LITERT_RETURN_IF_ERROR(
        (input.template WriteRandom<T>(modified_data_builder, device)));
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
    std::vector<T*> out_ptrs;
    out_ptrs.reserve(kNumSplits);
    auto add_ptrs = [&out_ptrs](auto&... bufs) {
      (out_ptrs.push_back(bufs.data.data()), ...);
    };
    std::apply(add_ptrs, ref_outputs);

    litert::internal::ReferenceSplit<T>(
        in.data.data(),
        reinterpret_cast<const int32_t*>(params_.input_shape.data()), kRank,
        params_.axis, params_.num_splits, absl::MakeSpan(out_ptrs));

    return {};
  }

  Split(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

 private:
  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;
    std::vector<int32_t> dims(params.input_shape.begin(),
                              params.input_shape.end());

    TensorTf input = litert::tensor::Create(
        std::string(kInputNames[0]), litert::tensor::ApiType<T>::value, dims);

    TensorTf axis_tensor = litert::tensor::Create(
        "axis", litert::tensor::Type::kI32, {1},
        litert::tensor::OwningCpuBuffer::Copy<litert::tensor::Type::kI32>(
            {params.axis}));

    std::vector<TensorTf> outputs =
        litert::tensor::Split(input, axis_tensor, params.num_splits);

    std::vector<litert::tensor::TensorHandle> out_handles;
    out_handles.reserve(outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
      outputs[i].SetName(std::string(kOutputNames[i]));
      out_handles.push_back(outputs[i]);
    }

    return SaveTensorGraph(std::move(out_handles));
  }

  Params params_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_SPLIT_H_
