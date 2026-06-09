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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_SLICE_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_SLICE_H_

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
#include "litert/core/model/model_load.h"
#include "litert/core/model/model_serialize.h"
#include "litert/core/model/ops/slice.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/core/util/flatbuffer_tools.h"
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

template <typename Rank, typename T,
          typename OpCode = OpCodeC<kLiteRtOpCodeTflSlice>>
class Slice : public TestGraph {
  static_assert(std::is_same_v<typename Rank::value_type, size_t>);
  static constexpr size_t kRank = Rank::value;

  static constexpr ElementType kElementType = GetElementType<T>();
  static constexpr TensorNames<1> kInputNames = {"input"};
  static constexpr TensorNames<1> kOutputNames = {"output"};

  struct Params {
    std::array<Layout::Dim, kRank> input_shape;
    std::array<int32_t, kRank> begin;
    std::array<int32_t, kRank> size;
    std::array<Layout::Dim, kRank> output_shape;
  };

 public:
  using Traits = TestLogicTraits<TypeList<T>, TypeList<T>, Params>;
  using Ptr = std::unique_ptr<Slice>;

  static constexpr absl::string_view Name() { return "Slice"; }

  template <typename Rng>
  static Expected<Slice::Ptr> Create(Rng& rng) {
    Params params;
    std::uniform_int_distribution<int> dim_dist(3, 8);

    for (size_t i = 0; i < kRank; ++i) {
      params.input_shape[i] = dim_dist(rng);
      params.begin[i] =
          std::uniform_int_distribution<int>(0, params.input_shape[i] - 2)(rng);
      params.size[i] = std::uniform_int_distribution<int>(
          1, params.input_shape[i] - params.begin[i])(rng);
      params.output_shape[i] = params.size[i];
    }

    LiteRtOpT op;
    op.SetOpCode(kLiteRtOpCodeTflSlice);

    std::vector<int32_t> begin_vec(params.begin.begin(),
                                   params.begin.begin() + kRank);
    auto begin_buf = MakeOwningBufferRef(begin_vec);
    LiteRtTensorT begin_tensor;
    begin_tensor.SetType(::MakeRankedTensorType(kLiteRtElementTypeInt32,
                                                {static_cast<int32_t>(kRank)}));
    ::SetWeightsFromUnownedBuffer(begin_tensor.Weights(), begin_buf);

    std::vector<int32_t> size_vec(params.size.begin(),
                                  params.size.begin() + kRank);
    auto size_buf = MakeOwningBufferRef(size_vec);
    LiteRtTensorT size_tensor;
    size_tensor.SetType(::MakeRankedTensorType(kLiteRtElementTypeInt32,
                                               {static_cast<int32_t>(kRank)}));
    ::SetWeightsFromUnownedBuffer(size_tensor.Weights(), size_buf);

    op.Inputs().push_back(nullptr);
    op.Inputs().push_back(&begin_tensor);
    op.Inputs().push_back(&size_tensor);

    std::vector<litert::internal::Dims> input_shapes = {
        {params.input_shape.begin(), params.input_shape.end()},
        {static_cast<int32_t>(kRank)},
        {static_cast<int32_t>(kRank)}};
    std::vector<litert::internal::Dims> output_shapes(1);

    LITERT_RETURN_IF_ERROR(litert::internal::InferSlice(
        op, absl::MakeSpan(input_shapes), output_shapes));

    std::copy(output_shapes[0].begin(), output_shapes[0].end(),
              params.output_shape.begin());

    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<Slice>(std::move(params), std::move(model));
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
    auto [output] = ref_outputs;

    litert::internal::ReferenceSlice<T>(
        in.data.data(),
        reinterpret_cast<const int32_t*>(params_.input_shape.data()),
        params_.begin.data(), params_.size.data(), kRank, output.data.data());

    return {};
  }

  Slice(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

 private:
  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;
    std::vector<int32_t> dims(params.input_shape.begin(),
                              params.input_shape.end());

    TensorTf input = litert::tensor::Create(
        std::string(kInputNames[0]), litert::tensor::ApiType<T>::value, dims);

    TensorTf begin_tensor = litert::tensor::Create(
        "begin", litert::tensor::Type::kI32, {static_cast<int32_t>(kRank)},
        litert::tensor::OwningCpuBuffer::Copy<litert::tensor::Type::kI32>(
            absl::MakeSpan(params.begin.data(), kRank)));

    TensorTf size_tensor = litert::tensor::Create(
        "size", litert::tensor::Type::kI32, {static_cast<int32_t>(kRank)},
        litert::tensor::OwningCpuBuffer::Copy<litert::tensor::Type::kI32>(
            absl::MakeSpan(params.size.data(), kRank)));

    TensorTf output = litert::tensor::Slice(input, begin_tensor, size_tensor);
    output.SetName(std::string(kOutputNames[0]));

    return SaveTensorGraph({output});
  }

  Params params_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_SLICE_H_
