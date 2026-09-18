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
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/internal/litert_rng.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/model/model.h"
#include "litert/core/model/ops/slice.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/graph_helpers.h"
#include "litert/test/simple_buffer.h"
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"

namespace litert::testing {

template <typename Rank, typename T,
          typename OpCode = OpCodeC<kLiteRtOpCodeTflSlice>>
class Slice : public TestGraph {
  static_assert(std::is_same_v<typename Rank::value_type, size_t>);
  static constexpr size_t kRank = Rank::value;

  static constexpr LiteRtOpCode kOpCode = OpCode::value;
  static_assert(kOpCode == kLiteRtOpCodeTflSlice);

  static constexpr ElementType kElementType = GetElementType<T>();
  static constexpr TensorNames<1> kInputNames = {"input"};
  static constexpr TensorNames<1> kOutputNames = {"output"};

 public:
  struct Params {
    std::array<Layout::Dim, kRank> input_shape;
    std::array<int32_t, kRank> begin;
    std::array<int32_t, kRank> size;
    std::array<Layout::Dim, kRank> output_shape;
  };

  using Traits = TestLogicTraits<TypeList<T>, TypeList<T>, Params>;
  using Ptr = std::unique_ptr<Slice>;

  static constexpr absl::string_view Name() { return "Slice"; }

  template <typename Rng>
  static Expected<Slice::Ptr> Create(Rng& rng) {
    Params params;
    std::uniform_int_distribution<int> dim_dist(2, 8);
    for (size_t i = 0; i < kRank; ++i) {
      params.input_shape[i] = dim_dist(rng);
      std::uniform_int_distribution<int> begin_dist(
          0, params.input_shape[i] - 1);
      params.begin[i] = begin_dist(rng);
      std::uniform_int_distribution<int> size_dist(
          1, params.input_shape[i] - params.begin[i]);
      params.size[i] = size_dist(rng);
      params.output_shape[i] = params.size[i];
    }

    return Create(std::move(params));
  }

  static Expected<Slice::Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<Slice>(std::move(params), std::move(model));
  }

  bool HasReference() const override { return true; }

  ConformanceSpec GetConformanceSpec() const override {
    ConformanceSpec spec;
    spec.comparator_kind = ConformanceComparatorKind::kExact;
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

    litert::internal::ReferenceSlice<T>(
        in.data.data(),
        reinterpret_cast<const int32_t*>(params_.input_shape.data()),
        params_.begin.data(),
        reinterpret_cast<const int32_t*>(params_.output_shape.data()),
        static_cast<int>(kRank),
        output.data.data());

    return {};
  }

  Slice(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

 private:
  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;
    std::vector<int32_t> dims(params.input_shape.begin(),
                              params.input_shape.end());
    std::vector<int> begin(params.begin.begin(), params.begin.end());
    std::vector<int> size(params.size.begin(), params.size.end());

    TensorTf input = litert::tensor::Create(
        std::string(kInputNames[0]), litert::tensor::ApiType<T>::value, dims);

    TensorTf output = litert::tensor::Slice(input, begin, size);
    output.SetName(std::string(kOutputNames[0]));

    return SaveTensorGraph({output});
  }

  Params params_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_SLICE_H_
