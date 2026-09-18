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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_SELECT_V2_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_SELECT_V2_H_

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
#include "litert/core/model/ops/select.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/graph_helpers.h"
#include "litert/test/simple_buffer.h"
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"

namespace litert::testing {

template <typename Rank, typename T,
          typename OpCode = OpCodeC<kLiteRtOpCodeTflSelectV2>>
class SelectV2 : public TestGraph {
  static_assert(std::is_same_v<typename Rank::value_type, size_t>);
  static constexpr size_t kRank = Rank::value;

  static constexpr LiteRtOpCode kOpCode = OpCode::value;
  static_assert(kOpCode == kLiteRtOpCodeTflSelectV2);

  static constexpr ElementType kElementType = GetElementType<T>();
  static constexpr TensorNames<3> kInputNames = {"condition", "x", "y"};
  static constexpr TensorNames<1> kOutputNames = {"output"};

 public:
  struct Params {
    std::array<Layout::Dim, kRank> cond_shape;
    std::array<Layout::Dim, kRank> x_shape;
    std::array<Layout::Dim, kRank> y_shape;
    std::array<Layout::Dim, kRank> output_shape;
  };

  using Traits = TestLogicTraits<TypeList<bool, T, T>, TypeList<T>, Params>;
  using Ptr = std::unique_ptr<SelectV2>;

  static constexpr absl::string_view Name() { return "SelectV2"; }

  template <typename Rng>
  static Expected<SelectV2::Ptr> Create(Rng& rng) {
    Params params;
    std::uniform_int_distribution<int> dim_dist(2, 8);
    std::bernoulli_distribution broadcast_dist(0.3);

    for (size_t i = 0; i < kRank; ++i) {
      int d = dim_dist(rng);
      params.cond_shape[i] = broadcast_dist(rng) ? 1 : d;
      params.x_shape[i] = broadcast_dist(rng) ? 1 : d;
      params.y_shape[i] = broadcast_dist(rng) ? 1 : d;
      params.output_shape[i] = std::max(
          {params.cond_shape[i], params.x_shape[i], params.y_shape[i]});
    }

    return Create(std::move(params));
  }

  static Expected<SelectV2::Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<SelectV2>(std::move(params), std::move(model));
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
    inputs.reserve(3);

    LITERT_ASSIGN_OR_RETURN(auto cond,
                            SimpleBuffer::Create<bool>(params_.cond_shape));
    LITERT_RETURN_IF_ERROR(
        (cond.template WriteRandom<bool>(data_builder, device)));
    inputs.push_back(std::move(cond));

    LITERT_ASSIGN_OR_RETURN(auto x, SimpleBuffer::Create<T>(params_.x_shape));
    LITERT_RETURN_IF_ERROR((x.template WriteRandom<T>(data_builder, device)));
    inputs.push_back(std::move(x));

    LITERT_ASSIGN_OR_RETURN(auto y, SimpleBuffer::Create<T>(params_.y_shape));
    LITERT_RETURN_IF_ERROR((y.template WriteRandom<T>(data_builder, device)));
    inputs.push_back(std::move(y));

    return inputs;
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    LITERT_ASSIGN_OR_RETURN(auto ref_inputs,
                            Traits::MakeReferenceInputs(inputs));
    LITERT_ASSIGN_OR_RETURN(auto ref_outputs,
                            Traits::MakeReferenceOutputs(outputs));

    auto [cond, x, y] = ref_inputs;
    auto [output] = ref_outputs;

    litert::internal::ReferenceSelect<T>(
        cond.data.data(),
        reinterpret_cast<const int32_t*>(params_.cond_shape.data()),
        static_cast<int>(kRank), x.data.data(),
        reinterpret_cast<const int32_t*>(params_.x_shape.data()),
        static_cast<int>(kRank), y.data.data(),
        reinterpret_cast<const int32_t*>(params_.y_shape.data()),
        static_cast<int>(kRank), output.data.data(),
        reinterpret_cast<const int32_t*>(params_.output_shape.data()),
        static_cast<int>(kRank));

    return {};
  }

  SelectV2(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

 private:
  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;
    std::vector<int32_t> cond_dims(params.cond_shape.begin(),
                                   params.cond_shape.end());
    std::vector<int32_t> x_dims(params.x_shape.begin(),
                                params.x_shape.end());
    std::vector<int32_t> y_dims(params.y_shape.begin(),
                                params.y_shape.end());

    TensorTf cond = litert::tensor::Create(
        std::string(kInputNames[0]), litert::tensor::Type::kBOOL, cond_dims);
    TensorTf x = litert::tensor::Create(
        std::string(kInputNames[1]), litert::tensor::ApiType<T>::value, x_dims);
    TensorTf y = litert::tensor::Create(
        std::string(kInputNames[2]), litert::tensor::ApiType<T>::value, y_dims);

    TensorTf output = litert::tensor::SelectV2(cond, x, y);
    output.SetName(std::string(kOutputNames[0]));

    return SaveTensorGraph({output});
  }

  Params params_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_SELECT_V2_H_
