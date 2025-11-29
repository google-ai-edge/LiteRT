// Copyright 2025 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_UNARY_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_UNARY_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_c_types_printing.h"  // IWYU pragma: keep
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/internal/litert_rng.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/model/model.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/graph_helpers.h"
#include "litert/test/simple_buffer.h"

namespace litert {
namespace testing {

template <typename T>
struct ReluReference {
  T operator()(T value) const { return std::max(value, static_cast<T>(0)); }
};

// clang-format off
template <
    typename Rank,
    typename T,
    typename OpCode,
    typename MaxTensorSize = SizeC<1024>
>
// clang-format on
class Unary : public TestGraph {
 private:
  static_assert(std::is_same_v<typename OpCode::value_type, LiteRtOpCode>);
  static constexpr LiteRtOpCode kOpCode = OpCode::value;

  static_assert(std::is_same_v<typename MaxTensorSize::value_type, size_t>);
  static constexpr size_t kMaxTensorSize = MaxTensorSize::value;

  static_assert(std::is_same_v<typename Rank::value_type, size_t>);
  static constexpr size_t kRank = Rank::value;

  // TODO finish support for other op codes and activation functions.
  static_assert(kOpCode == kLiteRtOpCodeTflRelu);

  static constexpr TensorNames<1> kInputNames = {"input"};
  static constexpr TensorNames<2> kOutputNames = {"output"};
  static constexpr absl::string_view kSignatureName = "default";

  // clang-format off
  using ReferenceOperator =
      SelectT<
          std::bool_constant<kOpCode == kLiteRtOpCodeTflRelu>,
            ReluReference<T>
            >;
  // clang-format on

  static constexpr ElementType kElementType = GetElementType<T>();

  using FbTypes = FbOpTypes<kOpCode>;

  struct Params {
    std::array<Layout::Dim, kRank> shape;
  };

 public:
  using Traits = TestLogicTraits<TypeList<T>, TypeList<T>, Params>;
  using Ptr = std::unique_ptr<Unary>;

  static constexpr absl::string_view Name() { return "Unary"; }

  template <typename Rng>
  static Expected<Unary::Ptr> Create(Rng& rng) {
    LITERT_ASSIGN_OR_RETURN(auto params, GenerateParams(rng));
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<Unary>(std::move(params), std::move(model));
  }

  static Expected<Unary::Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<Unary>(std::move(params), std::move(model));
  }

  bool HasReference() const override { return true; }

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    VarBuffers inputs;
    LITERT_ASSIGN_OR_RETURN(auto input, SimpleBuffer::Create<T>(params_.shape));
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
    return ReferenceImpl(ref_inputs, ref_outputs);
  }

  Unary(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

 private:
  template <typename Rng>
  static Expected<Params> GenerateParams(Rng& rng) {
    RandomTensorType<kRank, kMaxTensorSize, LiteRtElementType(kElementType)>
        type;
    LITERT_ASSIGN_OR_RETURN(const auto tensor_type, type(rng));
    Params p;
    std::copy(std::cbegin(tensor_type.layout.dimensions),
              std::cbegin(tensor_type.layout.dimensions) + kRank,
              std::begin(p.shape));
    return p;
  }
  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    const std::vector<int32_t> dims(params.shape.begin(), params.shape.end());

    std::vector<TensorDetails> inputs(1);
    std::vector<TensorDetails> outputs(1);

    inputs[0] = TensorDetails{dims, LiteRtElementType(kElementType),
                              std::string(kInputNames[0])};

    outputs[0] = TensorDetails{dims, LiteRtElementType(kElementType),
                               std::string(kOutputNames[0])};

    return SingleOpModel<kOpCode>(inputs, outputs);
  }

  Expected<void> ReferenceImpl(
      const typename Traits::ReferenceInputs& inputs,
      const typename Traits::ReferenceOutputs& outputs) const {
    auto [input] = inputs;
    auto [output] = outputs;
    if (input.dimensions != output.dimensions) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "input and output must have the same dimensions");
    }
    for (auto i = 0; i < input.NumElements(); ++i) {
      output.data[i] = ReferenceOperator()(input.data[i]);
    }
    return {};
  }

  Params params_;
};

}  // namespace testing
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_UNARY_H_
