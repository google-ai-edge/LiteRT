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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_BINARY_NO_BCAST_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_BINARY_NO_BCAST_H_

// Logic for generating models with a single non-broadcasted bin op.

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
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
#include "litert/core/model/model_load.h"
#include "litert/core/model/model_serialize.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/graph_helpers.h"
#include "litert/test/simple_buffer.h"
#include "tflite/schema/schema_generated.h"

namespace litert {
namespace testing {

using ::litert::internal::AttachInput;
using ::litert::internal::AttachOutput;
using ::litert::internal::LoadModelFromBuffer;
using ::litert::internal::SerializeModel;
using ::litert::internal::SetTflOpCodeInd;
using ::litert::internal::SetTflOpCodes;
using ::litert::internal::SetTflOptions;
using ::litert::internal::TflOpCode;
using ::litert::internal::TflOpCodePtr;
using ::litert::internal::TflOptions;

// clang-format off
template <
    typename Rank,
    typename T,
    typename OpCode,
    typename Fa = FaC<>,
    typename MaxTensorSize = SizeC<1024>
>
// clang-format on
class BinaryNoBroadcast : public TestGraph {
 private:
  static_assert(std::is_same_v<typename OpCode::value_type, LiteRtOpCode>);
  static constexpr LiteRtOpCode kOpCode = OpCode::value;

  static_assert(
      std::is_same_v<typename Fa::value_type, tflite::ActivationFunctionType>);
  static constexpr tflite::ActivationFunctionType kFa = Fa::value;

  static_assert(std::is_same_v<typename MaxTensorSize::value_type, size_t>);
  static constexpr size_t kMaxTensorSize = MaxTensorSize::value;

  static_assert(std::is_same_v<typename Rank::value_type, size_t>);
  static constexpr size_t kRank = Rank::value;

  // TODO finish support for other op codes and activation functions.
  static_assert(kOpCode == kLiteRtOpCodeTflAdd ||
                kOpCode == kLiteRtOpCodeTflSub);
  static_assert(kFa == tflite::ActivationFunctionType_NONE);

  static constexpr TensorNames<2> kInputNames = {"lhs", "rhs"};
  static constexpr TensorNames<2> kOutputNames = {"output"};
  static constexpr absl::string_view kSignatureName = "default";

  using ReferenceOperator =
      SelectT<std::bool_constant<kOpCode == kLiteRtOpCodeTflAdd>, std::plus<T>,
              std::bool_constant<kOpCode == kLiteRtOpCodeTflSub>,
              std::minus<T>>;

  static constexpr ElementType kElementType = GetElementType<T>();

  using FbTypes = FbOpTypes<kOpCode>;

  struct Params {
    std::array<Layout::Dim, kRank> shape;
  };

 public:
  using Traits = TestLogicTraits<TypeList<T, T>, TypeList<T>, Params>;
  using Ptr = std::unique_ptr<BinaryNoBroadcast>;

  static constexpr absl::string_view Name() { return "BinaryNoBroadcast"; }

  template <typename Rng>
  static Expected<BinaryNoBroadcast::Ptr> Create(Rng& rng) {
    LITERT_ASSIGN_OR_RETURN(auto params, GenerateParams(rng));
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<BinaryNoBroadcast>(std::move(params),
                                               std::move(model));
  }

  static Expected<BinaryNoBroadcast::Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<BinaryNoBroadcast>(std::move(params),
                                               std::move(model));
  }

  bool HasReference() const override { return true; }

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    LITERT_ASSIGN_OR_RETURN(auto lhs, SimpleBuffer::Create<T>(params_.shape));
    LITERT_ASSIGN_OR_RETURN(auto rhs, SimpleBuffer::Create<T>(params_.shape));
    LITERT_RETURN_IF_ERROR((lhs.template WriteRandom<T>(data_builder, device)));
    LITERT_RETURN_IF_ERROR((rhs.template WriteRandom<T>(data_builder, device)));
    // Prevent overflow.
    static const auto kScale = 3;
    auto lhs_dat = lhs.template Span<T>();
    auto rhs_dat = rhs.template Span<T>();
    ScaleDown(lhs_dat, kScale);
    ScaleDown(rhs_dat, kScale);
    VarBuffers inputs;
    inputs.push_back(std::move(lhs));
    inputs.push_back(std::move(rhs));
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

  BinaryNoBroadcast(Params params, LiteRtModelT::Ptr model)
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

    std::vector<TensorDetails> inputs(2);
    std::vector<TensorDetails> outputs(1);

    inputs[0] = TensorDetails{dims, LiteRtElementType(kElementType),
                              std::string(kInputNames[0])};
    inputs[1] = TensorDetails{dims, LiteRtElementType(kElementType),
                              std::string(kInputNames[1])};

    outputs[0] = TensorDetails{dims, LiteRtElementType(kElementType),
                               std::string(kOutputNames[0])};

    return SingleOpModel<kOpCode>(inputs, outputs, kFa,
                                  /*pot_scale_int16=*/false);
  }
  Expected<void> ReferenceImpl(
      const typename Traits::ReferenceInputs& inputs,
      const typename Traits::ReferenceOutputs& outputs) const {
    auto [lhs, rhs] = inputs;
    auto [output] = outputs;
    if (lhs.dimensions != rhs.dimensions ||
        lhs.dimensions != output.dimensions) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "lhs, rhs, and output must have the same dimensions");
    }
    for (auto i = 0; i < lhs.NumElements(); ++i) {
      output.data[i] = ReferenceOperator()(lhs.data[i], rhs.data[i]);
    }
    return {};
  }

  Params params_;
};

}  // namespace testing
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_BINARY_NO_BCAST_H_
