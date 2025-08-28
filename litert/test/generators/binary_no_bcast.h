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
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_c_types_printing.h"  // IWYU pragma: keep
#include "litert/cc/litert_detail.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_rng.h"
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
struct BinaryNoBroadcast {
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

  static constexpr absl::string_view Name() { return "BinaryNoBroadcast"; }

  Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
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

  template <typename Rng>
  Expected<Params> GenerateParams(Rng& rng) {
    RandomTensorType<kRank, kMaxTensorSize, LiteRtElementType(kElementType)>
        type;
    LITERT_ASSIGN_OR_RETURN(const auto tensor_type, type(rng));
    Params p;
    std::copy(std::cbegin(tensor_type.layout.dimensions),
              std::cbegin(tensor_type.layout.dimensions) + kRank,
              std::begin(p.shape));
    return p;
  }

  template <typename Rng>
  Expected<typename Traits::InputBuffers> MakeInputs(
      const RandomTensorDataBuilder& data_builder, Rng& rng,
      const Params& params) {
    LITERT_ASSIGN_OR_RETURN(auto lhs, SimpleBuffer::Create<T>(params.shape));
    LITERT_ASSIGN_OR_RETURN(auto rhs, SimpleBuffer::Create<T>(params.shape));
    LITERT_RETURN_IF_ERROR((lhs.template WriteRandom<T>(data_builder, rng)));
    LITERT_RETURN_IF_ERROR((rhs.template WriteRandom<T>(data_builder, rng)));
    static const auto kScale = 3.0f;
    const auto [min, max] = data_builder.Bounds<T>();
    // Prevent overflow.
    for (T& val : lhs.template Span<T>()) {
      if (val > (max / kScale)) {
        val = static_cast<T>(max / kScale);
      }
    }
    for (T& val : rhs.template Span<T>()) {
      if (val > (max / kScale)) {
        val = static_cast<T>(max / kScale);
      }
    }
    return typename Traits::InputBuffers{std::move(lhs), std::move(rhs)};
  }

  Expected<typename Traits::OutputBuffers> MakeOutputs(const Params& params) {
    LITERT_ASSIGN_OR_RETURN(auto output, SimpleBuffer::Create<T>(params.shape));
    return typename Traits::OutputBuffers{std::move(output)};
  }

  Expected<void> Reference(const Params& params,
                           const typename Traits::ReferenceInputs& inputs,
                           const typename Traits::ReferenceOutputs& outputs) {
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
};

}  // namespace testing
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_BINARY_NO_BCAST_H_
