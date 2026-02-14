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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_COMMON_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_COMMON_H_

#include <array>
#include <cstddef>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/internal/litert_rng.h"
#include "litert/cc/litert_expected.h"
#include "litert/core/model/model.h"
#include "litert/test/simple_buffer.h"
#include "tflite/schema/schema_generated.h"

namespace litert::testing {

// Helpers for defining generators and their consituent types.

// Short hand for dynamic array of i/o buffers.
using VarBuffers = std::vector<SimpleBuffer>;

// Shorthand for specifying i/o tensor names as constexpr.
template <size_t N>
using TensorNames = std::array<absl::string_view, N>;

// Convenience type for mapping litert ops to the related flatbuffer types.
template <typename Options, ::tflite::BuiltinOperator BuiltinOperator,
          ::tflite::BuiltinOptions BuiltinOptions>
struct FbOpTraits {
  using OptionsT = Options;
  static constexpr tflite::BuiltinOperator kBuiltinOperator = BuiltinOperator;
  static constexpr tflite::BuiltinOptions kBuiltinOptions = BuiltinOptions;
  static constexpr bool kHasOptions = !std::is_same_v<Options, std::monostate>;
};

template <::tflite::BuiltinOperator BuiltinOperator>
using FbOpTraitsNoOptions =
    FbOpTraits<std::monostate, BuiltinOperator, tflite::BuiltinOptions_NONE>;

// Selects the appropriate fb traits based on the op code.
// TODO finish for other op codes.
// clang-format off
template <LiteRtOpCode OpCode>
using FbOpTypes =
    SelectT<
        std::bool_constant<OpCode == kLiteRtOpCodeTflAdd>,
            FbOpTraits<tflite::AddOptionsT, tflite::BuiltinOperator_ADD,
                       tflite::BuiltinOptions_AddOptions>,
        std::bool_constant<OpCode == kLiteRtOpCodeTflFloor>,
            FbOpTraitsNoOptions<tflite::BuiltinOperator_FLOOR>,
        std::bool_constant<OpCode == kLiteRtOpCodeTflLogistic>,
            FbOpTraitsNoOptions<tflite::BuiltinOperator_LOGISTIC>,
        std::bool_constant<OpCode == kLiteRtOpCodeTflRelu>,
            FbOpTraitsNoOptions<tflite::BuiltinOperator_RELU>,
        std::bool_constant<OpCode == kLiteRtOpCodeTflReluN1To1>,
            FbOpTraitsNoOptions<tflite::BuiltinOperator_RELU_N1_TO_1>,
        std::bool_constant<OpCode == kLiteRtOpCodeTflRelu6>,
            FbOpTraitsNoOptions<tflite::BuiltinOperator_RELU6>,
        std::bool_constant<OpCode == kLiteRtOpCodeTflTanh>,
            FbOpTraitsNoOptions<tflite::BuiltinOperator_TANH>,
        std::bool_constant<OpCode == kLiteRtOpCodeTflSub>,
            FbOpTraits<tflite::SubOptionsT, tflite::BuiltinOperator_SUB,
                       tflite::BuiltinOptions_SubOptions>,
        std::bool_constant<OpCode == kLiteRtOpCodeTflExp>,
            FbOpTraitsNoOptions<tflite::BuiltinOperator_EXP>,
        std::bool_constant<OpCode == kLiteRtOpCodeTflNeg>,
            FbOpTraitsNoOptions<tflite::BuiltinOperator_NEG>,
        std::bool_constant<OpCode == kLiteRtOpCodeTflSin>,
            FbOpTraitsNoOptions<tflite::BuiltinOperator_SIN>,
        std::bool_constant<OpCode == kLiteRtOpCodeTflLog>,
            FbOpTraitsNoOptions<tflite::BuiltinOperator_LOG>,
        std::bool_constant<OpCode == kLiteRtOpCodeTflSqrt>,
            FbOpTraitsNoOptions<tflite::BuiltinOperator_SQRT>,
        std::bool_constant<OpCode == kLiteRtOpCodeTflRsqrt>,
            FbOpTraitsNoOptions<tflite::BuiltinOperator_RSQRT>,
        std::bool_constant<OpCode == kLiteRtOpCodeTflSquare>,
            FbOpTraitsNoOptions<tflite::BuiltinOperator_SQUARE>,
        std::bool_constant<OpCode == kLiteRtOpCodeTflZerosLike>,
            FbOpTraitsNoOptions<tflite::BuiltinOperator_ZEROS_LIKE>,
        std::bool_constant<OpCode == kLiteRtOpCodeTflAbs>,
            FbOpTraitsNoOptions<tflite::BuiltinOperator_ABS>,
        std::bool_constant<OpCode == kLiteRtOpCodeTflCeil>,
            FbOpTraitsNoOptions<tflite::BuiltinOperator_CEIL>,
        std::bool_constant<OpCode == kLiteRtOpCodeTflCos>,
            FbOpTraitsNoOptions<tflite::BuiltinOperator_COS>,
        std::bool_constant<OpCode == kLiteRtOpCodeTflElu>,
            FbOpTraitsNoOptions<tflite::BuiltinOperator_ELU>,
        std::bool_constant<OpCode == kLiteRtOpCodeTflRound>,
            FbOpTraitsNoOptions<tflite::BuiltinOperator_ROUND>,
        std::bool_constant<OpCode == kLiteRtOpCodeTflHardSwish>,
            FbOpTraitsNoOptions<tflite::BuiltinOperator_HARD_SWISH>,
        std::bool_constant<OpCode == kLiteRtOpCodeTflGelu>,
            FbOpTraits<tflite::GeluOptionsT, tflite::BuiltinOperator_GELU,
                       tflite::BuiltinOptions_GeluOptions>,
        std::bool_constant<OpCode == kLiteRtOpCodeTflRelu0To1>,
            FbOpTraitsNoOptions<tflite::BuiltinOperator_RELU_0_TO_1>,
        std::bool_constant<OpCode == kLiteRtOpCodeTflSign>,
            FbOpTraitsNoOptions<tflite::BuiltinOperator_SIGN>
    >;
// clang-format on
static_assert(FbOpTypes<kLiteRtOpCodeTflAdd>::kHasOptions);
static_assert(!FbOpTypes<kLiteRtOpCodeTflRelu>::kHasOptions);

// Traits defined by individual test logics. Each generator should provide
// one of these as a public member type. Upstream glue code will use these types
// to specialize its logic. These are defined by a set of primitive input
// and output data types, as well as the Params struct that encapsulates all
// of the randomized values for the test.
template <typename InputTypes, typename OutputTypes, typename P>
struct TestLogicTraits {
 private:
  // Typing utility for defining the signature of a reference implementation
  // from a set of input and output primitive types.
  template <typename... Ts>
  static auto RefInputTypes(TypeList<Ts...>)
      -> std::tuple<SimpleBuffer::CView<Ts>...>;
  template <typename... Ts>
  static auto RefOutputTypes(TypeList<Ts...>)
      -> std::tuple<SimpleBuffer::View<Ts>...>;

  // Metaprogramming helper to grab the typed views from the tensor buffers
  // to pass to the reference implementation.
  template <typename ReferenceTensors, typename Buffers, size_t... Is>
  static Expected<ReferenceTensors> MakeReferenceTensors(
      Buffers& inputs, std::index_sequence<Is...>) {
    const bool types_ok =
        (true && ... &&
         (inputs[Is].ElementType() == GetElementType<InputDataType<Is>>()));
    if (!types_ok) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Input types do not match reference implementation types");
    }
    return ReferenceTensors{inputs[Is].template AsView<InputDataType<Is>>()...};
  }
  template <typename ReferenceTensors, typename Buffers>
  static Expected<ReferenceTensors> MakeReferenceTensors(Buffers& inputs) {
    return MakeReferenceTensors<ReferenceTensors>(
        inputs,
        std::make_index_sequence<std::tuple_size_v<ReferenceTensors>>());
  }

 public:
  // Inputs for the reference implementation.
  using ReferenceInputs = decltype(RefInputTypes(InputTypes()));

  // Outputs for the reference implementation.
  using ReferenceOutputs = decltype(RefOutputTypes(OutputTypes()));

  // Parameters that encapsulate all of the randomized values for this test.
  using Params = P;

  // Get the primitive type for the associated input.
  template <size_t N>
  using InputDataType = typename std::tuple_element_t<N, ReferenceInputs>::Type;

  // Get the primitive type for the associated output.
  template <size_t N>
  using OutputDataType =
      typename std::tuple_element_t<N, ReferenceOutputs>::Type;

  // Number of inputs.
  static constexpr size_t kNumInputs = InputTypes::kSize;

  // Number of outputs.
  static constexpr size_t kNumOutputs = OutputTypes::kSize;

  // Get the typed reference input views from the input buffers.
  static Expected<ReferenceInputs> MakeReferenceInputs(
      const VarBuffers& inputs) {
    if (inputs.size() != kNumInputs) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   absl::StrFormat("Expected %d inputs, got %d", kNumInputs,
                                   inputs.size()));
    }
    return MakeReferenceTensors<ReferenceInputs>(inputs);
  }

  // Get the typed reference output views from the output buffers.
  static Expected<ReferenceOutputs> MakeReferenceOutputs(VarBuffers& outputs) {
    if (outputs.size() != kNumOutputs) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   absl::StrFormat("Expected %d inputs, got %d", kNumOutputs,
                                   outputs.size()));
    }
    return MakeReferenceTensors<ReferenceOutputs>(outputs);
  }
};

// Shorthand for compile time constants of misc types. Due to constraints
// in other metaprogramming utilities, all template parameters for test logic
// cannot be non-type, so we rely on std::integral_constant for value based
// template constants.
template <::tflite::ActivationFunctionType Fa =
              ::tflite::ActivationFunctionType_NONE>
using FaC = std::integral_constant<tflite::ActivationFunctionType, Fa>;
template <::tflite::ActivationFunctionType... Fa>
using FaListC = TypeList<FaC<Fa>...>;

template <size_t N>
using SizeC = std::integral_constant<size_t, N>;
template <size_t... Sizes>
using SizeListC = TypeList<SizeC<Sizes>...>;

template <LiteRtOpCode OpCode>
using OpCodeC = std::integral_constant<LiteRtOpCode, OpCode>;
template <LiteRtOpCode... OpCodes>
using OpCodeListC = TypeList<OpCodeC<OpCodes>...>;

// Base class for containers of test graphs and related logic.
class TestGraph {
 public:
  using Ptr = std::unique_ptr<TestGraph>;
  // Detect if this graph container implements its own custom reference
  // impl.
  virtual bool HasReference() const = 0;

  // Returns the graph under test.
  LiteRtModelT& Graph() const { return *model_; }

  // Generate input buffers for the graph under test.
  virtual Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const = 0;

  // Reference implementation of the graph under test.
  virtual Expected<void> Reference(const VarBuffers& inputs,
                                   VarBuffers& outputs) const = 0;

  virtual ~TestGraph() = default;

  TestGraph(const TestGraph&) = delete;
  TestGraph& operator=(const TestGraph&) = delete;
  TestGraph(TestGraph&&) = default;
  TestGraph& operator=(TestGraph&&) = default;

 protected:
  explicit TestGraph(LiteRtModelT::Ptr model) : model_(std::move(model)) {}

 private:
  LiteRtModelT::Ptr model_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_COMMON_H_
