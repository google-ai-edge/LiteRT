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
#include <tuple>
#include <type_traits>
#include <utility>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_detail.h"
#include "litert/cc/litert_rng.h"
#include "litert/test/simple_buffer.h"
#include "tflite/schema/schema_generated.h"

namespace litert::testing {

// Helpers for defining generators and their consituent types.

// Shorthand for specifying i/o tensor names as constexpr.
template <size_t N>
using TensorNames = std::array<absl::string_view, N>;

// Traits that configure behavior for random tensor data generation. These
// are attached to generators.
template <typename G>
struct RandomTensorBufferTraits {
  using Gen = G;
};

// The default random generator traits that can be used unless a specific
// distribution needs to be configured.
template <typename T>
using DefaultRandomTensorBufferTraits =
    RandomTensorBufferTraits<RandomTensorData<T>>;

// Convenience type for mapping litert ops to the related flatbuffer types.
template <typename Options, ::tflite::BuiltinOperator BuiltinOperator,
          ::tflite::BuiltinOptions BuiltinOptions>
struct FbOpTraits {
  using OptionsT = Options;
  static constexpr tflite::BuiltinOperator kBuiltinOperator = BuiltinOperator;
  static constexpr tflite::BuiltinOptions kBuiltinOptions = BuiltinOptions;
};

// Selects the appropriate fb traits based on the op code.
// TODO finish for other op codes.
template <LiteRtOpCode OpCode>
using FbOpTypes =
    SelectT<std::bool_constant<OpCode == kLiteRtOpCodeTflAdd>,
            FbOpTraits<tflite::AddOptionsT, tflite::BuiltinOperator_ADD,
                       tflite::BuiltinOptions_AddOptions>,
            std::bool_constant<OpCode == kLiteRtOpCodeTflSub>,
            FbOpTraits<tflite::SubOptionsT, tflite::BuiltinOperator_SUB,
                       tflite::BuiltinOptions_SubOptions>>;

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
  static ReferenceTensors MakeReferenceTensors(Buffers& inputs,
                                               std::index_sequence<Is...>) {
    return ReferenceTensors{inputs[Is].template AsView<InputDataType<Is>>()...};
  }
  template <typename ReferenceTensors, typename Buffers>
  static ReferenceTensors MakeReferenceTensors(Buffers& inputs) {
    return MakeReferenceTensors<ReferenceTensors>(
        inputs, std::make_index_sequence<kNumInputs>());
  }

 public:
  // Inputs for this test logic for the compiled model api.
  using InputBuffers = std::array<SimpleBuffer, InputTypes::kSize>;

  // Outputs for this test logic for the compiled model api.
  using OutputBuffers = std::array<SimpleBuffer, OutputTypes::kSize>;

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
  static ReferenceInputs MakeReferenceInputs(const InputBuffers& inputs) {
    return MakeReferenceTensors<ReferenceInputs>(inputs);
  }

  // Get the typed reference output views from the output buffers.
  static ReferenceOutputs MakeReferenceOutputs(OutputBuffers& outputs) {
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
template <size_t N>
using SizeC = std::integral_constant<size_t, N>;
template <LiteRtOpCode OpCode>
using OpCodeC = std::integral_constant<LiteRtOpCode, OpCode>;
template <size_t... Sizes>
using SizeListC = TypeList<SizeC<Sizes>...>;

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_COMMON_H_
