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

#include "litert/ats/register_fully_connected.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "litert/ats/compile_fixture.h"
#include "litert/ats/configure.h"
#include "litert/ats/inference_fixture.h"
#include "litert/ats/register.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/fully_connected.h"
#include "tensor/arithmetic_graph.h"
#include "tflite/schema/schema_generated.h"
#include "tflite/types/half.h"

namespace litert::testing {
namespace {

template <typename Fixture>
void RegisterFullyConnectedImpl(const AtsConf& options, size_t& test_id,
                                size_t iters, typename Fixture::Capture& cap) {
  // clang-format off
  // Floating-Point Static
  RegisterCombinations<
      Fixture,
      FullyConnected,
      SizeListC<2, 3, 4>,
      TypeList<TypeTuple<float, float>,              // Uniform FP32
               TypeTuple<tflite::half, tflite::half>>,  // Uniform FP16
      OpCodeListC<kLiteRtOpCodeTflFullyConnected>,
      TypeList<std::true_type, std::false_type>,     // KeepNumDims
      TypeList<FaC<tflite::ActivationFunctionType_NONE>,
               FaC<tflite::ActivationFunctionType_RELU>>,  // FusedActivation
      TypeList<std::true_type, std::false_type>,     // HasBias
      TypeList<std::false_type>,                     // AsymmetricQuantizeInputs
      TypeList<std::integral_constant<
          litert::tensor::FullyConnectedWeightsFormat,
                                      litert::tensor::kWeightsFormatDefault>>,
      TypeList<std::false_type>,                     // PerChannel
      TypeList<std::false_type>,                     // DynamicFilter
      TypeList<std::false_type>>                     // DynamicBias
    (iters, test_id, options, cap);

  // Floating-Point Dynamic Filter & Bias
  RegisterCombinations<
      Fixture,
      FullyConnected,
      SizeListC<2, 3>,
      TypeList<TypeTuple<float, float>>,             // Uniform FP32
      OpCodeListC<kLiteRtOpCodeTflFullyConnected>,
      TypeList<std::false_type>,                     // KeepNumDims
      TypeList<FaC<tflite::ActivationFunctionType_NONE>>,  // FusedActivation
      TypeList<std::true_type>,                      // HasBias
      TypeList<std::false_type>,                     // AsymmetricQuantizeInputs
      TypeList<std::integral_constant<
          litert::tensor::FullyConnectedWeightsFormat,
                             litert::tensor::kWeightsFormatDefault>>,
      TypeList<std::false_type>,                     // PerChannel
      TypeList<std::true_type, std::false_type>,     // DynamicFilter
      TypeList<std::true_type, std::false_type>>     // DynamicBias
    (iters, test_id, options, cap);

  // Hybrid Quantization (FP32 activations x INT8 weights)
  RegisterCombinations<
      Fixture,
      FullyConnected,
      SizeListC<2, 3>,
      TypeList<TypeTuple<float, float>>,             // T_in=float, T_out=float
      OpCodeListC<kLiteRtOpCodeTflFullyConnected>,
      TypeList<std::false_type>,                     // KeepNumDims
      TypeList<FaC<tflite::ActivationFunctionType_NONE>>,  // FusedActivation
      TypeList<std::true_type>,                      // HasBias
      TypeList<std::true_type, std::false_type>,     // AsymmetricQuantizeInputs
      TypeList<std::integral_constant<
          litert::tensor::FullyConnectedWeightsFormat,
          litert::tensor::kWeightsFormatDefault>>,
      TypeList<std::true_type, std::false_type>,     // PerChannel
      TypeList<std::false_type>,                     // DynamicFilter
      TypeList<std::false_type>>                     // DynamicBias
    (iters, test_id, options, cap);

  // Full Integer Quantization (INT8/UINT8 activations & weights)
  RegisterCombinations<
      Fixture,
      FullyConnected,
      SizeListC<2, 3>,
      TypeList<TypeTuple<int8_t, int8_t>,            // INT8 in/wt, INT8 out
               TypeTuple<uint8_t, uint8_t>>,         // UINT8 in/wt, UINT8 out
      OpCodeListC<kLiteRtOpCodeTflFullyConnected>,
      TypeList<std::true_type, std::false_type>,     // KeepNumDims
      TypeList<FaC<tflite::ActivationFunctionType_NONE>,
               FaC<tflite::ActivationFunctionType_RELU>>,  // FusedActivation
      TypeList<std::true_type, std::false_type>,     // HasBias
      TypeList<std::false_type>,                     // AsymmetricQuantizeInputs
      TypeList<std::integral_constant<
          litert::tensor::FullyConnectedWeightsFormat,
                                      litert::tensor::kWeightsFormatDefault>,
               std::integral_constant<
                   litert::tensor::FullyConnectedWeightsFormat,
                   litert::tensor::kWeightsFormatShuffled4x16Int8>>,
      TypeList<std::true_type, std::false_type>,     // PerChannel
      TypeList<std::false_type>,                     // DynamicFilter
      TypeList<std::false_type>>                     // DynamicBias
    (iters, test_id, options, cap);
  // clang-format on
}

}  // namespace

void RegisterFullyConnected(const AtsConf& options, size_t& test_id,
                            size_t iters, AtsInferenceTest::Capture& cap) {
  RegisterFullyConnectedImpl<AtsInferenceTest>(options, test_id, iters, cap);
}

void RegisterFullyConnected(const AtsConf& options, size_t& test_id,
                            size_t iters, AtsCompileTest::Capture& cap) {
  RegisterFullyConnectedImpl<AtsCompileTest>(options, test_id, iters, cap);
}

}  // namespace litert::testing
