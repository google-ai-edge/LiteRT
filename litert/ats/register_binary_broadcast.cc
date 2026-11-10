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

#include "litert/ats/register_binary_broadcast.h"

#include <cstddef>

#include "litert/ats/compile_fixture.h"
#include "litert/ats/configure.h"
#include "litert/ats/inference_fixture.h"
#include "litert/ats/register.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/test/generators/binary_broadcast.h"
#include "litert/test/generators/common.h"
#include "tflite/schema/schema_generated.h"

namespace litert::testing {
namespace {

template <typename Fixture>
void RegisterBinaryBroadcastImpl(const AtsConf& options, size_t& test_id,
                                 size_t iters, typename Fixture::Capture& cap) {
  // clang-format off
  RegisterCombinations<
      Fixture,
      BinaryBroadcast,
      SizeListC<1, 2, 3, 4>,
      SizeListC<1, 2, 3, 4>,
      TypeList<float>,
      OpCodeListC<kLiteRtOpCodeTflAdd, kLiteRtOpCodeTflMul,
                  kLiteRtOpCodeTflSub, kLiteRtOpCodeTflDiv>,
      TypeList<FaC<tflite::ActivationFunctionType_NONE>,
               FaC<tflite::ActivationFunctionType_RELU>>>
    (iters, test_id, options, cap);

  RegisterCombinations<
      Fixture,
      BinaryBroadcast,
      SizeListC<1, 2, 3, 4>,
      SizeListC<1, 2, 3, 4>,
      TypeList<float>,
      OpCodeListC<kLiteRtOpCodeTflMaximum, kLiteRtOpCodeTflMinimum,
                  kLiteRtOpCodeTflSquaredDifference,
                  kLiteRtOpCodeTflFloorDiv, kLiteRtOpCodeTflPow>,
      TypeList<FaC<tflite::ActivationFunctionType_NONE>>>
    (iters, test_id, options, cap);

  // Prelu requires Rank2 <= Rank1!
  // We generate all valid pairs (R1 >= R2) up to rank 4.
  RegisterCombinations<
      Fixture,
      BinaryBroadcast,
      SizeListC<1, 2, 3, 4>,
      SizeListC<1>,
      TypeList<float>,
      OpCodeListC<kLiteRtOpCodeTflPrelu>,
      TypeList<FaC<tflite::ActivationFunctionType_NONE>>>
    (iters, test_id, options, cap);
  RegisterCombinations<
      Fixture,
      BinaryBroadcast,
      SizeListC<2, 3, 4>,
      SizeListC<2>,
      TypeList<float>,
      OpCodeListC<kLiteRtOpCodeTflPrelu>,
      TypeList<FaC<tflite::ActivationFunctionType_NONE>>>
    (iters, test_id, options, cap);
  RegisterCombinations<
      Fixture,
      BinaryBroadcast,
      SizeListC<3, 4>,
      SizeListC<3>,
      TypeList<float>,
      OpCodeListC<kLiteRtOpCodeTflPrelu>,
      TypeList<FaC<tflite::ActivationFunctionType_NONE>>>
    (iters, test_id, options, cap);
  RegisterCombinations<
      Fixture,
      BinaryBroadcast,
      SizeListC<4>,
      SizeListC<4>,
      TypeList<float>,
      OpCodeListC<kLiteRtOpCodeTflPrelu>,
      TypeList<FaC<tflite::ActivationFunctionType_NONE>>>
    (iters, test_id, options, cap);
  // clang-format on
}

}  // namespace

void RegisterBinaryBroadcast(const AtsConf& options, size_t& test_id,
                             size_t iters, AtsInferenceTest::Capture& cap) {
  RegisterBinaryBroadcastImpl<AtsInferenceTest>(options, test_id, iters, cap);
}

void RegisterBinaryBroadcast(const AtsConf& options, size_t& test_id,
                             size_t iters, AtsCompileTest::Capture& cap) {
  RegisterBinaryBroadcastImpl<AtsCompileTest>(options, test_id, iters, cap);
}

}  // namespace litert::testing
