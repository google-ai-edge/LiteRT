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

#include "litert/ats/register_moe_experts.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "litert/ats/compile_fixture.h"
#include "litert/ats/configure.h"
#include "litert/ats/inference_fixture.h"
#include "litert/ats/register.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/test/generators/moe_experts.h"
#include "tensor/datatypes.h"
#include "tflite/types/half.h"

namespace litert::testing {
namespace {

template <typename Fixture>
void RegisterMoeExpertsImpl(const AtsConf& options, size_t& test_id,
                            size_t iters, typename Fixture::Capture& cap) {
  // 7-input layout (per-channel affine quantization on weights).
  RegisterCombinations<
      Fixture, MoeExperts, TypeList<float, tflite::half>,
      TypeList<float, tflite::half, int8_t, litert::tensor::int4_t,
               litert::tensor::int2_t>,
      TypeList<std::false_type>>(iters, test_id, options, cap);

  // 10-input layout (explicit per-expert scale tensors).
  RegisterCombinations<
      Fixture, MoeExperts, TypeList<float, tflite::half>,
      TypeList<int8_t, litert::tensor::int4_t, litert::tensor::int2_t>,
      TypeList<std::true_type>>(iters, test_id, options, cap);
}

}  // namespace

void RegisterMoeExperts(const AtsConf& options, size_t& test_id, size_t iters,
                        AtsInferenceTest::Capture& cap) {
  RegisterMoeExpertsImpl<AtsInferenceTest>(options, test_id, iters, cap);
}

void RegisterMoeExperts(const AtsConf& options, size_t& test_id, size_t iters,
                        AtsCompileTest::Capture& cap) {
  RegisterMoeExpertsImpl<AtsCompileTest>(options, test_id, iters, cap);
}

}  // namespace litert::testing
