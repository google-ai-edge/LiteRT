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

#include "litert/ats/register_runtime_bmm.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "litert/ats/compile_fixture.h"
#include "litert/ats/configure.h"
#include "litert/ats/inference_fixture.h"
#include "litert/ats/register.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/test/generators/runtime_bmm.h"
#include "tensor/datatypes.h"
#include "tflite/types/half.h"

namespace litert::testing {
namespace {

template <typename Fixture>
void RegisterRuntimeBmmImpl(const AtsConf& options, size_t& test_id,
                            size_t iters, typename Fixture::Capture& cap) {
  // clang-format off
  RegisterCombinations<
      Fixture,
      RuntimeBmm,
      TypeList<TypeTuple<float, float>,
               TypeTuple<tflite::half, tflite::half>,
               TypeTuple<float, int8_t>,
               TypeTuple<float, litert::tensor::int4_t>>,
      TypeList<std::false_type, std::true_type>>
    (iters, test_id, options, cap);
  // clang-format on
}

}  // namespace

void RegisterRuntimeBmm(const AtsConf& options, size_t& test_id, size_t iters,
                        AtsInferenceTest::Capture& cap) {
  RegisterRuntimeBmmImpl<AtsInferenceTest>(options, test_id, iters, cap);
}

void RegisterRuntimeBmm(const AtsConf& options, size_t& test_id, size_t iters,
                        AtsCompileTest::Capture& cap) {
  RegisterRuntimeBmmImpl<AtsCompileTest>(options, test_id, iters, cap);
}

}  // namespace litert::testing
