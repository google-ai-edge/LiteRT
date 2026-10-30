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

#include "litert/ats/register_reduction.h"

#include <cstddef>
#include <type_traits>

#include "litert/ats/compile_fixture.h"
#include "litert/ats/configure.h"
#include "litert/ats/inference_fixture.h"
#include "litert/ats/register.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/reduction.h"

namespace litert::testing {
namespace {

template <typename Fixture>
void RegisterReductionImpl(const AtsConf& options, size_t& test_id,
                           size_t iters, typename Fixture::Capture& cap) {
  // clang-format off
  RegisterCombinations<
      Fixture,
      Reduction,
      SizeListC<1, 2, 3, 4>,
      TypeList<float>,
      OpCodeListC<kLiteRtOpCodeTflReduceMax, kLiteRtOpCodeTflReduceMin,
                  kLiteRtOpCodeTflReduceProd, kLiteRtOpCodeTflSum>,
      TypeList<std::true_type, std::false_type>>
    (iters, test_id, options, cap);
  // clang-format on
}

}  // namespace

void RegisterReduction(const AtsConf& options, size_t& test_id, size_t iters,
                       AtsInferenceTest::Capture& cap) {
  RegisterReductionImpl<AtsInferenceTest>(options, test_id, iters, cap);
}

void RegisterReduction(const AtsConf& options, size_t& test_id, size_t iters,
                       AtsCompileTest::Capture& cap) {
  RegisterReductionImpl<AtsCompileTest>(options, test_id, iters, cap);
}

}  // namespace litert::testing
