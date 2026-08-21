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

#include "litert/ats/register_one_hot.h"

#include <cstddef>

#include "litert/ats/compile_fixture.h"
#include "litert/ats/configure.h"
#include "litert/ats/inference_fixture.h"
#include "litert/ats/register.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/one_hot.h"

namespace litert::testing {
namespace {

template <typename Fixture>
void RegisterOneHotImpl(const AtsConf& options, size_t& test_id, size_t iters,
                        typename Fixture::Capture& cap) {
  // clang-format off
  // Rank 1
  RegisterCombinations<
      Fixture,
      OneHot,
      TypeList<float>,
      SizeListC<1>,
      SizeListC<0, 1>>
    (iters, test_id, options, cap);

  // Rank 2
  RegisterCombinations<
      Fixture,
      OneHot,
      TypeList<float>,
      SizeListC<2>,
      SizeListC<0, 1, 2>>
    (iters, test_id, options, cap);

  // Rank 3
  RegisterCombinations<
      Fixture,
      OneHot,
      TypeList<float>,
      SizeListC<3>,
      SizeListC<0, 1, 2, 3>>
    (iters, test_id, options, cap);

  // Rank 4
  RegisterCombinations<
      Fixture,
      OneHot,
      TypeList<float>,
      SizeListC<4>,
      SizeListC<0, 1, 2, 3, 4>>
    (iters, test_id, options, cap);
  // clang-format on
}

}  // namespace

void RegisterOneHot(const AtsConf& options, size_t& test_id, size_t iters,
                    AtsInferenceTest::Capture& cap) {
  RegisterOneHotImpl<AtsInferenceTest>(options, test_id, iters, cap);
}

void RegisterOneHot(const AtsConf& options, size_t& test_id, size_t iters,
                    AtsCompileTest::Capture& cap) {
  RegisterOneHotImpl<AtsCompileTest>(options, test_id, iters, cap);
}

}  // namespace litert::testing
