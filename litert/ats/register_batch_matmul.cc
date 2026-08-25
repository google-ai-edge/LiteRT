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

#include "litert/ats/register_batch_matmul.h"

#include <cstddef>
#include <type_traits>

#include "litert/ats/compile_fixture.h"
#include "litert/ats/configure.h"
#include "litert/ats/inference_fixture.h"
#include "litert/ats/register.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/test/generators/batch_matmul.h"
#include "litert/test/generators/common.h"
#include "tflite/types/half.h"

namespace litert::testing {
namespace {

template <typename Fixture>
void RegisterBatchMatmulImpl(const AtsConf& options, size_t& test_id,
                             size_t iters, typename Fixture::Capture& cap) {
  // clang-format off
  RegisterCombinations<
      Fixture,
      BatchMatmul,
      SizeListC<2, 3, 4>,
      SizeListC<2, 3, 4>,
      TypeList<TypeTuple<float, float>,
               TypeTuple<tflite::half, tflite::half>>,
      TypeList<std::true_type, std::false_type>,
      TypeList<std::true_type, std::false_type>>
    (iters, test_id, options, cap);
  // clang-format on
}

}  // namespace

void RegisterBatchMatmul(const AtsConf& options, size_t& test_id, size_t iters,
                         AtsInferenceTest::Capture& cap) {
  RegisterBatchMatmulImpl<AtsInferenceTest>(options, test_id, iters, cap);
}

void RegisterBatchMatmul(const AtsConf& options, size_t& test_id, size_t iters,
                         AtsCompileTest::Capture& cap) {
  RegisterBatchMatmulImpl<AtsCompileTest>(options, test_id, iters, cap);
}

}  // namespace litert::testing
