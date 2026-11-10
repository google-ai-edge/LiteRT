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

#include "litert/ats/register_transpose.h"

#include <cstddef>
#include <cstdint>

#include "litert/ats/compile_fixture.h"
#include "litert/ats/configure.h"
#include "litert/ats/inference_fixture.h"
#include "litert/ats/register.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/transpose.h"
#include "tflite/types/half.h"

namespace litert::testing {
namespace {

template <typename Fixture>
void RegisterTransposeImpl(const AtsConf& options, size_t& test_id,
                           size_t iters, typename Fixture::Capture& cap) {
  // clang-format off
  RegisterCombinations<
      Fixture,
      Transpose,
      SizeListC<1, 2, 3, 4, 5, 6, 7, 8>,
      TypeList<
          bool,
          int8_t,
          uint8_t,
          int16_t,
          uint16_t,
          int32_t,
          uint32_t,
          int64_t,
          uint64_t,
          float,
          tflite::half>>
    (iters, test_id, options, cap);
  RegisterCombinations<
      Fixture,
      TransposeInt4,
      SizeListC<1, 2, 3, 4, 5, 6, 7, 8>>
    (iters, test_id, options, cap);
  // clang-format on
}

}  // namespace

void RegisterTranspose(const AtsConf& options, size_t& test_id, size_t iters,
                       AtsInferenceTest::Capture& cap) {
  RegisterTransposeImpl<AtsInferenceTest>(options, test_id, iters, cap);
}

void RegisterTranspose(const AtsConf& options, size_t& test_id, size_t iters,
                       AtsCompileTest::Capture& cap) {
  RegisterTransposeImpl<AtsCompileTest>(options, test_id, iters, cap);
}

}  // namespace litert::testing
