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

#include "litert/ats/register_sdpa.h"

#include <cstddef>
#include <type_traits>

#include "litert/ats/compile_fixture.h"
#include "litert/ats/configure.h"
#include "litert/ats/inference_fixture.h"
#include "litert/ats/register.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/test/generators/sdpa.h"
#include "tflite/types/half.h"

namespace litert::testing {
namespace {

template <typename Fixture>
void RegisterSdpaImpl(const AtsConf& options, size_t& test_id, size_t iters,
                      typename Fixture::Capture& cap) {
  // clang-format off
  RegisterCombinations<
      Fixture,
      Sdpa,
      TypeList<float, tflite::half>,
      TypeList<std::false_type, std::true_type>,
      TypeList<std::false_type, std::true_type>>
    (iters, test_id, options, cap);
  // clang-format on
}

}  // namespace

void RegisterSdpa(const AtsConf& options, size_t& test_id, size_t iters,
                  AtsInferenceTest::Capture& cap) {
  RegisterSdpaImpl<AtsInferenceTest>(options, test_id, iters, cap);
}

void RegisterSdpa(const AtsConf& options, size_t& test_id, size_t iters,
                  AtsCompileTest::Capture& cap) {
  RegisterSdpaImpl<AtsCompileTest>(options, test_id, iters, cap);
}

}  // namespace litert::testing
