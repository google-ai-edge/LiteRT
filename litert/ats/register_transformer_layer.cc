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

#include "litert/ats/register_transformer_layer.h"

#include <cstddef>
#include <type_traits>

#include "litert/ats/compile_fixture.h"
#include "litert/ats/configure.h"
#include "litert/ats/inference_fixture.h"
#include "litert/ats/register.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/test/generators/transformer_layer.h"

namespace litert::testing {
namespace {

template <typename Fixture>
void RegisterTransformerLayerImpl(const AtsConf& options, size_t& test_id,
                                  size_t iters,
                                  typename Fixture::Capture& cap) {
  // clang-format off
  RegisterCombinations<
      Fixture,
      TransformerLayer,
      TypeList<std::integral_constant<AttentionType, AttentionType::kMHA>,
               std::integral_constant<AttentionType, AttentionType::kGQA>>,
      TypeList<std::integral_constant<NormType, NormType::kLayerNorm>,
               std::integral_constant<NormType, NormType::kRMSNorm>>,
      TypeList<std::integral_constant<FfnType, FfnType::kStandard>,
               std::integral_constant<FfnType, FfnType::kSwiGLU>>,
      TypeList<float>>
    (iters, test_id, options, cap, "Transformer");
  // clang-format on
}

}  // namespace

void RegisterTransformerLayer(const AtsConf& options, size_t& test_id,
                              size_t iters, AtsInferenceTest::Capture& cap) {
  RegisterTransformerLayerImpl<AtsInferenceTest>(options, test_id, iters, cap);
}

void RegisterTransformerLayer(const AtsConf& options, size_t& test_id,
                              size_t iters, AtsCompileTest::Capture& cap) {
  RegisterTransformerLayerImpl<AtsCompileTest>(options, test_id, iters, cap);
}

}  // namespace litert::testing
