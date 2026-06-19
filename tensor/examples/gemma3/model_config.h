/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA3_MODEL_CONFIG_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA3_MODEL_CONFIG_H_

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "tensor/examples/gemma3/gemma3_graph.h"
#include "tensor/examples/gemma3/safetensor_loader.h"

namespace litert::tensor::examples {

class TfliteLoader;

enum class Gemma3ModelVariant {
  kAuto,
  k270M,
  k1B,
};

absl::StatusOr<Gemma3ModelVariant> ParseGemma3ModelVariant(
    absl::string_view variant);

absl::string_view Gemma3ModelVariantToString(Gemma3ModelVariant variant);

Gemma3Config GetGemma3BaseConfig(Gemma3ModelVariant variant);

absl::StatusOr<Gemma3Config> InferGemma3ConfigFromLoader(
    const SafetensorLoader& loader, const Gemma3Config& fallback);

absl::StatusOr<Gemma3Config> ResolveGemma3Config(const SafetensorLoader& loader,
                                                 Gemma3ModelVariant variant);

absl::StatusOr<Gemma3Config> ResolveGemma3Config(const TfliteLoader& loader,
                                                 Gemma3ModelVariant variant);

}  // namespace litert::tensor::examples

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA3_MODEL_CONFIG_H_
