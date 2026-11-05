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

#include "tensor/examples/gemma3/model_config.h"

#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "tensor/examples/gemma3/gemma3_config.h"

namespace litert::tensor::examples {

absl::StatusOr<Gemma3ModelVariant> ParseGemma3ModelVariant(
    absl::string_view variant) {
  std::string normalized = std::string(variant);
  absl::AsciiStrToLower(&normalized);
  if (normalized == "auto" || normalized == "infer") {
    return Gemma3ModelVariant::kAuto;
  }
  if (normalized == "270m" || normalized == "gemma3_270m" ||
      normalized == "gemma-3-270m" || normalized == "gemma-3-270m-it") {
    return Gemma3ModelVariant::k270M;
  }
  if (normalized == "1b" || normalized == "gemma3_1b" ||
      normalized == "gemma-3-1b" || normalized == "gemma-3-1b-it") {
    return Gemma3ModelVariant::k1B;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unsupported model variant: ", variant,
                   ". Expected one of: auto, 270m, 1b"));
}

absl::string_view Gemma3ModelVariantToString(Gemma3ModelVariant variant) {
  switch (variant) {
    case Gemma3ModelVariant::kAuto:
      return "auto";
    case Gemma3ModelVariant::k270M:
      return "270m";
    case Gemma3ModelVariant::k1B:
      return "1b";
  }
  return "unknown";
}

Gemma3Config GetGemma3BaseConfig(Gemma3ModelVariant variant) {
  switch (variant) {
    case Gemma3ModelVariant::k1B:
      return GetGemma3_1B_Config();
    case Gemma3ModelVariant::k270M:
    case Gemma3ModelVariant::kAuto:
      return GetGemma3_270M_Config();
  }
  return GetGemma3_270M_Config();
}

absl::StatusOr<Gemma3Config> ResolveGemma3Config(const TfliteLoader& loader,
                                                 Gemma3ModelVariant variant) {
  if (variant == Gemma3ModelVariant::kAuto) {
    return absl::InvalidArgumentError(
        "Automatic model variant inference is not supported for TFLite models. "
        "Please specify --model_variant explicitly (e.g., 1b or 270m).");
  }
  return GetGemma3BaseConfig(variant);
}

}  // namespace litert::tensor::examples
