// Copyright 2025 Google LLC.
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
#include "litert/c/options/litert_mediatek_options.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/litert_toml_parser.h"

struct LrtMediatekOptions {
  std::optional<LiteRtMediatekOptionsNeronSDKVersionType> neron_sdk_version;
  std::optional<bool> gemma_compiler_optimizations;
  std::optional<LiteRtMediatekNeuronAdapterPerformanceMode> performance_mode;
  std::optional<bool> l1_cache_optimizations;
  std::optional<LiteRtMediatekNeuronAdapterOptimizationHint> optimization_hint;
  std::optional<bool> disable_dla_dir_removal;
  std::string mediatek_dla_dir;
  std::string aot_compilation_options;
};

LiteRtStatus LrtCreateMediatekOptions(LrtMediatekOptions** options) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *options = new LrtMediatekOptions;
  return kLiteRtStatusOk;
}

void LrtDestroyMediatekOptions(LrtMediatekOptions* options) { delete options; }

LiteRtStatus LrtCreateMediatekOptionsFromToml(const char* toml_payload,
                                              LrtMediatekOptions** options) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  LITERT_RETURN_IF_ERROR(LrtCreateMediatekOptions(options));

  if (toml_payload == nullptr || toml_payload[0] == '\0') {
    return kLiteRtStatusOk;
  }

  auto status = ::litert::internal::ParseToml(
      toml_payload,
      [&](absl::string_view key, absl::string_view value) -> LiteRtStatus {
        if (key == "neron_sdk_version") {
          LITERT_ASSIGN_OR_RETURN(auto version,
                                  ::litert::internal::ParseTomlInt(value));
          return LrtSetMediatekOptionsNeronSDKVersionType(
              *options,
              static_cast<LiteRtMediatekOptionsNeronSDKVersionType>(version));
        }
        if (key == "gemma_compiler_optimizations") {
          LITERT_ASSIGN_OR_RETURN(auto gemma_opts,
                                  ::litert::internal::ParseTomlBool(value));
          return LrtSetMediatekOptionsGemmaCompilerOptimizations(*options,
                                                                 gemma_opts);
        }
        if (key == "performance_mode") {
          LITERT_ASSIGN_OR_RETURN(auto mode,
                                  ::litert::internal::ParseTomlInt(value));
          return LrtSetMediatekOptionsPerformanceMode(
              *options,
              static_cast<LiteRtMediatekNeuronAdapterPerformanceMode>(mode));
        }
        if (key == "l1_cache_optimizations") {
          LITERT_ASSIGN_OR_RETURN(auto l1,
                                  ::litert::internal::ParseTomlBool(value));
          return LrtSetMediatekOptionsL1CacheOptimizations(*options, l1);
        }
        if (key == "optimization_hint") {
          LITERT_ASSIGN_OR_RETURN(auto hint,
                                  ::litert::internal::ParseTomlInt(value));
          return LrtSetMediatekOptionsOptimizationHint(
              *options,
              static_cast<LiteRtMediatekNeuronAdapterOptimizationHint>(hint));
        }
        if (key == "disable_dla_dir_removal") {
          LITERT_ASSIGN_OR_RETURN(auto disable,
                                  ::litert::internal::ParseTomlBool(value));
          return LrtSetMediatekOptionsDisableDlaDirRemoval(*options, disable);
        }
        if (key == "mediatek_dla_dir") {
          return LrtSetMediatekOptionsMediatekDlaDir(
              *options, std::string(value).c_str());
        }
        if (key == "aot_compilation_options") {
          return LrtSetMediatekOptionsAotCompilationOptions(
              *options, std::string(value).c_str());
        }

        // Ignore unknown keys to allow for forward compatibility.
        return kLiteRtStatusOk;
      });

  if (status != kLiteRtStatusOk) {
    LrtDestroyMediatekOptions(*options);
    *options = nullptr;
  }
  return status;
}

LiteRtStatus LrtGetOpaqueMediatekOptionsData(const LrtMediatekOptions* options,
                                             const char** identifier,
                                             void** payload,
                                             void (**payload_deleter)(void*)) {
  if (options == nullptr || identifier == nullptr || payload == nullptr ||
      payload_deleter == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  // Construct a TOML string from the options.
  std::string toml_str;
  if (options->neron_sdk_version.has_value()) {
    absl::StrAppendFormat(&toml_str, "neron_sdk_version = %d\n",
                          static_cast<int>(*options->neron_sdk_version));
  }
  if (options->gemma_compiler_optimizations.has_value()) {
    absl::StrAppendFormat(
        &toml_str, "gemma_compiler_optimizations = %s\n",
        *options->gemma_compiler_optimizations ? "true" : "false");
  }
  if (options->performance_mode.has_value()) {
    absl::StrAppendFormat(&toml_str, "performance_mode = %d\n",
                          static_cast<int>(*options->performance_mode));
  }
  if (options->l1_cache_optimizations.has_value()) {
    absl::StrAppendFormat(&toml_str, "l1_cache_optimizations = %s\n",
                          *options->l1_cache_optimizations ? "true" : "false");
  }
  if (options->optimization_hint.has_value()) {
    absl::StrAppendFormat(&toml_str, "optimization_hint = %d\n",
                          static_cast<int>(*options->optimization_hint));
  }
  if (options->disable_dla_dir_removal.has_value()) {
    absl::StrAppendFormat(&toml_str, "disable_dla_dir_removal = %s\n",
                          *options->disable_dla_dir_removal ? "true" : "false");
  }
  if (!options->mediatek_dla_dir.empty()) {
    absl::StrAppendFormat(&toml_str, "mediatek_dla_dir = \"%s\"\n",
                          options->mediatek_dla_dir);
  }
  if (!options->aot_compilation_options.empty()) {
    absl::StrAppendFormat(&toml_str, "aot_compilation_options = \"%s\"\n",
                          options->aot_compilation_options);
  }

  *identifier = "mediatek";
  *payload = new std::string(toml_str);
  *payload_deleter = [](void* payload) {
    delete reinterpret_cast<std::string*>(payload);
  };
  return kLiteRtStatusOk;
}

// COMPILATION OPTIONS /////////////////////////////////////////////////////////
// sdk_version_type ----------------------------------------------------------
LiteRtStatus LrtSetMediatekOptionsNeronSDKVersionType(
    LrtMediatekOptions* options,
    LiteRtMediatekOptionsNeronSDKVersionType sdk_version_type) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->neron_sdk_version = sdk_version_type;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetMediatekOptionsNeronSDKVersionType(
    const LrtMediatekOptions* options,
    LiteRtMediatekOptionsNeronSDKVersionType* sdk_version_type) {
  if (options == nullptr || sdk_version_type == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *sdk_version_type = options->neron_sdk_version.value_or(
      kLiteRtMediatekOptionsNeronSDKVersionTypeVersion8);
  return kLiteRtStatusOk;
}

// gemma_compiler_optimizations ---------------------------------------------
LiteRtStatus LrtSetMediatekOptionsGemmaCompilerOptimizations(
    LrtMediatekOptions* options, bool gemma_compiler_optimizations) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->gemma_compiler_optimizations = gemma_compiler_optimizations;

  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetMediatekOptionsGemmaCompilerOptimizations(
    const LrtMediatekOptions* options, bool* gemma_compiler_optimizations) {
  if (gemma_compiler_optimizations == nullptr || options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *gemma_compiler_optimizations =
      options->gemma_compiler_optimizations.value_or(false);
  return kLiteRtStatusOk;
}

// neuron_adapter_peformance_mode --------------------------------------------
LiteRtStatus LrtSetMediatekOptionsPerformanceMode(
    LrtMediatekOptions* options,
    LiteRtMediatekNeuronAdapterPerformanceMode performance_mode) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->performance_mode = performance_mode;

  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetMediatekOptionsPerformanceMode(
    const LrtMediatekOptions* options,
    LiteRtMediatekNeuronAdapterPerformanceMode* performance_mode) {
  if (options == nullptr || performance_mode == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *performance_mode = options->performance_mode.value_or(
      kLiteRtMediatekNeuronAdapterPerformanceModeNeuronPreferSustainedSpeed);
  return kLiteRtStatusOk;
}

// l1_cache_optimizations ----------------------------------------------------
LiteRtStatus LrtSetMediatekOptionsL1CacheOptimizations(
    LrtMediatekOptions* options, bool l1_cache_optimizations) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->l1_cache_optimizations = l1_cache_optimizations;

  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetMediatekOptionsL1CacheOptimizations(
    const LrtMediatekOptions* options, bool* l1_cache_optimizations) {
  if (l1_cache_optimizations == nullptr || options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *l1_cache_optimizations = options->l1_cache_optimizations.value_or(false);
  return kLiteRtStatusOk;
}

// neuron_optimization_hints -------------------------------------------------
LiteRtStatus LrtSetMediatekOptionsOptimizationHint(
    LrtMediatekOptions* options,
    LiteRtMediatekNeuronAdapterOptimizationHint optimization_hint) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->optimization_hint = optimization_hint;

  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetMediatekOptionsOptimizationHint(
    const LrtMediatekOptions* options,
    LiteRtMediatekNeuronAdapterOptimizationHint* optimization_hint) {
  if (options == nullptr || optimization_hint == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *optimization_hint = options->optimization_hint.value_or(
      kLiteRtMediatekNeuronAdapterOptimizationHintNormal);
  return kLiteRtStatusOk;
}

// disable_dla_dir_removal ---------------------------------------------------
LiteRtStatus LrtSetMediatekOptionsDisableDlaDirRemoval(
    LrtMediatekOptions* options, bool disable_dla_dir_removal) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->disable_dla_dir_removal = disable_dla_dir_removal;

  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetMediatekOptionsDisableDlaDirRemoval(
    const LrtMediatekOptions* options, bool* disable_dla_dir_removal) {
  if (disable_dla_dir_removal == nullptr || options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *disable_dla_dir_removal = options->disable_dla_dir_removal.value_or(false);
  return kLiteRtStatusOk;
}

// mediatek_dla_dir -------------------------------------------------
LiteRtStatus LrtSetMediatekOptionsMediatekDlaDir(LrtMediatekOptions* options,
                                                 const char* mediatek_dla_dir) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->mediatek_dla_dir = mediatek_dla_dir;

  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetMediatekOptionsMediatekDlaDir(
    const LrtMediatekOptions* options, const char** mediatek_dla_dir) {
  if (options == nullptr || mediatek_dla_dir == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *mediatek_dla_dir = options->mediatek_dla_dir.c_str();

  return kLiteRtStatusOk;
}

// AoT compilation options --------------------------------------------
LiteRtStatus LrtSetMediatekOptionsAotCompilationOptions(
    LrtMediatekOptions* options, const char* aot_compilation_options) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->aot_compilation_options = aot_compilation_options;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetMediatekOptionsAotCompilationOptions(
    const LrtMediatekOptions* options, const char** aot_compilation_options) {
  if (options == nullptr || aot_compilation_options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *aot_compilation_options = options->aot_compilation_options.c_str();
  return kLiteRtStatusOk;
}
