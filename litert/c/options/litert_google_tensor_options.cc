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

#include "litert/c/options/litert_google_tensor_options.h"

#include <cstddef>
#include <iterator>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/btree_map.h"  // from @com_google_absl
#include "absl/strings/escaping.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/strip.h"  // from @com_google_absl
#include "litert/c/internal/litert_options_helper.h"
#include "litert/c/litert_common.h"
#include "litert/c/options/litert_google_tensor_options_type.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/litert_toml_parser.h"

// copybara:uncomment_begin(google-only)
// constexpr char kExperimentalEnableInputValidatorKey[] =
//     "experimental_enable_input_validator";
// copybara:uncomment_end

struct LrtGoogleTensorOptionsT {
  LrtGoogleTensorOptionsTruncationType float_truncation_type =
      kLiteRtGoogleTensorFloatTruncationTypeAuto;
  bool int64_to_int32_truncation = false;
  std::string output_dir = "";
  bool dump_op_timings = false;
  bool enable_large_model_support = false;
  bool enable_4bit_compilation = false;
  LrtGoogleTensorOptionsShardingIntensity sharding_intensity =
      kLiteRtGoogleTensorShardingIntensityUnspecified;
  bool enable_dynamic_range_quantization = false;
  std::optional<LiteRtGoogleTensorOptionsPerformanceMode> performance_mode =
      std::nullopt;
  std::string op_filters_proto = "";
  std::string extra_options_path = "";
  // Map from (signature_name, tensor_name) to coherency preference (true =
  // coherent).
  absl::btree_map<std::pair<std::string, std::string>, bool> input_coherency;
  // Map from (signature_name, tensor_name) to coherency preference (true =
  // coherent).
  absl::btree_map<std::pair<std::string, std::string>, bool> output_coherency;
  std::string extra_options = "";
  // copybara:uncomment_begin(google-only)
  // bool experimental_enable_input_validator = false;
  // copybara:uncomment_end
};

LiteRtStatus LrtCreateGoogleTensorOptions(LrtGoogleTensorOptions* options) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *options = new LrtGoogleTensorOptionsT();
  return kLiteRtStatusOk;
}

void LrtDestroyGoogleTensorOptions(LrtGoogleTensorOptions options) {
  delete options;
}

const char* LrtGoogleTensorOptionsGetIdentifier() { return "google_tensor"; }

LiteRtStatus LrtGetOpaqueGoogleTensorOptionsData(
    LrtGoogleTensorOptions options, const char** identifier, void** payload,
    void (**payload_deleter)(void*)) {
  if (options == nullptr || identifier == nullptr || payload == nullptr ||
      payload_deleter == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  std::string toml_str;
  if (options->float_truncation_type !=
      kLiteRtGoogleTensorFloatTruncationTypeAuto) {
    absl::StrAppendFormat(&toml_str, "float_truncation_type = %d\n",
                          static_cast<int>(options->float_truncation_type));
  }
  if (options->int64_to_int32_truncation) {
    absl::StrAppendFormat(&toml_str, "int64_to_int32_truncation = true\n");
  }
  if (!options->output_dir.empty()) {
    absl::StrAppendFormat(&toml_str, "output_dir = \"%s\"\n",
                          options->output_dir);
  }
  if (options->dump_op_timings) {
    absl::StrAppendFormat(&toml_str, "dump_op_timings = true\n");
  }
  if (options->enable_large_model_support) {
    absl::StrAppendFormat(&toml_str, "enable_large_model_support = true\n");
  }
  if (options->enable_4bit_compilation) {
    absl::StrAppendFormat(&toml_str, "enable_four_bit_compilation = true\n");
  }
  if (options->sharding_intensity !=
      kLiteRtGoogleTensorShardingIntensityUnspecified) {
    absl::StrAppendFormat(&toml_str, "sharding_intensity = %d\n",
                          static_cast<int>(options->sharding_intensity));
  }
  if (options->enable_dynamic_range_quantization) {
    absl::StrAppendFormat(&toml_str,
                          "enable_dynamic_range_quantization = true\n");
  }
  if (options->performance_mode.has_value()) {
    absl::StrAppendFormat(&toml_str, "performance_mode = %d\n",
                          static_cast<int>(*options->performance_mode));
  }

  if (!options->op_filters_proto.empty()) {
    absl::StrAppendFormat(&toml_str, "op_filters_proto = \"%s\"\n",
                          absl::Base64Escape(options->op_filters_proto));
  }
  if (!options->extra_options_path.empty()) {
    absl::StrAppendFormat(&toml_str, "extra_options_path = \"%s\"\n",
                          options->extra_options_path);
  }
  if (!options->extra_options.empty()) {
    absl::StrAppendFormat(&toml_str, "extra_options = \"%s\"\n",
                          absl::Base64Escape(options->extra_options));
  }
  for (const auto& [key, pref] : options->input_coherency) {
    absl::StrAppendFormat(&toml_str, "input_coherency_%s:%s = %s\n", key.first,
                          key.second, pref ? "true" : "false");
  }
  for (const auto& [key, pref] : options->output_coherency) {
    absl::StrAppendFormat(&toml_str, "output_coherency_%s:%s = %s\n", key.first,
                          key.second, pref ? "true" : "false");
  }

  // copybara:uncomment_begin(google-only)
  // if (options->experimental_enable_input_validator) {
    // absl::StrAppendFormat(&toml_str, "%s = true\n",
                          // kExperimentalEnableInputValidatorKey);
  // }
  // copybara:uncomment_end

  *identifier = LrtGoogleTensorOptionsGetIdentifier();
  litert::internal::MakeCStringPayload(toml_str, payload, payload_deleter);
  return kLiteRtStatusOk;
}

LiteRtStatus LrtCreateGoogleTensorOptionsFromToml(
    const char* toml_payload, LrtGoogleTensorOptions* options) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  LITERT_RETURN_IF_ERROR(LrtCreateGoogleTensorOptions(options));

  if (toml_payload == nullptr || toml_payload[0] == '\0') {
    return kLiteRtStatusOk;
  }

  LrtGoogleTensorOptionsT& options_ref = **options;

  auto status = litert::internal::ParseToml(
      toml_payload,
      [&options_ref](absl::string_view key,
                     absl::string_view value) -> LiteRtStatus {
        if (key == "float_truncation_type") {
          LITERT_ASSIGN_OR_RETURN(auto val,
                                  litert::internal::ParseTomlInt(value));
          options_ref.float_truncation_type =
              static_cast<LrtGoogleTensorOptionsTruncationType>(val);
        } else if (key == "int64_to_int32_truncation") {
          LITERT_ASSIGN_OR_RETURN(options_ref.int64_to_int32_truncation,
                                  litert::internal::ParseTomlBool(value));
        } else if (key == "output_dir") {
          options_ref.output_dir = std::string(value);
        } else if (key == "dump_op_timings") {
          LITERT_ASSIGN_OR_RETURN(options_ref.dump_op_timings,
                                  litert::internal::ParseTomlBool(value));
        } else if (key == "enable_large_model_support") {
          LITERT_ASSIGN_OR_RETURN(options_ref.enable_large_model_support,
                                  litert::internal::ParseTomlBool(value));
        } else if (key == "enable_four_bit_compilation") {
          LITERT_ASSIGN_OR_RETURN(options_ref.enable_4bit_compilation,
                                  litert::internal::ParseTomlBool(value));
        } else if (key == "sharding_intensity") {
          LITERT_ASSIGN_OR_RETURN(auto val,
                                  litert::internal::ParseTomlInt(value));
          options_ref.sharding_intensity =
              static_cast<LrtGoogleTensorOptionsShardingIntensity>(val);
        } else if (key == "enable_dynamic_range_quantization") {
          LITERT_ASSIGN_OR_RETURN(options_ref.enable_dynamic_range_quantization,
                                  litert::internal::ParseTomlBool(value));
        } else if (key == "performance_mode") {
          LITERT_ASSIGN_OR_RETURN(auto val,
                                  litert::internal::ParseTomlInt(value));
          options_ref.performance_mode =
              static_cast<LiteRtGoogleTensorOptionsPerformanceMode>(val);
        } else if (key == "op_filters_proto") {
          if (!absl::Base64Unescape(value, &options_ref.op_filters_proto)) {
            return kLiteRtStatusErrorInvalidArgument;
          }
        } else if (key == "extra_options_path") {
          options_ref.extra_options_path = std::string(value);
        } else if (absl::string_view rest = key;
                   absl::ConsumePrefix(&rest, "input_coherency_")) {
          size_t colon_pos = rest.find(':');
          if (colon_pos != absl::string_view::npos) {
            std::string sig(rest.substr(0, colon_pos));
            std::string tensor(rest.substr(colon_pos + 1));
            LITERT_ASSIGN_OR_RETURN(bool pref,
                                    litert::internal::ParseTomlBool(value));
            options_ref.input_coherency[{std::move(sig), std::move(tensor)}] =
                pref;
          }
        } else if (absl::string_view rest = key;
                   absl::ConsumePrefix(&rest, "output_coherency_")) {
          size_t colon_pos = rest.find(':');
          if (colon_pos != absl::string_view::npos) {
            std::string sig(rest.substr(0, colon_pos));
            std::string tensor(rest.substr(colon_pos + 1));
            LITERT_ASSIGN_OR_RETURN(bool pref,
                                    litert::internal::ParseTomlBool(value));
            options_ref.output_coherency[{std::move(sig), std::move(tensor)}] =
                pref;
          }
        } else if (key == "extra_options") {
          if (!absl::Base64Unescape(value, &options_ref.extra_options)) {
            return kLiteRtStatusErrorInvalidArgument;
          }
          // copybara:uncomment_begin(google-only)
        // } else if (key == kExperimentalEnableInputValidatorKey) {
          // LITERT_ASSIGN_OR_RETURN(
              // options_ref.experimental_enable_input_validator,
              // litert::internal::ParseTomlBool(value));
          // copybara:uncomment_end
        }
        return kLiteRtStatusOk;
      });

  if (status != kLiteRtStatusOk) {
    LrtDestroyGoogleTensorOptions(*options);
    *options = nullptr;
  }
  return status;
}

// COMPILATION OPTIONS /////////////////////////////////////////////////////////

// float_truncation_type -------------------------------------------------------

LiteRtStatus LrtGoogleTensorOptionsSetFloatTruncationType(
    LrtGoogleTensorOptions options,
    LrtGoogleTensorOptionsTruncationType truncation_type) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->float_truncation_type = truncation_type;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGoogleTensorOptionsGetFloatTruncationType(
    LrtGoogleTensorOptions options,
    LrtGoogleTensorOptionsTruncationType* truncation_type) {
  if (options == nullptr || truncation_type == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *truncation_type = options->float_truncation_type;
  return kLiteRtStatusOk;
}

// int64_to_int32_truncation ---------------------------------------------------

LiteRtStatus LrtGoogleTensorOptionsSetInt64ToInt32Truncation(
    LrtGoogleTensorOptions options, bool int64_to_int32_truncation) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->int64_to_int32_truncation = int64_to_int32_truncation;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGoogleTensorOptionsGetInt64ToInt32Truncation(
    LrtGoogleTensorOptions options, bool* int64_to_int32_truncation) {
  if (options == nullptr || int64_to_int32_truncation == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *int64_to_int32_truncation = options->int64_to_int32_truncation;
  return kLiteRtStatusOk;
}

// output_dir ------------------------------------------------------------------

LiteRtStatus LrtGoogleTensorOptionsSetOutputDir(LrtGoogleTensorOptions options,
                                                const char* output_dir) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (output_dir == nullptr) {
    options->output_dir = "";
  } else {
    options->output_dir = output_dir;
  }
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGoogleTensorOptionsGetOutputDir(LrtGoogleTensorOptions options,
                                                const char** output_dir) {
  if (options == nullptr || output_dir == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *output_dir = options->output_dir.c_str();
  return kLiteRtStatusOk;
}

// dump_op_timings -------------------------------------------------------------

LiteRtStatus LrtGoogleTensorOptionsSetDumpOpTimings(
    LrtGoogleTensorOptions options, bool dump_op_timings) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->dump_op_timings = dump_op_timings;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGoogleTensorOptionsGetDumpOpTimings(
    LrtGoogleTensorOptions options, bool* dump_op_timings) {
  if (options == nullptr || dump_op_timings == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *dump_op_timings = options->dump_op_timings;
  return kLiteRtStatusOk;
}

// enable_large_model_support --------------------------------------------------
LiteRtStatus LrtGoogleTensorOptionsSetEnableLargeModelSupport(
    LrtGoogleTensorOptions options, bool enable_large_model_support) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->enable_large_model_support = enable_large_model_support;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGoogleTensorOptionsGetEnableLargeModelSupport(
    LrtGoogleTensorOptions options, bool* enable_large_model_support) {
  if (options == nullptr || enable_large_model_support == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *enable_large_model_support = options->enable_large_model_support;
  return kLiteRtStatusOk;
}

// enable_4bit_compilation -----------------------------------------------------
LiteRtStatus LrtGoogleTensorOptionsSetEnable4BitCompilation(
    LrtGoogleTensorOptions options, bool enable_4bit_compilation) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->enable_4bit_compilation = enable_4bit_compilation;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGoogleTensorOptionsGetEnable4BitCompilation(
    LrtGoogleTensorOptions options, bool* enable_4bit_compilation) {
  if (options == nullptr || enable_4bit_compilation == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *enable_4bit_compilation = options->enable_4bit_compilation;
  return kLiteRtStatusOk;
}

// sharding intensity ----------------------------------------------------------
LiteRtStatus LrtGoogleTensorOptionsSetShardingIntensity(
    LrtGoogleTensorOptions options,
    LrtGoogleTensorOptionsShardingIntensity sharding_intensity) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->sharding_intensity = sharding_intensity;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGoogleTensorOptionsGetShardingIntensity(
    LrtGoogleTensorOptions options,
    LrtGoogleTensorOptionsShardingIntensity* sharding_intensity) {
  if (options == nullptr || sharding_intensity == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *sharding_intensity = options->sharding_intensity;
  return kLiteRtStatusOk;
}

// enable_dynamic_range_quantization -----------------------------------------
LiteRtStatus LrtGoogleTensorOptionsSetEnableDynamicRangeQuantization(
    LrtGoogleTensorOptions options, bool enable_dynamic_range_quantization) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->enable_dynamic_range_quantization =
      enable_dynamic_range_quantization;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGoogleTensorOptionsGetEnableDynamicRangeQuantization(
    LrtGoogleTensorOptions options, bool* enable_dynamic_range_quantization) {
  if (options == nullptr || enable_dynamic_range_quantization == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *enable_dynamic_range_quantization =
      options->enable_dynamic_range_quantization;
  return kLiteRtStatusOk;
}

// performance mode ----------------------------------------------------

LiteRtStatus LrtGoogleTensorOptionsSetPerformanceMode(
    LrtGoogleTensorOptions options,
    LiteRtGoogleTensorOptionsPerformanceMode mode) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->performance_mode = mode;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGoogleTensorOptionsGetPerformanceMode(
    LrtGoogleTensorOptions options,
    LiteRtGoogleTensorOptionsPerformanceMode* mode) {
  if (options == nullptr || mode == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!options->performance_mode.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *mode = *options->performance_mode;
  return kLiteRtStatusOk;
}

// op_filters_proto --------------------------------------------------

LiteRtStatus LrtGoogleTensorOptionsSetOpFiltersProto(
    LrtGoogleTensorOptions options, const char* op_filters_proto) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (op_filters_proto == nullptr) {
    options->op_filters_proto = "";
  } else {
    options->op_filters_proto = op_filters_proto;
  }
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGoogleTensorOptionsGetOpFiltersProto(
    LrtGoogleTensorOptions options, const char** op_filters_proto) {
  if (options == nullptr || op_filters_proto == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *op_filters_proto = options->op_filters_proto.c_str();
  return kLiteRtStatusOk;
}

// extra_options_path --------------------------------------------------

LiteRtStatus LrtGoogleTensorOptionsSetExtraOptionsPath(
    LrtGoogleTensorOptions options, const char* extra_options_path) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (extra_options_path == nullptr) {
    options->extra_options_path = "";
  } else {
    options->extra_options_path = extra_options_path;
  }
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGoogleTensorOptionsGetExtraOptionsPath(
    LrtGoogleTensorOptions options, const char** extra_options_path) {
  if (options == nullptr || extra_options_path == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *extra_options_path = options->extra_options_path.c_str();
  return kLiteRtStatusOk;
}

// input_coherency --------------------------------------------------

LiteRtStatus LrtGoogleTensorOptionsSetInputCoherency(
    LrtGoogleTensorOptions options, const char* signature_name,
    const char* tensor_name, bool prefer_coherent) {
  if (options == nullptr || signature_name == nullptr ||
      tensor_name == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->input_coherency[{std::string(signature_name),
                            std::string(tensor_name)}] = prefer_coherent;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGoogleTensorOptionsGetInputCoherency(
    LrtGoogleTensorOptions options, const char* signature_name,
    const char* tensor_name, bool* prefer_coherent) {
  if (options == nullptr || signature_name == nullptr ||
      tensor_name == nullptr || prefer_coherent == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto it = options->input_coherency.find(
      {std::string(signature_name), std::string(tensor_name)});
  if (it == options->input_coherency.end()) {
    *prefer_coherent = false;
  } else {
    *prefer_coherent = it->second;
  }
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGoogleTensorOptionsGetNumInputCoherencyEntries(
    LrtGoogleTensorOptions options, int* num_entries) {
  if (options == nullptr || num_entries == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_entries = static_cast<int>(options->input_coherency.size());
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGoogleTensorOptionsGetInputCoherencyEntry(
    LrtGoogleTensorOptions options, int entry_idx, const char** signature_name,
    const char** tensor_name, bool* prefer_coherent) {
  if (options == nullptr || signature_name == nullptr ||
      tensor_name == nullptr || prefer_coherent == nullptr || entry_idx < 0 ||
      static_cast<size_t>(entry_idx) >= options->input_coherency.size()) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto it = options->input_coherency.begin();
  std::advance(it, entry_idx);
  *signature_name = it->first.first.c_str();
  *tensor_name = it->first.second.c_str();
  *prefer_coherent = it->second;
  return kLiteRtStatusOk;
}

// output_coherency --------------------------------------------------

LiteRtStatus LrtGoogleTensorOptionsSetOutputCoherency(
    LrtGoogleTensorOptions options, const char* signature_name,
    const char* tensor_name, bool prefer_coherent) {
  if (options == nullptr || signature_name == nullptr ||
      tensor_name == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->output_coherency[{std::string(signature_name),
                             std::string(tensor_name)}] = prefer_coherent;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGoogleTensorOptionsGetOutputCoherency(
    LrtGoogleTensorOptions options, const char* signature_name,
    const char* tensor_name, bool* prefer_coherent) {
  if (options == nullptr || signature_name == nullptr ||
      tensor_name == nullptr || prefer_coherent == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto it = options->output_coherency.find(
      {std::string(signature_name), std::string(tensor_name)});
  if (it == options->output_coherency.end()) {
    *prefer_coherent = false;
  } else {
    *prefer_coherent = it->second;
  }
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGoogleTensorOptionsGetNumOutputCoherencyEntries(
    LrtGoogleTensorOptions options, int* num_entries) {
  if (options == nullptr || num_entries == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_entries = static_cast<int>(options->output_coherency.size());
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGoogleTensorOptionsGetOutputCoherencyEntry(
    LrtGoogleTensorOptions options, int entry_idx, const char** signature_name,
    const char** tensor_name, bool* prefer_coherent) {
  if (options == nullptr || signature_name == nullptr ||
      tensor_name == nullptr || prefer_coherent == nullptr || entry_idx < 0 ||
      static_cast<size_t>(entry_idx) >= options->output_coherency.size()) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto it = options->output_coherency.begin();
  std::advance(it, entry_idx);
  *signature_name = it->first.first.c_str();
  *tensor_name = it->first.second.c_str();
  *prefer_coherent = it->second;
  return kLiteRtStatusOk;
}

// extra_options --------------------------------------------------

LiteRtStatus LrtGoogleTensorOptionsSetExtraOptions(
    LrtGoogleTensorOptions options, const char* extra_options) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (extra_options == nullptr) {
    options->extra_options = "";
  } else {
    options->extra_options = extra_options;
  }
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGoogleTensorOptionsGetExtraOptions(
    LrtGoogleTensorOptions options, const char** extra_options) {
  if (options == nullptr || extra_options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *extra_options = options->extra_options.c_str();
  return kLiteRtStatusOk;
}

// copybara:uncomment_begin(google-only)
// // experimental_enable_input_validator ---------------------------------
// 
// LiteRtStatus LrtGoogleTensorOptionsSetExperimentalEnableInputValidator(
//     LrtGoogleTensorOptions options, bool experimental_enable_input_validator) {
//   if (options == nullptr) {
//     return kLiteRtStatusErrorInvalidArgument;
//   }
//   options->experimental_enable_input_validator =
//       experimental_enable_input_validator;
//   return kLiteRtStatusOk;
// }
// 
// LiteRtStatus LrtGoogleTensorOptionsGetExperimentalEnableInputValidator(
//     LrtGoogleTensorOptions options, bool* experimental_enable_input_validator) {
//   if (options == nullptr || experimental_enable_input_validator == nullptr) {
//     return kLiteRtStatusErrorInvalidArgument;
//   }
//   *experimental_enable_input_validator =
//       options->experimental_enable_input_validator;
//   return kLiteRtStatusOk;
// }
// copybara:uncomment_end
