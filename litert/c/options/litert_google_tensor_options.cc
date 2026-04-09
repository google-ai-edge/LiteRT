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
#include <sstream>
#include <string>
#include <vector>

#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_replace.h"  // from @com_google_absl
#include "litert/c/internal/litert_options_helper.h"
#include "litert/c/litert_common.h"
#include "litert/c/options/litert_google_tensor_options_type.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/litert_toml_parser.h"

struct LrtGoogleTensorOptionsT {
  LrtGoogleTensorOptionsTruncationType float_truncation_type =
      kLiteRtGoogleTensorFloatTruncationTypeAuto;
  bool int64_to_int32_truncation = false;
  std::string output_dir = "";
  bool dump_op_timings = false;
  bool enable_large_model_support = false;
  bool enable_4bit_compilation = false;
  LrtGoogleTensorOptionsShardingIntensity sharding_intensity =
      kLiteRtGoogleTensorShardingIntensityMinimal;
  bool enable_dynamic_range_quantization = false;
  std::vector<std::vector<std::string>> testing_flags = {};
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
      kLiteRtGoogleTensorShardingIntensityMinimal) {
    absl::StrAppendFormat(&toml_str, "sharding_intensity = %d\n",
                          static_cast<int>(options->sharding_intensity));
  }
  if (options->enable_dynamic_range_quantization) {
    absl::StrAppendFormat(&toml_str,
                          "enable_dynamic_range_quantization = true\n");
  }

  if (!options->testing_flags.empty()) {
    std::string testing_flags_str;
    for (const auto& group : options->testing_flags) {
      if (group.empty()) {
        continue;
      }
      if (!testing_flags_str.empty()) {
        testing_flags_str += ',';
      }
      if (group.size() >= 2) {
        std::string escaped_value =
            absl::StrReplaceAll(group[1], {{"\\", "\\\\"}, {"\"", "\\\""}});
        absl::StrAppendFormat(&testing_flags_str, "%s=%s", group[0],
                              escaped_value);
      } else {
        testing_flags_str += group[0];
      }
    }
    absl::StrAppendFormat(&toml_str, "testing_flags = \"%s\"\n",
                          testing_flags_str);
  }

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
        } else if (key == "testing_flags") {
          std::string unescaped =
              absl::StrReplaceAll(value, {{"\\\\", "\\"}, {"\\\"", "\""}});
          LITERT_RETURN_IF_ERROR(
              LrtGoogleTensorOptionsSetTestingFlags(&options_ref, unescaped));
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
  options->output_dir = output_dir;
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

// testing flags ---------------------------------------------------------------
LiteRtStatus LrtGoogleTensorOptionsSetTestingFlags(
    LrtGoogleTensorOptions options, const std::string& testing_flags) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (testing_flags.empty()) {
    options->testing_flags = {};
  }
  std::vector<std::vector<std::string>> result;
  std::stringstream ss(testing_flags);
  std::string segment;

  // Split the input string by ',' to get individual "key=value" segments
  while (std::getline(ss, segment, ',')) {
    std::vector<std::string> currentPair;
    size_t delimiterPos = segment.find('=');

    if (delimiterPos != std::string::npos) {
      // '=' found, split into key and value
      std::string key = segment.substr(0, delimiterPos);
      std::string value = segment.substr(delimiterPos + 1);
      currentPair.push_back(key);
      currentPair.push_back(value);
    } else {
      // '=' not found, consider the whole segment as the key and an
      // empty string as the value
      currentPair.push_back(segment);
      currentPair.push_back("");  // Empty value
    }
    result.push_back(currentPair);
  }
  options->testing_flags = result;

  return kLiteRtStatusOk;
}

LiteRtStatus LrtGoogleTensorOptionsGetTestingFlags(
    LrtGoogleTensorOptions options,
    std::vector<std::vector<std::string>>* testing_flags) {
  if (options == nullptr || testing_flags == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *testing_flags = options->testing_flags;
  return kLiteRtStatusOk;
}
