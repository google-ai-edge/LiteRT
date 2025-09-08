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
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/cache/hash_util.h"
#include "litert/runtime/litert_google_tensor.h"

LiteRtStatus LiteRtGoogleTensorOptionsCreate(LiteRtOpaqueOptions* options) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  auto options_data = std::make_unique<LiteRtGoogleTensorOptionsT>();

  LITERT_RETURN_IF_ERROR(LiteRtCreateOpaqueOptions(
      LiteRtGoogleTensorOptionsGetIdentifier(), options_data.get(),
      [](void* payload) {
        delete reinterpret_cast<LiteRtGoogleTensorOptions>(payload);
      },
      options));
  auto google_tensor_hash = [](const void* payload) -> uint64_t {
    const LiteRtGoogleTensorOptionsT* options =
        reinterpret_cast<const LiteRtGoogleTensorOptionsT*>(payload);
    uint64_t ans = 0;
    litert::HashCombine(
        ans, options->float_truncation_type, options->int64_to_int32_truncation,
        options->output_dir, options->dump_op_timings,
        options->enable_large_model_support, options->sharding_intensity);
    return ans;
  };
  LITERT_RETURN_IF_ERROR(
      LiteRtSetOpaqueOptionsHash(*options, google_tensor_hash));
  options_data.release();
  return kLiteRtStatusOk;
}

const char* LiteRtGoogleTensorOptionsGetIdentifier() { return "google_tensor"; }

LiteRtStatus LiteRtGoogleTensorOptionsGet(
    LiteRtOpaqueOptions options, LiteRtGoogleTensorOptions* options_data) {
  if (options_data == nullptr || options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  const char* identifier;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetOpaqueOptionsIdentifier(options, &identifier));
  if (absl::NullSafeStringView(identifier) !=
      LiteRtGoogleTensorOptionsGetIdentifier()) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  void* payload;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpaqueOptionsData(options, &payload));
  *options_data = reinterpret_cast<LiteRtGoogleTensorOptions>(payload);

  return kLiteRtStatusOk;
}

// COMPILATION OPTIONS /////////////////////////////////////////////////////////

// float_truncation_type -------------------------------------------------------

LiteRtStatus LiteRtGoogleTensorOptionsSetFloatTruncationType(
    LiteRtGoogleTensorOptions options,
    LiteRtGoogleTensorOptionsTruncationType truncation_type) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->float_truncation_type = truncation_type;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGoogleTensorOptionsGetFloatTruncationType(
    LiteRtGoogleTensorOptions options,
    LiteRtGoogleTensorOptionsTruncationType* truncation_type) {
  if (options == nullptr || truncation_type == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *truncation_type = options->float_truncation_type;
  return kLiteRtStatusOk;
}

// int64_to_int32_truncation ---------------------------------------------------

LiteRtStatus LiteRtGoogleTensorOptionsSetInt64ToInt32Truncation(
    LiteRtGoogleTensorOptions options, bool int64_to_int32_truncation) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->int64_to_int32_truncation = int64_to_int32_truncation;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGoogleTensorOptionsGetInt64ToInt32Truncation(
    LiteRtGoogleTensorOptions options, bool* int64_to_int32_truncation) {
  if (options == nullptr || int64_to_int32_truncation == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *int64_to_int32_truncation = options->int64_to_int32_truncation;
  return kLiteRtStatusOk;
}

// output_dir ------------------------------------------------------------------

LiteRtStatus LiteRtGoogleTensorOptionsSetOutputDir(
    LiteRtGoogleTensorOptions options, const char* output_dir) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->output_dir = output_dir;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGoogleTensorOptionsGetOutputDir(
    LiteRtGoogleTensorOptions options, const char** output_dir) {
  if (options == nullptr || output_dir == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *output_dir = options->output_dir.c_str();
  return kLiteRtStatusOk;
}

// dump_op_timings -------------------------------------------------------------

LiteRtStatus LiteRtGoogleTensorOptionsSetDumpOpTimings(
    LiteRtGoogleTensorOptions options, bool dump_op_timings) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->dump_op_timings = dump_op_timings;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGoogleTensorOptionsGetDumpOpTimings(
    LiteRtGoogleTensorOptions options, bool* dump_op_timings) {
  if (options == nullptr || dump_op_timings == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *dump_op_timings = options->dump_op_timings;
  return kLiteRtStatusOk;
}

// enable_large_model_support --------------------------------------------------
LiteRtStatus LiteRtGoogleTensorOptionsSetEnableLargeModelSupport(
    LiteRtGoogleTensorOptions options, bool enable_large_model_support) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->enable_large_model_support = enable_large_model_support;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGoogleTensorOptionsGetEnableLargeModelSupport(
    LiteRtGoogleTensorOptions options, bool* enable_large_model_support) {
  if (options == nullptr || enable_large_model_support == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *enable_large_model_support = options->enable_large_model_support;
  return kLiteRtStatusOk;
}

// enable_4bit_compilation -----------------------------------------------------
LiteRtStatus LiteRtGoogleTensorOptionsSetEnable4BitCompilation(
    LiteRtGoogleTensorOptions options, bool enable_4bit_compilation) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->enable_4bit_compilation = enable_4bit_compilation;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGoogleTensorOptionsGetEnable4BitCompilation(
    LiteRtGoogleTensorOptions options, bool* enable_4bit_compilation) {
  if (options == nullptr || enable_4bit_compilation == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *enable_4bit_compilation = options->enable_4bit_compilation;
  return kLiteRtStatusOk;
}

// sharding intensity ----------------------------------------------------------
LiteRtStatus LiteRtGoogleTensorOptionsSetShardingIntensity(
    LiteRtGoogleTensorOptions options,
    LiteRtGoogleTensorOptionsShardingIntensity sharding_intensity) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->sharding_intensity = sharding_intensity;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGoogleTensorOptionsGetShardingIntensity(
    LiteRtGoogleTensorOptions options,
    LiteRtGoogleTensorOptionsShardingIntensity* sharding_intensity) {
  if (options == nullptr || sharding_intensity == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *sharding_intensity = options->sharding_intensity;
  return kLiteRtStatusOk;
}

// testing flags ---------------------------------------------------------------
LiteRtStatus LiteRtGoogleTensorOptionsSetTestingFlags(
    LiteRtGoogleTensorOptions options, const std::string& testing_flags) {
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

LiteRtStatus LiteRtGoogleTensorOptionsGetTestingFlags(
    LiteRtGoogleTensorOptions options,
    std::vector<std::vector<std::string>>* testing_flags) {
  if (options == nullptr || testing_flags == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *testing_flags = options->testing_flags;
  return kLiteRtStatusOk;
}
