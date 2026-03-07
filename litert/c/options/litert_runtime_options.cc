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

#include "litert/c/options/litert_runtime_options.h"

#include <string.h>  // NOLINT: To use strdup in some environments.

#include <cstdlib>
#include <optional>
#include <sstream>

#include "litert/c/litert_common.h"

struct LrtRuntimeOptions {
  std::optional<bool> enable_profiling;
  std::optional<LiteRtErrorReporterMode> error_reporter_mode;
  std::optional<bool> compress_quantization_zero_points;
};

LiteRtStatus LrtCreateRuntimeOptions(LrtRuntimeOptions** options) {
  if (!options) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *options = new LrtRuntimeOptions();
  if (!*options) {
    return kLiteRtStatusErrorMemoryAllocationFailure;
  }
  return kLiteRtStatusOk;
}

void LrtDestroyRuntimeOptions(LrtRuntimeOptions* options) {
  if (options) {
    delete options;
  }
}

LiteRtStatus LrtGetOpaqueRuntimeOptionsData(const LrtRuntimeOptions* options,
                                            const char** identifier,
                                            void** payload,
                                            void (**payload_deleter)(void*)) {
  if (!options || !identifier || !payload || !payload_deleter) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  std::stringstream ss;
  if (options->enable_profiling.has_value()) {
    ss << "enable_profiling = "
       << (options->enable_profiling.value() ? "true" : "false") << "\n";
  }
  if (options->error_reporter_mode.has_value()) {
    ss << "error_reporter_mode = "
       << static_cast<int>(options->error_reporter_mode.value()) << "\n";
  }
  if (options->compress_quantization_zero_points.has_value()) {
    ss << "compress_quantization_zero_points = "
       << (options->compress_quantization_zero_points.value() ? "true"
                                                              : "false")
       << "\n";
  }

  char* payload_str = strdup(ss.str().c_str());

  *identifier = LrtGetRuntimeOptionsIdentifier();
  *payload = payload_str;
  *payload_deleter = [](void* p) { free(p); };

  return kLiteRtStatusOk;
}

const char* LrtGetRuntimeOptionsIdentifier() {
  return "runtime_options_string";
}

LiteRtStatus LrtSetRuntimeOptionsEnableProfiling(LrtRuntimeOptions* options,
                                                 bool enable_profiling) {
  if (!options) return kLiteRtStatusErrorInvalidArgument;
  options->enable_profiling = enable_profiling;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetRuntimeOptionsEnableProfiling(
    const LrtRuntimeOptions* options, bool* enable_profiling) {
  if (!options || !enable_profiling) return kLiteRtStatusErrorInvalidArgument;
  if (!options->enable_profiling.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *enable_profiling = options->enable_profiling.value();
  return kLiteRtStatusOk;
}

LiteRtStatus LrtSetRuntimeOptionsErrorReporterMode(
    LrtRuntimeOptions* options, LiteRtErrorReporterMode error_reporter_mode) {
  if (!options) return kLiteRtStatusErrorInvalidArgument;
  options->error_reporter_mode = error_reporter_mode;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetRuntimeOptionsErrorReporterMode(
    const LrtRuntimeOptions* options,
    LiteRtErrorReporterMode* error_reporter_mode) {
  if (!options || !error_reporter_mode)
    return kLiteRtStatusErrorInvalidArgument;
  if (!options->error_reporter_mode.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *error_reporter_mode = options->error_reporter_mode.value();
  return kLiteRtStatusOk;
}

LiteRtStatus LrtSetRuntimeOptionsCompressQuantizationZeroPoints(
    LrtRuntimeOptions* options, bool compress_zero_points) {
  if (!options) return kLiteRtStatusErrorInvalidArgument;
  options->compress_quantization_zero_points = compress_zero_points;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetRuntimeOptionsCompressQuantizationZeroPoints(
    const LrtRuntimeOptions* options, bool* compress_zero_points) {
  if (!options || !compress_zero_points)
    return kLiteRtStatusErrorInvalidArgument;
  if (!options->compress_quantization_zero_points.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *compress_zero_points = options->compress_quantization_zero_points.value();
  return kLiteRtStatusOk;
}
