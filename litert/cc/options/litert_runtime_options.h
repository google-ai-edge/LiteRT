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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_RUNTIME_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_RUNTIME_OPTIONS_H_

#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_opaque_options.h"

namespace litert {

/// @brief Defines the C++ wrapper for LiteRT runtime options.
class RuntimeOptions {
 public:
  static Expected<RuntimeOptions> Create();

  Expected<void> SetEnableProfiling(bool enable_profiling) {
    enable_profiling_ = enable_profiling;
    return {};
  }
  Expected<bool> GetEnableProfiling() const {
    return enable_profiling_;
  }
  Expected<void> SetErrorReporterMode(
      LiteRtErrorReporterMode error_reporter_mode) {
    error_reporter_mode_ = error_reporter_mode;
    return {};
  }
  Expected<LiteRtErrorReporterMode> GetErrorReporterMode() const {
    return error_reporter_mode_;
  }
  Expected<void> SetCompressQuantizationZeroPoints(bool compress_zero_points) {
    compress_quantization_zero_points_ = compress_zero_points;
    return {};
  }
  Expected<bool> GetCompressQuantizationZeroPoints() const {
    return compress_quantization_zero_points_;
  }

  Expected<OpaqueOptions> GetOpaqueOptions();

 private:
   static constexpr const absl::string_view kPayloadIdentifier = "runtime_toml_payload";

  // If true, the interpreter will enable profiling.
  bool enable_profiling_ = false;
  // Error reporter mode to use for this model
  LiteRtErrorReporterMode error_reporter_mode_ =
      LiteRtErrorReporterMode::kLiteRtErrorReporterModeNone;
  // If true, per-channel quantization zero-points that are all identical will
  // be stored as a single value to reduce memory usage.
  bool compress_quantization_zero_points_ = false;
};

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_RUNTIME_OPTIONS_H_
