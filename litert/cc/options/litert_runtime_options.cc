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

#include "litert/cc/options/litert_runtime_options.h"

#include <memory>

#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

namespace litert {

Expected<RuntimeOptions> RuntimeOptions::Create() {
  LrtRuntimeOptions* options = nullptr;
  LITERT_RETURN_IF_ERROR(LrtCreateRuntimeOptions(&options));
  return RuntimeOptions(options);
}

RuntimeOptions::RuntimeOptions(LrtRuntimeOptions* options)
    : options_(options) {}

Expected<void> RuntimeOptions::SetEnableProfiling(bool enable_profiling) {
  LITERT_RETURN_IF_ERROR(
      LrtSetRuntimeOptionsEnableProfiling(options_.get(), enable_profiling));
  return {};
}

Expected<bool> RuntimeOptions::GetEnableProfiling() const {
  bool enable_profiling;
  LITERT_RETURN_IF_ERROR(
      LrtGetRuntimeOptionsEnableProfiling(options_.get(), &enable_profiling));
  return enable_profiling;
}

Expected<void> RuntimeOptions::SetErrorReporterMode(
    LiteRtErrorReporterMode error_reporter_mode) {
  LITERT_RETURN_IF_ERROR(LrtSetRuntimeOptionsErrorReporterMode(
      options_.get(), error_reporter_mode));
  return {};
}

Expected<LiteRtErrorReporterMode> RuntimeOptions::GetErrorReporterMode() const {
  LiteRtErrorReporterMode error_reporter_mode;
  LITERT_RETURN_IF_ERROR(LrtGetRuntimeOptionsErrorReporterMode(
      options_.get(), &error_reporter_mode));
  return error_reporter_mode;
}

Expected<void> RuntimeOptions::SetCompressQuantizationZeroPoints(
    bool compress_zero_points) {
  LITERT_RETURN_IF_ERROR(LrtSetRuntimeOptionsCompressQuantizationZeroPoints(
      options_.get(), compress_zero_points));
  return {};
}

Expected<bool> RuntimeOptions::GetCompressQuantizationZeroPoints() const {
  bool compress_zero_points;
  LITERT_RETURN_IF_ERROR(LrtGetRuntimeOptionsCompressQuantizationZeroPoints(
      options_.get(), &compress_zero_points));
  return compress_zero_points;
}

}  // namespace litert
