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

#include <memory>

#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/cc/litert_macros.h"
#include "litert/runtime/litert_runtime_options.h"

LiteRtStatus LiteRtCreateRuntimeOptions(LiteRtOpaqueOptions* options) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  auto options_data = std::make_unique<LiteRtRuntimeOptionsT>();
  LITERT_RETURN_IF_ERROR(LiteRtCreateOpaqueOptions(
      LiteRtGetRuntimeOptionsIdentifier(), options_data.get(),
      [](void* payload) {
        delete reinterpret_cast<LiteRtRuntimeOptions>(payload);
      },
      options));
  options_data.release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtFindRuntimeOptions(LiteRtOpaqueOptions opaque_options,
                                      LiteRtRuntimeOptions* runtime_options) {
  LITERT_RETURN_IF_ERROR(runtime_options,
                         litert::ErrorStatusBuilder::InvalidArgument())
      << "runtime_options is null.";
  void* options_data = nullptr;
  LITERT_RETURN_IF_ERROR(LiteRtFindOpaqueOptionsData(
      opaque_options, LiteRtGetRuntimeOptionsIdentifier(), &options_data));
  *runtime_options = reinterpret_cast<LiteRtRuntimeOptions>(options_data);
  return kLiteRtStatusOk;
}

const char* LiteRtGetRuntimeOptionsIdentifier() { return "runtime"; }

LiteRtStatus LiteRtSetRuntimeOptionsEnableProfiling(
    LiteRtRuntimeOptions options, bool enable_profiling) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  options->enable_profiling = enable_profiling;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetRuntimeOptionsEnableProfiling(
    LiteRtRuntimeOptions options, bool* enable_profiling) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  LITERT_RETURN_IF_ERROR(enable_profiling,
                         litert::ErrorStatusBuilder::InvalidArgument())
      << "enable_profiling is null.";
  *enable_profiling = options->enable_profiling;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetRuntimeOptionsErrorReporterMode(
    LiteRtRuntimeOptions options, LiteRtErrorReporterMode error_reporter_mode) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  options->error_reporter_mode = error_reporter_mode;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetRuntimeOptionsErrorReporterMode(
    LiteRtRuntimeOptions options,
    LiteRtErrorReporterMode* error_reporter_mode) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  LITERT_RETURN_IF_ERROR(error_reporter_mode,
                         litert::ErrorStatusBuilder::InvalidArgument())
      << "error_reporter_mode is null.";
  *error_reporter_mode = options->error_reporter_mode;
  return kLiteRtStatusOk;
}
