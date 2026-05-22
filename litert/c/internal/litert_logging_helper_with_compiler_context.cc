// Copyright 2026 Google LLC.
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

#include "litert/c/internal/litert_logging_helper_with_compiler_context.h"

#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment_options.h"

LiteRtStatus LiteRtPropagateMinLoggerSeverityWithCompilerContext(
    const LiteRtCompilerContext* compiler_context,
    LiteRtEnvironmentOptions options) {
  LiteRtAny min_logger_severity;
  if (!compiler_context || !compiler_context->get_environment_options_value) {
    return kLiteRtStatusErrorNotFound;
  }
  auto status = compiler_context->get_environment_options_value(
      options, kLiteRtEnvOptionTagMinLoggerSeverity, &min_logger_severity);
  if (status == kLiteRtStatusOk) {
    if (min_logger_severity.type == kLiteRtAnyTypeInt) {
      LiteRtSetMinLoggerSeverity(
          LiteRtGetDefaultLogger(),
          static_cast<LiteRtLogSeverity>(min_logger_severity.int_value));
    } else {
      return kLiteRtStatusErrorInvalidArgument;
    }
  }
  return status;
}
