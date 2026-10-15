/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <stdarg.h>

#include <cstdio>

#if defined(__OHOS__)
#include <hilog/log.h>
#endif  // defined(__OHOS__)

#include "tflite/minimal_logging.h"

namespace tflite {
namespace logging_internal {

#ifndef NDEBUG
// In debug builds, default is VERBOSE.
LogSeverity MinimalLogger::minimum_log_severity_ = TFLITE_LOG_VERBOSE;
#else
// In prod builds, default is INFO.
LogSeverity MinimalLogger::minimum_log_severity_ = TFLITE_LOG_INFO;
#endif

void MinimalLogger::LogFormatted(LogSeverity severity, const char* format,
                                 va_list args) {
  if (severity >= MinimalLogger::minimum_log_severity_) {
    fprintf(stderr, "%s: ", GetSeverityName(severity));
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-nonliteral"
    vfprintf(stderr, format, args);
#pragma clang diagnostic pop
    fputc('\n', stderr);
#if defined(__OHOS__)
    // OpenHarmony discards a native app's stderr, so mirror into hilog.
    va_list args_copy;
    va_copy(args_copy, args);
    char hilog_buffer[2048];
    vsnprintf(hilog_buffer, sizeof(hilog_buffer), format, args_copy);
    va_end(args_copy);
    LogLevel level = LOG_DEBUG;
    switch (severity) {
      case TFLITE_LOG_INFO:
        level = LOG_INFO;
        break;
      case TFLITE_LOG_WARNING:
        level = LOG_WARN;
        break;
      case TFLITE_LOG_ERROR:
        level = LOG_ERROR;
        break;
      default:
        break;
    }
    OH_LOG_Print(LOG_APP, level, 0xFF02, "tflite", "%{public}s", hilog_buffer);
#endif  // defined(__OHOS__)
  }
}

}  // namespace logging_internal
}  // namespace tflite
