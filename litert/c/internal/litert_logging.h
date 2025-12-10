// Copyright 2024 Google LLC.
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

#ifndef ODML_LITERT_LITERT_C_INTERNAL_LITERT_LOGGING_H_
#define ODML_LITERT_LITERT_C_INTERNAL_LITERT_LOGGING_H_

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>

#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// WARNING: The values of the following enum are to be kept in sync with
// tflite::LogSeverity.
typedef enum : int8_t {
  kLiteRtLogSeverityDebug = -1,
  kLiteRtLogSeverityVerbose = 0,
  kLiteRtLogSeverityInfo = 1,
  kLiteRtLogSeverityWarning = 2,
  kLiteRtLogSeverityError = 3,
  kLiteRtLogSeveritySilent = 4,
} LiteRtLogSeverity;

#ifdef NDEBUG
#define LITERT_DEBUG kLiteRtLogSeverityDebug
#else
#define LITERT_DEBUG kLiteRtLogSeverityInfo
#endif
#define LITERT_VERBOSE kLiteRtLogSeverityVerbose
#define LITERT_INFO kLiteRtLogSeverityInfo
#define LITERT_WARNING kLiteRtLogSeverityWarning
#define LITERT_ERROR kLiteRtLogSeverityError
#define LITERT_SILENT kLiteRtLogSeveritySilent

const char* LiteRtGetLogSeverityName(LiteRtLogSeverity severity);

// Creates a default LiteRT logger.
LiteRtStatus LiteRtCreateLogger(LiteRtLogger* logger);

LiteRtStatus LiteRtGetMinLoggerSeverity(LiteRtLogger logger,
                                        LiteRtLogSeverity* min_severity);

LiteRtStatus LiteRtSetMinLoggerSeverity(LiteRtLogger logger,
                                        LiteRtLogSeverity min_severity);

LiteRtStatus LiteRtLoggerLog(LiteRtLogger logger, LiteRtLogSeverity severity,
                             const char* format, ...);

// Returns the identifier of the logger.
// The returned string pointer is owned by the logger. It becomes invalid when
// the LiteRtLogger is destroyed.
LiteRtStatus LiteRtGetLoggerIdentifier(LiteRtLoggerConst logger,
                                       const char** identifier);

void LiteRtDestroyLogger(LiteRtLogger logger);

// Creates a sink logger.
//
// A sink logger will store the logs instead of outputting them. This allows the
// C++ public API to silence logs generated at the C boundary and control
// when/if to output them.
LiteRtStatus LiteRtCreateSinkLogger(LiteRtLogger* logger);

// Returns the number of log calls that were done to the sink logger.
LiteRtStatus LiteRtGetSinkLoggerSize(LiteRtLogger logger, size_t* size);

// Returns the idx_th log in the sink logger.
// The returned string pointer is owned by the sink logger. It becomes invalid
// when the LiteRtLogger is destroyed.
LiteRtStatus LiteRtGetSinkLoggerMessage(LiteRtLogger logger, size_t idx,
                                        const char** message);

// Clears the sink logger.
LiteRtStatus LiteRtClearSinkLogger(LiteRtLogger logger);

LiteRtLogger LiteRtGetDefaultLogger();

LiteRtStatus LiteRtSetDefaultLogger(LiteRtLogger logger);

LiteRtStatus LiteRtDefaultLoggerLog(LiteRtLogSeverity severity,
                                    const char* format, ...);

// Use the library provided standard logger instead of creating a new one.
LiteRtStatus LiteRtUseStandardLogger();

// Use the library provided sink logger instead of creating a new one.
LiteRtStatus LiteRtUseSinkLogger();

#ifdef __cplusplus
}  // extern "C"

// Compile statement only in debug mode.
#ifndef NDEBUG
#define LITERT_DEBUG_CODE(stmt) stmt;
#else
#define LITERT_DEBUG_CODE(stmt)
#endif

#endif  // __cplusplus

#ifndef __FILE_NAME__
#define __FILE_NAME__ __FILE__
#endif

#define LITERT_LOGGER_LOG_PROD(logger, severity, format, ...)             \
  {                                                                       \
    LiteRtLogSeverity __min_severity__;                                   \
    if (LiteRtGetMinLoggerSeverity(logger, &__min_severity__) !=          \
        kLiteRtStatusOk) {                                                \
      __min_severity__ = kLiteRtLogSeverityVerbose;                       \
    }                                                                     \
    if (severity >= __min_severity__) {                                   \
      LiteRtLoggerLog(logger, severity, "[%s:%d] " format, __FILE_NAME__, \
                      __LINE__, ##__VA_ARGS__);                           \
    }                                                                     \
  }

#ifndef NDEBUG
#define LITERT_LOGGER_LOG LITERT_LOGGER_LOG_PROD
#else
#define LITERT_LOGGER_LOG(logger, severity, format, ...)             \
  do {                                                               \
    LITERT_LOGGER_LOG_PROD(logger, severity, format, ##__VA_ARGS__); \
  } while (false)
#endif

#define LITERT_LOG(severity, format, ...) \
  LITERT_LOGGER_LOG(LiteRtGetDefaultLogger(), severity, format, ##__VA_ARGS__);

#define LITERT_ABORT abort()

#define LITERT_FATAL(format, ...)                              \
  do {                                                         \
    LITERT_LOG(kLiteRtLogSeverityError, format, ##__VA_ARGS__) \
    LITERT_ABORT;                                              \
  } while (0)

#endif  // ODML_LITERT_LITERT_C_INTERNAL_LITERT_LOGGING_H_
