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

#ifndef ODML_LITERT_LITERT_C_LITERT_ERROR_REPORTER_H_
#define ODML_LITERT_LITERT_C_LITERT_ERROR_REPORTER_H_

#include <stdarg.h>

#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Opaque handle for the error reporter.
LITERT_DEFINE_HANDLE(LiteRtErrorReporter);

// Function type for reporting errors.
// The `reporter` is the error reporter handle.
// The `format` is the format string for the error message.
// The `args` is the list of arguments for the format string.
typedef int (*LiteRtErrorReporterReportFunc)(LiteRtErrorReporter reporter,
                                             const char* format, va_list args);

// Creates a new error reporter that prints errors to stderr.
// The `reporter` is the pointer to the error reporter handle.
// Returns LITERT_STATUS_OK on success.
LiteRtStatus LiteRtCreateStderrErrorReporter(LiteRtErrorReporter* reporter);

// Creates a new error reporter that stores errors in a buffer.
// The `reporter` is the pointer to the error reporter handle.
// Returns LITERT_STATUS_OK on success.
LiteRtStatus LiteRtCreateBufferErrorReporter(LiteRtErrorReporter* reporter);

// Reports an error using the given error reporter.
// The `reporter` is the error reporter handle.
// The `format` is the format string for the error message.
// The `...` is the list of arguments for the format string.
// Returns LITERT_STATUS_OK on success.
LiteRtStatus LiteRtErrorReporterReport(LiteRtErrorReporter reporter,
                                       const char* format, ...);

// Gets the error message from the given error reporter.
// The `reporter` is the error reporter handle.
// The `message` is the pointer to the error message.
// Returns LITERT_STATUS_OK on success.
LiteRtStatus LiteRtErrorReporterGetMessage(LiteRtErrorReporter reporter,
                                           const char** message);

// Clears the error message from the given error reporter.
// The `reporter` is the error reporter handle.
// Returns LITERT_STATUS_OK on success.
LiteRtStatus LiteRtErrorReporterClear(LiteRtErrorReporter reporter);

// Gets the default error reporter.
// Returns the default error reporter handle.
LiteRtErrorReporter LiteRtGetDefaultErrorReporter();

// Destroys the given error reporter.
// The `reporter` is the error reporter handle.
void LiteRtDestroyErrorReporter(LiteRtErrorReporter reporter);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // ODML_LITERT_LITERT_C_LITERT_ERROR_REPORTER_H_
