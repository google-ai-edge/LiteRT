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

LITERT_DEFINE_HANDLE(LiteRtErrorReporter);

typedef int (*LiteRtErrorReporterReportFunc)(LiteRtErrorReporter reporter,
                                              const char* format, va_list args);

LiteRtStatus LiteRtCreateStderrErrorReporter(LiteRtErrorReporter* reporter);

LiteRtStatus LiteRtCreateBufferErrorReporter(LiteRtErrorReporter* reporter);

LiteRtStatus LiteRtErrorReporterReport(LiteRtErrorReporter reporter,
                                       const char* format, ...);

LiteRtStatus LiteRtErrorReporterGetMessage(LiteRtErrorReporter reporter,
                                           const char** message);

LiteRtStatus LiteRtErrorReporterClear(LiteRtErrorReporter reporter);

LiteRtErrorReporter LiteRtGetDefaultErrorReporter();

void LiteRtDestroyErrorReporter(LiteRtErrorReporter reporter);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // ODML_LITERT_LITERT_C_LITERT_ERROR_REPORTER_H_