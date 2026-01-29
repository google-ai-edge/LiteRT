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

// This file defines the C API for LiteRt runtime options.
// It contains the following methods:
//   - LiteRtCreateRuntimeOptions: Creates an opaque options object holding
//     runtime options.
//   - LiteRtFindRuntimeOptions: Gets the underlying runtime options from an
//     opaque options handle.
//   - LiteRtGetRuntimeOptionsIdentifier: Gets the identifier for Runtime
//     options stored in opaque options.
// Example usage:
//   LiteRtOpaqueOptions options = nullptr;
//   LITERT_ASSERT_OK(LiteRtCreateRuntimeOptions(&options));
//   LiteRtRuntimeOptions runtime_options = nullptr;
//   LITERT_ASSERT_OK(LiteRtFindRuntimeOptions(options, &runtime_options));

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_RUNTIME_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_RUNTIME_OPTIONS_H_

#include "litert/c/litert_common.h"
#ifdef __cplusplus
extern "C" {
#endif

LITERT_DEFINE_HANDLE(LiteRtRuntimeOptions);

// Creates an opaque options object holding CPU options.
LiteRtStatus LiteRtCreateRuntimeOptions(LiteRtOpaqueOptions* options);

// Gets the underlying CPU options from an opaque options handle.
LiteRtStatus LiteRtFindRuntimeOptions(LiteRtOpaqueOptions opaque_options,
                                      LiteRtRuntimeOptions* runtime_options);

// Gets the identifier for Runtime options stored in opaque options.
const char* LiteRtGetRuntimeOptionsIdentifier();

// Sets the profiling flag in runtime options. The options is
// being modified in the this setter method.
LiteRtStatus LiteRtSetRuntimeOptionsEnableProfiling(
  LiteRtRuntimeOptions options, bool enable_profiling);

// Gets the profiling flag from runtime options. Reads the
// value from the options and writes it to the pointer.
LiteRtStatus LiteRtGetRuntimeOptionsEnableProfiling(
  LiteRtRuntimeOptions options, bool* enable_profiling);

// Sets the error reporter mode in runtime options. The options is
// being modified in the this setter method.
LiteRtStatus LiteRtSetRuntimeOptionsErrorReporterMode(
    LiteRtRuntimeOptions options, LiteRtErrorReporterMode error_reporter_mode);

// Gets the error reporter mode from runtime options. Reads the
// value from the options and writes it to the pointer.
LiteRtStatus LiteRtGetRuntimeOptionsErrorReporterMode(
    LiteRtRuntimeOptions options, LiteRtErrorReporterMode* error_reporter_mode);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_RUNTIME_OPTIONS_H_
