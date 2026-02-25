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
//   - LrtCreateRuntimeOptions: Creates a concrete options object holding
//     runtime options.
//   - LrtDestroyRuntimeOptions: Destroys a created runtime options object.
//   - LrtCreateOpaqueRuntimeOptions: Creates an opaque options object
//   holding
//     serialized options.
//   - LrtGetRuntimeOptionsIdentifier: Gets the identifier for Runtime
//     options stored in opaque options.
// Example usage:
//   LrtRuntimeOptions* options = nullptr;
//   LITERT_ASSERT_OK(LrtCreateRuntimeOptions(&options));
//   LITERT_ASSERT_OK(LrtSetRuntimeOptionsEnableProfiling(options, true);
//   LiteRtOpaqueOptions opaque_options = nullptr;
//   LITERT_ASSERT_OK(LrtCreateOpaqueRuntimeOptions(options,
//   &opaque_options));
//   LrtDestroyRuntimeOptions(options);

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_RUNTIME_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_RUNTIME_OPTIONS_H_

#include "litert/c/litert_common.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct LrtRuntimeOptions LrtRuntimeOptions;

// Creates a runtime options object.
// The caller is responsible for freeing the returned options using
// `LrtDestroyRuntimeOptions`.
LiteRtStatus LrtCreateRuntimeOptions(LrtRuntimeOptions** options);

// Destroys a runtime options object.
void LrtDestroyRuntimeOptions(LrtRuntimeOptions* options);

// Serializes runtime options and returns the components needed to create opaque
// options. The caller is responsible for passing these to
// `LiteRtCreateOpaqueOptions` and freeing the returned payload using
// `payload_deleter`.
LiteRtStatus LrtGetOpaqueRuntimeOptionsData(const LrtRuntimeOptions* options,
                                            const char** identifier,
                                            void** payload,
                                            void (**payload_deleter)(void*));

// Gets the identifier for Runtime options stored in opaque options.
const char* LrtGetRuntimeOptionsIdentifier();

// Sets the profiling flag in runtime options.
LiteRtStatus LrtSetRuntimeOptionsEnableProfiling(LrtRuntimeOptions* options,
                                                 bool enable_profiling);

// Gets the profiling flag from runtime options.
LiteRtStatus LrtGetRuntimeOptionsEnableProfiling(
    const LrtRuntimeOptions* options, bool* enable_profiling);

// Sets the error reporter mode in runtime options.
LiteRtStatus LrtSetRuntimeOptionsErrorReporterMode(
    LrtRuntimeOptions* options, LiteRtErrorReporterMode error_reporter_mode);

// Gets the error reporter mode from runtime options.
LiteRtStatus LrtGetRuntimeOptionsErrorReporterMode(
    const LrtRuntimeOptions* options,
    LiteRtErrorReporterMode* error_reporter_mode);

// Sets whether to compress per-channel quantization zero-points when all
// zero-points are identical.
LiteRtStatus LrtSetRuntimeOptionsCompressQuantizationZeroPoints(
    LrtRuntimeOptions* options, bool compress_zero_points);

// Gets whether per-channel quantization zero-points compression is enabled.
LiteRtStatus LrtGetRuntimeOptionsCompressQuantizationZeroPoints(
    const LrtRuntimeOptions* options, bool* compress_zero_points);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_RUNTIME_OPTIONS_H_
