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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_GOOGLE_TENSOR_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_GOOGLE_TENSOR_OPTIONS_H_

#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/options/litert_google_tensor_options_type.h"

#ifdef __cplusplus
#include <string>
#include <vector>

extern "C" {
#endif  // __cplusplus

// Create a google tensor options object that is type erased. The actual option
// data can be accessed from the payload.
LiteRtStatus LiteRtGoogleTensorOptionsCreate(LiteRtOpaqueOptions* options);

LITERT_DEFINE_HANDLE(LiteRtGoogleTensorOptions);

// The a string identifier that discriminates qualcomm options within
// type erased options.
const char* LiteRtGoogleTensorOptionsGetIdentifier();

// Attempt to retieve google tensor options from the opaque options. Fails
// unlesss the opaque options are of another type.
LiteRtStatus LiteRtGoogleTensorOptionsGet(
    LiteRtOpaqueOptions options, LiteRtGoogleTensorOptions* options_data);

// COMPILATION OPTIONS /////////////////////////////////////////////////////////


LiteRtStatus LiteRtGoogleTensorOptionsSetFloatTruncationType(
    LiteRtGoogleTensorOptions options,
    LiteRtGoogleTensorOptionsTruncationType truncation_type);

LiteRtStatus LiteRtGoogleTensorOptionsGetFloatTruncationType(
    LiteRtGoogleTensorOptions options,
    LiteRtGoogleTensorOptionsTruncationType* truncation_type);

// int64_to_int32_truncation ---------------------------------------------------

LiteRtStatus LiteRtGoogleTensorOptionsSetInt64ToInt32Truncation(
    LiteRtGoogleTensorOptions options, bool int64_to_int32_truncation);

LiteRtStatus LiteRtGoogleTensorOptionsGetInt64ToInt32Truncation(
    LiteRtGoogleTensorOptions options, bool* int64_to_int32_truncation);

// output_dir ------------------------------------------------------------------

// Sets the output directory for the generated files.
// The `output_dir` string is copied and stored in the `options` object.
LiteRtStatus LiteRtGoogleTensorOptionsSetOutputDir(
    LiteRtGoogleTensorOptions options, const char* output_dir);

// Returns the output directory for the generated files.
// The `output_dir` string is owned by the `options` object.
LiteRtStatus LiteRtGoogleTensorOptionsGetOutputDir(
    LiteRtGoogleTensorOptions options, const char** output_dir);

// dump_op_timings -------------------------------------------------------------

LiteRtStatus LiteRtGoogleTensorOptionsSetDumpOpTimings(
    LiteRtGoogleTensorOptions options, bool dump_op_timings);

LiteRtStatus LiteRtGoogleTensorOptionsGetDumpOpTimings(
    LiteRtGoogleTensorOptions options, bool* dump_op_timings);

// enable_large_model_support --------------------------------------------------

LiteRtStatus LiteRtGoogleTensorOptionsSetEnableLargeModelSupport(
    LiteRtGoogleTensorOptions options, bool enable_large_model_support);

LiteRtStatus LiteRtGoogleTensorOptionsGetEnableLargeModelSupport(
    LiteRtGoogleTensorOptions options, bool* enable_large_model_support);

// enable_4bit_compilation -----------------------------------------------------
LiteRtStatus LiteRtGoogleTensorOptionsSetEnable4BitCompilation(
    LiteRtGoogleTensorOptions options, bool enable_4bit_compilation);

LiteRtStatus LiteRtGoogleTensorOptionsGetEnable4BitCompilation(
    LiteRtGoogleTensorOptions options, bool* enable_4bit_compilation);

// sharding intensity ---------------------------------------------------------

LiteRtStatus LiteRtGoogleTensorOptionsSetShardingIntensity(
    LiteRtGoogleTensorOptions options,
    LiteRtGoogleTensorOptionsShardingIntensity sharding_intensity);

LiteRtStatus LiteRtGoogleTensorOptionsGetShardingIntensity(
    LiteRtGoogleTensorOptions options,
    LiteRtGoogleTensorOptionsShardingIntensity* sharding_intensity);

#ifdef __cplusplus
// testing flags ---------------------------------------------------------------
LiteRtStatus LiteRtGoogleTensorOptionsSetTestingFlags(
    LiteRtGoogleTensorOptions options, const std::string& testing_flags);

LiteRtStatus LiteRtGoogleTensorOptionsGetTestingFlags(
    LiteRtGoogleTensorOptions options,
    std::vector<std::vector<std::string>>* testing_flags);
}  // extern "C"
#endif  // __cplusplus

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_GOOGLE_TENSOR_OPTIONS_H_
