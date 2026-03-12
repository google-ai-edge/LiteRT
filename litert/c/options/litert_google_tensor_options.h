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
#include "litert/c/options/litert_google_tensor_options_type.h"

#ifdef __cplusplus
#include <string>
#include <vector>

extern "C" {
#endif  // __cplusplus

LITERT_DEFINE_HANDLE(LrtGoogleTensorOptions);

LiteRtStatus LrtCreateGoogleTensorOptions(LrtGoogleTensorOptions* options);

void LrtDestroyGoogleTensorOptions(LrtGoogleTensorOptions options);

// The a string identifier that discriminates GoogleTensor options within
// type erased options.
const char* LrtGoogleTensorOptionsGetIdentifier();

LiteRtStatus LrtGetOpaqueGoogleTensorOptionsData(
    LrtGoogleTensorOptions options, const char** identifier, void** payload,
    void (**payload_deleter)(void*));

// Creates a google_tensor options object from a TOML payload.
LiteRtStatus LrtCreateGoogleTensorOptionsFromToml(
    const char* toml_payload, LrtGoogleTensorOptions* options);

// COMPILATION OPTIONS /////////////////////////////////////////////////////////

LiteRtStatus LrtGoogleTensorOptionsSetFloatTruncationType(
    LrtGoogleTensorOptions options,
    LrtGoogleTensorOptionsTruncationType truncation_type);

LiteRtStatus LrtGoogleTensorOptionsGetFloatTruncationType(
    LrtGoogleTensorOptions options,
    LrtGoogleTensorOptionsTruncationType* truncation_type);

// int64_to_int32_truncation ---------------------------------------------------

LiteRtStatus LrtGoogleTensorOptionsSetInt64ToInt32Truncation(
    LrtGoogleTensorOptions options, bool int64_to_int32_truncation);

LiteRtStatus LrtGoogleTensorOptionsGetInt64ToInt32Truncation(
    LrtGoogleTensorOptions options, bool* int64_to_int32_truncation);

// output_dir ------------------------------------------------------------------

// Sets the output directory for the generated files.
// The `output_dir` string is copied and stored in the `options` object.
LiteRtStatus LrtGoogleTensorOptionsSetOutputDir(LrtGoogleTensorOptions options,
                                                const char* output_dir);

// Returns the output directory for the generated files.
// The `output_dir` string is owned by the `options` object.
LiteRtStatus LrtGoogleTensorOptionsGetOutputDir(LrtGoogleTensorOptions options,
                                                const char** output_dir);

// dump_op_timings -------------------------------------------------------------

LiteRtStatus LrtGoogleTensorOptionsSetDumpOpTimings(
    LrtGoogleTensorOptions options, bool dump_op_timings);

LiteRtStatus LrtGoogleTensorOptionsGetDumpOpTimings(
    LrtGoogleTensorOptions options, bool* dump_op_timings);

// enable_large_model_support --------------------------------------------------

LiteRtStatus LrtGoogleTensorOptionsSetEnableLargeModelSupport(
    LrtGoogleTensorOptions options, bool enable_large_model_support);

LiteRtStatus LrtGoogleTensorOptionsGetEnableLargeModelSupport(
    LrtGoogleTensorOptions options, bool* enable_large_model_support);

// enable_4bit_compilation -----------------------------------------------------
LiteRtStatus LrtGoogleTensorOptionsSetEnable4BitCompilation(
    LrtGoogleTensorOptions options, bool enable_4bit_compilation);

LiteRtStatus LrtGoogleTensorOptionsGetEnable4BitCompilation(
    LrtGoogleTensorOptions options, bool* enable_4bit_compilation);

// sharding intensity ---------------------------------------------------------

LiteRtStatus LrtGoogleTensorOptionsSetShardingIntensity(
    LrtGoogleTensorOptions options,
    LrtGoogleTensorOptionsShardingIntensity sharding_intensity);

LiteRtStatus LrtGoogleTensorOptionsGetShardingIntensity(
    LrtGoogleTensorOptions options,
    LrtGoogleTensorOptionsShardingIntensity* sharding_intensity);

// enable_dynamic_range_quantization -----------------------------------------
LiteRtStatus LrtGoogleTensorOptionsSetEnableDynamicRangeQuantization(
    LrtGoogleTensorOptions options, bool enable_dynamic_range_quantization);

LiteRtStatus LrtGoogleTensorOptionsGetEnableDynamicRangeQuantization(
    LrtGoogleTensorOptions options, bool* enable_dynamic_range_quantization);

#ifdef __cplusplus
// testing flags ---------------------------------------------------------------
LiteRtStatus LrtGoogleTensorOptionsSetTestingFlags(
    LrtGoogleTensorOptions options, const std::string& testing_flags);

LiteRtStatus LrtGoogleTensorOptionsGetTestingFlags(
    LrtGoogleTensorOptions options,
    std::vector<std::vector<std::string>>* testing_flags);
}  // extern "C"
#endif  // __cplusplus

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_GOOGLE_TENSOR_OPTIONS_H_
