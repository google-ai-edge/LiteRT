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
#include "litert/cc/litert_opaque_options.h"

#ifdef __cplusplus
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

// float_truncation_type -------------------------------------------------------

typedef enum LiteRtGoogleTensorOptionsTruncationType {
  kLiteRtGoogleTensorFloatTruncationTypeUnspecified = 0,
  kLiteRtGoogleTensorFloatTruncationTypeNoTruncation = 1,
  kLiteRtGoogleTensorFloatTruncationTypeBfloat16 = 2,
  kLiteRtGoogleTensorFloatTruncationTypeHalf = 3,
} LiteRtGoogleTensorOptionsTruncationType;

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

LiteRtStatus LiteRtGoogleTensorOptionsSetOutputDir(
    LiteRtGoogleTensorOptions options, const char* output_dir);

LiteRtStatus LiteRtGoogleTensorOptionsGetOutputDir(
    LiteRtGoogleTensorOptions options, const char** output_dir);

// dump_op_timings -------------------------------------------------------------

LiteRtStatus LiteRtGoogleTensorOptionsSetDumpOpTimings(
    LiteRtGoogleTensorOptions options, bool dump_op_timings);

LiteRtStatus LiteRtGoogleTensorOptionsGetDumpOpTimings(
    LiteRtGoogleTensorOptions options, bool* dump_op_timings);

// enable_reference ------------------------------------------------------------

LiteRtStatus LiteRtGoogleTensorOptionsSetEnableReference(
    LiteRtGoogleTensorOptions options, bool enable_reference);

LiteRtStatus LiteRtGoogleTensorOptionsGetEnableReference(
    LiteRtGoogleTensorOptions options, bool* enable_reference);

#ifdef __cplusplus
}  // extern "C"

// C++ WRAPPERS ////////////////////////////////////////////////////////////////

namespace litert::google_tensor {

// Wraps a LiteRtGoogleTensorOptions object for convenience.
class GoogleTensorOptions : public OpaqueOptions {
 public:
  using OpaqueOptions::OpaqueOptions;

  GoogleTensorOptions() = delete;

  static const char* Discriminator();

  static Expected<GoogleTensorOptions> Create(OpaqueOptions& options);
  static Expected<GoogleTensorOptions> Create();

  void SetFloatTruncationType(
      LiteRtGoogleTensorOptionsTruncationType truncation_type);

  LiteRtGoogleTensorOptionsTruncationType GetFloatTruncationType();

  void SetInt64ToInt32Truncation(bool int64_to_int32_truncation);

  bool GetInt64ToInt32Truncation();

  void SetOutputDir(absl::string_view output_dir);

  absl::string_view GetOutputDir();

  void SetDumpOpTimings(bool dump_op_timings);

  bool GetDumpOpTimings();

  void SetEnableReference(bool enable_reference);

  bool GetEnableReference();

 private:
  LiteRtGoogleTensorOptions Data() const;
};

}  // namespace litert::google_tensor
#endif  // __cplusplus

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_GOOGLE_TENSOR_OPTIONS_H_
