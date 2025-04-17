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

#include "litert/tools/flags/vendors/google_tensor_flags.h"

#include <string>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/options/litert_google_tensor_options.h"
#include "litert/cc/litert_expected.h"

// NOLINTBEGIN(*alien-types*)
// TODO: Move absl parse/unparse function to same file as enum types if
// it becomes an issue.

bool AbslParseFlag(absl::string_view text,
                   LiteRtGoogleTensorOptionsTruncationType* options,
                   std::string* error) {
  if (text == "unspecified") {
    *options = kLiteRtGoogleTensorFloatTruncationTypeUnspecified;
    return true;
  }
  if (text == "no_truncation") {
    *options = kLiteRtGoogleTensorFloatTruncationTypeNoTruncation;
    return true;
  }
  if (text == "bf16") {
    *options = kLiteRtGoogleTensorFloatTruncationTypeBfloat16;
    return true;
  }
  if (text == "half") {
    *options = kLiteRtGoogleTensorFloatTruncationTypeHalf;
    return true;
  }
  *error = "Unknown truncation type";
  return false;
}

std::string AbslUnparseFlag(LiteRtGoogleTensorOptionsTruncationType options) {
  switch (options) {
    case kLiteRtGoogleTensorFloatTruncationTypeUnspecified:
      return "unspecified";
    case kLiteRtGoogleTensorFloatTruncationTypeNoTruncation:
      return "no_truncation";
    case kLiteRtGoogleTensorFloatTruncationTypeBfloat16:
      return "bf16";
    case kLiteRtGoogleTensorFloatTruncationTypeHalf:
      return "half";
  }
}

ABSL_FLAG(LiteRtGoogleTensorOptionsTruncationType,
          google_tensor_truncation_type,
          kLiteRtGoogleTensorFloatTruncationTypeUnspecified,
          "Float truncation type for Google Tensor.");

ABSL_FLAG(bool, google_tensor_int64_to_int32, false,
          "Whether to truncate int64 to int32.");

ABSL_FLAG(std::string, google_tensor_output_dir, "",
          "Output directory for Google Tensor.");

ABSL_FLAG(bool, google_tensor_dump_op_timings, false,
          "Whether to dump op timings.");

ABSL_FLAG(bool, google_tensor_enable_reference, false,
          "Whether to enable reference.");

// NOLINTEND(*alien-types*)

namespace litert::google_tensor {

Expected<GoogleTensorOptions> GoogleTensorOptionsFromFlags() {
  GoogleTensorOptions options;
  options.SetFloatTruncationType(
      absl::GetFlag(FLAGS_google_tensor_truncation_type));
  options.SetInt64ToInt32Truncation(
      absl::GetFlag(FLAGS_google_tensor_int64_to_int32));
  options.SetOutputDir(absl::GetFlag(FLAGS_google_tensor_output_dir));
  options.SetDumpOpTimings(absl::GetFlag(FLAGS_google_tensor_dump_op_timings));
  options.SetEnableReference(
      absl::GetFlag(FLAGS_google_tensor_enable_reference));
  return options;
}

}  // namespace litert::google_tensor
