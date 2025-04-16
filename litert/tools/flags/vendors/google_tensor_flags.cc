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

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/options/litert_google_tensor_options.h"

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

// NOLINTEND(*alien-types*)

namespace litert::google_tensor {
namespace {}

}  // namespace litert::google_tensor
