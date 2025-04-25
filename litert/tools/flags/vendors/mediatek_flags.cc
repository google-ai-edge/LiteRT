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

#include "litert/tools/flags/vendors/mediatek_flags.h"

#include <string>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/options/litert_mediatek_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

// NOLINTBEGIN(*alien-types*)
// TODO: Move absl parse/unparse function to same file as enum types if
// it becomes an issue.

bool AbslParseFlag(absl::string_view text,
                   LiteRtMediatekOptionsNeronSDKVersionType* options,
                   std::string* error) {
  if (text == "version8") {
    *options = kLiteRtMediatekOptionsNeronSDKVersionTypeVersion8;
    return true;
  }
  if (text == "version7") {
    *options = kLiteRtMediatekOptionsNeronSDKVersionTypeVersion7;
    return true;
  }

  *error = "Unknown truncation type";
  return false;
}

std::string AbslUnparseFlag(LiteRtMediatekOptionsNeronSDKVersionType options) {
  switch (options) {
    case kLiteRtMediatekOptionsNeronSDKVersionTypeVersion8:
      return "version8";
    case kLiteRtMediatekOptionsNeronSDKVersionTypeVersion7:
      return "version7";
  }
}

ABSL_FLAG(LiteRtMediatekOptionsNeronSDKVersionType, mediatek_sdk_version_type,
          kLiteRtMediatekOptionsNeronSDKVersionTypeVersion8,
          "Version for neuron sdk for Mediatek.");

// NOLINTEND(*alien-types*)

namespace litert::mediatek {

Expected<MediatekOptions> MediatekOptionsFromFlags() {
  LITERT_ASSIGN_OR_RETURN(auto options, MediatekOptions::Create());
  options.SetNeronSDKVersionType(
      absl::GetFlag(FLAGS_mediatek_sdk_version_type));
  return options;
}

}  // namespace litert::mediatek
