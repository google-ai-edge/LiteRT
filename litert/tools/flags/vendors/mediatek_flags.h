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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_FLAGS_VENDORS_MEDIATEK_FLAGS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_FLAGS_VENDORS_MEDIATEK_FLAGS_H_

#include <string>

#include "absl/flags/declare.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_expected.h"
#include "litert/cc/options/litert_mediatek_options.h"

// COMPILATION OPTIONS /////////////////////////////////////////////////////////

ABSL_DECLARE_FLAG(LiteRtMediatekOptionsNeronSDKVersionType,
                  mediatek_sdk_version_type);

// PARSERS (internal) //////////////////////////////////////////////////////////

bool AbslParseFlag(absl::string_view text,
                   LiteRtMediatekOptionsNeronSDKVersionType* options,
                   std::string* error);

std::string AbslUnparseFlag(LiteRtMediatekOptionsNeronSDKVersionType options);

namespace litert::mediatek {

Expected<MediatekOptions> MediatekOptionsFromFlags();

}  // namespace litert::mediatek

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_FLAGS_VENDORS_MEDIATEK_FLAGS_H_
