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

#include "litert/core/environment_options.h"

#include <cstring>
#include <memory>

#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment_options.h"
#include "litert/cc/litert_expected.h"

litert::Expected<LiteRtAny> LiteRtEnvironmentOptionsT::GetOption(
    LiteRtEnvOptionTag tag) const {
  if (auto it = options_.find(tag); it != options_.end()) {
    return it->second;
  }
  return litert::Error(kLiteRtStatusErrorNotFound,
                       "Option was not set for this environment.");
}

litert::Expected<void> LiteRtEnvironmentOptionsT::SetOption(
    LiteRtEnvOption option, bool overwrite) {
  // Used count instead of contains to support compiling pre C++20.
  if (!overwrite && options_.count(option.tag)) {
    return litert::Error(kLiteRtStatusErrorAlreadyExists,
                         "Option was already set for this environment.");
  }
  if (option.value.type == kLiteRtAnyTypeString) {
    const int size = strlen(option.value.str_value) + 1;
    auto [string_it, _] = string_option_values_.insert_or_assign(
        option.tag, std::unique_ptr<char[]>(new char[size]));
    std::memcpy(string_it->second.get(), option.value.str_value, size);
    LiteRtAny value{/*type=*/kLiteRtAnyTypeString};
    value.str_value = string_it->second.get();
    options_[option.tag] = value;
  } else {
    options_[option.tag] = option.value;
  }
  return {};
}
