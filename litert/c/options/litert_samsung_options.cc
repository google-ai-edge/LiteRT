// Copyright 2026 Google LLC.
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

#include "litert/c/options/litert_samsung_options.h"

#include <cstring>
#include <optional>
#include <sstream>
#include <string>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/litert_toml_parser.h"

struct LrtSamsungOptionsT {
  // Set to true when compile LLM.
  std::optional<bool> enable_large_model_support;
};

const char* LrtSamsungOptionsGetIdentifier() { return "samsung"; }

LiteRtStatus LrtCreateSamsungOptions(LrtSamsungOptions* options) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *options = new LrtSamsungOptionsT;
  return kLiteRtStatusOk;
}

void LrtDestroySamsungOptions(LrtSamsungOptions options) {
  if (options != nullptr) {
    delete options;
  }
}

LiteRtStatus LrtCreateSamsungOptionsFromToml(const char* toml_payload,
                                             LrtSamsungOptions* options) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LrtCreateSamsungOptions(options));

  if (toml_payload == nullptr || toml_payload[0] == '\0') {
    return kLiteRtStatusOk;
  }

  LrtSamsungOptionsT& options_ref = **options;
  auto status = litert::internal::ParseToml(
      toml_payload,
      [&options_ref](absl::string_view key,
                     absl::string_view value) -> LiteRtStatus {
        if (key == "enable_large_model_support") {
          LITERT_ASSIGN_OR_RETURN(options_ref.enable_large_model_support,
                                  litert::internal::ParseTomlBool(value));
          return kLiteRtStatusOk;
        }
        return kLiteRtStatusOk;
      });

  if (status != kLiteRtStatusOk) {
    LrtDestroySamsungOptions(*options);
    *options = nullptr;
  }

  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetOpaqueSamsungOptionsData(LrtSamsungOptions options,
                                            const char** identifier,
                                            void** payload,
                                            void (**payload_deleter)(void*)) {
  if (options == nullptr || identifier == nullptr || payload == nullptr ||
      payload_deleter == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *identifier = LrtSamsungOptionsGetIdentifier();

  std::ostringstream toml;
  if (options->enable_large_model_support.has_value()) {
    toml << "enable_large_model_support = "
         << (*options->enable_large_model_support ? "true" : "false") << "\n";
  }

  std::string toml_str = toml.str();
  *payload = new char[toml_str.size() + 1];
  memcpy(*payload, toml_str.c_str(), toml_str.size() + 1);
  *payload_deleter = [](void* p) { delete[] static_cast<char*>(p); };
  return kLiteRtStatusOk;
}

// enable_large_model_support ------------------------------------------------
LiteRtStatus LrtSamsungOptionsSetEnableLargeModelSupport(
    LrtSamsungOptions options, bool enable_large_model_support) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->enable_large_model_support = enable_large_model_support;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtSamsungOptionsGetEnableLargeModelSupport(
    LrtSamsungOptions options, bool* enable_large_model_support) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *enable_large_model_support =
      options->enable_large_model_support.value_or(false);
  return kLiteRtStatusOk;
}
