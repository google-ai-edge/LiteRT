// Copyright 2026 Amlogic, Inc.
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

#include "litert/c/options/litert_amlogic_options.h"

#include <cstddef>
#include <cstring>
#include <optional>
#include <string>

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/litert_toml_parser.h"

struct LrtAmlogicOptionsT {
  std::optional<LrtAmlogicOptionsLogLevel> log_level;
  std::optional<std::string> model_path;
  std::optional<std::string> model_name;
};

namespace {

// LiteRT's TOML parser splits on newlines and strips quotes, but does not
// unescape. Encode the characters that would break a single-line quoted value.
std::string EscapeTomlString(absl::string_view value) {
  std::string out;
  out.reserve(value.size());
  for (const char c : value) {
    switch (c) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      default:
        out.push_back(c);
        break;
    }
  }
  return out;
}

LiteRtStatus UnescapeTomlString(absl::string_view value, std::string* out) {
  out->clear();
  out->reserve(value.size());
  for (size_t i = 0; i < value.size(); ++i) {
    if (value[i] != '\\') {
      out->push_back(value[i]);
      continue;
    }
    if (i + 1 >= value.size()) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    switch (value[++i]) {
      case '\\':
        out->push_back('\\');
        break;
      case '"':
        out->push_back('"');
        break;
      case 'n':
        out->push_back('\n');
        break;
      case 'r':
        out->push_back('\r');
        break;
      default:
        return kLiteRtStatusErrorInvalidArgument;
    }
  }
  return kLiteRtStatusOk;
}

}  // namespace

const char* LrtAmlogicOptionsGetIdentifier() { return "Amlogic"; }

LiteRtStatus LrtCreateAmlogicOptions(LrtAmlogicOptions* options) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *options = new LrtAmlogicOptionsT;
  return kLiteRtStatusOk;
}

void LrtDestroyAmlogicOptions(LrtAmlogicOptions options) { delete options; }

LiteRtStatus LrtCreateAmlogicOptionsFromToml(const char* toml_payload,
                                         LrtAmlogicOptions* options) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LrtCreateAmlogicOptions(options));
  if (toml_payload == nullptr || toml_payload[0] == '\0') {
    return kLiteRtStatusOk;
  }

  LrtAmlogicOptionsT& options_ref = **options;
  auto status = litert::internal::ParseToml(
      toml_payload,
      [&options_ref](absl::string_view key,
                     absl::string_view value) -> LiteRtStatus {
        if (key == "log_level") {
          LITERT_ASSIGN_OR_RETURN(const int64_t parsed,
                                  litert::internal::ParseTomlInt(value));
          if (parsed < kLrtAmlogicLogOff ||
              parsed > kLrtAmlogicLogLevelDebug) {
            return kLiteRtStatusErrorInvalidArgument;
          }
          options_ref.log_level = static_cast<LrtAmlogicOptionsLogLevel>(parsed);
          return kLiteRtStatusOk;
        }
        if (key == "model_path" || key == "model_name") {
          std::string unescaped;
          LITERT_RETURN_IF_ERROR(UnescapeTomlString(value, &unescaped));
          if (key == "model_path") {
            options_ref.model_path = std::move(unescaped);
          } else {
            options_ref.model_name = std::move(unescaped);
          }
          return kLiteRtStatusOk;
        }
        // Ignore keys this API no longer publishes.
        return kLiteRtStatusOk;
      });
  if (status != kLiteRtStatusOk) {
    LrtDestroyAmlogicOptions(*options);
    *options = nullptr;
  }
  return status;
}

LiteRtStatus LrtGetOpaqueAmlogicOptionsData(LrtAmlogicOptions options,
                                        const char** identifier, void** payload,
                                        void (**payload_deleter)(void*)) {
  if (options == nullptr || identifier == nullptr || payload == nullptr ||
      payload_deleter == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *identifier = LrtAmlogicOptionsGetIdentifier();
  std::string toml_str;
  if (options->log_level.has_value()) {
    absl::StrAppend(&toml_str, "log_level = ",
                    static_cast<int>(*options->log_level), "\n");
  }
  if (options->model_path.has_value()) {
    absl::StrAppend(&toml_str, "model_path = \"",
                    EscapeTomlString(*options->model_path), "\"\n");
  }
  if (options->model_name.has_value()) {
    absl::StrAppend(&toml_str, "model_name = \"",
                    EscapeTomlString(*options->model_name), "\"\n");
  }
  *payload = new char[toml_str.size() + 1];
  memcpy(*payload, toml_str.c_str(), toml_str.size() + 1);
  *payload_deleter = [](void* p) { delete[] static_cast<char*>(p); };
  return kLiteRtStatusOk;
}

LiteRtStatus LrtAmlogicOptionsSetLogLevel(LrtAmlogicOptions options,
                                      LrtAmlogicOptionsLogLevel log_level) {
  if (options == nullptr || log_level < kLrtAmlogicLogOff ||
      log_level > kLrtAmlogicLogLevelDebug) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->log_level = log_level;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtAmlogicOptionsGetLogLevel(LrtAmlogicOptions options,
                                      LrtAmlogicOptionsLogLevel* log_level) {
  if (options == nullptr || log_level == nullptr ||
      !options->log_level.has_value()) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *log_level = *options->log_level;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtAmlogicOptionsSetModelPath(LrtAmlogicOptions options,
                                       const char* model_path) {
  if (options == nullptr || model_path == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->model_path = model_path;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtAmlogicOptionsGetModelPath(LrtAmlogicOptions options,
                                       const char** model_path) {
  if (options == nullptr || model_path == nullptr ||
      !options->model_path.has_value()) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *model_path = options->model_path->c_str();
  return kLiteRtStatusOk;
}

LiteRtStatus LrtAmlogicOptionsSetModelName(LrtAmlogicOptions options,
                                       const char* model_name) {
  if (options == nullptr || model_name == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->model_name = model_name;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtAmlogicOptionsGetModelName(LrtAmlogicOptions options,
                                       const char** model_name) {
  if (options == nullptr || model_name == nullptr ||
      !options->model_name.has_value()) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *model_name = options->model_name->c_str();
  return kLiteRtStatusOk;
}
