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

#include "litert/c/options/litert_aml_options.h"

#include <cstdint>
#include <cstring>
#include <optional>
#include <string>

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/litert_toml_parser.h"

struct LrtAmlOptionsT {
  std::optional<LrtAmlOptionsLogLevel> log_level;
  std::optional<LrtAmlOptionsProfiling> profiling;
  std::optional<LrtAmlOptionsPerfMode> perf_mode;
  std::optional<LrtAmlOptionsMemoryPolicy> memory_policy;
  std::optional<std::string> model_path;
  std::optional<std::string> model_name;
};

namespace {

LiteRtStatus ParseEnum(absl::string_view value, int* out) {
  LITERT_ASSIGN_OR_RETURN(const int64_t parsed,
                          litert::internal::ParseTomlInt(value));
  *out = static_cast<int>(parsed);
  return kLiteRtStatusOk;
}

}  // namespace

const char* LrtAmlOptionsGetIdentifier() { return "amlogic"; }

LiteRtStatus LrtCreateAmlOptions(LrtAmlOptions* options) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *options = new LrtAmlOptionsT;
  return kLiteRtStatusOk;
}

void LrtDestroyAmlOptions(LrtAmlOptions options) { delete options; }

LiteRtStatus LrtCreateAmlOptionsFromToml(const char* toml_payload,
                                         LrtAmlOptions* options) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LrtCreateAmlOptions(options));
  if (toml_payload == nullptr || toml_payload[0] == '\0') {
    return kLiteRtStatusOk;
  }

  LrtAmlOptionsT& options_ref = **options;
  auto status = litert::internal::ParseToml(
      toml_payload,
      [&options_ref](absl::string_view key,
                     absl::string_view value) -> LiteRtStatus {
        if (key == "log_level") {
          int parsed = 0;
          LITERT_RETURN_IF_ERROR(ParseEnum(value, &parsed));
          options_ref.log_level = static_cast<LrtAmlOptionsLogLevel>(parsed);
          return kLiteRtStatusOk;
        }
        if (key == "profiling") {
          int parsed = 0;
          LITERT_RETURN_IF_ERROR(ParseEnum(value, &parsed));
          options_ref.profiling = static_cast<LrtAmlOptionsProfiling>(parsed);
          return kLiteRtStatusOk;
        }
        if (key == "perf_mode") {
          int parsed = 0;
          LITERT_RETURN_IF_ERROR(ParseEnum(value, &parsed));
          options_ref.perf_mode = static_cast<LrtAmlOptionsPerfMode>(parsed);
          return kLiteRtStatusOk;
        }
        if (key == "memory_policy") {
          int parsed = 0;
          LITERT_RETURN_IF_ERROR(ParseEnum(value, &parsed));
          options_ref.memory_policy =
              static_cast<LrtAmlOptionsMemoryPolicy>(parsed);
          return kLiteRtStatusOk;
        }
        if (key == "model_path") {
          options_ref.model_path = std::string(value);
          return kLiteRtStatusOk;
        }
        if (key == "model_name") {
          options_ref.model_name = std::string(value);
          return kLiteRtStatusOk;
        }
        return kLiteRtStatusOk;
      });
  if (status != kLiteRtStatusOk) {
    LrtDestroyAmlOptions(*options);
    *options = nullptr;
  }
  return status;
}

LiteRtStatus LrtGetOpaqueAmlOptionsData(LrtAmlOptions options,
                                        const char** identifier, void** payload,
                                        void (**payload_deleter)(void*)) {
  if (options == nullptr || identifier == nullptr || payload == nullptr ||
      payload_deleter == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *identifier = LrtAmlOptionsGetIdentifier();
  std::string toml_str;
  if (options->log_level.has_value()) {
    absl::StrAppend(&toml_str, "log_level = ",
                    static_cast<int>(*options->log_level), "\n");
  }
  if (options->profiling.has_value()) {
    absl::StrAppend(&toml_str, "profiling = ",
                    static_cast<int>(*options->profiling), "\n");
  }
  if (options->perf_mode.has_value()) {
    absl::StrAppend(&toml_str, "perf_mode = ",
                    static_cast<int>(*options->perf_mode), "\n");
  }
  if (options->memory_policy.has_value()) {
    absl::StrAppend(&toml_str, "memory_policy = ",
                    static_cast<int>(*options->memory_policy), "\n");
  }
  if (options->model_path.has_value()) {
    absl::StrAppend(&toml_str, "model_path = \"", *options->model_path, "\"\n");
  }
  if (options->model_name.has_value()) {
    absl::StrAppend(&toml_str, "model_name = \"", *options->model_name, "\"\n");
  }
  *payload = new char[toml_str.size() + 1];
  memcpy(*payload, toml_str.c_str(), toml_str.size() + 1);
  *payload_deleter = [](void* p) { delete[] static_cast<char*>(p); };
  return kLiteRtStatusOk;
}

LiteRtStatus LrtAmlOptionsSetLogLevel(LrtAmlOptions options,
                                      LrtAmlOptionsLogLevel log_level) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->log_level = log_level;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtAmlOptionsGetLogLevel(LrtAmlOptions options,
                                      LrtAmlOptionsLogLevel* log_level) {
  if (options == nullptr || log_level == nullptr ||
      !options->log_level.has_value()) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *log_level = *options->log_level;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtAmlOptionsSetPerfMode(LrtAmlOptions options,
                                      LrtAmlOptionsPerfMode mode) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->perf_mode = mode;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtAmlOptionsGetPerfMode(LrtAmlOptions options,
                                      LrtAmlOptionsPerfMode* mode) {
  if (options == nullptr || mode == nullptr || !options->perf_mode.has_value()) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *mode = *options->perf_mode;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtAmlOptionsSetProfiling(LrtAmlOptions options,
                                       LrtAmlOptionsProfiling profiling) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->profiling = profiling;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtAmlOptionsGetProfiling(LrtAmlOptions options,
                                       LrtAmlOptionsProfiling* profiling) {
  if (options == nullptr || profiling == nullptr ||
      !options->profiling.has_value()) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *profiling = *options->profiling;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtAmlOptionsSetMemoryPolicy(LrtAmlOptions options,
                                          LrtAmlOptionsMemoryPolicy policy) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->memory_policy = policy;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtAmlOptionsGetMemoryPolicy(LrtAmlOptions options,
                                          LrtAmlOptionsMemoryPolicy* policy) {
  if (options == nullptr || policy == nullptr ||
      !options->memory_policy.has_value()) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *policy = *options->memory_policy;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtAmlOptionsSetModelPath(LrtAmlOptions options,
                                       const char* model_path) {
  if (options == nullptr || model_path == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->model_path = model_path;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtAmlOptionsGetModelPath(LrtAmlOptions options,
                                       const char** model_path) {
  if (options == nullptr || model_path == nullptr ||
      !options->model_path.has_value()) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *model_path = options->model_path->c_str();
  return kLiteRtStatusOk;
}

LiteRtStatus LrtAmlOptionsSetModelName(LrtAmlOptions options,
                                       const char* model_name) {
  if (options == nullptr || model_name == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->model_name = model_name;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtAmlOptionsGetModelName(LrtAmlOptions options,
                                       const char** model_name) {
  if (options == nullptr || model_name == nullptr ||
      !options->model_name.has_value()) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *model_name = options->model_name->c_str();
  return kLiteRtStatusOk;
}
