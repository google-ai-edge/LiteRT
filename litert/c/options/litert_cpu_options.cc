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

#include "litert/c/options/litert_cpu_options.h"

#include <stdint.h>
#include <string.h>  // NOLINT: To use strdup in some environments.

#include <cstdlib>
#include <optional>
#include <string>

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_macros.h"

struct LrtCpuOptions {
  std::optional<int32_t> num_threads;
  std::optional<uint32_t> flags;
  std::optional<std::string> weight_cache_file_path;
  std::optional<int> weight_cache_file_descriptor;
};

LiteRtStatus LrtCreateCpuOptions(LrtCpuOptions** options) {
  LITERT_ENSURE(options != nullptr, kLiteRtStatusErrorInvalidArgument,
                "options is null.");
  *options = new LrtCpuOptions();
  return kLiteRtStatusOk;
}

void LrtDestroyCpuOptions(LrtCpuOptions* options) { delete options; }

LiteRtStatus LrtGetOpaqueCpuOptionsData(const LrtCpuOptions* options,
                                        const char** identifier, void** payload,
                                        void (**payload_deleter)(void*)) {
  LITERT_ENSURE(options != nullptr, kLiteRtStatusErrorInvalidArgument,
                "options is null.");
  LITERT_ENSURE(identifier != nullptr, kLiteRtStatusErrorInvalidArgument,
                "identifier is null.");
  LITERT_ENSURE(payload != nullptr, kLiteRtStatusErrorInvalidArgument,
                "payload is null.");
  LITERT_ENSURE(payload_deleter != nullptr, kLiteRtStatusErrorInvalidArgument,
                "payload_deleter is null.");

  std::string toml_data;
  if (options->num_threads.has_value()) {
    absl::StrAppend(&toml_data, absl::StrFormat("num_threads = %d\n",
                                                *options->num_threads));
  }
  if (options->flags.has_value()) {
    absl::StrAppend(&toml_data,
                    absl::StrFormat("flags = %u\n", *options->flags));
  }
  if (options->weight_cache_file_path.has_value()) {
    absl::StrAppend(&toml_data,
                    absl::StrFormat("weight_cache_file_path = \"%s\"\n",
                                    *options->weight_cache_file_path));
  }
  if (options->weight_cache_file_descriptor.has_value()) {
    absl::StrAppend(&toml_data,
                    absl::StrFormat("weight_cache_file_descriptor = %d\n",
                                    *options->weight_cache_file_descriptor));
  }

  char* data_buffer = strdup(toml_data.c_str());

  *identifier = LrtGetCpuOptionsIdentifier();
  *payload = data_buffer;
  *payload_deleter = [](void* p) { free(static_cast<char*>(p)); };

  return kLiteRtStatusOk;
}

const char* LrtGetCpuOptionsIdentifier() { return "xnnpack"; }

LiteRtStatus LrtSetCpuOptionsNumThread(LrtCpuOptions* options,
                                       int num_threads) {
  LITERT_ENSURE(options != nullptr, kLiteRtStatusErrorInvalidArgument,
                "options is null.");
  options->num_threads = num_threads;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetCpuOptionsNumThread(const LrtCpuOptions* options,
                                       int* const num_threads) {
  LITERT_ENSURE(options != nullptr, kLiteRtStatusErrorInvalidArgument,
                "options is null.");
  LITERT_ENSURE(num_threads != nullptr, kLiteRtStatusErrorInvalidArgument,
                "num_threads is null.");
  if (!options->num_threads.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *num_threads = *options->num_threads;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtSetCpuOptionsXNNPackFlags(LrtCpuOptions* options,
                                          uint32_t flags) {
  LITERT_ENSURE(options != nullptr, kLiteRtStatusErrorInvalidArgument,
                "options is null.");
  options->flags = flags;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetCpuOptionsXNNPackFlags(const LrtCpuOptions* options,
                                          uint32_t* const flags) {
  LITERT_ENSURE(options != nullptr, kLiteRtStatusErrorInvalidArgument,
                "options is null.");
  LITERT_ENSURE(flags != nullptr, kLiteRtStatusErrorInvalidArgument,
                "flags is null.");
  if (!options->flags.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *flags = *options->flags;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtSetCpuOptionsXnnPackWeightCachePath(LrtCpuOptions* options,
                                                    const char* path) {
  LITERT_ENSURE(options != nullptr, kLiteRtStatusErrorInvalidArgument,
                "options is null.");
  if (options->weight_cache_file_descriptor.has_value()) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->weight_cache_file_path = std::string(path);
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetCpuOptionsXnnPackWeightCachePath(
    const LrtCpuOptions* options, const char** const path) {
  LITERT_ENSURE(options != nullptr, kLiteRtStatusErrorInvalidArgument,
                "options is null.");
  LITERT_ENSURE(path != nullptr, kLiteRtStatusErrorInvalidArgument,
                "path is null.");
  if (!options->weight_cache_file_path.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *path = options->weight_cache_file_path->c_str();
  return kLiteRtStatusOk;
}

LiteRtStatus LrtSetCpuOptionsXnnPackWeightCacheFileDescriptor(
    LrtCpuOptions* options, int fd) {
  LITERT_ENSURE(options != nullptr, kLiteRtStatusErrorInvalidArgument,
                "options is null.");
  if (options->weight_cache_file_path.has_value()) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->weight_cache_file_descriptor = fd;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetCpuOptionsXnnPackWeightCacheFileDescriptor(
    const LrtCpuOptions* options, int* const fd) {
  LITERT_ENSURE(options != nullptr, kLiteRtStatusErrorInvalidArgument,
                "options is null.");
  LITERT_ENSURE(fd != nullptr, kLiteRtStatusErrorInvalidArgument,
                "fd is null.");
  if (!options->weight_cache_file_descriptor.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *fd = *options->weight_cache_file_descriptor;
  return kLiteRtStatusOk;
}
