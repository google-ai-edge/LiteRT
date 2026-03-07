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

#include "litert/c/options/litert_compiler_options.h"

#include <string.h>  // NOLINT: To use strdup in some environments.

#include <cstdlib>
#include <optional>
#include <sstream>

#include "litert/c/litert_common.h"

struct LrtCompilerOptions {
  std::optional<LiteRtCompilerOptionsPartitionStrategy> partition_strategy;
  std::optional<bool> dummy_option;
};

LiteRtStatus LrtCreateCompilerOptions(LrtCompilerOptions** options) {
  if (!options) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *options = new LrtCompilerOptions();
  if (!*options) {
    return kLiteRtStatusErrorMemoryAllocationFailure;
  }
  return kLiteRtStatusOk;
}

void LrtDestroyCompilerOptions(LrtCompilerOptions* options) {
  if (options) {
    delete options;
  }
}

LiteRtStatus LrtGetOpaqueCompilerOptionsData(const LrtCompilerOptions* options,
                                             const char** identifier,
                                             void** payload,
                                             void (**payload_deleter)(void*)) {
  if (!options || !identifier || !payload || !payload_deleter) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  std::stringstream ss;
  if (options->partition_strategy.has_value()) {
    ss << "partition_strategy = "
       << static_cast<int>(options->partition_strategy.value()) << "\n";
  }
  if (options->dummy_option.has_value()) {
    ss << "dummy_option = "
       << (options->dummy_option.value() ? "true" : "false") << "\n";
  }

  *identifier = LrtGetCompilerOptionsIdentifier();
  *payload = strdup(ss.str().c_str());
  *payload_deleter = [](void* p) { free(p); };

  return kLiteRtStatusOk;
}

const char* LrtGetCompilerOptionsIdentifier() {
  return "compiler_options_string";
}

LiteRtStatus LrtSetCompilerOptionsPartitionStrategy(
    LrtCompilerOptions* options,
    LiteRtCompilerOptionsPartitionStrategy partition_strategy) {
  if (!options) return kLiteRtStatusErrorInvalidArgument;
  options->partition_strategy = partition_strategy;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetCompilerOptionsPartitionStrategy(
    const LrtCompilerOptions* options,
    LiteRtCompilerOptionsPartitionStrategy* partition_strategy) {
  if (!options || !partition_strategy) return kLiteRtStatusErrorInvalidArgument;
  if (!options->partition_strategy.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *partition_strategy = options->partition_strategy.value();
  return kLiteRtStatusOk;
}

LiteRtStatus LrtSetCompilerOptionsDummyOption(LrtCompilerOptions* options,
                                              bool dummy_option) {
  if (!options) return kLiteRtStatusErrorInvalidArgument;
  options->dummy_option = dummy_option;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetCompilerOptionsDummyOption(const LrtCompilerOptions* options,
                                              bool* dummy_option) {
  if (!options || !dummy_option) return kLiteRtStatusErrorInvalidArgument;
  if (!options->dummy_option.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *dummy_option = options->dummy_option.value();
  return kLiteRtStatusOk;
}
