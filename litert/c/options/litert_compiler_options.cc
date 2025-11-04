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

#include <cstdint>
#include <memory>

#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/cache/hash_util.h"

struct LiteRtCompilerOptionsT {
  LiteRtCompilerOptionsPartitionStrategy partition_strategy =
      kLiteRtCompilerOptionsPartitionStrategyDefault;
  bool dummy_option = false;
};

LiteRtStatus LiteRtCreateCompilerOptions(LiteRtOpaqueOptions* options) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  auto options_data = std::make_unique<LiteRtCompilerOptionsT>();
  LITERT_RETURN_IF_ERROR(LiteRtCreateOpaqueOptions(
      LiteRtGetCompilerOptionsIdentifier(), options_data.get(),
      [](void* payload) {
        delete reinterpret_cast<LiteRtCompilerOptions>(payload);
      },
      options));

  // Hashable options for JIT cache.
  auto compiler_hash = [](const void* payload) -> uint64_t {
    const LiteRtCompilerOptionsT* options =
        reinterpret_cast<const LiteRtCompilerOptionsT*>(payload);
    uint64_t ans = 0;
    litert::HashCombine(ans, options->dummy_option);
    return ans;
  };
  LITERT_RETURN_IF_ERROR(LiteRtSetOpaqueOptionsHash(*options, compiler_hash));

  options_data.release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtFindCompilerOptions(
    LiteRtOpaqueOptions opaque_options,
    LiteRtCompilerOptions* compiler_options) {
  LITERT_RETURN_IF_ERROR(compiler_options,
                         litert::ErrorStatusBuilder::InvalidArgument())
      << "compiler_options is null.";
  void* options_data = nullptr;
  LITERT_RETURN_IF_ERROR(LiteRtFindOpaqueOptionsData(
      opaque_options, LiteRtGetCompilerOptionsIdentifier(), &options_data));
  *compiler_options = reinterpret_cast<LiteRtCompilerOptions>(options_data);
  return kLiteRtStatusOk;
}

const char* LiteRtGetCompilerOptionsIdentifier() { return "litert_compiler"; }

LiteRtStatus LiteRtSetCompilerOptionsPartitionStrategy(
    LiteRtCompilerOptions options,
    LiteRtCompilerOptionsPartitionStrategy partition_strategy) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->partition_strategy = partition_strategy;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompilerOptionsPartitionStrategy(
    LiteRtCompilerOptionsConst options,
    LiteRtCompilerOptionsPartitionStrategy* partition_strategy) {
  if (options == nullptr || partition_strategy == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *partition_strategy = options->partition_strategy;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetDummyCompilerOptions(LiteRtCompilerOptions options,
                                           bool dummy_option) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->dummy_option = dummy_option;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetDummyCompilerOptions(LiteRtCompilerOptionsConst options,
                                           bool* dummy_option) {
  if (options == nullptr || dummy_option == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *dummy_option = options->dummy_option;
  return kLiteRtStatusOk;
}
