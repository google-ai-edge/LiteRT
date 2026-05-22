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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_MAGIC_NUMBER_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_MAGIC_NUMBER_OPTIONS_H_

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>

#include "litert/c/litert_environment_options.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_expected.h"

/// @file
/// @brief Defines C++ wrappers and utilities for LiteRT magic number options.

namespace litert::options {

struct MagicNumberConfigsDeleter {
  void operator()(LiteRtMagicNumberConfigs* ptr) const { std::free(ptr); }
};

struct MagicNumberVerificationsDeleter {
  void operator()(LiteRtMagicNumberVerifications* ptr) const { std::free(ptr); }
};

using MagicNumberConfigsPtr =
    std::unique_ptr<LiteRtMagicNumberConfigs, MagicNumberConfigsDeleter>;
using MagicNumberVerificationsPtr =
    std::unique_ptr<LiteRtMagicNumberVerifications,
                    MagicNumberVerificationsDeleter>;

/// @brief Allocates a `LiteRtMagicNumberConfigs` structure large enough to
/// hold `num_configs` entries.
///
/// The returned pointer owns the allocated memory and will automatically free
/// it when destroyed.
Expected<MagicNumberConfigsPtr> CreateMagicNumberConfigs(
    std::size_t num_configs);

/// @brief Allocates a `LiteRtMagicNumberVerifications` structure large enough
/// to hold `num_verifications` entries.
///
/// The returned pointer owns the allocated memory and will automatically free
/// it when destroyed.
Expected<MagicNumberVerificationsPtr> CreateMagicNumberVerifications(
    std::size_t num_verifications);

inline Expected<MagicNumberConfigsPtr> CreateMagicNumberConfigs(
    std::size_t num_configs) {
  constexpr std::size_t kHeaderSize = sizeof(LiteRtMagicNumberConfigs);
  constexpr std::size_t kElementSize = sizeof(LiteRtMagicNumberConfig);
  if (num_configs == 0 ||

      num_configs > (std::numeric_limits<std::size_t>::max() - kHeaderSize) /
                        kElementSize) {
    return Unexpected(Error(Status::kErrorInvalidArgument,
                            "Magic number configs allocation size overflow"));
  }

  const std::size_t total_size = kHeaderSize + num_configs * kElementSize;
  auto* raw = static_cast<LiteRtMagicNumberConfigs*>(std::malloc(total_size));

  raw->num_configs = static_cast<int64_t>(num_configs);
  return MagicNumberConfigsPtr(raw);
}

inline Expected<MagicNumberVerificationsPtr> CreateMagicNumberVerifications(
    std::size_t num_verifications) {
  if (num_verifications >
      static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
    return Unexpected(
        Error(Status::kErrorInvalidArgument,
              "Number of magic number verifications exceeds supported range"));
  }

  constexpr std::size_t kHeaderSize = sizeof(LiteRtMagicNumberVerifications);
  constexpr std::size_t kElementSize = sizeof(LiteRtMagicNumberVerification);
  if (num_verifications == 0 ||

      num_verifications >
          (std::numeric_limits<std::size_t>::max() - kHeaderSize) /
              kElementSize) {
    return Unexpected(
        Error(Status::kErrorInvalidArgument,
              "Magic number verifications allocation size overflow"));
  }

  const std::size_t total_size = kHeaderSize + num_verifications * kElementSize;
  auto* raw =
      static_cast<LiteRtMagicNumberVerifications*>(std::malloc(total_size));

  raw->num_verifications = static_cast<int64_t>(num_verifications);
  return MagicNumberVerificationsPtr(raw);
}

}  // namespace litert::options

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_MAGIC_NUMBER_OPTIONS_H_
