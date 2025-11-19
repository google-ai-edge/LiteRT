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

#include "litert/cc/options/litert_magic_number_options.h"

#include <cstdint>
#include <cstdlib>
#include <limits>

#include "litert/c/litert_common.h"
#include "litert/c/litert_environment_options.h"
#include "litert/cc/litert_expected.h"

namespace litert::options {

void MagicNumberConfigsDeleter::operator()(
    LiteRtMagicNumberConfigs* ptr) const {
  std::free(ptr);
}

void MagicNumberVerificationsDeleter::operator()(
    LiteRtMagicNumberVerifications* ptr) const {
  std::free(ptr);
}

Expected<MagicNumberConfigsPtr> CreateMagicNumberConfigs(
    std::size_t num_configs) {
  constexpr std::size_t kHeaderSize = sizeof(LiteRtMagicNumberConfigs);
  constexpr std::size_t kElementSize = sizeof(LiteRtMagicNumberConfig);
  if (num_configs == 0 ||

      num_configs > (std::numeric_limits<std::size_t>::max() - kHeaderSize) /
                        kElementSize) {
    return Unexpected(Error(kLiteRtStatusErrorInvalidArgument,
                            "Magic number configs allocation size overflow"));
  }

  const std::size_t total_size = kHeaderSize + num_configs * kElementSize;
  auto* raw =
      static_cast<LiteRtMagicNumberConfigs*>(std::malloc(total_size));

  raw->num_configs = static_cast<int64_t>(num_configs);
  return MagicNumberConfigsPtr(raw);
}

Expected<MagicNumberVerificationsPtr> CreateMagicNumberVerifications(
    std::size_t num_verifications) {
  if (num_verifications >
      static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
    return Unexpected(
        Error(kLiteRtStatusErrorInvalidArgument,
              "Number of magic number verifications exceeds supported range"));
  }

  constexpr std::size_t kHeaderSize = sizeof(LiteRtMagicNumberVerifications);
  constexpr std::size_t kElementSize = sizeof(LiteRtMagicNumberVerification);
  if (num_verifications == 0 ||

      num_verifications >
          (std::numeric_limits<std::size_t>::max() - kHeaderSize) /
              kElementSize) {
    return Unexpected(
        Error(kLiteRtStatusErrorInvalidArgument,
              "Magic number verifications allocation size overflow"));
  }

  const std::size_t total_size = kHeaderSize + num_verifications * kElementSize;
  auto* raw =
      static_cast<LiteRtMagicNumberVerifications*>(std::malloc(total_size));

  raw->num_verifications = static_cast<int64_t>(num_verifications);
  return MagicNumberVerificationsPtr(raw);
}

}  // namespace litert::options
