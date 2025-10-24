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
#include <memory>

#include "litert/c/litert_environment_options.h"
#include "litert/cc/litert_expected.h"

namespace litert::options {

struct MagicNumberConfigsDeleter {
  void operator()(LiteRtMagicNumberConfigs* ptr) const;
};

struct MagicNumberVerificationsDeleter {
  void operator()(LiteRtMagicNumberVerifications* ptr) const;
};

using MagicNumberConfigsPtr =
    std::unique_ptr<LiteRtMagicNumberConfigs, MagicNumberConfigsDeleter>;
using MagicNumberVerificationsPtr =
    std::unique_ptr<LiteRtMagicNumberVerifications,
                    MagicNumberVerificationsDeleter>;

// Allocates a LiteRtMagicNumberConfigs structure large enough to hold
// `num_configs` entries. The returned pointer owns the allocated memory and
// will automatically free it when destroyed.
Expected<MagicNumberConfigsPtr> CreateMagicNumberConfigs(
    std::size_t num_configs);

// Allocates a LiteRtMagicNumberVerifications structure large enough to hold
// `num_verifications` entries. The returned pointer owns the allocated memory
// and will automatically free it when destroyed.
Expected<MagicNumberVerificationsPtr> CreateMagicNumberVerifications(
    std::size_t num_verifications);

}  // namespace litert::options

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_MAGIC_NUMBER_OPTIONS_H_
