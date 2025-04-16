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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_CC_OPTIONS_HELPER_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_CC_OPTIONS_HELPER_H_

#include <optional>
#include <utility>

#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_options.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_handle.h"
#include "litert/cc/litert_opaque_options.h"
#include "litert/cc/litert_options.h"

namespace litert {

// Null check wrapper around options arguments for vendor code.
class OptionsHelper {
 public:
  OptionsHelper(LiteRtEnvironmentOptions env, LiteRtOptions options)
      : env_(env), options_(options) {}

  std::optional<EnvironmentOptions> Environment() {
    if (env_) {
      return EnvironmentOptions(env_);
    }
    return std::nullopt;
  }

  std::optional<litert::Options> Options() {
    if (options_) {
      return litert::Options(options_, OwnHandle::kNo);
    }
    return std::nullopt;
  }

  std::optional<litert::OpaqueOptions> OpaqueOptions() {
    auto opts = this->Options();
    if (!opts) {
      return std::nullopt;
    }
    auto opq = opts->GetOpaqueOptions();
    if (!opq) {
      return std::nullopt;
    }
    return std::move(*opq);
  }

  template <class Discriminated>
  std::optional<Discriminated> FindOptions() {
    auto opq = this->OpaqueOptions();
    if (!opq) {
      return std::nullopt;
    }
    auto disc = Find<Discriminated>(*opq);
    if (!disc) {
      return std::nullopt;
    }
    return std::move(*disc);
  }

 private:
  LiteRtEnvironmentOptions env_;
  LiteRtOptions options_;
};

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_CC_OPTIONS_HELPER_H_
