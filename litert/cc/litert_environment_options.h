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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_ENVIRONMENT_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_ENVIRONMENT_OPTIONS_H_

#include <any>

#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment_options.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_any.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

namespace litert {

class EnvironmentOptions
    : public internal::NonOwnedHandle<LiteRtEnvironmentOptions> {
 public:
  // EnvironmentOptions are always owned by some environment, this can never be
  // an owning handle.
  explicit EnvironmentOptions(LiteRtEnvironmentOptions env)
      : NonOwnedHandle(env) {}

  using OptionTag = LiteRtEnvOptionTag;

  Expected<LiteRtVariant> GetOption(OptionTag tag) const {
    if (Get() == nullptr) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Environment options are null");
    }
    LiteRtAny option;
    LITERT_RETURN_IF_ERROR(
        LiteRtGetEnvironmentOptionsValue(Get(), tag, &option));
    return ToStdAny(option);
  }
};

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_ENVIRONMENT_OPTIONS_H_
