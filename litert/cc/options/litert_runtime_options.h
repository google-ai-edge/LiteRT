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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_RUNTIME_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_RUNTIME_OPTIONS_H_

#include "litert/c/litert_common.h"
#include "litert/c/options/litert_runtime_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_handle.h"
#include "litert/cc/litert_macros.h"

namespace litert {

class RuntimeOptions : public internal::Handle<LiteRtRuntimeOptions,
                                               LiteRtDestroyRuntimeOptions> {
 public:
  explicit RuntimeOptions(LiteRtRuntimeOptions options, OwnHandle owned)
      : internal::Handle<LiteRtRuntimeOptions, LiteRtDestroyRuntimeOptions>(
            options, owned) {}

  RuntimeOptions() = default;

  static Expected<RuntimeOptions> Create() {
    LiteRtRuntimeOptions options;
    LITERT_RETURN_IF_ERROR(LiteRtCreateRuntimeOptions(&options));
    return RuntimeOptions(options, OwnHandle::kYes);
  }

  Expected<void> SetShloCompositeInlining(bool shlo_composite_inlining) {
    LITERT_RETURN_IF_ERROR(LiteRtSetRuntimeOptionsShloCompositeInlining(
        Get(), shlo_composite_inlining));
    return {};
  }
  Expected<bool> GetShloCompositeInlining() {
    bool shlo_composite_inlining;
    LITERT_RETURN_IF_ERROR(LiteRtGetRuntimeOptionsShloCompositeInlining(
        Get(), &shlo_composite_inlining));
    return shlo_composite_inlining;
  }
};

}  // namespace litert
#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_RUNTIME_OPTIONS_H_
