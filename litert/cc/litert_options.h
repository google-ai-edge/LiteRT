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

#ifndef ODML_LITERT_LITERT_CC_LITERT_COMPILATION_OPTIONS_H_
#define ODML_LITERT_LITERT_CC_LITERT_COMPILATION_OPTIONS_H_

#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/litert_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_handle.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_opaque_options.h"

namespace litert {

class Options : public internal::Handle<LiteRtOptions, LiteRtDestroyOptions> {
 public:
  Options() = default;

  // Parameter `owned` indicates if the created CompilationOptions object
  // should take ownership of the provided `compilation_options` handle.
  explicit Options(LiteRtOptions compilation_options, OwnHandle owned)
      : internal::Handle<LiteRtOptions, LiteRtDestroyOptions>(
            compilation_options, owned) {}

  static Expected<Options> Create(
      LiteRtHwAcceleratorSet accelerators = kLiteRtHwAcceleratorNone) {
    LiteRtOptions lrt_options;
    LITERT_RETURN_IF_ERROR(LiteRtCreateOptions(&lrt_options));
    Options options(lrt_options, OwnHandle::kYes);
    if (accelerators != kLiteRtHwAcceleratorNone) {
      LITERT_RETURN_IF_ERROR(options.SetHardwareAccelerators(accelerators));
    }
    return options;
  }

  Expected<void> SetHardwareAccelerators(LiteRtHwAcceleratorSet accelerators) {
    LITERT_RETURN_IF_ERROR(
        LiteRtSetOptionsHardwareAccelerators(Get(), accelerators));
    return {};
  }

  Expected<LiteRtHwAcceleratorSet> GetHardwareAccelerators() {
    LiteRtHwAcceleratorSet accelerators;
    LITERT_RETURN_IF_ERROR(
        LiteRtGetOptionsHardwareAccelerators(Get(), &accelerators));
    return accelerators;
  }

  Expected<void> AddOpaqueOptions(OpaqueOptions&& options) {
    LITERT_RETURN_IF_ERROR(LiteRtAddOpaqueOptions(Get(), options.Release()));
    return {};
  }

  Expected<OpaqueOptions> GetOpaqueOptions() {
    LiteRtOpaqueOptions options;
    LITERT_RETURN_IF_ERROR(LiteRtGetOpaqueOptions(Get(), &options));
    return OpaqueOptions(options, OwnHandle::kNo);
  }
};

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_LITERT_COMPILATION_OPTIONS_H_
