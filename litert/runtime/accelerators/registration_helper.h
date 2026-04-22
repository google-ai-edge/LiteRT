// Copyright 2026 Google LLC.
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

#ifndef ODML_LITERT_LITERT_RUNTIME_ACCELERATORS_REGISTRATION_HELPER_H_
#define ODML_LITERT_LITERT_RUNTIME_ACCELERATORS_REGISTRATION_HELPER_H_

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_accelerator_def.h"
#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_shared_library.h"
#include "litert/cc/litert_expected.h"
#include "litert/core/environment.h"

namespace litert::internal {

// Registers an accelerator and tensor buffer handlers using the provided
// accelerator definition.
LiteRtStatus RegisterAcceleratorFromDef(
    LiteRtEnvironment env, const LiteRtAcceleratorDef* accelerator_def);

Expected<SharedLibrary> LoadSharedLibrary(absl::string_view shlib_path,
                                          bool try_default_on_failure);

Expected<void> RegisterSharedObjectAcceleratorViaFunctionPointer(
    LiteRtEnvironmentT& environment, absl::string_view shlib_path,
    absl::string_view registration_function_name, bool try_default_on_failure);

Expected<void> RegisterSharedObjectAcceleratorViaAcceleratorDef(
    LiteRtEnvironmentT& environment, absl::string_view shlib_path,
    absl::string_view accelerator_def_name, bool try_default_on_failure);

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_RUNTIME_ACCELERATORS_REGISTRATION_HELPER_H_
