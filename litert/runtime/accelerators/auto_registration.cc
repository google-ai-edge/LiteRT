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

#include "litert/runtime/accelerators/auto_registration.h"

#include <dlfcn.h>

#include <utility>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_shared_library.h"
#include "litert/core/environment.h"

namespace litert {

Expected<void> RegisterSharedObjectAccelerator(
    LiteRtEnvironmentT& environment, absl::string_view plugin_path,
    absl::string_view registration_function_name) {
  LITERT_ASSIGN_OR_RETURN(
      SharedLibrary lib,
      SharedLibrary::Load(plugin_path, RtldFlags::Now().Local().DeepBind()));
  LITERT_ASSIGN_OR_RETURN(auto registration_function,
                          lib.LookupSymbol<LiteRtStatus (*)(LiteRtEnvironment)>(
                              registration_function_name.data()));
  LITERT_RETURN_IF_ERROR(registration_function(&environment));
  environment.GetAcceleratorRegistry().TakeOwnershipOfSharedLibrary(
      std::move(lib));
  return {};
}

}  // namespace litert
