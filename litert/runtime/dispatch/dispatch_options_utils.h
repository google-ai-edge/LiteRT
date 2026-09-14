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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_DISPATCH_DISPATCH_OPTIONS_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_DISPATCH_DISPATCH_OPTIONS_UTILS_H_

#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/options/litert_dispatch_delegate_vendor_options.h"

namespace litert::internal {

// Finds an existing `DispatchDelegateVendorOptions` in `opaque_options`, or
// creates a new one, appends it to the chain, and returns a pointer to it.
Expected<DispatchDelegateVendorOptions*>
GetOrCreateDispatchDelegateVendorOptions(LiteRtOpaqueOptions& opaque_options);

}  // namespace litert::internal

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_DISPATCH_DISPATCH_OPTIONS_UTILS_H_
