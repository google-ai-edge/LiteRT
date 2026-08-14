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

#include "litert/runtime/dispatch/dispatch_options_utils.h"

#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/options/litert_dispatch_delegate_vendor_options.h"

namespace litert::internal {

Expected<DispatchDelegateVendorOptions*>
GetOrCreateDispatchDelegateVendorOptions(LiteRtOpaqueOptions& opaque_options) {
  void* payload = nullptr;
  if (LiteRtFindOpaqueOptionsData(opaque_options,
                                  DispatchDelegateVendorOptions::kIdentifier,
                                  &payload) == kLiteRtStatusOk &&
      payload != nullptr) {
    return static_cast<DispatchDelegateVendorOptions*>(payload);
  }

  auto* vendor_options = new DispatchDelegateVendorOptions();
  LiteRtOpaqueOptions opaque_node = nullptr;
  LITERT_RETURN_IF_ERROR(LiteRtCreateOpaqueOptions(
      DispatchDelegateVendorOptions::kIdentifier, vendor_options,
      [](void* ptr) {
        delete static_cast<DispatchDelegateVendorOptions*>(ptr);
      },
      &opaque_node));
  LITERT_RETURN_IF_ERROR(
      LiteRtAppendOpaqueOptions(&opaque_options, opaque_node));

  return vendor_options;
}

}  // namespace litert::internal
