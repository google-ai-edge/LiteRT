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

#include "litert/cc/litert_opaque_options.h"

#include <string>

#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_handle.h"

namespace litert {
namespace {}  // namespace

Expected<OpaqueOptions> Find(OpaqueOptions& options,
                             const std::string& payload_identifier) {
  Expected<OpaqueOptions> chain(OpaqueOptions(options.Get(), OwnHandle::kNo));
  while (chain) {
    // TODO: lukeboyer - Error out in all the cases where there isn't
    // a valid identifier.
    const auto next_id = chain->GetIdentifier();
    if (next_id && *next_id == payload_identifier) {
      return OpaqueOptions(chain->Get(), OwnHandle::kNo);
    }
    chain = chain->Next();
  }
  return Error(kLiteRtStatusErrorInvalidArgument);
}

}  // namespace litert
