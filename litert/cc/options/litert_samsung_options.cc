// Copyright (C) 2026 Samsung Electronics Co. LTD.
// SPDX-License-Identifier: Apache-2.0
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

#include "litert/cc/options/litert_samsung_options.h"

#include "litert/cc/litert_expected.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_opaque_options.h"

namespace litert::samsung {

const char *SamsungOptions::Discriminator() {
  return LiteRtSamsungOptionsGetIdentifier();
}

Expected<SamsungOptions> SamsungOptions::Create(OpaqueOptions &options) {
  const auto id = options.GetIdentifier();
  if (!id || *id != Discriminator()) {
    return Error(kLiteRtStatusErrorInvalidArgument);
  }
  return SamsungOptions(options.Get(), OwnHandle::kNo);
}

Expected<SamsungOptions> SamsungOptions::Create() {
  LiteRtOpaqueOptions options;
  LITERT_RETURN_IF_ERROR(LiteRtSamsungOptionsCreate(&options));
  return SamsungOptions(options, OwnHandle::kYes);
}

LiteRtSamsungOptions SamsungOptions::Data() const {
  LiteRtSamsungOptions options;
  internal::AssertOk(LiteRtSamsungOptionsGet, Get(), &options);
  return options;
}

}

