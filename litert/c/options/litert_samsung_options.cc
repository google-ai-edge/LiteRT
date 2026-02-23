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
//
// Copyright (C) 2026 Samsung Electronics Co. LTD.
// SPDX-License-Identifier: Apache-2.0

#include <memory>

#include "litert/c/options/litert_samsung_options.h"

#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"

#include "litert/cc/litert_macros.h"

struct LiteRtSamsungOptionsT {
  //
};

LiteRtStatus LiteRtSamsungOptionsCreate(LiteRtOpaqueOptions *options) {
    if (options == nullptr) {
        return kLiteRtStatusErrorInvalidArgument;
    }

    auto options_data = std::make_unique<LiteRtSamsungOptionsT>();

    LITERT_RETURN_IF_ERROR(LiteRtCreateOpaqueOptions(
            LiteRtSamsungOptionsGetIdentifier(), options_data.get(),
            [](void* payload) {
                delete reinterpret_cast<LiteRtSamsungOptions>(payload);
            },
            options));

    options_data.release();
    return kLiteRtStatusOk;

}

const char *LiteRtSamsungOptionsGetIdentifier() { return "samsung"; }

LiteRtStatus LiteRtSamsungOptionsGet(LiteRtOpaqueOptions options,
                                     LiteRtSamsungOptions *options_data) {
    if (options_data == nullptr || options == nullptr) {
        return kLiteRtStatusErrorInvalidArgument;
    }
    const char* identifier;
    LITERT_RETURN_IF_ERROR(
            LiteRtGetOpaqueOptionsIdentifier(options, &identifier));
    if (absl::NullSafeStringView(identifier) !=
            LiteRtSamsungOptionsGetIdentifier()) {
        return kLiteRtStatusErrorInvalidArgument;
    }
    void* payload;
    LITERT_RETURN_IF_ERROR(LiteRtGetOpaqueOptionsData(options, &payload));
    *options_data = reinterpret_cast<LiteRtSamsungOptions>(payload);
    return kLiteRtStatusOk;
}

