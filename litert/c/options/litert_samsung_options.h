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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_SAMSUNG_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_SAMSUNG_OPTIONS_H_

#include <stdint.h>

#include "litert/c/litert_common.h"

// C-API for an opaque options type relevant to Samsung (both dspatch and
// plugin).
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

LITERT_DEFINE_HANDLE(LiteRtSamsungOptions);


LiteRtStatus LiteRtSamsungOptionsCreate(LiteRtOpaqueOptions *options);

const char *LiteRtSamsungOptionsGetIdentifier();


LiteRtStatus LiteRtSamsungOptionsGet(LiteRtOpaqueOptions options,
                                     LiteRtSamsungOptions *options_data);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_SAMSUNG_OPTIONS_H_
