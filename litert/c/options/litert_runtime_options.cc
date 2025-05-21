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
// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/c/options/litert_runtime_options.h"
#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif

LiteRtStatus LiteRtCreateRuntimeOptions(LiteRtRuntimeOptions* options) {
  *options = new LiteRtRuntimeOptionsT;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetRuntimeOptionsShloCompositeInlining(
    LiteRtRuntimeOptions options, bool shlo_composite_inlining) {
  options->shlo_composite_inlining = shlo_composite_inlining;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetRuntimeOptionsShloCompositeInlining(
    LiteRtRuntimeOptions options, bool* shlo_composite_inlining) {
  *shlo_composite_inlining = options->shlo_composite_inlining;
  return kLiteRtStatusOk;
}

void LiteRtDestroyRuntimeOptions(LiteRtRuntimeOptions options) {
  delete options;
}

#ifdef __cplusplus
}  // extern "C"
#endif
