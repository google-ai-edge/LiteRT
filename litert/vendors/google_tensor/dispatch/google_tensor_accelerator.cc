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

#if defined(LITERT_USE_STATIC_LINKED_NPU_ACCELERATOR)

#include "litert/vendors/c/litert_dispatch_api.h"

namespace {
// Strongly reference LiteRtDispatchGetApi to ensure it is linked.
[[maybe_unused]] volatile auto kLiteRtDispatchGetApiReference =
    &LiteRtDispatchGetApi;
}  // namespace

#endif  // defined(LITERT_USE_STATIC_LINKED_NPU_ACCELERATOR)
