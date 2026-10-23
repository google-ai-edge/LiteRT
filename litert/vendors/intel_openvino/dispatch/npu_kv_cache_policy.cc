// Copyright (C) 2025 Intel Corporation
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

#include "litert/vendors/intel_openvino/dispatch/npu_kv_cache_policy.h"

#include <cstddef>
#include <cstdint>

namespace litert::openvino::npu_kv_cache {

bool ShouldRelease(int64_t incoming_tier, int64_t candidate_tier) {
  return incoming_tier > 0 && candidate_tier > 0 &&
         candidate_tier != incoming_tier;
}

}  // namespace litert::openvino::npu_kv_cache
