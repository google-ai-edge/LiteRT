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

#ifndef ODML_LITERT_LITERT_VENDORS_INTEL_OPENVINO_DISPATCH_NPU_KV_CACHE_POLICY_H_
#define ODML_LITERT_LITERT_VENDORS_INTEL_OPENVINO_DISPATCH_NPU_KV_CACHE_POLICY_H_

// Policy for multi-context (tiered) NPU KV caches. Everything model- or
// tuning-specific is here; the device and invocation contexts only provide the
// mechanism and ask this file what to do.

#include <cstddef>  // NOLINT
#include <cstdint>

namespace litert::openvino::npu_kv_cache {

// --- Release -----------------------------------------------------------------

// Release an imported node of `candidate_tier` when a node of `incoming_tier`
// imports: only across tiers (prefill and decode of one tier never evict each
// other), and never for an unknown tier.
bool ShouldRelease(int64_t incoming_tier, int64_t candidate_tier);

}  // namespace litert::openvino::npu_kv_cache

#endif  // ODML_LITERT_LITERT_VENDORS_INTEL_OPENVINO_DISPATCH_NPU_KV_CACHE_POLICY_H_
