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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_FUSED_SDPA_CACHE_UPDATE_KERNEL_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_FUSED_SDPA_CACHE_UPDATE_KERNEL_H_

#include <cstdint>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/fused_sdpa_cache_update_parser.h"

namespace litert::ml_drift {

// Builds the GPU graph of `odml.fused_sdpa_cache_update`.
//
// `input_ids` are the 7 composite inputs (query, key_cache, value_cache,
// key_new, value_new, mask, param) and `output_ids` holds 1 (attention) or 3
// (attention, key_cache', value_cache') outputs. The caches use the packed
// layouts written by `odml.cache_update` (AddValuesToCache) and must have
// BUFFER storage.
//
// The attention reads the caches as they were before this step, and the ring
// buffer write is emitted as a separate GPU operation after every attention
// operation (and data-dependent on the attention output), so the write cannot
// race with the reads even when the updated caches alias the input caches.
absl::Status BuildFusedSdpaCacheUpdateGpuGraph(
    const std::vector<uint32_t>& input_ids,
    const std::vector<uint32_t>& output_ids,
    const FusedSdpaCacheUpdateAttributes& attr,
    ::ml_drift::GpuModelBuilder* model_builder);

absl::Status CreateFusedSdpaCacheUpdateFromNode(
    const std::vector<::ml_drift::Value*>& inputs,
    const std::vector<::ml_drift::Value*>& outputs,
    const ::ml_drift::Node& node, ::ml_drift::GpuModelBuilder* model_builder);

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_FUSED_SDPA_CACHE_UPDATE_KERNEL_H_
