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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_FUSE_SPLIT_SDPA_CACHE_UPDATE_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_FUSE_SPLIT_SDPA_CACHE_UPDATE_H_

#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/model.h"  // from @ml_drift

namespace ml_drift::ir {
class IrModel;
}  // namespace ml_drift::ir

namespace litert::ml_drift {

// Recognizes the decomposed ring-buffer sliding-window prefill SDPA subgraph
// (runtime_batched_matmul on past KV cache + batched_matmul on new KV +
// concat + optional softcap + select_v2 bool mask + softmax + split slices +
// add + optional ring-buffer add_values_to_cache) and rewrites each matched
// layer into a single `fused_sdpa_cache_update` node.
//
// This enables the fused NAX / FullyConnectedExternalWeights prefill path in
// `fused_sdpa_cache_update_kernel.cc` for existing models exported without
// `--use_fused_sdpa_cache_update=True`.
absl::Status FuseSplitSdpaCacheUpdate(::ml_drift::GraphFloat32* graph);

namespace ir {

absl::Status FuseSplitSdpaCacheUpdate(::ml_drift::ir::IrModel* model);

}  // namespace ir

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_FUSE_SPLIT_SDPA_CACHE_UPDATE_H_
