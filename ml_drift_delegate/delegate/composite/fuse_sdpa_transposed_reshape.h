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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_FUSE_SDPA_TRANSPOSED_RESHAPE_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_FUSE_SDPA_TRANSPOSED_RESHAPE_H_

#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/model.h"  // from @ml_drift

namespace ml_drift::ir {
class IrModel;
}  // namespace ml_drift::ir

namespace litert::ml_drift {

// Fuses single-token decode `sdpa_transposed` nodes whose 4D output
// [1, N, 1, H] is immediately reshaped (or transposed [0, 2, 1, 3] + reshaped)
// to [1, 1, 1, N*H], allowing the fused Flash-Decode GPU kernel to write
// directly to the flattened destination tensor without intermediate reshape or
// transpose dispatches.
//
// Note: This must run on GraphFloat32 / IrModel prior to
// BuildSdpaTransposedGpuGraph (so output_id has shape [1, 1, 1, N*H] and
// activates CreateFusedFlashDecodeSdpa's is_flattened_dst path), rather than
// via GPUOperation::AllowFuseInputReorder(true) in MergeReorderNodes, which
// only fuses a preceding reorder op into a consumer's Read(X, Y, S) selector.
absl::Status FuseSdpaTransposedReshape(::ml_drift::GraphFloat32* graph);

namespace ir {

absl::Status FuseSdpaTransposedReshape(::ml_drift::ir::IrModel* model);

}  // namespace ir

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_FUSE_SDPA_TRANSPOSED_RESHAPE_H_
