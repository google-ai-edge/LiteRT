// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ml_drift_delegate/delegate/composite/custom_transformations.h"

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/fuse_qkv_norm_rope.h"
#include "ml_drift_delegate/delegate/delegate_options.h"

namespace litert::ml_drift {

absl::Status ApplyCustomTransformations(
    ::ml_drift::GraphFloat32* graph,
    const MlDriftDelegateOptions& options) {
  if (options.enable_qkv_norm_rope_fusion) {
    ABSL_RETURN_IF_ERROR(FuseQkvNormRoPE(graph));
  }
  return absl::OkStatus();
}

absl::Status ApplyCustomTransformations(
    ::ml_drift::ir::IrModel* ir_model,
    const MlDriftDelegateOptions& options) {
  return absl::OkStatus();
}

}  // namespace litert::ml_drift
