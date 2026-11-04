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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_FUSE_SHORT_CONV_STEP_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_FUSE_SHORT_CONV_STEP_H_

#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/model.h"  // from @ml_drift

namespace ml_drift::ir {
class IrModel;
}  // namespace ml_drift::ir

namespace litert::ml_drift {

// Scans the GraphFloat32 graph for short convolution step patterns
// (slice state -> concat state -> concat window -> mul -> reduce_sum) and
// fuses them into a single fused odml.short_conv_step node. Only channels-last
// patterns are fused: conv_state / next_state BHWC(1, 1, L - 1, H), the window
// BHWC(1, 1, L, H) concatenated along W and a BHWC(1, 1, L, H) weight, which is
// the layout the fused kernel reads. Channels-first patterns are left as is.
absl::Status FuseShortConvStep(::ml_drift::GraphFloat32* graph);

namespace ir {

// Scans the IrModel graph for short convolution step patterns and fuses them
// into a single fused short_conv_step operation. As above, only channels-last
// patterns are fused.
absl::Status FuseShortConvStep(::ml_drift::ir::IrModel* model);

}  // namespace ir

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_FUSE_SHORT_CONV_STEP_H_
