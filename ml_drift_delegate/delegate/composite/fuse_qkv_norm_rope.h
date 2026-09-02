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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_FUSE_QKV_NORM_ROPE_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_FUSE_QKV_NORM_ROPE_H_

#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/model.h"  // from @ml_drift

namespace litert::ml_drift {

// Scans the GraphFloat32 graph for QKV projection split -> RMSNorm -> RoPE
// patterns and fuses them into a single fused odml.qkv_norm_rope node.
absl::Status FuseQkvNormRoPE(::ml_drift::GraphFloat32* graph);

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_FUSE_QKV_NORM_ROPE_H_
