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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_CUSTOM_TRANSFORMATIONS_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_CUSTOM_TRANSFORMATIONS_H_

#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/delegate_options.h"

namespace litert::ml_drift {

// Applies custom graph transformations and fusions for the LiteRT delegate,
// controlled by delegate options.
absl::Status ApplyCustomTransformations(
    ::ml_drift::GraphFloat32* graph,
    const MlDriftDelegateOptions& options);

// Applies custom graph transformations and fusions for the LiteRT delegate (IrModel path),
// controlled by delegate options.
absl::Status ApplyCustomTransformations(
    ::ml_drift::ir::IrModel* ir_model,
    const MlDriftDelegateOptions& options);

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_CUSTOM_TRANSFORMATIONS_H_
