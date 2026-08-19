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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_GATED_DELTA_UPDATE_KERNEL_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_GATED_DELTA_UPDATE_KERNEL_H_

#include <memory>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift

namespace litert::ml_drift {

absl::StatusOr<std::unique_ptr<::ml_drift::GPUOperation>>
CreateGatedDeltaUpdateFromNode(const ::ml_drift::OperationDef& op_def,
                               const ::ml_drift::Node& node,
                               const ::ml_drift::GpuInfo* gpu_info = nullptr);

absl::StatusOr<std::unique_ptr<::ml_drift::GPUOperation>>
CreateGatedDeltaUpdateFromIrOp(const ::ml_drift::OperationDef& op_def,
                               const ::ml_drift::ir::IrOp& ir_op,
                               const ::ml_drift::GpuInfo* gpu_info = nullptr);

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_GATED_DELTA_UPDATE_KERNEL_H_
