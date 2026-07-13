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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_IR_LITERT_OP_SELECTOR_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_IR_LITERT_OP_SELECTOR_H_

#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/ir_model_util.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift

namespace litert::ml_drift::ir {

class LiteRtOpSelector : public ::ml_drift::ir::IrModelOpSelector {
 public:
  LiteRtOpSelector() = default;
  explicit LiteRtOpSelector(const ::ml_drift::CreateGpuModelInfo* create_info,
                            const ::ml_drift::GpuInfo* gpu_info);
  ~LiteRtOpSelector() override = default;

  absl::Status GPUOperationFromNode(
      const ::ml_drift::OperationDef& op_def,
      const std::vector<const ::ml_drift::ir::IrTensor*>& inputs,
      const std::vector<const ::ml_drift::ir::IrTensor*>& outputs,
      const ::ml_drift::ir::IrOp& op,
      ::ml_drift::GpuModelBuilder* model_builder) override;

  absl::Status GPUSubgraphFromIrModel(
      const ::ml_drift::ir::IrModel& ir_model,
      ::ml_drift::ir::IrOpId first_op_id,
      const absl::flat_hash_set<::ml_drift::ir::IrOpId>& consumed_ops,
      absl::flat_hash_set<::ml_drift::ir::IrOpId>* new_consumed_ops,
      ::ml_drift::GpuModelBuilder* model_builder) override;

 private:
  ::ml_drift::CreateGpuModelInfo create_info_;
  ::ml_drift::GpuInfo gpu_info_;
};

}  // namespace litert::ml_drift::ir

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_IR_LITERT_OP_SELECTOR_H_
