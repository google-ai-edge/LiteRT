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

#include "third_party/odml/litert/ml_drift/delegate/composite/ir/litert_op_selector.h"

#include <cstddef>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/selectors/operation_selector.h"  // from @ml_drift
#include "ml_drift/common/selectors/special_selector.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "third_party/odml/litert/ml_drift/delegate/composite/add_values_to_cache_kernel.h"
#include "third_party/odml/litert/ml_drift/delegate/composite/moe_experts_kernel.h"
#include "third_party/odml/litert/ml_drift/delegate/composite/runtime_batched_matmul_kernel.h"

namespace litert::ml_drift::ir {

LiteRtOpSelector::LiteRtOpSelector(
    const ::ml_drift::CreateGpuModelInfo* create_info,
    const ::ml_drift::GpuInfo* gpu_info)
    : create_info_(*create_info), gpu_info_(*gpu_info) {}

absl::Status LiteRtOpSelector::GPUOperationFromNode(
    const ::ml_drift::OperationDef& op_def,
    const std::vector<const ::ml_drift::ir::IrTensor*>& inputs,
    const std::vector<const ::ml_drift::ir::IrTensor*>& outputs,
    const ::ml_drift::ir::IrOp& op,
    ::ml_drift::GpuModelBuilder* model_builder) {
  if (op.name == "add_values_to_cache") {
    ASSIGN_OR_RETURN(auto op, CreateAddValuesToCacheFromNode(op_def, op));

    std::vector<::ml_drift::ValueId> src_ids(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
      src_ids[i] = inputs[i]->id;
    }
    std::vector<::ml_drift::ValueId> dst_ids(outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
      dst_ids[i] = outputs[i]->id;
    }
    model_builder->AddGpuOperation(src_ids, dst_ids, std::move(op),
                                   "add_values_to_cache");
    return absl::OkStatus();
  } else if (op.name == "moe_experts") {
    return CreateMoeExpertsFromIrOp(create_info_, inputs, outputs, op,
                                    model_builder);
  } else if (op.name == "runtime_batched_matmul") {
    return CreateRuntimeBatchedMatMulFromIrOp(inputs, outputs, op,
                                              model_builder);
  }

  return ::ml_drift::GPUOperationFromNode(gpu_info_, op_def, create_info_,
                                          inputs, outputs, op, model_builder);
}

absl::Status LiteRtOpSelector::GPUSubgraphFromIrModel(
    const ::ml_drift::ir::IrModel& ir_model, ::ml_drift::ir::IrOpId first_op_id,
    const absl::flat_hash_set<::ml_drift::ir::IrOpId>& consumed_ops,
    absl::flat_hash_set<::ml_drift::ir::IrOpId>* new_consumed_ops,
    ::ml_drift::GpuModelBuilder* model_builder) {
  return ::ml_drift::GPUSubgraphFromIrModel(create_info_.hints, gpu_info_,
                                            ir_model, first_op_id, consumed_ops,
                                            new_consumed_ops, model_builder);
}

}  // namespace litert::ml_drift::ir
