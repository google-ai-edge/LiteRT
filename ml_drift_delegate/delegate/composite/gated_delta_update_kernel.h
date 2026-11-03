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
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/gated_delta_update_parser.h"

namespace litert::ml_drift {

absl::Status CreateGatedDeltaUpdateFromNode(
    const ::ml_drift::OperationDef& op_def,
    const std::vector<::ml_drift::Value*>& inputs,
    const std::vector<::ml_drift::Value*>& outputs,
    const ::ml_drift::Node& node, const ::ml_drift::GpuInfo* gpu_info,
    ::ml_drift::GpuModelBuilder* model_builder);

absl::Status CreateGatedDeltaUpdateFromIrOp(
    const ::ml_drift::OperationDef& op_def,
    const std::vector<const ::ml_drift::ir::IrTensor*>& inputs,
    const std::vector<const ::ml_drift::ir::IrTensor*>& outputs,
    const ::ml_drift::ir::IrOp& ir_op, const ::ml_drift::GpuInfo* gpu_info,
    ::ml_drift::GpuModelBuilder* model_builder);

// Block-mode kernels of `gated_delta_update` (the GatedDeltaNet block from the
// short conv output and the fused [z | b | a] projection to the gated RMSNorm
// output, updating the recurrent state in place), exposed for tests.
//
// Single-token decode runs as one fused kernel (`CreateGatedDeltaNetDecode`,
// using 32-lane wave shuffles when supported or workgroup-memory reductions
// otherwise). Multi-token prefill runs `CreateGatedDeltaNetPrep`,
// `CreateGatedDeltaNetRecurrence` and `CreateGatedDeltaNetNorm`.
//
// Single-token kernel, with the subgroup shader if `use_wave_ops`, otherwise
// with workgroup-memory reductions.
//   Sources: conv_out [1, 1, 2 * H_k * D_k + H_v * D_v],
//            zba [1, 1, H_v * D_v + 2 * H_v], a_log [H_v], dt_bias [H_v],
//            norm_weight [D_v], optional valid_mask [1, 1].
//   Destinations: output [1, 1, H_v * D_v], recurrent_state
//                 [1, H_v, D_k, D_v] (float32, read and updated in place).
absl::StatusOr<std::unique_ptr<::ml_drift::GPUOperation>>
CreateGatedDeltaNetDecode(const ::ml_drift::OperationDef& definition,
                          const GatedDeltaNetAttributes& attr,
                          const ::ml_drift::GpuInfo& gpu_info,
                          bool use_wave_ops);

// Parallel token preparation for the recurrence kernels: masks and
// L2-normalizes q and k (with 1 / sqrt(D_k) folded into q) and computes
// per-head (beta, decay, mask) in float32.
//   Sources: conv_out [1, T, ...], zba [1, T, ...], a_log, dt_bias, optional
//            valid_mask [1, T].
//   Destinations: qk_norm [1, T, 2 * H_k * D_k] (float32),
//                 gates [1, T, 4 * H_v] (float32).
absl::StatusOr<std::unique_ptr<::ml_drift::GPUOperation>>
CreateGatedDeltaNetPrep(const ::ml_drift::OperationDef& definition,
                        const GatedDeltaNetAttributes& attr);

// Recurrence for any number of tokens T, with the subgroup shader if
// `use_wave_ops`, otherwise with 64-thread workgroup-memory reductions.
//   Sources: conv_out [1, T, ...], qk_norm [1, T, 2 * H_k * D_k] (float32),
//            gates [1, T, 4 * H_v] (float32).
//   Destinations: un-normalized output [1, T, H_v * D_v] (float32),
//                 recurrent_state (float32, read and updated in place).
absl::StatusOr<std::unique_ptr<::ml_drift::GPUOperation>>
CreateGatedDeltaNetRecurrence(const ::ml_drift::OperationDef& definition,
                              const GatedDeltaNetAttributes& attr,
                              const ::ml_drift::GpuInfo& gpu_info,
                              bool use_wave_ops);

// Gated RMSNorm x SiLU(z) of the recurrence output.
//   Sources: un-normalized output [1, T, H_v * D_v] (float32), zba,
//            norm_weight.
//   Destination: output [1, T, H_v * D_v].
absl::StatusOr<std::unique_ptr<::ml_drift::GPUOperation>>
CreateGatedDeltaNetNorm(const ::ml_drift::OperationDef& definition,
                        const GatedDeltaNetAttributes& attr);

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_GATED_DELTA_UPDATE_KERNEL_H_
