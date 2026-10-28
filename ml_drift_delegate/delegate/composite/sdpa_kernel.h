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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_SDPA_KERNEL_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_SDPA_KERNEL_H_

#include <cstdint>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift

namespace litert::ml_drift {

// Returns whether the custom fused 2D-tiled Flash Attention kernel supports
// the given device and tensor descriptors for
// `odml.scaled_dot_product_attention`:
// - Target GPU API is OpenCL, Metal, or WebGPU (`IsApiOpenCl()`,
//   `IsApiMetal()`, or `IsApiWebGpu()`).
// - `Q` and `Dst` have shape `[B, T, N_q, H]` and `K` and `V` have shape
//   `[B, S, N_kv, H]` in `BHWC` layout (`Batch=B`, `Height=T/S`,
//   `Width=N_q/N_kv`, `Channels=H`), with matching batch size `B > 0` and head
//   dimension `H`, query length `T > 1` (single-token decode `T == 1` falls
//   back to ML Drift's decode-parallel path), and key length `S > 0`.
// - Head dimension `H` is a positive multiple of 4 (for 4-channel vector
//   slices) and `<= 128` (`<= 32` slices, so each thread holds at most 16
//   accumulator slices with `s_splits = 1` or `2`).
// - `N_kv > 0`, `N_q >= N_kv`, and `N_q % N_kv == 0` (supports both MHA where
//   `N_q == N_kv` and GQA where `N_q` is a multiple of `N_kv`).
// - Unmasked MHA shapes already handled by ML Drift's tuned `ConvSoftmaxConv`
//   kernel (`T >= 1024`, `H <= 40`, `kF16`) return `false` so they continue to
//   use `ConvSoftmaxConv`.
// - Optional `mask_desc` has shape `[B_m, N_m, T_m, S]` covering all `S` keys
//   and broadcasting across batch (`B_m` is `1` or `B`), heads (`N_m` is `1`
//   or `N_q`), and query tokens (`T_m` is `1` or `T`).
// - `Q`, `K`, `V`, and `Dst` have `DataType::kFloat16` or `DataType::kFloat32`,
//   and optional `mask_desc` has `DataType::kBool`, `DataType::kFloat16`, or
//   `DataType::kFloat32`.
// - The work-group size and `__local` SRAM footprint fit within the target
//   GPU's limits (`GetMaxWorkGroupTotalSize()` and local memory budget).
bool IsSupportedFlashSdpa(const ::ml_drift::GpuInfo& gpu_info,
                          const ::ml_drift::TensorDescriptor& q_desc,
                          const ::ml_drift::TensorDescriptor& k_desc,
                          const ::ml_drift::TensorDescriptor& v_desc,
                          const ::ml_drift::TensorDescriptor* mask_desc,
                          const ::ml_drift::TensorDescriptor& dst_desc,
                          bool use_f32_accum = false);

// Builds the fused Flash Attention GPU operation for
// `odml.scaled_dot_product_attention`. Returns `absl::UnimplementedError` if
// the shape or device is not supported so the caller can fall back to ML
// Drift's default lowering.
absl::Status BuildSdpaGpuGraph(
    const std::vector<uint32_t>& input_ids, uint32_t output_id,
    const ::ml_drift::ScaledDotProductAttentionAttributes& attr,
    ::ml_drift::GpuModelBuilder* model_builder);

absl::Status CreateSdpaFromNode(const std::vector<::ml_drift::Value*>& inputs,
                                const std::vector<::ml_drift::Value*>& outputs,
                                const ::ml_drift::Node& node,
                                ::ml_drift::GpuModelBuilder* model_builder);

absl::Status CreateSdpaFromIrOp(
    const std::vector<const ::ml_drift::ir::IrTensor*>& inputs,
    const std::vector<const ::ml_drift::ir::IrTensor*>& outputs,
    const ::ml_drift::ir::IrOp& op, ::ml_drift::GpuModelBuilder* model_builder);

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_SDPA_KERNEL_H_
