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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_BLOCK_FFW_KERNEL_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_BLOCK_FFW_KERNEL_H_

#include <set>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift

namespace litert::ml_drift {

// Lowers an internal `kBlockFfwUnpackPartType` node into `model_builder`.
absl::Status CreateBlockFfwUnpackPartFromNode(
    const ::ml_drift::OperationDef& op_def,
    const std::vector<::ml_drift::Value*>& inputs,
    const std::vector<::ml_drift::Value*>& outputs,
    const ::ml_drift::Node& node, ::ml_drift::GpuModelBuilder* model_builder);

// Lowers an internal `kBlockFfwPackPartsType` node into `model_builder`.
absl::Status CreateBlockFfwPackPartsFromNode(
    const ::ml_drift::OperationDef& op_def,
    const std::vector<::ml_drift::Value*>& inputs,
    const std::vector<::ml_drift::Value*>& outputs,
    const ::ml_drift::Node& node, ::ml_drift::GpuModelBuilder* model_builder);

// Lowers an internal `kBlockFfwSiluType` node into `model_builder`.
absl::Status CreateBlockFfwSiluFromNode(
    const ::ml_drift::OperationDef& op_def,
    const std::vector<::ml_drift::Value*>& inputs,
    const std::vector<::ml_drift::Value*>& outputs,
    ::ml_drift::GpuModelBuilder* model_builder);

// Lowers an internal `kBlockFfwUnpackPartType` IrModel op into `model_builder`.
absl::Status CreateBlockFfwUnpackPartFromIrOp(
    const ::ml_drift::OperationDef& op_def,
    const std::vector<const ::ml_drift::ir::IrTensor*>& inputs,
    const std::vector<const ::ml_drift::ir::IrTensor*>& outputs,
    const ::ml_drift::ir::IrOp& op, ::ml_drift::GpuModelBuilder* model_builder);

// Lowers an internal `kBlockFfwPackPartsType` IrModel op into `model_builder`.
absl::Status CreateBlockFfwPackPartsFromIrOp(
    const ::ml_drift::OperationDef& op_def,
    const std::vector<const ::ml_drift::ir::IrTensor*>& inputs,
    const std::vector<const ::ml_drift::ir::IrTensor*>& outputs,
    const ::ml_drift::ir::IrOp& op, ::ml_drift::GpuModelBuilder* model_builder);

// Lowers an internal `kBlockFfwSiluType` IrModel op into `model_builder`.
absl::Status CreateBlockFfwSiluFromIrOp(
    const ::ml_drift::OperationDef& op_def,
    const std::vector<const ::ml_drift::ir::IrTensor*>& inputs,
    const std::vector<const ::ml_drift::ir::IrTensor*>& outputs,
    ::ml_drift::GpuModelBuilder* model_builder);

// Single-token decode fusion for an edge stage (U1 or D2) of
// `odml.blocked_ffw`. Fuses input split/permute + optional input QDQ + K1
// quantized FullyConnected projections (one `[O, 1, 1, I]` weight tensor per
// part) + optional output QDQs + output permute/merge into a single GPU kernel
// dispatch.
struct BlockFfwEdgeFcSpec {
  // Number of edge partitions (`K1`) fused into this kernel dispatch.
  int num_parts = 0;
  // `num_parts` quantized weight descriptors with `weights.shape.h == 1`.
  std::vector<::ml_drift::GpuModelBuilder::Weights> part_weights;
  // Ordered activation QDQ operations applied inline after reading `src`.
  std::vector<::ml_drift::QuantizeAndDequantizeAttributes> src_qdqs;
  // Ordered activation QDQ operations applied inline before writing `dst`.
  std::vector<::ml_drift::QuantizeAndDequantizeAttributes> dst_qdqs;
};

// Appends a fused single-token `BlockFfwEdgeFcOp` GPU kernel computing all
// `spec.num_parts` edge projections from `src` into `dst`.
absl::Status AddBlockFfwEdgeFc(
    ::ml_drift::GpuModelBuilder* builder,
    const ::ml_drift::GpuModelBuilder::TensorHandle& src,
    const ::ml_drift::GpuModelBuilder::TensorHandle& dst,
    const BlockFfwEdgeFcSpec& spec);

// Matches and fuses single-token (`B == 1, W == 1`) U1 or D2 subgraphs emitted
// by BlockFfwOperationParser into a single GPU kernel via AddBlockFfwEdgeFc.
absl::Status TryBlockFfwEdgeFc(
    const ::ml_drift::CreateGpuModelInfo& create_info,
    const ::ml_drift::GpuInfo& gpu_info, const ::ml_drift::GraphFloat32& graph,
    ::ml_drift::NodeId first_node_id,
    const std::set<::ml_drift::NodeId>& consumed_nodes,
    std::set<::ml_drift::NodeId>* new_consumed_nodes,
    ::ml_drift::GpuModelBuilder* model_builder);
}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_BLOCK_FFW_KERNEL_H_
