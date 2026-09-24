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

#include "ml_drift_delegate/delegate/composite/fuse_sdpa_transposed_reshape.h"

#include <any>
#include <cstddef>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/sdpa_transposed_parser.h"

namespace litert::ml_drift {
namespace {

bool IsDecode4dOutput(const ::ml_drift::BHWC& q_shape,
                      const ::ml_drift::BHWC& mid_shape) {
  return mid_shape.b == 1 && mid_shape.w == 1 && mid_shape.h > 1 &&
         mid_shape.h == q_shape.h && mid_shape.c == q_shape.c;
}

bool IsFlattenedDecodeShape(const ::ml_drift::BHWC& mid_shape,
                            const ::ml_drift::BHWC& out_shape) {
  return out_shape.b == 1 && out_shape.h == 1 && out_shape.w == 1 &&
         out_shape.c == mid_shape.h * mid_shape.c;
}

bool IsUnitWidthSwapTranspose(const std::any& attr) {
  if (attr.type() != typeid(::ml_drift::TransposeAttributes)) {
    return false;
  }
  const auto& t_attr =
      std::any_cast<const ::ml_drift::TransposeAttributes&>(attr);
  return t_attr.perm == ::ml_drift::BHWC(0, 2, 1, 3);
}

}  // namespace

absl::Status FuseSdpaTransposedReshape(::ml_drift::GraphFloat32* graph) {
  if (!graph) return absl::OkStatus();

  const std::string reshape_op_name =
      ToString(::ml_drift::OperationType::RESHAPE);
  const std::string transpose_op_name =
      ToString(::ml_drift::OperationType::TRANSPOSE);

  std::vector<::ml_drift::NodeId> sdpa_node_ids;
  for (::ml_drift::Node* node : graph->nodes()) {
    if (node && node->operation.type == kSdpaTransposedType) {
      sdpa_node_ids.push_back(node->id);
    }
  }

  for (::ml_drift::NodeId sdpa_id : sdpa_node_ids) {
    ::ml_drift::Node* node = graph->GetNode(sdpa_id);
    if (!node || node->operation.type != kSdpaTransposedType) {
      continue;
    }
    const auto inputs = graph->FindInputs(node->id);
    const auto outputs = graph->FindOutputs(node->id);
    if (inputs.empty() || outputs.size() != 1) {
      continue;
    }
    ::ml_drift::Value* q_val = inputs[0];
    ::ml_drift::Value* mid_val = outputs[0];
    if (!q_val || !mid_val) continue;

    const auto& q_shape = q_val->tensor.shape;
    const auto& mid_shape = mid_val->tensor.shape;
    if (!IsDecode4dOutput(q_shape, mid_shape)) {
      continue;
    }
    if (graph->IsGraphOutput(mid_val->id)) {
      continue;
    }
    const auto mid_consumers = graph->FindConsumers(mid_val->id);
    if (mid_consumers.size() != 1) {
      continue;
    }
    ::ml_drift::Node* c0 = mid_consumers[0];
    if (!c0) continue;

    if (c0->operation.type == reshape_op_name) {
      const auto c0_outs = graph->FindOutputs(c0->id);
      if (c0_outs.size() == 1 &&
          IsFlattenedDecodeShape(mid_shape, c0_outs[0]->tensor.shape)) {
        ABSL_RETURN_IF_ERROR(::ml_drift::RemoveSimpleNodeKeepOutput(graph, c0));
      }
    } else if (c0->operation.type == transpose_op_name &&
               IsUnitWidthSwapTranspose(c0->operation.attributes)) {
      const auto c0_outs = graph->FindOutputs(c0->id);
      if (c0_outs.size() != 1) continue;
      ::ml_drift::Value* perm_val = c0_outs[0];
      if (!perm_val || graph->IsGraphOutput(perm_val->id)) continue;
      const auto perm_consumers = graph->FindConsumers(perm_val->id);
      if (perm_consumers.size() != 1) continue;
      ::ml_drift::Node* c1 = perm_consumers[0];
      if (!c1 || c1->operation.type != reshape_op_name) continue;
      const auto c1_outs = graph->FindOutputs(c1->id);
      if (c1_outs.size() == 1 &&
          IsFlattenedDecodeShape(mid_shape, c1_outs[0]->tensor.shape)) {
        ABSL_RETURN_IF_ERROR(::ml_drift::RemoveSimpleNodeKeepOutput(graph, c0));
        ABSL_RETURN_IF_ERROR(::ml_drift::RemoveSimpleNodeKeepOutput(graph, c1));
      }
    }
  }
  return absl::OkStatus();
}

namespace ir {

absl::Status FuseSdpaTransposedReshape(::ml_drift::ir::IrModel* model) {
  if (!model) return absl::OkStatus();

  const std::string reshape_op_name =
      ToString(::ml_drift::OperationType::RESHAPE);
  const std::string transpose_op_name =
      ToString(::ml_drift::OperationType::TRANSPOSE);

  for (size_t i = 0; i < model->ops().size(); ++i) {
    ::ml_drift::ir::IrOp* sdpa_op = model->ops()[i].get();
    if (!sdpa_op || sdpa_op->name != kSdpaTransposedType) {
      continue;
    }
    if (sdpa_op->inputs.empty() || sdpa_op->outputs.size() != 1) {
      continue;
    }
    const ::ml_drift::ir::IrTensorId q_id = sdpa_op->inputs[0];
    const ::ml_drift::ir::IrTensorId mid_id = sdpa_op->outputs[0];
    const auto* q_tensor = model->tensor(q_id);
    const auto* mid_tensor = model->tensor(mid_id);
    if (!q_tensor || !mid_tensor) continue;

    const auto q_shape = q_tensor->desc.GetBHWCShape();
    const auto mid_shape = mid_tensor->desc.GetBHWCShape();
    if (!IsDecode4dOutput(q_shape, mid_shape)) {
      continue;
    }
    if (model->IsGraphOutput(mid_id)) {
      continue;
    }
    const auto mid_consumers = model->FindConsumers(mid_id);
    if (mid_consumers.size() != 1) {
      continue;
    }
    ::ml_drift::ir::IrOp* c0 = mid_consumers[0];
    if (!c0 || c0->inputs.size() != 1 || c0->outputs.size() != 1) continue;

    if (c0->name == reshape_op_name) {
      const ::ml_drift::ir::IrTensorId out_id = c0->outputs[0];
      const auto* out_tensor = model->tensor(out_id);
      if (out_tensor &&
          IsFlattenedDecodeShape(mid_shape, out_tensor->desc.GetBHWCShape())) {
        sdpa_op->outputs.clear();
        if (auto* mid_mut = model->GetMutableTensor(mid_id)) {
          mid_mut->producer.reset();
        }
        c0->outputs.clear();
        model->SetProducer(out_id, sdpa_op->id);
        ABSL_RETURN_IF_ERROR(model->RemoveOp(c0->id));
      }
    } else if (c0->name == transpose_op_name &&
               IsUnitWidthSwapTranspose(c0->attr)) {
      const ::ml_drift::ir::IrTensorId perm_id = c0->outputs[0];
      if (model->IsGraphOutput(perm_id)) continue;
      const auto perm_consumers = model->FindConsumers(perm_id);
      if (perm_consumers.size() != 1) continue;
      ::ml_drift::ir::IrOp* c1 = perm_consumers[0];
      if (!c1 || c1->name != reshape_op_name || c1->inputs.size() != 1 ||
          c1->outputs.size() != 1) {
        continue;
      }
      const ::ml_drift::ir::IrTensorId out_id = c1->outputs[0];
      const auto* out_tensor = model->tensor(out_id);
      if (out_tensor &&
          IsFlattenedDecodeShape(mid_shape, out_tensor->desc.GetBHWCShape())) {
        // Save op IDs before RemoveOp(c1_id), which erases from model->ops_
        // and invalidates the raw IrOp* pointer c0.
        const auto c0_id = c0->id;
        const auto c1_id = c1->id;
        sdpa_op->outputs.clear();
        if (auto* mid_mut = model->GetMutableTensor(mid_id)) {
          mid_mut->producer.reset();
        }
        c1->outputs.clear();
        model->SetProducer(out_id, sdpa_op->id);
        ABSL_RETURN_IF_ERROR(model->RemoveOp(c1_id));
        ABSL_RETURN_IF_ERROR(model->RemoveOp(c0_id));
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace ir
}  // namespace litert::ml_drift
