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

#include "ml_drift_delegate/delegate/composite/fuse_short_conv_step.h"

#include <any>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/short_conv_step_parser.h"

namespace litert::ml_drift {
namespace {

struct ShortConvStepMatch {
  int32_t conv_l_cache = 3;

  ::ml_drift::Value* in_proj_val = nullptr;
  ::ml_drift::Value* conv_state_val = nullptr;
  ::ml_drift::Value* conv_weight_val = nullptr;
  ::ml_drift::Value* conv_bias_val = nullptr;

  ::ml_drift::Value* final_out_val = nullptr;
  ::ml_drift::Value* next_state_val = nullptr;

  ::ml_drift::BHWC weight_shape;
  std::vector<float> weight_data;
  ::ml_drift::DataType weight_dtype = ::ml_drift::DataType::FLOAT32;

  std::set<::ml_drift::NodeId> nodes_to_delete;
  ::ml_drift::NodeId insert_after_id = 0;
};

::ml_drift::Value* CreateConstTensor(
    ::ml_drift::GraphFloat32* graph, const ::ml_drift::BHWC& shape,
    const std::vector<float>& data, ::ml_drift::NodeId insert_after_id,
    ::ml_drift::Node** out_node,
    ::ml_drift::DataType dtype = ::ml_drift::DataType::FLOAT32) {
  ::ml_drift::Node* node = nullptr;
  if (!graph->InsertNodeAfter(insert_after_id, &node).ok() || !node) {
    return nullptr;
  }
  node->operation.type = ToString(::ml_drift::OperationType::CONSTANT);
  ::ml_drift::Value* value = graph->NewValue();
  value->tensor.type = dtype;
  value->tensor.shape = shape;
  value->tensor.ref = -1;
  graph->SetProducer(node->id, value->id);

  ::ml_drift::ConstTensorAttributes attr;
  ::ml_drift::TensorFloat32 tensor;
  tensor.shape = shape;
  tensor.data = data;
  attr.tensor = std::move(tensor);
  node->operation.attributes = std::move(attr);
  if (out_node) *out_node = node;
  return value;
}

int32_t DetermineConvLCache(const ::ml_drift::Value* state_val,
                           const ::ml_drift::Node* slice_node) {
  int state_len = 0;
  if (state_val) {
    const auto& shape = state_val->tensor.shape;
    if (shape.c > 1 && shape.c <= 16) {
      state_len = shape.c;
    } else if (shape.w > 1 && shape.w <= 16) {
      state_len = shape.w;
    } else if (shape.h > 1 && shape.h <= 16) {
      state_len = shape.h;
    } else if (shape.b > 1 && shape.b <= 16) {
      state_len = shape.b;
    }
  }
  if (state_len == 0 && slice_node &&
      slice_node->operation.attributes.has_value()) {
    if (slice_node->operation.attributes.type() ==
        typeid(::ml_drift::SliceAttributes)) {
      const auto& attr = std::any_cast<const ::ml_drift::SliceAttributes&>(
          slice_node->operation.attributes);
      if (attr.ends.c > attr.starts.c && attr.ends.c - attr.starts.c <= 16) {
        state_len = attr.ends.c - attr.starts.c;
      } else if (attr.ends.w > attr.starts.w &&
                 attr.ends.w - attr.starts.w <= 16) {
        state_len = attr.ends.w - attr.starts.w;
      }
    }
  }
  if (state_len <= 0) state_len = 2;
  // conv_L_cache represents filter size = state cache length + 1 (e.g. 2+1=3).
  return state_len + 1;
}

}  // namespace

absl::Status FuseShortConvStep(::ml_drift::GraphFloat32* graph) {
  if (!graph) return absl::OkStatus();

  std::vector<ShortConvStepMatch> matches;
  std::set<::ml_drift::NodeId> matched_node_ids;

  for (::ml_drift::Node* node : graph->nodes()) {
    if (!node || matched_node_ids.count(node->id)) continue;
    if (node->operation.type !=
        ToString(::ml_drift::OperationType::REDUCE_SUM)) {
      continue;
    }

    auto reduce_inputs = graph->FindInputs(node->id);
    if (reduce_inputs.empty()) continue;
    ::ml_drift::Value* reduce_in = reduce_inputs[0];

    ::ml_drift::Node* mul_conv = graph->FindProducer(reduce_in->id);
    if (!mul_conv ||
        mul_conv->operation.type != ToString(::ml_drift::OperationType::MUL)) {
      continue;
    }

    auto mul_inputs = graph->FindInputs(mul_conv->id);
    ::ml_drift::Value* win_val = nullptr;
    ::ml_drift::Value* conv_weight_val = nullptr;
    ::ml_drift::BHWC weight_shape;
    std::vector<float> weight_data;

    if (mul_inputs.size() == 1) {
      win_val = mul_inputs[0];
      if (mul_conv->operation.attributes.has_value() &&
          mul_conv->operation.attributes.type() ==
              typeid(::ml_drift::ElementwiseAttributes)) {
        const auto& attr =
            std::any_cast<const ::ml_drift::ElementwiseAttributes&>(
                mul_conv->operation.attributes);
        if (auto* t_f32 =
                std::get_if<::ml_drift::Tensor<::ml_drift::BHWC,
                                               ::ml_drift::DataType::FLOAT32>>(
                    &attr.param)) {
          weight_shape = t_f32->shape;
          weight_data = t_f32->data;
        } else if (auto* t_lin = std::get_if<::ml_drift::Tensor<
                       ::ml_drift::Linear, ::ml_drift::DataType::FLOAT32>>(
                       &attr.param)) {
          weight_data = t_lin->data;
        }
      }
    } else if (mul_inputs.size() >= 2) {
      for (::ml_drift::Value* in_val : mul_inputs) {
        ::ml_drift::Node* producer = graph->FindProducer(in_val->id);
        if (producer && producer->operation.type ==
                            ToString(::ml_drift::OperationType::CONCAT)) {
          win_val = in_val;
        } else {
          conv_weight_val = in_val;
        }
      }
    }

    if (!win_val || (!conv_weight_val && weight_data.empty())) continue;

    ::ml_drift::Node* concat_win_node = graph->FindProducer(win_val->id);
    if (!concat_win_node ||
        concat_win_node->operation.type !=
            ToString(::ml_drift::OperationType::CONCAT)) {
      continue;
    }

    auto win_inputs = graph->FindInputs(concat_win_node->id);
    if (win_inputs.size() < 2) continue;

    // Check if next_state is produced by slicing win_val directly
    ::ml_drift::Node* slice_state_node = nullptr;
    ::ml_drift::Value* next_state_val = nullptr;
    for (::ml_drift::Node* c : graph->FindConsumers(win_val->id)) {
      if (c &&
          (c->operation.type == ToString(::ml_drift::OperationType::SLICE) ||
           c->operation.type == "strided_slice")) {
        slice_state_node = c;
        auto outs = graph->FindOutputs(c->id);
        if (!outs.empty()) next_state_val = outs[0];
        break;
      }
    }

    // Separate conv_state and px from concat_win inputs
    ::ml_drift::Value* conv_state_val = nullptr;
    ::ml_drift::Value* px_val = nullptr;
    ::ml_drift::Node* px_reshape_node = nullptr;

    for (::ml_drift::Value* v : win_inputs) {
      ::ml_drift::Node* prod = graph->FindProducer(v->id);
      if (prod && (prod->operation.type ==
                       ToString(::ml_drift::OperationType::RESHAPE) ||
                   prod->operation.type ==
                       ToString(::ml_drift::OperationType::MUL))) {
        px_val = v;
        if (prod->operation.type ==
            ToString(::ml_drift::OperationType::RESHAPE)) {
          px_reshape_node = prod;
        }
      } else {
        conv_state_val = v;
      }
    }
    if (!conv_state_val || !px_val) {
      conv_state_val = win_inputs[0];
      px_val = win_inputs[1];
    }
    if (!conv_state_val || !px_val || !next_state_val || !slice_state_node) {
      continue;
    }

    // Match Pattern: Gated Short Conv (LFM2: b * x -> conv -> c * conv_out)
    ::ml_drift::Node* bx_mul_node = nullptr;
    if (px_reshape_node) {
      auto r_ins = graph->FindInputs(px_reshape_node->id);
      if (!r_ins.empty()) bx_mul_node = graph->FindProducer(r_ins[0]->id);
    } else {
      bx_mul_node = graph->FindProducer(px_val->id);
    }

    if (!bx_mul_node ||
        bx_mul_node->operation.type !=
            ToString(::ml_drift::OperationType::MUL)) {
      continue;
    }

    auto bx_inputs = graph->FindInputs(bx_mul_node->id);
    if (bx_inputs.size() != 2) continue;

    ::ml_drift::Node* slice_b = graph->FindProducer(bx_inputs[0]->id);
    ::ml_drift::Node* slice_x = graph->FindProducer(bx_inputs[1]->id);
    if (!slice_b || !slice_x) continue;
    if ((slice_b->operation.type !=
             ToString(::ml_drift::OperationType::SLICE) &&
         slice_b->operation.type != "strided_slice") ||
        (slice_x->operation.type !=
             ToString(::ml_drift::OperationType::SLICE) &&
         slice_x->operation.type != "strided_slice")) {
      continue;
    }

    auto b_ins = graph->FindInputs(slice_b->id);
    auto x_ins = graph->FindInputs(slice_x->id);
    if (b_ins.empty() || x_ins.empty() || b_ins[0] != x_ins[0]) continue;
    ::ml_drift::Value* in_proj_val = b_ins[0];

    // Validate in_proj dimensions and vector slice alignment (multiple of 4)
    const auto& in_shape = in_proj_val->tensor.shape;
    int total_c = in_shape.c > 0 ? in_shape.c : in_shape.w;
    if (total_c <= 0 || total_c % 3 != 0) continue;
    int hidden_size = total_c / 3;
    if (hidden_size % 4 != 0) continue;

    int32_t conv_l_cache =
        DetermineConvLCache(conv_state_val, slice_state_node);
    // The GPU kernel supports 3-tap convolution filters.
    if (conv_l_cache != 3) continue;

    // Normalize weight shape to [1, 1, hidden_size, conv_l_cache] so it matches
    // args.conv_weight.Read($0, 0, 0) in the GPU shader.
    if (!weight_data.empty()) {
      if (weight_shape.w != hidden_size && weight_shape.b != hidden_size) {
        if (weight_data.size() == hidden_size * conv_l_cache) {
          weight_shape = ::ml_drift::BHWC(1, 1, hidden_size, conv_l_cache);
        } else {
          continue;
        }
      }
    }

    auto red_outs = graph->FindOutputs(node->id);
    if (red_outs.empty()) continue;
    ::ml_drift::Value* red_out = red_outs[0];
    ::ml_drift::Node* red_reshape = nullptr;

    // Optional reshape after reduce_sum: [1, 1, H, 1] -> [1, 1, 1, H]
    for (::ml_drift::Node* c : graph->FindConsumers(red_out->id)) {
      if (c &&
          c->operation.type == ToString(::ml_drift::OperationType::RESHAPE)) {
        red_reshape = c;
        auto outs = graph->FindOutputs(c->id);
        if (!outs.empty()) red_out = outs[0];
        break;
      }
    }

    // Optional bias addition: conv_out + bias
    ::ml_drift::Node* bias_add_node = nullptr;
    ::ml_drift::Value* conv_bias_val = nullptr;
    for (::ml_drift::Node* c : graph->FindConsumers(red_out->id)) {
      if (c && c->operation.type == ToString(::ml_drift::OperationType::ADD)) {
        auto add_ins = graph->FindInputs(c->id);
        if (add_ins.size() == 2) {
          ::ml_drift::Value* other =
              (add_ins[0] == red_out) ? add_ins[1] : add_ins[0];
          bias_add_node = c;
          conv_bias_val = other;
          auto outs = graph->FindOutputs(c->id);
          if (!outs.empty()) red_out = outs[0];
          break;
        }
      }
    }

    // Gating multiply: conv_out * c
    ::ml_drift::Node* gating_mul = nullptr;
    ::ml_drift::Node* slice_c = nullptr;
    for (::ml_drift::Node* c : graph->FindConsumers(red_out->id)) {
      if (c && c->operation.type == ToString(::ml_drift::OperationType::MUL)) {
        auto g_ins = graph->FindInputs(c->id);
        if (g_ins.size() == 2) {
          ::ml_drift::Value* c_val =
              (g_ins[0] == red_out) ? g_ins[1] : g_ins[0];
          slice_c = graph->FindProducer(c_val->id);
          if (slice_c &&
              (slice_c->operation.type ==
                   ToString(::ml_drift::OperationType::SLICE) ||
               slice_c->operation.type == "strided_slice")) {
            auto c_ins = graph->FindInputs(slice_c->id);
            if (!c_ins.empty() && c_ins[0] == in_proj_val) {
              gating_mul = c;
              break;
            }
          }
        }
      }
    }
    if (!gating_mul || !slice_c) continue;

    auto g_outs = graph->FindOutputs(gating_mul->id);
    if (g_outs.empty()) continue;

    // Check that intermediate values have no external consumers
    if (graph->FindConsumers(bx_inputs[0]->id).size() != 1) continue;
    if (graph->FindConsumers(bx_inputs[1]->id).size() != 1) continue;
    auto slice_c_outs = graph->FindOutputs(slice_c->id);
    if (slice_c_outs.empty() ||
        graph->FindConsumers(slice_c_outs[0]->id).size() != 1) {
      continue;
    }
    auto bx_outs = graph->FindOutputs(bx_mul_node->id);
    if (bx_outs.empty() || graph->FindConsumers(bx_outs[0]->id).size() != 1) {
      continue;
    }
    if (px_reshape_node) {
      auto px_outs = graph->FindOutputs(px_reshape_node->id);
      if (px_outs.empty() ||
          graph->FindConsumers(px_outs[0]->id).size() != 1) {
        continue;
      }
    }
    // win_val should only be consumed by slice_state_node and mul_conv
    if (graph->FindConsumers(win_val->id).size() > 2) continue;
    if (graph->FindConsumers(reduce_in->id).size() != 1) continue;
    if (red_reshape) {
      if (graph->FindConsumers(red_outs[0]->id).size() != 1) continue;
    }
    if (bias_add_node) {
      auto bias_ins = graph->FindInputs(bias_add_node->id);
      ::ml_drift::Value* add_in_val =
          (bias_ins[0] == conv_bias_val) ? bias_ins[1] : bias_ins[0];
      if (graph->FindConsumers(add_in_val->id).size() != 1) continue;
    }

    ShortConvStepMatch match;
    match.in_proj_val = in_proj_val;
    match.conv_state_val = conv_state_val;
    match.conv_weight_val = conv_weight_val;
    match.conv_bias_val = conv_bias_val;
    match.weight_shape = weight_shape;
    match.weight_data = weight_data;
    match.weight_dtype = in_proj_val->tensor.type;
    match.final_out_val = g_outs[0];
    match.next_state_val = next_state_val;
    match.conv_l_cache = conv_l_cache;
    match.insert_after_id = gating_mul->id;

    match.nodes_to_delete = {
        slice_b->id,         slice_x->id,        slice_c->id,
        bx_mul_node->id,     concat_win_node->id, slice_state_node->id,
        mul_conv->id,        node->id,           gating_mul->id};
    if (px_reshape_node) {
      match.nodes_to_delete.insert(px_reshape_node->id);
    }
    if (red_reshape) {
      match.nodes_to_delete.insert(red_reshape->id);
    }
    if (bias_add_node) {
      match.nodes_to_delete.insert(bias_add_node->id);
    }

    for (::ml_drift::NodeId nid : match.nodes_to_delete) {
      matched_node_ids.insert(nid);
    }
    matches.push_back(std::move(match));
  }

  if (matches.empty()) {
    return absl::OkStatus();
  }

  ABSL_LOG(INFO) << "[FuseShortConvStep] Found " << matches.size()
                 << " short_conv_step clusters to fuse into single kernels!";

  for (const auto& match : matches) {
    ::ml_drift::Node* weight_const_node = nullptr;
    ::ml_drift::Value* conv_weight_val = match.conv_weight_val;

    if (!conv_weight_val && !match.weight_data.empty()) {
      conv_weight_val = CreateConstTensor(
          graph, match.weight_shape, match.weight_data, match.insert_after_id,
          &weight_const_node, match.weight_dtype);
    }
    if (!conv_weight_val) continue;

    ::ml_drift::NodeId insert_id =
        weight_const_node ? weight_const_node->id : match.insert_after_id;
    ::ml_drift::Node* fused_node = nullptr;
    ABSL_CHECK_OK(graph->InsertNodeAfter(insert_id, &fused_node));
    fused_node->operation.type = kShortConvStepType;

    ShortConvStepAttributes attr;
    attr.conv_L_cache = match.conv_l_cache;
    fused_node->operation.attributes = std::move(attr);

    // Inputs: in_proj, conv_state, conv_weight, (conv_bias)
    graph->AddConsumer(fused_node->id, match.in_proj_val->id);
    graph->AddConsumer(fused_node->id, match.conv_state_val->id);
    graph->AddConsumer(fused_node->id, conv_weight_val->id);
    if (match.conv_bias_val) {
      graph->AddConsumer(fused_node->id, match.conv_bias_val->id);
    }

    // Outputs: dst, next_state
    graph->SetProducer(fused_node->id, match.final_out_val->id);
    graph->SetProducer(fused_node->id, match.next_state_val->id);

    // Clean up replaced intermediate nodes
    for (::ml_drift::NodeId id : match.nodes_to_delete) {
      if (id != fused_node->id &&
          (!weight_const_node || id != weight_const_node->id)) {
        (void)graph->DeleteNode(id);
      }
    }
  }

  return absl::OkStatus();
}

namespace ir {
namespace {

struct IrShortConvStepMatch {
  int32_t conv_l_cache = 3;

  ::ml_drift::ir::IrTensorId in_proj_id = 0;
  ::ml_drift::ir::IrTensorId conv_state_id = 0;
  ::ml_drift::ir::IrTensorId conv_weight_id = 0;
  std::optional<::ml_drift::ir::IrTensorId> conv_bias_id;

  ::ml_drift::ir::IrTensorId final_out_id = 0;
  ::ml_drift::ir::IrTensorId next_state_id = 0;

  ::ml_drift::BHWC weight_shape;
  std::vector<float> weight_data;

  ::ml_drift::ir::IrOpId primary_op_id = 0;
  std::set<::ml_drift::ir::IrOpId> ops_to_delete;
  std::set<::ml_drift::ir::IrTensorId> tensors_to_delete;
};

::ml_drift::ir::IrTensor* CreateIrConstTensor(
    ::ml_drift::ir::IrModel* ir_model, const ::ml_drift::BHWC& shape,
    const std::vector<float>& data) {
  ::ml_drift::ir::IrOp* const_op = ir_model->add_op();
  const_op->name = ToString(::ml_drift::OperationType::CONSTANT);

  ::ml_drift::ir::IrTensor* const_tensor =
      ir_model->add_tensor(::ml_drift::DataType::FLOAT32, shape);
  ir_model->SetProducer(const_tensor->id, const_op->id);

  ::ml_drift::TensorFloat32 tensor;
  tensor.shape = shape;
  tensor.data = data;

  ::ml_drift::ConstTensorAttributes attr;
  attr.tensor = std::move(tensor);
  const_op->attr = std::move(attr);
  return const_tensor;
}

::ml_drift::ir::IrTensorId PeelForward(
    const ::ml_drift::ir::IrModel* model, ::ml_drift::ir::IrTensorId tid,
    std::vector<::ml_drift::ir::IrOp*>* intermediate_ops = nullptr) {
  while (true) {
    auto consumers = model->FindConsumers(tid);
    if (consumers.size() == 1 &&
        (consumers[0]->name == "reshape" ||
         consumers[0]->name == "transpose" ||
         consumers[0]->name == ToString(::ml_drift::OperationType::RESHAPE) ||
         consumers[0]->name ==
             ToString(::ml_drift::OperationType::TRANSPOSE)) &&
        !consumers[0]->outputs.empty()) {
      if (intermediate_ops) intermediate_ops->push_back(consumers[0]);
      tid = consumers[0]->outputs[0];
    } else {
      break;
    }
  }
  return tid;
}

::ml_drift::ir::IrTensorId PeelBackward(
    const ::ml_drift::ir::IrModel* model, ::ml_drift::ir::IrTensorId tid,
    std::vector<::ml_drift::ir::IrOp*>* intermediate_ops = nullptr) {
  while (true) {
    auto* prod = model->FindProducer(tid);
    if (prod &&
        (prod->name == "reshape" ||
         prod->name == "transpose" ||
         prod->name == ToString(::ml_drift::OperationType::RESHAPE) ||
         prod->name == ToString(::ml_drift::OperationType::TRANSPOSE)) &&
        !prod->inputs.empty()) {
      if (intermediate_ops) intermediate_ops->push_back(prod);
      tid = prod->inputs[0];
    } else {
      break;
    }
  }
  return tid;
}

int32_t DetermineIrConvLCache(const ::ml_drift::ir::IrTensor* state_tensor,
                             const ::ml_drift::ir::IrOp* slice_op) {
  int state_len = 0;
  if (state_tensor) {
    auto shape = state_tensor->desc.GetBHWDCShape();
    if (shape.c > 1 && shape.c <= 16) {
      state_len = shape.c;
    } else if (shape.w > 1 && shape.w <= 16) {
      state_len = shape.w;
    } else if (shape.h > 1 && shape.h <= 16) {
      state_len = shape.h;
    } else if (shape.b > 1 && shape.b <= 16) {
      state_len = shape.b;
    }
  }
  if (state_len == 0 && slice_op && slice_op->attr.has_value()) {
    if (slice_op->attr.type() == typeid(::ml_drift::SliceAttributes)) {
      const auto& attr =
          std::any_cast<const ::ml_drift::SliceAttributes&>(slice_op->attr);
      if (attr.ends.c > attr.starts.c && attr.ends.c - attr.starts.c <= 16) {
        state_len = attr.ends.c - attr.starts.c;
      } else if (attr.ends.w > attr.starts.w &&
                 attr.ends.w - attr.starts.w <= 16) {
        state_len = attr.ends.w - attr.starts.w;
      }
    } else if (slice_op->attr.type() == typeid(::ml_drift::Slice3DAttributes)) {
      const auto& attr =
          std::any_cast<const ::ml_drift::Slice3DAttributes&>(slice_op->attr);
      if (attr.ends.c > attr.starts.c && attr.ends.c - attr.starts.c <= 16) {
        state_len = attr.ends.c - attr.starts.c;
      } else if (attr.ends.w > attr.starts.w &&
                 attr.ends.w - attr.starts.w <= 16) {
        state_len = attr.ends.w - attr.starts.w;
      }
    }
  }
  if (state_len <= 0) state_len = 2;
  return state_len + 1;
}

}  // namespace

absl::Status FuseShortConvStep(::ml_drift::ir::IrModel* model) {
  if (!model) return absl::OkStatus();

  std::vector<IrShortConvStepMatch> matches;
  std::set<::ml_drift::ir::IrOpId> matched_op_ids;

  for (const auto& op_ptr : model->ops()) {
    if (!op_ptr || matched_op_ids.count(op_ptr->id)) continue;
    if (op_ptr->name != ToString(::ml_drift::OperationType::REDUCE_SUM) &&
        op_ptr->name != "reduce_sum" && op_ptr->name != "sum") {
      continue;
    }
    ::ml_drift::ir::IrOp* reduce_sum_op = op_ptr.get();
    if (reduce_sum_op->inputs.empty() || reduce_sum_op->outputs.empty()) {
      continue;
    }

    std::vector<::ml_drift::ir::IrOp*> intermediate_ops;

    ::ml_drift::ir::IrTensorId reduce_in_tid =
        PeelBackward(model, reduce_sum_op->inputs[0], &intermediate_ops);
    ::ml_drift::ir::IrOp* mul_conv = model->FindProducer(reduce_in_tid);
    if (!mul_conv ||
        (mul_conv->name != ToString(::ml_drift::OperationType::MUL) &&
         mul_conv->name != "mul")) {
      continue;
    }

    ::ml_drift::ir::IrTensorId win_tid = 0;
    ::ml_drift::ir::IrTensorId conv_weight_tid = 0;
    ::ml_drift::BHWC weight_shape;
    std::vector<float> weight_data;

    if (mul_conv->inputs.size() == 1) {
      win_tid = mul_conv->inputs[0];
      if (mul_conv->attr.has_value() &&
          mul_conv->attr.type() == typeid(::ml_drift::ElementwiseAttributes)) {
        const auto& attr =
            std::any_cast<const ::ml_drift::ElementwiseAttributes&>(
                mul_conv->attr);
        if (auto* t_f32 =
                std::get_if<::ml_drift::Tensor<::ml_drift::BHWC,
                                               ::ml_drift::DataType::FLOAT32>>(
                    &attr.param)) {
          weight_shape = t_f32->shape;
          weight_data = t_f32->data;
        } else if (auto* t_lin = std::get_if<::ml_drift::Tensor<
                       ::ml_drift::Linear, ::ml_drift::DataType::FLOAT32>>(
                       &attr.param)) {
          weight_data = t_lin->data;
        }
      }
    } else if (mul_conv->inputs.size() >= 2) {
      for (::ml_drift::ir::IrTensorId in_tid : mul_conv->inputs) {
        ::ml_drift::ir::IrTensorId peeled =
            PeelBackward(model, in_tid, &intermediate_ops);
        ::ml_drift::ir::IrOp* prod = model->FindProducer(peeled);
        if (prod && (prod->name == "concat" ||
                     prod->name ==
                         ToString(::ml_drift::OperationType::CONCAT))) {
          win_tid = peeled;
        } else {
          conv_weight_tid = peeled;
        }
      }
    }

    if (win_tid == 0) continue;
    win_tid = PeelBackward(model, win_tid, &intermediate_ops);

    ::ml_drift::ir::IrOp* concat_win_node = model->FindProducer(win_tid);
    if (!concat_win_node ||
        (concat_win_node->name != "concat" &&
         concat_win_node->name !=
             ToString(::ml_drift::OperationType::CONCAT))) {
      continue;
    }
    if (concat_win_node->inputs.size() < 2) continue;

    ::ml_drift::ir::IrOp* slice_state_node = nullptr;
    ::ml_drift::ir::IrTensorId next_state_tid = 0;
    for (::ml_drift::ir::IrOp* c : model->FindConsumers(win_tid)) {
      if (!c) continue;
      if (c->name == "slice" || c->name == "strided_slice" ||
          c->name == ToString(::ml_drift::OperationType::SLICE)) {
        slice_state_node = c;
        if (!c->outputs.empty()) next_state_tid = c->outputs[0];
        break;
      }
    }
    if (!slice_state_node || next_state_tid == 0) continue;

    ::ml_drift::ir::IrTensorId conv_state_tid = 0;
    ::ml_drift::ir::IrTensorId px_tid = 0;
    for (::ml_drift::ir::IrTensorId v : concat_win_node->inputs) {
      ::ml_drift::ir::IrTensorId peeled =
          PeelBackward(model, v, &intermediate_ops);
      ::ml_drift::ir::IrOp* prod = model->FindProducer(peeled);
      if (prod && (prod->name == "mul" ||
                   prod->name == ToString(::ml_drift::OperationType::MUL))) {
        px_tid = peeled;
      } else {
        conv_state_tid = peeled;
      }
    }
    if (conv_state_tid == 0 || px_tid == 0) {
      conv_state_tid =
          PeelBackward(model, concat_win_node->inputs[0], &intermediate_ops);
      px_tid =
          PeelBackward(model, concat_win_node->inputs[1], &intermediate_ops);
    }
    if (conv_state_tid == 0 || px_tid == 0) continue;

    ::ml_drift::ir::IrOp* bx_mul_node = model->FindProducer(px_tid);
    if (!bx_mul_node ||
        (bx_mul_node->name != "mul" &&
         bx_mul_node->name != ToString(::ml_drift::OperationType::MUL))) {
      continue;
    }
    if (bx_mul_node->inputs.size() != 2) continue;

    ::ml_drift::ir::IrTensorId b_in =
        PeelBackward(model, bx_mul_node->inputs[0], &intermediate_ops);
    ::ml_drift::ir::IrTensorId x_in =
        PeelBackward(model, bx_mul_node->inputs[1], &intermediate_ops);
    ::ml_drift::ir::IrOp* slice_b = model->FindProducer(b_in);
    ::ml_drift::ir::IrOp* slice_x = model->FindProducer(x_in);
    if (!slice_b || !slice_x) continue;
    if ((slice_b->name != "slice" && slice_b->name != "strided_slice" &&
         slice_b->name != ToString(::ml_drift::OperationType::SLICE)) ||
        (slice_x->name != "slice" && slice_x->name != "strided_slice" &&
         slice_x->name != ToString(::ml_drift::OperationType::SLICE))) {
      continue;
    }

    if (slice_b->inputs.empty() || slice_x->inputs.empty()) continue;
    ::ml_drift::ir::IrTensorId in_proj_tid_b =
        PeelBackward(model, slice_b->inputs[0], &intermediate_ops);
    ::ml_drift::ir::IrTensorId in_proj_tid_x =
        PeelBackward(model, slice_x->inputs[0], &intermediate_ops);
    if (in_proj_tid_b != in_proj_tid_x) continue;
    ::ml_drift::ir::IrTensorId in_proj_tid = in_proj_tid_b;

    const auto* in_proj_tensor = model->tensor(in_proj_tid);
    if (!in_proj_tensor) continue;
    auto in_shape = in_proj_tensor->desc.GetBHWDCShape();
    int total_c = in_shape.c > 0 ? in_shape.c : in_shape.w;
    if (total_c <= 0 || total_c % 3 != 0) continue;
    int hidden_size = total_c / 3;
    if (hidden_size % 4 != 0) continue;

    int32_t conv_l_cache = DetermineIrConvLCache(
        model->tensor(conv_state_tid), slice_state_node);
    if (conv_l_cache != 3) continue;

    if (!weight_data.empty()) {
      if (weight_shape.w != hidden_size && weight_shape.b != hidden_size) {
        if (weight_data.size() == hidden_size * conv_l_cache) {
          weight_shape = ::ml_drift::BHWC(1, 1, hidden_size, conv_l_cache);
        } else {
          continue;
        }
      }
    } else if (conv_weight_tid != 0) {
      auto* w_prod = model->FindProducer(conv_weight_tid);
      if (w_prod && (w_prod->name == "constant" ||
                     w_prod->name ==
                         ToString(::ml_drift::OperationType::CONSTANT))) {
        if (w_prod->attr.type() ==
            typeid(::ml_drift::ConstTensorAttributes)) {
          const auto& c_attr =
              std::any_cast<const ::ml_drift::ConstTensorAttributes&>(
                  w_prod->attr);
          if (auto* t_f32 =
                  std::get_if<::ml_drift::TensorFloat32>(&c_attr.tensor)) {
            weight_data = t_f32->data;
            weight_shape = ::ml_drift::BHWC(1, 1, hidden_size, conv_l_cache);
          } else if (auto* t_f16 = std::get_if<::ml_drift::TensorFloat16>(
                         &c_attr.tensor)) {
            weight_data.resize(t_f16->data.size());
            for (size_t i = 0; i < t_f16->data.size(); ++i) {
              weight_data[i] = static_cast<float>(t_f16->data[i]);
            }
            weight_shape = ::ml_drift::BHWC(1, 1, hidden_size, conv_l_cache);
          }
        }
      }
    }

    ::ml_drift::ir::IrTensorId curr_conv_out =
        PeelForward(model, reduce_sum_op->outputs[0], &intermediate_ops);

    ::ml_drift::ir::IrOp* bias_add_op = nullptr;
    std::optional<::ml_drift::ir::IrTensorId> conv_bias_tid;
    for (::ml_drift::ir::IrOp* c : model->FindConsumers(curr_conv_out)) {
      if (c && (c->name == "add" ||
                c->name == ToString(::ml_drift::OperationType::ADD)) &&
          c->inputs.size() == 2) {
        bias_add_op = c;
        conv_bias_tid = (c->inputs[0] == curr_conv_out) ? c->inputs[1]
                                                        : c->inputs[0];
        if (!c->outputs.empty()) {
          curr_conv_out =
              PeelForward(model, c->outputs[0], &intermediate_ops);
        }
        break;
      }
    }

    ::ml_drift::ir::IrOp* gating_mul = nullptr;
    ::ml_drift::ir::IrOp* slice_c = nullptr;
    for (::ml_drift::ir::IrOp* c : model->FindConsumers(curr_conv_out)) {
      if (c && (c->name == "mul" ||
                c->name == ToString(::ml_drift::OperationType::MUL)) &&
          c->inputs.size() == 2) {
        ::ml_drift::ir::IrTensorId c_val_tid =
            (c->inputs[0] == curr_conv_out) ? c->inputs[1] : c->inputs[0];
        c_val_tid = PeelBackward(model, c_val_tid, &intermediate_ops);
        ::ml_drift::ir::IrOp* prod_c = model->FindProducer(c_val_tid);
        if (prod_c &&
            (prod_c->name == "slice" || prod_c->name == "strided_slice" ||
             prod_c->name == ToString(::ml_drift::OperationType::SLICE))) {
          if (!prod_c->inputs.empty() &&
              PeelBackward(model, prod_c->inputs[0], &intermediate_ops) ==
                  in_proj_tid) {
            gating_mul = c;
            slice_c = prod_c;
            break;
          }
        }
      }
    }
    if (!gating_mul || !slice_c || gating_mul->outputs.empty()) continue;

    ::ml_drift::ir::IrTensorId final_out_tid =
        PeelForward(model, gating_mul->outputs[0], &intermediate_ops);

    IrShortConvStepMatch match;
    match.in_proj_id = in_proj_tid;
    match.conv_state_id = conv_state_tid;
    match.conv_weight_id = conv_weight_tid;
    match.conv_bias_id = conv_bias_tid;
    match.final_out_id = final_out_tid;
    match.next_state_id = next_state_tid;
    match.conv_l_cache = conv_l_cache;
    match.weight_shape = weight_shape;
    match.weight_data = std::move(weight_data);
    match.primary_op_id = slice_b->id;

    for (::ml_drift::ir::IrOp* op :
         {slice_b, slice_x, slice_c, bx_mul_node, concat_win_node,
          slice_state_node, mul_conv, reduce_sum_op, gating_mul}) {
      if (op) {
        match.ops_to_delete.insert(op->id);
        matched_op_ids.insert(op->id);
      }
    }
    if (bias_add_op) {
      match.ops_to_delete.insert(bias_add_op->id);
      matched_op_ids.insert(bias_add_op->id);
    }
    for (::ml_drift::ir::IrOp* op : intermediate_ops) {
      if (op) {
        match.ops_to_delete.insert(op->id);
        matched_op_ids.insert(op->id);
      }
    }

    for (::ml_drift::ir::IrOpId op_id : match.ops_to_delete) {
      const auto* op = model->op(op_id);
      if (!op) continue;
      for (::ml_drift::ir::IrTensorId tid : op->outputs) {
        if (tid != match.final_out_id && tid != match.next_state_id) {
          match.tensors_to_delete.insert(tid);
        }
      }
    }

    matches.push_back(std::move(match));
  }

  if (matches.empty()) {
    return absl::OkStatus();
  }

  ABSL_LOG(INFO) << "[FuseShortConvStep IrModel] Found " << matches.size()
                 << " short_conv_step clusters to fuse into single kernels!";

  for (auto& match : matches) {
    ::ml_drift::ir::IrTensorId conv_weight_tid = match.conv_weight_id;
    if (conv_weight_tid == 0 && !match.weight_data.empty()) {
      auto* weight_tensor = CreateIrConstTensor(
          model, match.weight_shape, match.weight_data);
      if (!weight_tensor) {
        return absl::InternalError(
            "Failed to create constant tensor for short_conv_step weights in "
            "IrModel");
      }
      conv_weight_tid = weight_tensor->id;
    }
    if (conv_weight_tid == 0) continue;

    ABSL_RETURN_IF_ERROR(model->DeleteOp(match.primary_op_id));
    ::ml_drift::ir::IrOp* fused_op = model->replace_op(match.primary_op_id);
    if (!fused_op) {
      return absl::InternalError("Failed to replace op at primary_op_id");
    }
    fused_op->name = "short_conv_step";
    ShortConvStepAttributes attr;
    attr.conv_L_cache = match.conv_l_cache;
    fused_op->attr = std::move(attr);

    model->AddConsumer(match.in_proj_id, fused_op->id);
    model->AddConsumer(match.conv_state_id, fused_op->id);
    model->AddConsumer(conv_weight_tid, fused_op->id);
    if (match.conv_bias_id.has_value()) {
      model->AddConsumer(*match.conv_bias_id, fused_op->id);
    }

    model->SetProducer(match.final_out_id, fused_op->id);
    model->SetProducer(match.next_state_id, fused_op->id);

    for (::ml_drift::ir::IrOpId op_id : match.ops_to_delete) {
      if (op_id != match.primary_op_id) {
        ABSL_RETURN_IF_ERROR(model->DeleteOp(op_id));
      }
    }

    for (::ml_drift::ir::IrTensorId tid : match.tensors_to_delete) {
      if (!model->IsGraphInput(tid) && !model->IsGraphOutput(tid)) {
        (void)model->DeleteTensor(tid);
      }
    }
  }

  return absl::OkStatus();
}

}  // namespace ir

}  // namespace litert::ml_drift
