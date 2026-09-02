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

#include "ml_drift_delegate/delegate/composite/fuse_qkv_norm_rope.h"

#include <any>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/qkv_norm_rope_parser.h"

namespace litert::ml_drift {
namespace {

struct QkvNormRopeMatch {
  ::ml_drift::Node* rope_q = nullptr;
  ::ml_drift::Node* pre_rope_q = nullptr;
  ::ml_drift::Node* norm_q = nullptr;
  ::ml_drift::Node* pre_norm_q = nullptr;
  ::ml_drift::Node* slice_q = nullptr;

  ::ml_drift::Node* rope_k = nullptr;
  ::ml_drift::Node* pre_rope_k = nullptr;
  ::ml_drift::Node* norm_k = nullptr;
  ::ml_drift::Node* pre_norm_k = nullptr;
  ::ml_drift::Node* slice_k = nullptr;

  std::vector<::ml_drift::Node*> v_intermediates;
  ::ml_drift::Node* slice_v = nullptr;

  ::ml_drift::Value* qkv_val = nullptr;
  ::ml_drift::Value* pos_val = nullptr;

  ::ml_drift::Value* q_out_val = nullptr;
  ::ml_drift::Value* k_out_val = nullptr;
  ::ml_drift::Value* v_out_val = nullptr;

  int num_heads = 0;
  int num_kv_heads = 0;
  int head_dim = 0;
  float max_timescale = 1000000.0f;
  float min_timescale = 1.0f;
  float proportion = 1.0f;
  float epsilon = 1e-6f;

  std::vector<float> q_weight_data;
  std::vector<float> k_weight_data;
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

  ::ml_drift::TensorFloat32 tensor;
  tensor.shape = shape;
  tensor.data = data;

  ::ml_drift::ConstTensorAttributes attr;
  attr.tensor = std::move(tensor);
  node->operation.attributes = std::move(attr);
  *out_node = node;
  return value;
}

}  // namespace

absl::Status FuseQkvNormRoPE(::ml_drift::GraphFloat32* graph) {
  std::vector<QkvNormRopeMatch> matches;

  for (::ml_drift::Node* node : graph->nodes()) {
    if (!node ||
        node->operation.type != ToString(::ml_drift::OperationType::ROPE)) {
      continue;
    }

    auto rope_inputs = graph->FindInputs(node->id);
    if (rope_inputs.size() < 2) continue;
    ::ml_drift::Value* rope_in_q = rope_inputs[0];
    ::ml_drift::Value* pos_val = rope_inputs[1];

    ::ml_drift::Node* pre_rope_q = graph->FindProducer(rope_in_q->id);
    if (!pre_rope_q) continue;
    if (pre_rope_q->operation.type != "reshape" &&
        pre_rope_q->operation.type != "transpose") {
      continue;
    }

    auto pre_rope_inputs = graph->FindInputs(pre_rope_q->id);
    if (pre_rope_inputs.empty()) continue;
    ::ml_drift::Value* norm_out_q = pre_rope_inputs[0];

    ::ml_drift::Node* norm_q = graph->FindProducer(norm_out_q->id);
    if (!norm_q || norm_q->operation.type != "rms_norm") continue;

    auto norm_inputs = graph->FindInputs(norm_q->id);
    if (norm_inputs.empty()) continue;
    ::ml_drift::Value* norm_in_q = norm_inputs[0];

    ::ml_drift::Node* pre_norm_q = graph->FindProducer(norm_in_q->id);
    if (!pre_norm_q) continue;
    ::ml_drift::Value* slice_out_val = nullptr;
    ::ml_drift::Node* slice_q = nullptr;

    if (pre_norm_q->operation.type == "reshape" ||
        pre_norm_q->operation.type == "transpose") {
      auto pre_norm_inputs = graph->FindInputs(pre_norm_q->id);
      if (pre_norm_inputs.empty()) continue;
      slice_out_val = pre_norm_inputs[0];
      slice_q = graph->FindProducer(slice_out_val->id);
    } else if (pre_norm_q->operation.type == "slice") {
      slice_q = pre_norm_q;
      slice_out_val = norm_in_q;
    } else {
      continue;
    }

    // Verify producer node is a slice op.
    if (!slice_q || slice_q->operation.type != "slice") continue;

    auto slice_q_inputs = graph->FindInputs(slice_q->id);
    if (slice_q_inputs.empty()) continue;
    ::ml_drift::Value* qkv_val = slice_q_inputs[0];

    // Check if slice_q starts at channel 0
    const auto& slice_q_attr =
        std::any_cast<const ::ml_drift::SliceAttributes&>(
            slice_q->operation.attributes);
    if (slice_q_attr.starts.c != 0) {
      continue;
    }

    // Find slice_k and slice_v among consumers of qkv_val in a single pass.
    ::ml_drift::Node* slice_k = nullptr;
    ::ml_drift::Node* slice_v = nullptr;
    int q_end = slice_q_attr.ends.c;

    std::vector<::ml_drift::Node*> slice_candidates;
    for (::ml_drift::Node* c : graph->FindConsumers(qkv_val->id)) {
      if (c->operation.type != "slice") continue;
      const auto& sa = std::any_cast<const ::ml_drift::SliceAttributes&>(
          c->operation.attributes);
      if (sa.starts.c == q_end) {
        slice_k = c;
      } else {
        slice_candidates.push_back(c);
      }
    }
    if (!slice_k) continue;

    const auto& slice_k_attr =
        std::any_cast<const ::ml_drift::SliceAttributes&>(
            slice_k->operation.attributes);
    int k_end = slice_k_attr.ends.c;

    for (::ml_drift::Node* c : slice_candidates) {
      const auto& sa = std::any_cast<const ::ml_drift::SliceAttributes&>(
          c->operation.attributes);
      if (sa.starts.c == k_end) {
        slice_v = c;
        break;
      }
    }
    if (!slice_v) continue;

    // Trace K stream
    auto slice_k_outputs = graph->FindOutputs(slice_k->id);
    if (slice_k_outputs.empty()) continue;
    ::ml_drift::Node* pre_norm_k = nullptr;
    ::ml_drift::Node* norm_k = nullptr;

    for (::ml_drift::Node* c : graph->FindConsumers(slice_k_outputs[0]->id)) {
      if (c->operation.type == "reshape" || c->operation.type == "transpose") {
        pre_norm_k = c;
        auto outs = graph->FindOutputs(c->id);
        if (!outs.empty()) {
          for (::ml_drift::Node* cc : graph->FindConsumers(outs[0]->id)) {
            if (cc->operation.type == "rms_norm") norm_k = cc;
          }
        }
      } else if (c->operation.type == "rms_norm") {
        norm_k = c;
      }
    }
    if (!norm_k) continue;

    auto norm_k_outputs = graph->FindOutputs(norm_k->id);
    if (norm_k_outputs.empty()) continue;
    ::ml_drift::Node* pre_rope_k = nullptr;
    ::ml_drift::Node* rope_k = nullptr;

    // Reshape or transpose may exist between norm and RoPE depending on
    // whether the model unrolls head dimensions before or after RMSNorm.
    for (::ml_drift::Node* c : graph->FindConsumers(norm_k_outputs[0]->id)) {
      if (c->operation.type == "reshape" || c->operation.type == "transpose") {
        pre_rope_k = c;
        auto outs = graph->FindOutputs(c->id);
        if (!outs.empty()) {
          for (::ml_drift::Node* cc : graph->FindConsumers(outs[0]->id)) {
            if (cc->operation.type == ToString(::ml_drift::OperationType::ROPE))
              rope_k = cc;
          }
        }
      } else if (c->operation.type ==
                 ToString(::ml_drift::OperationType::ROPE)) {
        rope_k = c;
      }
    }
    if (!rope_k) continue;

    // Trace V stream through any sequence of reshape / transpose until the
    // final consumer
    auto slice_v_outputs = graph->FindOutputs(slice_v->id);
    if (slice_v_outputs.empty()) continue;
    std::vector<::ml_drift::Node*> v_intermediates;
    ::ml_drift::Value* v_curr_val = slice_v_outputs[0];

    while (true) {
      auto consumers = graph->FindConsumers(v_curr_val->id);
      if (consumers.size() == 1 &&
          (consumers[0]->operation.type == "reshape" ||
           consumers[0]->operation.type == "transpose")) {
        v_intermediates.push_back(consumers[0]);
        auto outs = graph->FindOutputs(consumers[0]->id);
        if (outs.empty()) break;
        v_curr_val = outs[0];
      } else {
        break;
      }
    }
    ::ml_drift::Value* v_out_val = v_curr_val;

    // Extract attributes
    const auto& rms_attr_q =
        std::any_cast<const ::ml_drift::RmsNormAttributes&>(
            norm_q->operation.attributes);
    const auto& rms_attr_k =
        std::any_cast<const ::ml_drift::RmsNormAttributes&>(
            norm_k->operation.attributes);
    const auto& rope_attr = std::any_cast<const ::ml_drift::RoPEAttributes&>(
        node->operation.attributes);

    if (!rms_attr_q.scale.has_value() || !rms_attr_k.scale.has_value()) {
      continue;
    }

    int head_dim = rms_attr_q.scale->shape.v;
    int num_heads = (slice_q_attr.ends.c - slice_q_attr.starts.c) / head_dim;
    int num_kv_heads = (slice_k_attr.ends.c - slice_k_attr.starts.c) / head_dim;

    // Verify that V-stream output layout assumptions match the fused kernel's
    // expected shape [batch, num_kv_heads, seq_len, head_dim].
    if (!v_intermediates.empty()) {
      if (v_out_val->tensor.shape.h != num_kv_heads ||
          v_out_val->tensor.shape.c != head_dim) {
        continue;
      }
    }

    auto rope_q_outs = graph->FindOutputs(node->id);
    auto rope_k_outs = graph->FindOutputs(rope_k->id);
    if (rope_q_outs.empty() || rope_k_outs.empty()) continue;

    QkvNormRopeMatch match;
    match.rope_q = node;
    match.pre_rope_q = pre_rope_q;
    match.norm_q = norm_q;
    match.pre_norm_q = pre_norm_q;
    match.slice_q = slice_q;

    match.rope_k = rope_k;
    match.pre_rope_k = pre_rope_k;
    match.norm_k = norm_k;
    match.pre_norm_k = pre_norm_k;
    match.slice_k = slice_k;

    match.v_intermediates = std::move(v_intermediates);
    match.slice_v = slice_v;

    match.qkv_val = qkv_val;
    match.pos_val = pos_val;

    match.q_out_val = rope_q_outs[0];
    match.k_out_val = rope_k_outs[0];
    match.v_out_val = v_out_val;

    match.num_heads = num_heads;
    match.num_kv_heads = num_kv_heads;
    match.head_dim = head_dim;
    match.max_timescale = rope_attr.max_timescale;
    match.min_timescale = rope_attr.min_timescale;
    match.proportion = rope_attr.proportion;
    match.epsilon = rms_attr_q.epsilon;

    match.q_weight_data = rms_attr_q.scale->data;
    match.k_weight_data = rms_attr_k.scale->data;

    matches.push_back(std::move(match));
  }

  if (matches.empty()) {
    return absl::OkStatus();
  }

  ABSL_LOG(INFO) << "[FuseQkvNormRoPE] Found " << matches.size()
                 << " QKV Norm RoPE clusters to fuse into single kernels!";

  for (const auto& match : matches) {
    ::ml_drift::Node* qkv_producer = graph->FindProducer(match.qkv_val->id);
    ::ml_drift::NodeId insert_after =
        qkv_producer ? qkv_producer->id : match.slice_q->id;

    ::ml_drift::DataType weight_dtype =
        match.qkv_val ? match.qkv_val->tensor.type
                      : ::ml_drift::DataType::FLOAT32;

    ::ml_drift::Node* q_w_node = nullptr;
    ::ml_drift::Value* q_w_val =
        CreateConstTensor(graph, ::ml_drift::BHWC(match.head_dim, 1, 1, 1),
                          match.q_weight_data, insert_after, &q_w_node,
                          weight_dtype);

    ::ml_drift::Node* k_w_node = nullptr;
    ::ml_drift::Value* k_w_val =
        CreateConstTensor(graph, ::ml_drift::BHWC(match.head_dim, 1, 1, 1),
                          match.k_weight_data, q_w_node->id, &k_w_node,
                          weight_dtype);

    ::ml_drift::Node* fused_node = nullptr;
    ABSL_CHECK_OK(graph->InsertNodeAfter(k_w_node->id, &fused_node));
    fused_node->operation.type = std::string(kQkvNormRopeType);

    QkvNormRopeAttributes fused_attr;
    fused_attr.num_heads = match.num_heads;
    fused_attr.num_kv_heads = match.num_kv_heads;
    fused_attr.head_dim = match.head_dim;
    fused_attr.max_timescale = match.max_timescale;
    fused_attr.min_timescale = match.min_timescale;
    fused_attr.proportion = match.proportion;
    fused_attr.epsilon = match.epsilon;
    fused_node->operation.attributes = std::move(fused_attr);

    // Inputs: qkv, position, q_weight, k_weight
    graph->AddConsumer(fused_node->id, match.qkv_val->id);
    graph->AddConsumer(fused_node->id, match.pos_val->id);
    graph->AddConsumer(fused_node->id, q_w_val->id);
    graph->AddConsumer(fused_node->id, k_w_val->id);

    // Outputs: reassign producers of final consumer values
    graph->SetProducer(fused_node->id, match.q_out_val->id);
    graph->SetProducer(fused_node->id, match.k_out_val->id);
    graph->SetProducer(fused_node->id, match.v_out_val->id);

    // Delete replaced intermediate nodes
    std::set<::ml_drift::NodeId> to_delete = {
        match.rope_q->id,  match.rope_k->id,  match.norm_q->id,
        match.norm_k->id,  match.slice_q->id, match.slice_k->id,
        match.slice_v->id,
    };
    if (match.pre_rope_q) to_delete.insert(match.pre_rope_q->id);
    if (match.pre_rope_k) to_delete.insert(match.pre_rope_k->id);
    if (match.pre_norm_q) to_delete.insert(match.pre_norm_q->id);
    if (match.pre_norm_k) to_delete.insert(match.pre_norm_k->id);
    for (auto* n : match.v_intermediates) {
      to_delete.insert(n->id);
    }

    for (::ml_drift::NodeId id : to_delete) {
      if (id != fused_node->id && id != q_w_node->id && id != k_w_node->id) {
        (void)graph->DeleteNode(id);
      }
    }
  }

  return absl::OkStatus();
}

}  // namespace litert::ml_drift
