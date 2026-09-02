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

#include "ml_drift_delegate/delegate/composite/fuse_qkv_norm_rope.h"

#include <any>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/qkv_norm_rope_parser.h"

namespace litert::ml_drift {
namespace {

using ::testing::Eq;
using ::testing::NotNull;

TEST(FuseQkvNormRoPETest, FusesQkvNormRopeSubgraphSuccessfully) {
  ::ml_drift::GraphFloat32 graph;

  // Constants for Qwen3-like dimensions:
  // Q heads = 16, KV heads = 8, head_dim = 128
  // Q channels = 2048, K channels = 1024, V channels = 1024. Total = 4096.
  constexpr int kHeadDim = 128;
  constexpr int kNumHeads = 16;
  constexpr int kNumKvHeads = 8;
  constexpr int kQChannels = kNumHeads * kHeadDim;
  constexpr int kKChannels = kNumKvHeads * kHeadDim;
  constexpr int kVChannels = kNumKvHeads * kHeadDim;
  constexpr int kTotalChannels = kQChannels + kKChannels + kVChannels;

  // Graph inputs
  auto* qkv_input = graph.NewValue();
  qkv_input->tensor.type = ::ml_drift::DataType::FLOAT32;
  qkv_input->tensor.shape = ::ml_drift::BHWC(1, 1, 1, kTotalChannels);

  auto* pos_input = graph.NewValue();
  pos_input->tensor.type = ::ml_drift::DataType::INT32;
  pos_input->tensor.shape = ::ml_drift::BHWC(1, 1, 1, 1);

  // Producer for qkv
  auto* dummy_prod = graph.NewNode();
  dummy_prod->operation.type = "dummy";
  graph.SetProducer(dummy_prod->id, qkv_input->id);

  // 1. Slices: Q, K, V
  auto* slice_q = graph.NewNode();
  slice_q->operation.type = "slice";
  ::ml_drift::SliceAttributes slice_q_attr;
  slice_q_attr.starts = ::ml_drift::BHWC(0, 0, 0, 0);
  slice_q_attr.ends = ::ml_drift::BHWC(1, 1, 1, kQChannels);
  slice_q_attr.strides = ::ml_drift::BHWC(1, 1, 1, 1);
  slice_q->operation.attributes = slice_q_attr;
  auto* slice_q_out = graph.NewValue();
  slice_q_out->tensor.type = ::ml_drift::DataType::FLOAT32;
  slice_q_out->tensor.shape = ::ml_drift::BHWC(1, 1, 1, kQChannels);
  graph.AddConsumer(slice_q->id, qkv_input->id);
  graph.SetProducer(slice_q->id, slice_q_out->id);

  auto* slice_k = graph.NewNode();
  slice_k->operation.type = "slice";
  ::ml_drift::SliceAttributes slice_k_attr;
  slice_k_attr.starts = ::ml_drift::BHWC(0, 0, 0, kQChannels);
  slice_k_attr.ends = ::ml_drift::BHWC(1, 1, 1, kQChannels + kKChannels);
  slice_k_attr.strides = ::ml_drift::BHWC(1, 1, 1, 1);
  slice_k->operation.attributes = slice_k_attr;
  auto* slice_k_out = graph.NewValue();
  slice_k_out->tensor.type = ::ml_drift::DataType::FLOAT32;
  slice_k_out->tensor.shape = ::ml_drift::BHWC(1, 1, 1, kKChannels);
  graph.AddConsumer(slice_k->id, qkv_input->id);
  graph.SetProducer(slice_k->id, slice_k_out->id);

  auto* slice_v = graph.NewNode();
  slice_v->operation.type = "slice";
  ::ml_drift::SliceAttributes slice_v_attr;
  slice_v_attr.starts = ::ml_drift::BHWC(0, 0, 0, kQChannels + kKChannels);
  slice_v_attr.ends = ::ml_drift::BHWC(1, 1, 1, kTotalChannels);
  slice_v_attr.strides = ::ml_drift::BHWC(1, 1, 1, 1);
  slice_v->operation.attributes = slice_v_attr;
  auto* slice_v_out = graph.NewValue();
  slice_v_out->tensor.type = ::ml_drift::DataType::FLOAT32;
  slice_v_out->tensor.shape = ::ml_drift::BHWC(1, 1, 1, kVChannels);
  graph.AddConsumer(slice_v->id, qkv_input->id);
  graph.SetProducer(slice_v->id, slice_v_out->id);

  // 2. Q Stream: Reshape -> RMSNorm -> Reshape -> RoPE
  auto* q_pre_norm = graph.NewNode();
  q_pre_norm->operation.type = "reshape";
  auto* q_norm_in = graph.NewValue();
  q_norm_in->tensor.type = ::ml_drift::DataType::FLOAT32;
  q_norm_in->tensor.shape = ::ml_drift::BHWC(1, kNumHeads, 1, kHeadDim);
  graph.AddConsumer(q_pre_norm->id, slice_q_out->id);
  graph.SetProducer(q_pre_norm->id, q_norm_in->id);

  auto* norm_q = graph.NewNode();
  norm_q->operation.type = "rms_norm";
  ::ml_drift::RmsNormAttributes norm_q_attr;
  norm_q_attr.epsilon = 1e-6f;
  ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::FLOAT32> scale_q;
  scale_q.shape = ::ml_drift::Linear(kHeadDim);
  scale_q.data = std::vector<float>(kHeadDim, 1.0f);
  norm_q_attr.scale = std::move(scale_q);
  norm_q->operation.attributes = std::move(norm_q_attr);
  auto* q_norm_out = graph.NewValue();
  q_norm_out->tensor.type = ::ml_drift::DataType::FLOAT32;
  q_norm_out->tensor.shape = ::ml_drift::BHWC(1, kNumHeads, 1, kHeadDim);
  graph.AddConsumer(norm_q->id, q_norm_in->id);
  graph.SetProducer(norm_q->id, q_norm_out->id);

  auto* q_pre_rope = graph.NewNode();
  q_pre_rope->operation.type = "transpose";
  auto* q_rope_in = graph.NewValue();
  q_rope_in->tensor.type = ::ml_drift::DataType::FLOAT32;
  q_rope_in->tensor.shape = ::ml_drift::BHWC(1, kNumHeads, 1, kHeadDim);
  graph.AddConsumer(q_pre_rope->id, q_norm_out->id);
  graph.SetProducer(q_pre_rope->id, q_rope_in->id);

  auto* rope_q = graph.NewNode();
  rope_q->operation.type = ToString(::ml_drift::OperationType::ROPE);
  ::ml_drift::RoPEAttributes rope_q_attr;
  rope_q_attr.max_timescale = 1000000.0f;
  rope_q_attr.min_timescale = 1.0f;
  rope_q_attr.proportion = 1.0f;
  rope_q->operation.attributes = rope_q_attr;
  auto* q_final_out = graph.NewValue();
  q_final_out->tensor.type = ::ml_drift::DataType::FLOAT32;
  q_final_out->tensor.shape = ::ml_drift::BHWC(1, kNumHeads, 1, kHeadDim);
  graph.AddConsumer(rope_q->id, q_rope_in->id);
  graph.AddConsumer(rope_q->id, pos_input->id);
  graph.SetProducer(rope_q->id, q_final_out->id);

  // 3. K Stream: Reshape -> RMSNorm -> Reshape -> RoPE
  auto* k_pre_norm = graph.NewNode();
  k_pre_norm->operation.type = "reshape";
  auto* k_norm_in = graph.NewValue();
  k_norm_in->tensor.type = ::ml_drift::DataType::FLOAT32;
  k_norm_in->tensor.shape = ::ml_drift::BHWC(1, kNumKvHeads, 1, kHeadDim);
  graph.AddConsumer(k_pre_norm->id, slice_k_out->id);
  graph.SetProducer(k_pre_norm->id, k_norm_in->id);

  auto* norm_k = graph.NewNode();
  norm_k->operation.type = "rms_norm";
  ::ml_drift::RmsNormAttributes norm_k_attr;
  norm_k_attr.epsilon = 1e-6f;
  ::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::FLOAT32> scale_k;
  scale_k.shape = ::ml_drift::Linear(kHeadDim);
  scale_k.data = std::vector<float>(kHeadDim, 1.0f);
  norm_k_attr.scale = std::move(scale_k);
  norm_k->operation.attributes = std::move(norm_k_attr);
  auto* k_norm_out = graph.NewValue();
  k_norm_out->tensor.type = ::ml_drift::DataType::FLOAT32;
  k_norm_out->tensor.shape = ::ml_drift::BHWC(1, kNumKvHeads, 1, kHeadDim);
  graph.AddConsumer(norm_k->id, k_norm_in->id);
  graph.SetProducer(norm_k->id, k_norm_out->id);

  auto* k_pre_rope = graph.NewNode();
  k_pre_rope->operation.type = "transpose";
  auto* k_rope_in = graph.NewValue();
  k_rope_in->tensor.type = ::ml_drift::DataType::FLOAT32;
  k_rope_in->tensor.shape = ::ml_drift::BHWC(1, kNumKvHeads, 1, kHeadDim);
  graph.AddConsumer(k_pre_rope->id, k_norm_out->id);
  graph.SetProducer(k_pre_rope->id, k_rope_in->id);

  auto* rope_k = graph.NewNode();
  rope_k->operation.type = ToString(::ml_drift::OperationType::ROPE);
  ::ml_drift::RoPEAttributes rope_k_attr;
  rope_k_attr.max_timescale = 1000000.0f;
  rope_k_attr.min_timescale = 1.0f;
  rope_k_attr.proportion = 1.0f;
  rope_k->operation.attributes = rope_k_attr;
  auto* k_final_out = graph.NewValue();
  k_final_out->tensor.type = ::ml_drift::DataType::FLOAT32;
  k_final_out->tensor.shape = ::ml_drift::BHWC(1, kNumKvHeads, 1, kHeadDim);
  graph.AddConsumer(rope_k->id, k_rope_in->id);
  graph.AddConsumer(rope_k->id, pos_input->id);
  graph.SetProducer(rope_k->id, k_final_out->id);

  // 4. V Stream: Reshape to [1, num_kv_heads, 1, head_dim]
  auto* v_reshape = graph.NewNode();
  v_reshape->operation.type = "reshape";
  auto* v_final_out = graph.NewValue();
  v_final_out->tensor.type = ::ml_drift::DataType::FLOAT32;
  v_final_out->tensor.shape = ::ml_drift::BHWC(1, kNumKvHeads, 1, kHeadDim);
  graph.AddConsumer(v_reshape->id, slice_v_out->id);
  graph.SetProducer(v_reshape->id, v_final_out->id);

  // Downstream consumers to anchor the graph outputs
  auto* dummy_consumer = graph.NewNode();
  dummy_consumer->operation.type = "dummy_consumer";
  graph.AddConsumer(dummy_consumer->id, q_final_out->id);
  graph.AddConsumer(dummy_consumer->id, k_final_out->id);
  graph.AddConsumer(dummy_consumer->id, v_final_out->id);

  // Run the fusion pass
  EXPECT_TRUE(FuseQkvNormRoPE(&graph).ok());

  // Verify that an odml.qkv_norm_rope node was created
  ::ml_drift::Node* fused_node = nullptr;
  for (::ml_drift::Node* node : graph.nodes()) {
    if (node && node->operation.type == "odml.qkv_norm_rope") {
      fused_node = node;
      break;
    }
  }
  ASSERT_THAT(fused_node, NotNull());

  // Verify fused attributes
  const auto& attr = std::any_cast<const QkvNormRopeAttributes&>(
      fused_node->operation.attributes);
  EXPECT_THAT(attr.num_heads, Eq(kNumHeads));
  EXPECT_THAT(attr.num_kv_heads, Eq(kNumKvHeads));
  EXPECT_THAT(attr.head_dim, Eq(kHeadDim));
  EXPECT_FLOAT_EQ(attr.max_timescale, 1000000.0f);
  EXPECT_FLOAT_EQ(attr.min_timescale, 1.0f);
  EXPECT_FLOAT_EQ(attr.proportion, 1.0f);
  EXPECT_FLOAT_EQ(attr.epsilon, 1e-6f);

  // Verify fused node outputs
  auto fused_outputs = graph.FindOutputs(fused_node->id);
  EXPECT_THAT(fused_outputs.size(), Eq(3));
  EXPECT_THAT(fused_outputs[0]->id, Eq(q_final_out->id));
  EXPECT_THAT(fused_outputs[1]->id, Eq(k_final_out->id));
  EXPECT_THAT(fused_outputs[2]->id, Eq(v_final_out->id));

  // Verify that old slice, norm, rope nodes were deleted
  for (::ml_drift::Node* node : graph.nodes()) {
    if (!node) continue;
    EXPECT_NE(node->id, slice_q->id);
    EXPECT_NE(node->id, slice_k->id);
    EXPECT_NE(node->id, slice_v->id);
    EXPECT_NE(node->id, norm_q->id);
    EXPECT_NE(node->id, norm_k->id);
    EXPECT_NE(node->id, rope_q->id);
    EXPECT_NE(node->id, rope_k->id);
  }
}

}  // namespace
}  // namespace litert::ml_drift
