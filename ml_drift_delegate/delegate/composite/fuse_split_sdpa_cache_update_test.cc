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

#include "ml_drift_delegate/delegate/composite/fuse_split_sdpa_cache_update.h"

#include <any>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_util.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/ir_model_util.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/add_values_to_cache_parser.h"
#include "ml_drift_delegate/delegate/composite/fused_sdpa_cache_update_parser.h"
#include "ml_drift_delegate/delegate/composite/ir/litert_op_selector.h"
#include "ml_drift_delegate/delegate/composite/litert_op_selector.h"
#include "ml_drift_delegate/delegate/composite/runtime_batched_matmul_parser.h"

namespace litert::ml_drift {
namespace {

using ::ml_drift::Axis;
using ::ml_drift::BatchedMatMulAttributes;
using ::ml_drift::BHWC;
using ::ml_drift::CalculationsPrecision;
using ::ml_drift::ConcatAttributes;
using ::ml_drift::ConstTensorAttributes;
using ::ml_drift::CreateGpuModelInfo;
using ::ml_drift::DataType;
using ::ml_drift::ElementwiseAttributes;
using ::ml_drift::GpuApi;
using ::ml_drift::GpuInfo;
using ::ml_drift::GpuModel;
using ::ml_drift::GraphFloat32;
using ::ml_drift::GraphToGpuModel;
using ::ml_drift::kMaxHalf;
using ::ml_drift::Node;
using ::ml_drift::OperationType;
using ::ml_drift::SliceAttributes;
using ::ml_drift::SoftmaxAttributes;
using ::ml_drift::TensorDescriptor;
using ::ml_drift::TensorFloat32;
using ::ml_drift::TensorStorageType;
using ::ml_drift::TransposeAttributes;
using ::ml_drift::Value;
using ::ml_drift::ir::IrModel;
using ::ml_drift::ir::IrModelToGpuModel;
using ::ml_drift::ir::IrOp;
using ::ml_drift::ir::IrTensor;
using ::ml_drift::ir::IrTensorId;

struct SplitSdpaGraphOptions {
  int heads = 2;
  int g = 2;
  int new_len = 8;
  int cache_size = 16;
  int head_dim = 8;
  bool with_cache_update = true;
  bool value_new_transposed = false;
  bool with_float_capping = false;
  std::optional<float> softcap;
  float mask_fill_value = -10000.0f;
  bool is_ring_buffer = true;
  std::optional<int> kv_cache_batch_size_override;
  int num_layers = 1;
};

struct BuiltGraph {
  GraphFloat32 graph;
  Value* mask_raw = nullptr;
  Value* param_raw = nullptr;
  std::vector<Value*> queries;
  std::vector<Value*> k_caches;
  std::vector<Value*> v_caches;
  std::vector<Value*> k_news;
  std::vector<Value*> v_news;
  std::vector<Value*> attentions;
  std::vector<Value*> k_cache_outs;
  std::vector<Value*> v_cache_outs;
};

Value* AddValue(GraphFloat32& graph, DataType dtype, const BHWC& shape) {
  Value* v = graph.NewValue();
  v->tensor.type = dtype;
  v->tensor.shape = shape;
  return v;
}

Value* AddCopy(GraphFloat32& graph, Value* src) {
  Node* copy = graph.NewNode();
  copy->operation.type = ToString(OperationType::kCopy);
  graph.AddConsumer(copy->id, src->id);
  Value* dst = AddValue(graph, src->tensor.type, src->tensor.shape);
  graph.SetProducer(copy->id, dst->id);
  return dst;
}

Value* AddSlice(GraphFloat32& graph, Value* src, const BHWC& starts,
                const BHWC& ends) {
  Node* slice = graph.NewNode();
  slice->operation.type = ToString(OperationType::kSlice);
  SliceAttributes attr;
  attr.starts = starts;
  attr.ends = ends;
  attr.strides = BHWC(1, 1, 1, 1);
  slice->operation.attributes = attr;
  graph.AddConsumer(slice->id, src->id);
  Value* dst = AddValue(graph, src->tensor.type,
                        BHWC(ends.b - starts.b, ends.h - starts.h,
                             ends.w - starts.w, ends.c - starts.c));
  graph.SetProducer(slice->id, dst->id);
  return dst;
}

Value* AddConcat(GraphFloat32& graph, const std::vector<Value*>& raw_inputs,
                 Axis axis, const BHWC& out_shape) {
  std::vector<Value*> inputs = raw_inputs;
  for (size_t i = 0; i < inputs.size(); ++i) {
    for (size_t j = 0; j < i; ++j) {
      if (inputs[i] == inputs[j]) {
        inputs[i] = AddCopy(graph, inputs[j]);
        break;
      }
    }
  }
  Node* concat = graph.NewNode();
  concat->operation.type = ToString(OperationType::kConcat);
  ConcatAttributes attr;
  attr.axis = axis;
  concat->operation.attributes = attr;
  for (Value* in : inputs) {
    graph.AddConsumer(concat->id, in->id);
  }
  Value* dst = AddValue(graph, inputs[0]->tensor.type, out_shape);
  graph.SetProducer(concat->id, dst->id);
  return dst;
}

Value* AddElementwiseScalar(GraphFloat32& graph, OperationType op_type,
                            Value* src, const ::ml_drift::ScalarValue& param) {
  Node* node = graph.NewNode();
  node->operation.type = ToString(op_type);
  ElementwiseAttributes attr;
  attr.param = param;
  node->operation.attributes = attr;
  graph.AddConsumer(node->id, src->id);
  Value* dst = AddValue(graph, src->tensor.type, src->tensor.shape);
  graph.SetProducer(node->id, dst->id);
  return dst;
}

BuiltGraph BuildSplitSdpaGraph(const SplitSdpaGraphOptions& opts) {
  BuiltGraph built;
  GraphFloat32& g = built.graph;
  const int rows = opts.g * opts.new_len;
  const int w = opts.cache_size;
  const int t = opts.new_len;
  const int h = opts.heads;
  const int d = opts.head_dim;

  built.mask_raw = AddValue(g, DataType::kBool, BHWC(1, 1, t, w + t));
  built.param_raw = AddValue(g, DataType::kInt32, BHWC(1, 1, 1, 7));

  // Shared param subgraph matching sliding-window ring-buffer prefill export.
  Value* p03 = AddSlice(g, built.param_raw, BHWC(0, 0, 0, 0), BHWC(1, 1, 1, 3));
  Value* p_start = AddSlice(g, p03, BHWC(0, 0, 0, 0), BHWC(1, 1, 1, 1));
  Value* p_rest =
      AddSlice(g, built.param_raw, BHWC(0, 0, 0, 3), BHWC(1, 1, 1, 7));

  Value* offset = AddElementwiseScalar(g, OperationType::kFloorMod, p_start, w);
  Value* active_past =
      AddElementwiseScalar(g, OperationType::kMinimum, p_start, w);

  Value* param_past = AddConcat(g, {offset, active_past, active_past, p_rest},
                                Axis::kChannels, BHWC(1, 1, 1, 7));
  Value* param_upd = nullptr;
  if (opts.with_cache_update) {
    Value* p_end = AddSlice(g, p03, BHWC(0, 0, 0, 1), BHWC(1, 1, 1, 2));
    Value* p_aligned = AddSlice(g, p03, BHWC(0, 0, 0, 2), BHWC(1, 1, 1, 3));
    Value* active_end =
        AddElementwiseScalar(g, OperationType::kMinimum, p_end, w);
    Value* active_aligned =
        AddElementwiseScalar(g, OperationType::kMinimum, p_aligned, w);
    param_upd = AddConcat(g, {offset, active_end, active_aligned, p_rest},
                          Axis::kChannels, BHWC(1, 1, 1, 7));
  }

  // Shared mask subgraph matching sliding-window ring-buffer prefill export.
  Value* m_past =
      AddSlice(g, built.mask_raw, BHWC(0, 0, 0, 0), BHWC(1, 1, t, w));
  Value* m_new =
      AddSlice(g, built.mask_raw, BHWC(0, 0, 0, w), BHWC(1, 1, t, w + t));
  Value* m_past_rep = m_past;
  Value* m_new_rep = m_new;
  if (opts.g > 1) {
    m_past_rep = AddConcat(g, std::vector<Value*>(opts.g, m_past), Axis::kWidth,
                           BHWC(1, 1, rows, w));
    m_new_rep = AddConcat(g, std::vector<Value*>(opts.g, m_new), Axis::kWidth,
                          BHWC(1, 1, rows, t));
  }
  Value* mask_all = AddConcat(g, {m_past_rep, m_new_rep}, Axis::kChannels,
                              BHWC(1, 1, rows, w + t));

  Node* fill_const_node = g.NewNode();
  fill_const_node->operation.type = ToString(OperationType::kConstant);
  ConstTensorAttributes fill_attr;
  TensorFloat32 fill_tensor;
  fill_tensor.shape = BHWC(1, 1, 1, 1);
  fill_tensor.data = {opts.mask_fill_value};
  fill_attr.tensor = std::move(fill_tensor);
  fill_const_node->operation.attributes = std::move(fill_attr);
  Value* fill_val = AddValue(g, DataType::kFloat32, BHWC(1, 1, 1, 1));
  g.SetProducer(fill_const_node->id, fill_val->id);

  for (int layer = 0; layer < opts.num_layers; ++layer) {
    Value* q = AddValue(g, DataType::kFloat32, BHWC(1, h, rows, d));
    Value* kc = AddValue(g, DataType::kFloat32, BHWC(1, h, w, d));
    Value* vc = AddValue(g, DataType::kFloat32, BHWC(1, h, d, w));
    Value* kn = AddValue(g, DataType::kFloat32, BHWC(1, h, t, d));
    Value* vn = AddValue(
        g, DataType::kFloat32,
        opts.value_new_transposed ? BHWC(1, h, d, t) : BHWC(1, h, t, d));

    Node* bmm_past_k = g.NewNode();
    bmm_past_k->operation.type = kRuntimeBatchedMatMulType;
    RuntimeBatchedMatMulAttributes pk_attr;
    pk_attr.external_weights.emplace();
    pk_attr.runtime_check.dst_end_ch_index = 2;
    bmm_past_k->operation.attributes = pk_attr;
    g.AddConsumer(bmm_past_k->id, q->id);
    g.AddConsumer(bmm_past_k->id, kc->id);
    g.AddConsumer(bmm_past_k->id, param_past->id);
    Value* logits_past = AddValue(g, DataType::kFloat32, BHWC(1, h, rows, w));
    g.SetProducer(bmm_past_k->id, logits_past->id);

    Node* bmm_new_k = g.NewNode();
    bmm_new_k->operation.type = ToString(OperationType::kBatchedMatmul);
    BatchedMatMulAttributes nk_attr;
    nk_attr.transpose_left = false;
    nk_attr.transpose_right = true;
    bmm_new_k->operation.attributes = nk_attr;
    g.AddConsumer(bmm_new_k->id, q->id);
    g.AddConsumer(bmm_new_k->id, kn->id);
    Value* logits_new = AddValue(g, DataType::kFloat32, BHWC(1, h, rows, t));
    g.SetProducer(bmm_new_k->id, logits_new->id);

    Value* logits_cur = AddConcat(g, {logits_past, logits_new}, Axis::kChannels,
                                  BHWC(1, h, rows, w + t));
    if (opts.softcap.has_value()) {
      const float cap = *opts.softcap;
      logits_cur =
          AddElementwiseScalar(g, OperationType::kMul, logits_cur, 1.0f / cap);
      Node* tanh_node = g.NewNode();
      tanh_node->operation.type = ToString(OperationType::kTanh);
      g.AddConsumer(tanh_node->id, logits_cur->id);
      Value* tanh_out =
          AddValue(g, DataType::kFloat32, BHWC(1, h, rows, w + t));
      g.SetProducer(tanh_node->id, tanh_out->id);
      logits_cur = AddElementwiseScalar(g, OperationType::kMul, tanh_out, cap);
    }

    Node* select_node = g.NewNode();
    select_node->operation.type = ToString(OperationType::kSelectV2);
    g.AddConsumer(select_node->id, mask_all->id);
    g.AddConsumer(select_node->id, logits_cur->id);
    g.AddConsumer(select_node->id, fill_val->id);
    Value* masked = AddValue(g, DataType::kFloat32, BHWC(1, h, rows, w + t));
    g.SetProducer(select_node->id, masked->id);

    Value* softmax_in = masked;
    if (opts.with_float_capping) {
      Value* maxed = AddElementwiseScalar(g, OperationType::kMaximum,
                                          softmax_in, -kMaxHalf);
      softmax_in =
          AddElementwiseScalar(g, OperationType::kMinimum, maxed, kMaxHalf);
    }

    Node* softmax_node = g.NewNode();
    softmax_node->operation.type = ToString(OperationType::kSoftmax);
    SoftmaxAttributes sm_attr;
    sm_attr.axis = Axis::kChannels;
    softmax_node->operation.attributes = sm_attr;
    g.AddConsumer(softmax_node->id, softmax_in->id);
    Value* probs_all = AddValue(g, DataType::kFloat32, BHWC(1, h, rows, w + t));
    g.SetProducer(softmax_node->id, probs_all->id);

    Value* probs_past =
        AddSlice(g, probs_all, BHWC(0, 0, 0, 0), BHWC(1, h, rows, w));
    Value* probs_new =
        AddSlice(g, probs_all, BHWC(0, 0, 0, w), BHWC(1, h, rows, w + t));

    Node* bmm_past_v = g.NewNode();
    bmm_past_v->operation.type = kRuntimeBatchedMatMulType;
    RuntimeBatchedMatMulAttributes pv_attr;
    pv_attr.external_weights.emplace();
    pv_attr.runtime_check.src_end_ch_index = 2;
    bmm_past_v->operation.attributes = pv_attr;
    g.AddConsumer(bmm_past_v->id, probs_past->id);
    g.AddConsumer(bmm_past_v->id, vc->id);
    g.AddConsumer(bmm_past_v->id, param_past->id);
    Value* out_past = AddValue(g, DataType::kFloat32, BHWC(1, h, rows, d));
    g.SetProducer(bmm_past_v->id, out_past->id);

    Node* bmm_new_v = g.NewNode();
    bmm_new_v->operation.type = ToString(OperationType::kBatchedMatmul);
    BatchedMatMulAttributes nv_attr;
    nv_attr.transpose_left = false;
    nv_attr.transpose_right = opts.value_new_transposed;
    bmm_new_v->operation.attributes = nv_attr;
    g.AddConsumer(bmm_new_v->id, probs_new->id);
    g.AddConsumer(bmm_new_v->id, vn->id);
    Value* out_new = AddValue(g, DataType::kFloat32, BHWC(1, h, rows, d));
    g.SetProducer(bmm_new_v->id, out_new->id);

    Node* add_out = g.NewNode();
    add_out->operation.type = ToString(OperationType::kAdd);
    g.AddConsumer(add_out->id, out_past->id);
    g.AddConsumer(add_out->id, out_new->id);
    Value* attention = AddValue(g, DataType::kFloat32, BHWC(1, h, rows, d));
    g.SetProducer(add_out->id, attention->id);

    built.queries.push_back(q);
    built.k_caches.push_back(kc);
    built.v_caches.push_back(vc);
    built.k_news.push_back(kn);
    built.v_news.push_back(vn);
    built.attentions.push_back(attention);

    if (opts.with_cache_update) {
      Node* cu = g.NewNode();
      cu->operation.type = kAddValuesToCacheType;
      AddValuesToCacheAttributes cu_attr;
      cu_attr.kv_cache_batch_size =
          opts.kv_cache_batch_size_override.value_or(h);
      cu_attr.cache_size = w;
      cu_attr.head_size = d;
      cu_attr.is_ring_buffer = opts.is_ring_buffer;
      cu_attr.k_cache_ts_idx = 2;
      cu_attr.v_cache_ts_idx = 3;
      cu_attr.k_update_ts_idx = 2;
      cu_attr.v_update_ts_idx = opts.value_new_transposed ? 3 : 2;
      cu->operation.attributes = cu_attr;
      g.AddConsumer(cu->id, kn->id);
      g.AddConsumer(cu->id, vn->id);
      g.AddConsumer(cu->id, param_upd->id);
      Value* kc_out = AddValue(g, DataType::kFloat32, BHWC(1, h, w, d));
      Value* vc_out = AddValue(g, DataType::kFloat32, BHWC(1, h, d, w));
      g.SetProducer(cu->id, kc_out->id);
      g.SetProducer(cu->id, vc_out->id);
      built.k_cache_outs.push_back(kc_out);
      built.v_cache_outs.push_back(vc_out);
    }
  }
  return built;
}

GpuInfo GetTestMetalGpuInfo() {
  GpuInfo gpu_info;
  ::ml_drift::GetGpuInfoFromDeviceDescription("Apple A17 Pro", GpuApi::kMetal,
                                              &gpu_info);
  gpu_info.metal_info.buffer_max_size = 256 * 1024 * 1024;
  gpu_info.metal_info.image2d_max_width = 16384;
  gpu_info.metal_info.image2d_max_height = 16384;
  gpu_info.metal_info.image_array_max_layers = 2048;
  gpu_info.metal_info.image3d_max_width = 2048;
  gpu_info.metal_info.image3d_max_height = 2048;
  gpu_info.metal_info.image3d_max_depth = 2048;
  gpu_info.metal_info.max_work_group_size_x = 1024;
  gpu_info.metal_info.max_work_group_size_y = 1024;
  gpu_info.metal_info.max_work_group_size_z = 1024;
  gpu_info.metal_info.language_version =
      ::ml_drift::MetalLanguageVersion::kMetal3_0;
  return gpu_info;
}

TEST(FuseSplitSdpaCacheUpdateTest,
     FusesMultiLayerGqaBoolMaskWithFloatCappingAndCacheUpdate) {
  SplitSdpaGraphOptions opts;
  opts.num_layers = 2;
  opts.with_float_capping = true;
  opts.with_cache_update = true;
  opts.value_new_transposed = false;
  BuiltGraph built = BuildSplitSdpaGraph(opts);

  ASSERT_OK(FuseSplitSdpaCacheUpdate(&built.graph));

  // Each layer should become 1 Transpose(v_new) + 1 fused_sdpa_cache_update,
  // and all shared mask/param/const helper ops should be pruned.
  int fused_count = 0;
  int transpose_count = 0;
  for (const Node* node : built.graph.nodes()) {
    if (node->operation.type == kFusedSdpaCacheUpdateType) {
      ++fused_count;
      const auto* attr = std::any_cast<FusedSdpaCacheUpdateAttributes>(
          &node->operation.attributes);
      ASSERT_NE(attr, nullptr);
      EXPECT_TRUE(attr->update_cache);
      EXPECT_FALSE(attr->softcap.has_value());

      std::vector<Value*> ins = built.graph.FindInputs(node->id);
      std::vector<Value*> outs = built.graph.FindOutputs(node->id);
      ASSERT_EQ(ins.size(), 7u);
      ASSERT_EQ(outs.size(), 3u);
      EXPECT_EQ(ins[4]->tensor.shape,
                BHWC(1, opts.heads, opts.head_dim, opts.new_len));
      EXPECT_EQ(ins[5], built.mask_raw);
      EXPECT_EQ(ins[6], built.param_raw);
    } else if (node->operation.type == ToString(OperationType::kTranspose)) {
      ++transpose_count;
      const auto* t_attr =
          std::any_cast<TransposeAttributes>(&node->operation.attributes);
      ASSERT_NE(t_attr, nullptr);
      EXPECT_EQ(t_attr->perm, BHWC(0, 1, 3, 2));
    } else {
      ADD_FAILURE() << "Unexpected leftover op: " << node->operation.type;
    }
  }
  EXPECT_EQ(fused_count, 2);
  EXPECT_EQ(transpose_count, 2);

  // Verify the fused graph lowers cleanly through LiteRtOpSelector into a
  // GpuModel with fused softmax + ring cache write ops.
  GpuInfo gpu_info = GetTestMetalGpuInfo();
  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::kF16;
  create_info.storage_type = TensorStorageType::kTexture2D;
  auto set_buffer = [&](const Value* v) {
    TensorDescriptor desc(v->tensor.type, TensorStorageType::kBuffer,
                          ::ml_drift::Layout::kHWC);
    desc.SetBHWCShape(v->tensor.shape);
    create_info.predefined[v->id] = desc;
  };
  for (int i = 0; i < opts.num_layers; ++i) {
    set_buffer(built.k_caches[i]);
    set_buffer(built.v_caches[i]);
    set_buffer(built.k_cache_outs[i]);
    set_buffer(built.v_cache_outs[i]);
  }
  LiteRtOpSelector selector(&create_info, &gpu_info);
  GpuModel gpu_model;
  ASSERT_OK(GraphToGpuModel(built.graph, create_info, gpu_info, &gpu_model,
                            &selector));

  int masked_softmax_ops = 0;
  int ring_write_ops = 0;
  for (const auto& gpu_node : gpu_model.nodes) {
    if (gpu_node.name == "fused_sdpa_masked_softmax") {
      ++masked_softmax_ops;
    } else if (gpu_node.name == "fused_sdpa_ring_cache_write") {
      ++ring_write_ops;
    }
  }
  EXPECT_EQ(masked_softmax_ops, 2);
  EXPECT_EQ(ring_write_ops, 2);
}

TEST(FuseSplitSdpaCacheUpdateTest,
     FusesTransposedValueNewWithSoftcapAndNoCacheUpdate) {
  SplitSdpaGraphOptions opts;
  opts.g = 1;
  opts.with_cache_update = false;
  opts.value_new_transposed = true;
  opts.softcap = 50.0f;
  BuiltGraph built = BuildSplitSdpaGraph(opts);

  ASSERT_OK(FuseSplitSdpaCacheUpdate(&built.graph));

  ASSERT_EQ(built.graph.nodes().size(), 1u);
  const Node* fused = built.graph.nodes()[0];
  EXPECT_EQ(fused->operation.type, kFusedSdpaCacheUpdateType);
  const auto* attr = std::any_cast<FusedSdpaCacheUpdateAttributes>(
      &fused->operation.attributes);
  ASSERT_NE(attr, nullptr);
  EXPECT_FALSE(attr->update_cache);
  ASSERT_TRUE(attr->softcap.has_value());
  EXPECT_FLOAT_EQ(*attr->softcap, 50.0f);

  std::vector<Value*> ins = built.graph.FindInputs(fused->id);
  std::vector<Value*> outs = built.graph.FindOutputs(fused->id);
  ASSERT_EQ(ins.size(), 7u);
  ASSERT_EQ(outs.size(), 1u);
  EXPECT_EQ(ins[4], built.v_news[0]);
  EXPECT_EQ(ins[5], built.mask_raw);
  EXPECT_EQ(ins[6], built.param_raw);
  EXPECT_EQ(outs[0], built.attentions[0]);
}

TEST(FuseSplitSdpaCacheUpdateTest, SkipsNonRingBufferCacheUpdate) {
  SplitSdpaGraphOptions opts;
  opts.is_ring_buffer = false;
  BuiltGraph built = BuildSplitSdpaGraph(opts);
  const size_t initial_node_count = built.graph.nodes().size();

  ASSERT_OK(FuseSplitSdpaCacheUpdate(&built.graph));
  EXPECT_EQ(built.graph.nodes().size(), initial_node_count);
  for (const Node* node : built.graph.nodes()) {
    EXPECT_NE(node->operation.type, kFusedSdpaCacheUpdateType);
  }
}

TEST(FuseSplitSdpaCacheUpdateTest, SkipsMismatchedMaskFillValue) {
  SplitSdpaGraphOptions opts;
  opts.mask_fill_value = -1e9f;
  BuiltGraph built = BuildSplitSdpaGraph(opts);
  const size_t initial_node_count = built.graph.nodes().size();

  ASSERT_OK(FuseSplitSdpaCacheUpdate(&built.graph));
  EXPECT_EQ(built.graph.nodes().size(), initial_node_count);
}

TEST(FuseSplitSdpaCacheUpdateTest, SkipsMismatchedCacheBatchSize) {
  SplitSdpaGraphOptions opts;
  opts.heads = 8;
  opts.kv_cache_batch_size_override = 1;
  BuiltGraph built = BuildSplitSdpaGraph(opts);
  const size_t initial_node_count = built.graph.nodes().size();

  ASSERT_OK(FuseSplitSdpaCacheUpdate(&built.graph));
  EXPECT_EQ(built.graph.nodes().size(), initial_node_count);
}

TEST(FuseSplitSdpaCacheUpdateTest, SkipsSharedKvCacheWithMultipleReaders) {
  SplitSdpaGraphOptions opts;
  BuiltGraph built = BuildSplitSdpaGraph(opts);

  // Add an extra reader on `k_caches[0]` to simulate a KV-shared donor layer.
  Node* extra_reader = built.graph.NewNode();
  extra_reader->operation.type = ToString(OperationType::kCopy);
  built.graph.AddConsumer(extra_reader->id, built.k_caches[0]->id);
  Value* extra_out = AddValue(built.graph, DataType::kFloat32,
                              built.k_caches[0]->tensor.shape);
  built.graph.SetProducer(extra_reader->id, extra_out->id);

  const size_t initial_node_count = built.graph.nodes().size();
  ASSERT_OK(FuseSplitSdpaCacheUpdate(&built.graph));
  EXPECT_EQ(built.graph.nodes().size(), initial_node_count);
}

// ============================================================================
// IrModel tests
// ============================================================================

IrTensorId AddCopyIr(IrModel& model, IrTensorId src) {
  IrOp* copy = model.add_op();
  copy->name = ToString(OperationType::kCopy);
  model.AddConsumer(src, copy->id);
  const IrTensor* src_t = model.tensor(src);
  IrTensor* dst =
      model.add_tensor(src_t->desc.GetDataType(), src_t->desc.GetBHWCShape());
  model.SetProducer(dst->id, copy->id);
  return dst->id;
}

IrTensorId AddSliceIr(IrModel& model, IrTensorId src, const BHWC& starts,
                      const BHWC& ends) {
  IrOp* slice = model.add_op();
  slice->name = ToString(OperationType::kSlice);
  SliceAttributes attr;
  attr.starts = starts;
  attr.ends = ends;
  attr.strides = BHWC(1, 1, 1, 1);
  slice->attr = attr;
  model.AddConsumer(src, slice->id);
  IrTensor* dst = model.add_tensor(model.tensor(src)->desc.GetDataType(),
                                   BHWC(ends.b - starts.b, ends.h - starts.h,
                                        ends.w - starts.w, ends.c - starts.c));
  model.SetProducer(dst->id, slice->id);
  return dst->id;
}

IrTensorId AddConcatIr(IrModel& model, std::vector<IrTensorId> inputs,
                       Axis axis, const BHWC& out_shape) {
  for (size_t i = 0; i < inputs.size(); ++i) {
    for (size_t j = 0; j < i; ++j) {
      if (inputs[i] == inputs[j]) {
        inputs[i] = AddCopyIr(model, inputs[j]);
        break;
      }
    }
  }
  IrOp* concat = model.add_op();
  concat->name = ToString(OperationType::kConcat);
  ConcatAttributes attr;
  attr.axis = axis;
  concat->attr = attr;
  for (IrTensorId in : inputs) {
    model.AddConsumer(in, concat->id);
  }
  IrTensor* dst =
      model.add_tensor(model.tensor(inputs[0])->desc.GetDataType(), out_shape);
  model.SetProducer(dst->id, concat->id);
  return dst->id;
}

IrTensorId AddElementwiseScalarIr(IrModel& model, OperationType op_type,
                                  IrTensorId src,
                                  const ::ml_drift::ScalarValue& param) {
  IrOp* op = model.add_op();
  op->name = ToString(op_type);
  ElementwiseAttributes attr;
  attr.param = param;
  op->attr = attr;
  model.AddConsumer(src, op->id);
  const IrTensor* src_t = model.tensor(src);
  IrTensor* dst =
      model.add_tensor(src_t->desc.GetDataType(), src_t->desc.GetBHWCShape());
  model.SetProducer(dst->id, op->id);
  return dst->id;
}

TEST(FuseSplitSdpaCacheUpdateTest, FusesAndLowersIrModelInTopologicalOrder) {
  constexpr int kH = 2;
  constexpr int kG = 2;
  constexpr int kT = 8;
  constexpr int kW = 16;
  constexpr int kD = 8;
  constexpr int kRows = kG * kT;

  IrModel model;
  IrTensorId q =
      model.add_tensor(DataType::kFloat32, BHWC(1, kH, kRows, kD))->id;
  IrTensorId kc = model.add_tensor(DataType::kFloat32, BHWC(1, kH, kW, kD))->id;
  IrTensorId vc = model.add_tensor(DataType::kFloat32, BHWC(1, kH, kD, kW))->id;
  IrTensorId kn = model.add_tensor(DataType::kFloat32, BHWC(1, kH, kT, kD))->id;
  IrTensorId vn = model.add_tensor(DataType::kFloat32, BHWC(1, kH, kT, kD))->id;
  IrTensorId mask_raw =
      model.add_tensor(DataType::kBool, BHWC(1, 1, kT, kW + kT))->id;
  IrTensorId param_raw =
      model.add_tensor(DataType::kInt32, BHWC(1, 1, 1, 7))->id;
  for (IrTensorId in : {q, kc, vc, kn, vn, mask_raw, param_raw}) {
    model.add_input(in);
  }

  IrTensorId p03 =
      AddSliceIr(model, param_raw, BHWC(0, 0, 0, 0), BHWC(1, 1, 1, 3));
  IrTensorId p_start =
      AddSliceIr(model, p03, BHWC(0, 0, 0, 0), BHWC(1, 1, 1, 1));
  IrTensorId p_end = AddSliceIr(model, p03, BHWC(0, 0, 0, 1), BHWC(1, 1, 1, 2));
  IrTensorId p_aligned =
      AddSliceIr(model, p03, BHWC(0, 0, 0, 2), BHWC(1, 1, 1, 3));
  IrTensorId p_rest =
      AddSliceIr(model, param_raw, BHWC(0, 0, 0, 3), BHWC(1, 1, 1, 7));
  IrTensorId offset =
      AddElementwiseScalarIr(model, OperationType::kFloorMod, p_start, kW);
  IrTensorId active_past =
      AddElementwiseScalarIr(model, OperationType::kMinimum, p_start, kW);
  IrTensorId active_end =
      AddElementwiseScalarIr(model, OperationType::kMinimum, p_end, kW);
  IrTensorId active_aligned =
      AddElementwiseScalarIr(model, OperationType::kMinimum, p_aligned, kW);
  IrTensorId param_past =
      AddConcatIr(model, {offset, active_past, active_past, p_rest},
                  Axis::kChannels, BHWC(1, 1, 1, 7));
  IrTensorId param_upd =
      AddConcatIr(model, {offset, active_end, active_aligned, p_rest},
                  Axis::kChannels, BHWC(1, 1, 1, 7));

  IrTensorId m_past =
      AddSliceIr(model, mask_raw, BHWC(0, 0, 0, 0), BHWC(1, 1, kT, kW));
  IrTensorId m_new =
      AddSliceIr(model, mask_raw, BHWC(0, 0, 0, kW), BHWC(1, 1, kT, kW + kT));
  IrTensorId m_past_rep =
      AddConcatIr(model, {m_past, m_past}, Axis::kWidth, BHWC(1, 1, kRows, kW));
  IrTensorId m_new_rep =
      AddConcatIr(model, {m_new, m_new}, Axis::kWidth, BHWC(1, 1, kRows, kT));
  IrTensorId mask_all =
      AddConcatIr(model, {m_past_rep, m_new_rep}, Axis::kChannels,
                  BHWC(1, 1, kRows, kW + kT));

  IrOp* fill_op = model.add_op();
  fill_op->name = ToString(OperationType::kConstant);
  ConstTensorAttributes fill_attr;
  TensorFloat32 fill_tensor;
  fill_tensor.shape = BHWC(1, 1, 1, 1);
  fill_tensor.data = {-10000.0f};
  fill_attr.tensor = std::move(fill_tensor);
  fill_op->attr = std::move(fill_attr);
  IrTensorId fill_id =
      model.add_tensor(DataType::kFloat32, BHWC(1, 1, 1, 1))->id;
  model.SetProducer(fill_id, fill_op->id);

  IrOp* bmm_past_k = model.add_op();
  bmm_past_k->name = kRuntimeBatchedMatMulType;
  RuntimeBatchedMatMulAttributes pk_attr;
  pk_attr.external_weights.emplace();
  pk_attr.runtime_check.dst_end_ch_index = 2;
  bmm_past_k->attr = pk_attr;
  model.AddConsumer(q, bmm_past_k->id);
  model.AddConsumer(kc, bmm_past_k->id);
  model.AddConsumer(param_past, bmm_past_k->id);
  IrTensorId logits_past =
      model.add_tensor(DataType::kFloat32, BHWC(1, kH, kRows, kW))->id;
  model.SetProducer(logits_past, bmm_past_k->id);

  IrOp* bmm_new_k = model.add_op();
  bmm_new_k->name = ToString(OperationType::kBatchedMatmul);
  BatchedMatMulAttributes nk_attr;
  nk_attr.transpose_right = true;
  bmm_new_k->attr = nk_attr;
  model.AddConsumer(q, bmm_new_k->id);
  model.AddConsumer(kn, bmm_new_k->id);
  IrTensorId logits_new =
      model.add_tensor(DataType::kFloat32, BHWC(1, kH, kRows, kT))->id;
  model.SetProducer(logits_new, bmm_new_k->id);

  IrTensorId logits_all =
      AddConcatIr(model, {logits_past, logits_new}, Axis::kChannels,
                  BHWC(1, kH, kRows, kW + kT));

  IrOp* select_op = model.add_op();
  select_op->name = ToString(OperationType::kSelectV2);
  model.AddConsumer(mask_all, select_op->id);
  model.AddConsumer(logits_all, select_op->id);
  model.AddConsumer(fill_id, select_op->id);
  IrTensorId masked =
      model.add_tensor(DataType::kFloat32, BHWC(1, kH, kRows, kW + kT))->id;
  model.SetProducer(masked, select_op->id);

  IrTensorId maxed =
      AddElementwiseScalarIr(model, OperationType::kMaximum, masked, -kMaxHalf);
  IrTensorId capped =
      AddElementwiseScalarIr(model, OperationType::kMinimum, maxed, kMaxHalf);

  IrOp* softmax_op = model.add_op();
  softmax_op->name = ToString(OperationType::kSoftmax);
  SoftmaxAttributes sm_attr;
  sm_attr.axis = Axis::kChannels;
  softmax_op->attr = sm_attr;
  model.AddConsumer(capped, softmax_op->id);
  IrTensorId probs_all =
      model.add_tensor(DataType::kFloat32, BHWC(1, kH, kRows, kW + kT))->id;
  model.SetProducer(probs_all, softmax_op->id);

  IrTensorId probs_past =
      AddSliceIr(model, probs_all, BHWC(0, 0, 0, 0), BHWC(1, kH, kRows, kW));
  IrTensorId probs_new = AddSliceIr(model, probs_all, BHWC(0, 0, 0, kW),
                                    BHWC(1, kH, kRows, kW + kT));

  IrOp* bmm_past_v = model.add_op();
  bmm_past_v->name = kRuntimeBatchedMatMulType;
  RuntimeBatchedMatMulAttributes pv_attr;
  pv_attr.external_weights.emplace();
  pv_attr.runtime_check.src_end_ch_index = 2;
  bmm_past_v->attr = pv_attr;
  model.AddConsumer(probs_past, bmm_past_v->id);
  model.AddConsumer(vc, bmm_past_v->id);
  model.AddConsumer(param_past, bmm_past_v->id);
  IrTensorId out_past =
      model.add_tensor(DataType::kFloat32, BHWC(1, kH, kRows, kD))->id;
  model.SetProducer(out_past, bmm_past_v->id);

  IrOp* bmm_new_v = model.add_op();
  bmm_new_v->name = ToString(OperationType::kBatchedMatmul);
  BatchedMatMulAttributes nv_attr;
  nv_attr.transpose_right = false;
  bmm_new_v->attr = nv_attr;
  model.AddConsumer(probs_new, bmm_new_v->id);
  model.AddConsumer(vn, bmm_new_v->id);
  IrTensorId out_new =
      model.add_tensor(DataType::kFloat32, BHWC(1, kH, kRows, kD))->id;
  model.SetProducer(out_new, bmm_new_v->id);

  IrOp* add_out = model.add_op();
  add_out->name = ToString(OperationType::kAdd);
  model.AddConsumer(out_past, add_out->id);
  model.AddConsumer(out_new, add_out->id);
  IrTensorId attention =
      model.add_tensor(DataType::kFloat32, BHWC(1, kH, kRows, kD))->id;
  model.SetProducer(attention, add_out->id);

  // Add a downstream op consuming `attention` to verify topological order.
  IrTensorId final_out =
      AddElementwiseScalarIr(model, OperationType::kMul, attention, 2.0f);

  IrOp* cu_op = model.add_op();
  cu_op->name = kAddValuesToCacheType;
  AddValuesToCacheAttributes cu_attr;
  cu_attr.kv_cache_batch_size = kH;
  cu_attr.cache_size = kW;
  cu_attr.head_size = kD;
  cu_attr.is_ring_buffer = true;
  cu_attr.k_cache_ts_idx = 2;
  cu_attr.v_cache_ts_idx = 3;
  cu_attr.k_update_ts_idx = 2;
  cu_attr.v_update_ts_idx = 2;
  cu_op->attr = cu_attr;
  model.AddConsumer(kn, cu_op->id);
  model.AddConsumer(vn, cu_op->id);
  model.AddConsumer(param_upd, cu_op->id);
  IrTensorId kc_out =
      model.add_tensor(DataType::kFloat32, BHWC(1, kH, kW, kD))->id;
  IrTensorId vc_out =
      model.add_tensor(DataType::kFloat32, BHWC(1, kH, kD, kW))->id;
  model.SetProducer(kc_out, cu_op->id);
  model.SetProducer(vc_out, cu_op->id);

  model.add_output(final_out);
  model.add_output(kc_out);
  model.add_output(vc_out);

  ASSERT_OK(ir::FuseSplitSdpaCacheUpdate(&model));

  // Active ops must be: [Transpose(vn), fused_sdpa_cache_update, Mul(2.0f)]
  // in exact topological order!
  std::vector<std::string> live_op_names;
  for (const auto& op_ptr : model.ops()) {
    if (op_ptr != nullptr) {
      live_op_names.push_back(op_ptr->name);
    }
  }
  ASSERT_EQ(live_op_names.size(), 3u);
  EXPECT_EQ(live_op_names[0], ToString(OperationType::kTranspose));
  EXPECT_EQ(live_op_names[1], kFusedSdpaCacheUpdateType);
  EXPECT_EQ(live_op_names[2], ToString(OperationType::kMul));
  EXPECT_EQ(model.outputs().size(), 3u);

  // Lower through ir::LiteRtOpSelector to verify end-to-end GPU model build.
  GpuInfo gpu_info = GetTestMetalGpuInfo();
  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::kF16;
  create_info.storage_type = TensorStorageType::kTexture2D;
  auto set_buffer = [&](IrTensorId tid) {
    const IrTensor* t = model.tensor(tid);
    TensorDescriptor desc(t->desc.GetDataType(), TensorStorageType::kBuffer,
                          ::ml_drift::Layout::kHWC);
    desc.SetBHWCShape(t->desc.GetBHWCShape());
    create_info.predefined[tid] = desc;
  };
  for (IrTensorId tid : {kc, vc, kc_out, vc_out}) {
    set_buffer(tid);
  }
  ir::LiteRtOpSelector selector(&create_info, &gpu_info);
  GpuModel gpu_model;
  ASSERT_OK(
      IrModelToGpuModel(model, create_info, gpu_info, &gpu_model, &selector));

  int masked_softmax_ops = 0;
  int ring_write_ops = 0;
  for (const auto& gpu_node : gpu_model.nodes) {
    if (gpu_node.name == "fused_sdpa_masked_softmax") {
      ++masked_softmax_ops;
    } else if (gpu_node.name == "fused_sdpa_ring_cache_write") {
      ++ring_write_ops;
    }
  }
  EXPECT_EQ(masked_softmax_ops, 1);
  EXPECT_EQ(ring_write_ops, 1);
}

}  // namespace
}  // namespace litert::ml_drift
