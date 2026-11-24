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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/kernels/fully_connected.h"  // from @ml_drift
#include "ml_drift/common/kernels/tests/kernel_test.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/testing_util.h"  // from @ml_drift
#include "ml_drift/common/task/weights_conversion.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift/common/transformations/ir/transform.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/block_ffw_kernel.h"
#include "ml_drift_delegate/delegate/composite/block_ffw_parser.h"
#include "ml_drift_delegate/delegate/composite/ir/litert_op_selector.h"
#include "ml_drift_delegate/delegate/composite/litert_op_selector.h"

namespace litert::ml_drift {
namespace {

using ::ml_drift::BHWC;
using ::ml_drift::CalculationsPrecision;
using ::ml_drift::CreateGpuModelInfo;
using ::ml_drift::DataType;
using ::ml_drift::GpuInfo;
using ::ml_drift::GpuModel;
using ::ml_drift::GpuModelBuilder;
using ::ml_drift::GraphFloat32;
using ::ml_drift::Node;
using ::ml_drift::OHWI;
using ::ml_drift::OperationType;
using ::ml_drift::QuantizeAndDequantizeAttributes;
using ::ml_drift::TensorDescriptor;
using ::ml_drift::TensorStorageType;
using ::ml_drift::Value;
using ::ml_drift::ValueId;
using ::testing::Combine;
using ::testing::FloatNear;
using ::testing::Pointwise;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;

struct HostInt2Weights {
  ::ml_drift::Tensor<OHWI, DataType::kInt8> values;
  ::ml_drift::Tensor<OHWI, DataType::kFloat32> scales;
  ::ml_drift::Tensor<OHWI, DataType::kFloat32> zero_points;
};

HostInt2Weights MakeRandomInt2Weights(const OHWI& shape, uint32_t seed,
                                        float base_scale) {
  HostInt2Weights w;
  w.values.shape = shape;
  w.values.data.resize(shape.DimensionsProduct());
  std::mt19937 rng(seed);
  for (int8_t& v : w.values.data) {
    v = static_cast<int8_t>(static_cast<int>(rng() % 4) - 2);
  }
  const OHWI param_shape(shape.o, shape.h, 1, 1);
  w.scales.shape = param_shape;
  w.scales.data.resize(param_shape.DimensionsProduct());
  w.zero_points.shape = param_shape;
  w.zero_points.data.assign(param_shape.DimensionsProduct(), -0.5f);
  for (size_t i = 0; i < w.scales.data.size(); ++i) {
    w.scales.data[i] = base_scale * (0.75f + 0.05f * static_cast<float>(i % 7));
  }
  return w;
}

float ApplyHostQdq(float x, float scale) {
  const float min_v = -128.0f * scale;
  const float max_v = 127.0f * scale;
  const float clamped = std::min(max_v, std::max(min_v, x));
  const float q =
      std::clamp(std::floor(clamped / scale + 0.5f), -128.0f, 127.0f);
  return q * scale;
}

// Reference CPU execution of the 4-stage blocked FFW layer.
std::vector<float> ComputeBlockedFfwReference(
    const std::vector<float>& x, int tokens, int k1, int k2, int s,
    const std::vector<HostInt2Weights>& u1_ws, const HostInt2Weights& u2_w,
    const HostInt2Weights& d1_w, const std::vector<HostInt2Weights>& d2_ws,
    float u1_in_sc, float u1_out_sc, float u2_in_sc, float u2_out_sc,
    float d1_in_sc, float d1_out_sc, float d2_in_sc, float d2_out_sc) {
  const int i1 = u1_ws[0].values.shape.i;
  const int i2 = k1 * s;
  const int o2 = u2_w.values.shape.o;
  const int i4 = k2 * s;
  const int o4 = d2_ws[0].values.shape.o;
  const int d_model = k1 * i1;

  auto run_batched_fc = [](const HostInt2Weights& w, int h_idx,
                           const float* in_vec) {
    const int O = w.values.shape.o;
    const int H = w.values.shape.h;
    const int I = w.values.shape.i;
    std::vector<float> out(O, 0.0f);
    for (int o = 0; o < O; ++o) {
      const float sc = w.scales.data[o * H + h_idx];
      const float zp = w.zero_points.data[o * H + h_idx];
      double acc = 0.0;
      for (int i = 0; i < I; ++i) {
        const float w_val =
            sc *
            (static_cast<float>(w.values.data[(o * H + h_idx) * I + i]) - zp);
        acc += static_cast<double>(in_vec[i]) * static_cast<double>(w_val);
      }
      out[o] = static_cast<float>(acc);
    }
    return out;
  };

  std::vector<float> final_out(tokens * d_model, 0.0f);
  for (int t = 0; t < tokens; ++t) {
    const float* x_tok = &x[t * d_model];
    // Stage 1: U1 per k1 -> [k1][o1]
    std::vector<std::vector<float>> y_u1(k1);
    for (int b1 = 0; b1 < k1; ++b1) {
      std::vector<float> x_sub(i1);
      for (int c = 0; c < i1; ++c) {
        x_sub[c] = ApplyHostQdq(x_tok[b1 * i1 + c], u1_in_sc);
      }
      y_u1[b1] = run_batched_fc(u1_ws[b1], 0, x_sub.data());
      for (float& v : y_u1[b1]) v = ApplyHostQdq(v, u1_out_sc);
    }

    // Permute [k1, k2 * s] -> [k2, k1 * s] and U2 + SiLU + D1
    std::vector<std::vector<float>> y_d1(k2);
    for (int b2 = 0; b2 < k2; ++b2) {
      std::vector<float> x_u2(i2);
      for (int b1 = 0; b1 < k1; ++b1) {
        for (int c = 0; c < s; ++c) {
          float val = y_u1[b1][b2 * s + c];
          if (u2_in_sc != u1_out_sc) val = ApplyHostQdq(val, u2_in_sc);
          x_u2[b1 * s + c] = val;
        }
      }
      std::vector<float> u2_vec = run_batched_fc(u2_w, b2, x_u2.data());
      std::vector<float> h_vec(o2);
      for (int c = 0; c < o2; ++c) {
        const float u = ApplyHostQdq(u2_vec[c], u2_out_sc);
        const float silu = u / (1.0f + std::exp(-u));
        h_vec[c] = ApplyHostQdq(silu, d1_in_sc);
      }
      y_d1[b2] = run_batched_fc(d1_w, b2, h_vec.data());
      for (float& v : y_d1[b2]) v = ApplyHostQdq(v, d1_out_sc);
    }

    // Permute [k2, k1 * s] -> [k1, k2 * s] and D2 -> merge [k1 * o4]
    for (int b1 = 0; b1 < k1; ++b1) {
      std::vector<float> x_d2(i4);
      for (int b2 = 0; b2 < k2; ++b2) {
        for (int c = 0; c < s; ++c) {
          float val = y_d1[b2][b1 * s + c];
          if (d2_in_sc != d1_out_sc) val = ApplyHostQdq(val, d2_in_sc);
          x_d2[b2 * s + c] = val;
        }
      }
      std::vector<float> d2_vec = run_batched_fc(d2_ws[b1], 0, x_d2.data());
      for (int c = 0; c < o4; ++c) {
        final_out[t * d_model + b1 * o4 + c] =
            ApplyHostQdq(d2_vec[c], d2_out_sc);
      }
    }
  }
  return final_out;
}

Value* AddGraphValue(GraphFloat32* graph, DataType dtype, const BHWC& shape) {
  Value* v = graph->NewValue();
  v->tensor.type = dtype;
  v->tensor.shape = shape;
  return v;
}

Value* AddGraphQdq(GraphFloat32* graph, Value* src, float scale) {
  Value* dst = AddGraphValue(graph, src->tensor.type, src->tensor.shape);
  Node* node = graph->NewNode();
  node->operation.type = ToString(OperationType::kQuantizeAndDequantize);
  QuantizeAndDequantizeAttributes attr{
      .min = -128.0f * scale, .max = 127.0f * scale, .scale = scale};
  node->operation.attributes = attr;
  graph->AddConsumer(node->id, src->id);
  graph->SetProducer(node->id, dst->id);
  return dst;
}

Value* AddPredefinedInt2Fc(
    GraphFloat32* graph,
    absl::flat_hash_map<ValueId, TensorDescriptor>* const_tensors,
    const GpuInfo& gpu_info, DataType activation_dtype, Value* src,
    const HostInt2Weights& host_w, const BHWC& dst_shape) {
  const OHWI shape = host_w.values.shape;
  const auto desc =
      ::ml_drift::GetFullyConnectedInt2WeightsDesc(gpu_info, shape);
  TensorDescriptor w_desc =
      ::ml_drift::GetTensorDescriptorForWeightsLayout(host_w.values, desc);
  TensorDescriptor s_desc = ::ml_drift::ScaleOrZeroPointToTensorDesc(
      gpu_info, host_w.scales, activation_dtype);
  TensorDescriptor zp_desc = ::ml_drift::ScaleOrZeroPointToTensorDesc(
      gpu_info, host_w.zero_points, activation_dtype);

  Value* w_val =
      AddGraphValue(graph, w_desc.GetDataType(), w_desc.GetBHWCShape());
  Value* s_val = AddGraphValue(graph, activation_dtype, s_desc.GetBHWCShape());
  Value* zp_val =
      AddGraphValue(graph, activation_dtype, zp_desc.GetBHWCShape());
  (*const_tensors)[w_val->id] = std::move(w_desc);
  (*const_tensors)[s_val->id] = std::move(s_desc);
  (*const_tensors)[zp_val->id] = std::move(zp_desc);

  Value* dst = AddGraphValue(graph, activation_dtype, dst_shape);
  Node* fc = graph->NewNode();
  fc->operation.type = ToString(OperationType::kFullyConnectedInt2);
  ::ml_drift::FullyConnectedInt2Attributes attr;
  ::ml_drift::Tensor<OHWI, DataType::kInt2> dummy_w;
  dummy_w.shape = shape;
  attr.weights = dummy_w;
  attr.scale.shape = host_w.scales.shape;
  attr.zero_point.shape = host_w.zero_points.shape;
  fc->operation.attributes = std::move(attr);
  graph->AddConsumer(fc->id, src->id);
  graph->AddConsumer(fc->id, w_val->id);
  graph->AddConsumer(fc->id, s_val->id);
  graph->AddConsumer(fc->id, zp_val->id);
  graph->SetProducer(fc->id, dst->id);
  return dst;
}

struct BlockedFfwTestCase {
  int tokens = 0;
  int k1 = 0;
  int k2 = 0;
  int s = 8;
  int i1 = 16;
  int o1 = 0;
  int i2 = 0;
  int o2 = 24;
  int i4 = 0;
  int o4 = 16;
  int d_model = 0;
  bool distinct_adjacent_scales = false;
  std::vector<HostInt2Weights> u1_ws;
  HostInt2Weights u2_w;
  HostInt2Weights d1_w;
  std::vector<HostInt2Weights> d2_ws;
  float u1_in_sc = 0.025f;
  float u1_out_sc = 0.060f;
  float u2_in_sc = 0.060f;
  float u2_out_sc = 0.090f;
  float d1_in_sc = 0.045f;
  float d1_out_sc = 0.030f;
  float d2_in_sc = 0.030f;
  float d2_out_sc = 0.025f;
  std::vector<float> x_data;
  std::vector<float> expected;
};

BlockedFfwTestCase MakeBlockedFfwTestCase(int tokens,
                                          bool distinct_adjacent_scales,
                                          int k1 = 2, int k2 = 4) {
  BlockedFfwTestCase tc;
  tc.tokens = tokens;
  tc.k1 = k1;
  tc.k2 = k2;
  tc.o1 = k2 * tc.s;
  tc.i2 = k1 * tc.s;
  tc.i4 = k2 * tc.s;
  tc.d_model = k1 * tc.i1;
  tc.distinct_adjacent_scales = distinct_adjacent_scales;
  tc.u2_in_sc = distinct_adjacent_scales ? 0.055f : tc.u1_out_sc;
  tc.d2_in_sc = distinct_adjacent_scales ? 0.035f : tc.d1_out_sc;

  for (int k = 0; k < k1; ++k) {
    tc.u1_ws.push_back(
        MakeRandomInt2Weights(OHWI(tc.o1, 1, 1, tc.i1), 101 + k, 0.08f));
    tc.d2_ws.push_back(
        MakeRandomInt2Weights(OHWI(tc.o4, 1, 1, tc.i4), 404 + k, 0.06f));
  }
  tc.u2_w = MakeRandomInt2Weights(OHWI(tc.o2, k2, 1, tc.i2), 202, 0.07f);
  tc.d1_w = MakeRandomInt2Weights(OHWI(tc.i2, k2, 1, tc.o2), 303, 0.05f);

  tc.x_data.resize(tokens * tc.d_model);
  for (size_t i = 0; i < tc.x_data.size(); ++i) {
    tc.x_data[i] = 0.8f * std::sin(static_cast<float>(i + 1) * 0.23f) -
                   0.4f * std::cos(static_cast<float>(i + 2) * 0.17f);
  }
  tc.expected = ComputeBlockedFfwReference(
      tc.x_data, tokens, k1, k2, tc.s, tc.u1_ws, tc.u2_w, tc.d1_w, tc.d2_ws,
      tc.u1_in_sc, tc.u1_out_sc, tc.u2_in_sc, tc.u2_out_sc, tc.d1_in_sc,
      tc.d1_out_sc, tc.d2_in_sc, tc.d2_out_sc);
  return tc;
}

absl::Status RunBlockedFfwEndToEndTest(
    ::ml_drift::TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage, int tokens, bool distinct_adjacent_scales,
    int k1 = 2, int k2 = 4, bool expect_four_dispatches = true) {
  const DataType dtype = precision == CalculationsPrecision::kF16
                             ? DataType::kFloat16
                             : DataType::kFloat32;
  const BlockedFfwTestCase tc =
      MakeBlockedFfwTestCase(tokens, distinct_adjacent_scales, k1, k2);

  GraphFloat32 graph;
  CreateGpuModelInfo create_info;
  create_info.precision = precision;
  create_info.storage_type = storage;
  absl::flat_hash_map<ValueId, TensorDescriptor> const_tensors;

  const GpuInfo& gpu_info = env.GetGpuInfo();
  Value* x_val = AddGraphValue(&graph, dtype, BHWC(1, 1, tokens, tc.d_model));
  Value* out_val = AddGraphValue(&graph, dtype, BHWC(1, 1, tokens, tc.d_model));

  // Build the exact GraphFloat32 topology emitted by BlockFfwOperationParser.
  Value* u2_in = nullptr;
  {
    std::vector<Value*> parts(tc.k1);
    for (int k = 0; k < tc.k1; ++k) {
      Value* unp = AddGraphValue(&graph, dtype, BHWC(1, 1, tokens, tc.i1));
      Node* unp_n = graph.NewNode();
      unp_n->operation.type = kBlockFfwUnpackPartType;
      unp_n->operation.attributes =
          BlockFfwUnpackPartAttributes{.part = k, .num_parts = tc.k1};
      graph.AddConsumer(unp_n->id, x_val->id);
      graph.SetProducer(unp_n->id, unp->id);

      Value* q_in = AddGraphQdq(&graph, unp, tc.u1_in_sc);
      Value* fc =
          AddPredefinedInt2Fc(&graph, &const_tensors, gpu_info, dtype, q_in,
                              tc.u1_ws[k], BHWC(1, 1, tokens, tc.o1));
      parts[k] = AddGraphQdq(&graph, fc, tc.u1_out_sc);
    }
    Value* pack = AddGraphValue(&graph, dtype, BHWC(1, tc.k2, tokens, tc.i2));
    Node* pack_n = graph.NewNode();
    pack_n->operation.type = kBlockFfwPackPartsType;
    pack_n->operation.attributes =
        BlockFfwPackPartsAttributes{.num_parts = tc.k1};
    for (Value* p : parts) graph.AddConsumer(pack_n->id, p->id);
    graph.SetProducer(pack_n->id, pack->id);
    u2_in = pack;
  }
  if (tc.distinct_adjacent_scales) {
    u2_in = AddGraphQdq(&graph, u2_in, tc.u2_in_sc);
  }

  Value* u2_raw =
      AddPredefinedInt2Fc(&graph, &const_tensors, gpu_info, dtype, u2_in,
                          tc.u2_w, BHWC(1, tc.k2, tokens, tc.o2));
  Value* u2_q = AddGraphQdq(&graph, u2_raw, tc.u2_out_sc);
  Value* act = AddGraphValue(&graph, dtype, BHWC(1, tc.k2, tokens, tc.o2));
  Node* silu_n = graph.NewNode();
  silu_n->operation.type = kBlockFfwSiluType;
  graph.AddConsumer(silu_n->id, u2_q->id);
  graph.SetProducer(silu_n->id, act->id);

  Value* d1_in = AddGraphQdq(&graph, act, tc.d1_in_sc);
  Value* d1_raw =
      AddPredefinedInt2Fc(&graph, &const_tensors, gpu_info, dtype, d1_in,
                          tc.d1_w, BHWC(1, tc.k2, tokens, tc.i2));
  Value* d1_q = AddGraphQdq(&graph, d1_raw, tc.d1_out_sc);

  std::vector<Value*> parts(tc.k1);
  for (int k = 0; k < tc.k1; ++k) {
    Value* unp = AddGraphValue(&graph, dtype, BHWC(1, 1, tokens, tc.i4));
    Node* unp_n = graph.NewNode();
    unp_n->operation.type = kBlockFfwUnpackPartType;
    unp_n->operation.attributes =
        BlockFfwUnpackPartAttributes{.part = k, .num_parts = tc.k1};
    graph.AddConsumer(unp_n->id, d1_q->id);
    graph.SetProducer(unp_n->id, unp->id);

    Value* q_in =
        tc.distinct_adjacent_scales ? AddGraphQdq(&graph, unp, tc.d2_in_sc)
                                    : unp;
    Value* fc =
        AddPredefinedInt2Fc(&graph, &const_tensors, gpu_info, dtype, q_in,
                            tc.d2_ws[k], BHWC(1, 1, tokens, tc.o4));
    parts[k] = AddGraphQdq(&graph, fc, tc.d2_out_sc);
  }
  Node* pack_n = graph.NewNode();
  pack_n->operation.type = kBlockFfwPackPartsType;
  pack_n->operation.attributes =
      BlockFfwPackPartsAttributes{.num_parts = tc.k1};
  for (Value* p : parts) graph.AddConsumer(pack_n->id, p->id);
  graph.SetProducer(pack_n->id, out_val->id);

  absl::flat_hash_map<ValueId, TensorDescriptor> runtime_tensors;
  for (const Value* v : graph.values()) {
    if (const_tensors.contains(v->id)) continue;
    TensorDescriptor desc(v->tensor.type, storage,
                          v->tensor.shape.b == 1 ? ::ml_drift::Layout::kHWC
                                                 : ::ml_drift::Layout::kBHWC);
    ABSL_RETURN_IF_ERROR(
        desc.UpdateToSupportedStorageType(gpu_info, v->tensor.shape));
    desc.SetBHWCShape(v->tensor.shape);
    runtime_tensors[v->id] = std::move(desc);
  }

  GpuModelBuilder builder(gpu_info,
                          {.storage = storage,
                           .use_f32_accum_for_f16_convolutions =
                               precision == CalculationsPrecision::kF32F16},
                          std::move(runtime_tensors), std::move(const_tensors));
  LiteRtOpSelector selector(&create_info, &gpu_info);
  std::set<::ml_drift::NodeId> consumed_nodes;
  for (const Node* node : graph.nodes()) {
    if (consumed_nodes.count(node->id) != 0) continue;
    std::set<::ml_drift::NodeId> new_consumed;
    if (selector
            .GPUSubgraphFromGraph(graph, node->id, consumed_nodes,
                                  &new_consumed, &builder)
            .ok()) {
      consumed_nodes.insert(new_consumed.begin(), new_consumed.end());
      continue;
    }
    const std::vector<Value*> ins = graph.FindInputs(node->id);
    const std::vector<Value*> outs = graph.FindOutputs(node->id);
    ::ml_drift::OperationDef op_def;
    for (const Value* in : ins) {
      ABSL_ASSIGN_OR_RETURN(auto th, builder.GetTensor(in->id));
      op_def.src_tensors.push_back(th.tensor_desc);
    }
    for (const Value* out : outs) {
      ABSL_ASSIGN_OR_RETURN(auto th, builder.GetTensor(out->id));
      op_def.dst_tensors.push_back(th.tensor_desc);
    }
    ABSL_RETURN_IF_ERROR(
        selector.GPUOperationFromNode(op_def, ins, outs, *node, &builder));
    consumed_nodes.insert(node->id);
  }

  GpuModel gpu_model;
  ABSL_RETURN_IF_ERROR(builder.GetGpuModel(std::vector<ValueId>{x_val->id},
                                           std::vector<ValueId>{out_val->id},
                                           &gpu_model));

  if (tokens == 1 && expect_four_dispatches) {
    // Single-token decode must fuse U1 and D2 edge subgraphs + all QDQs + SiLU
    // down to exactly 4 GPU kernel dispatches.
    EXPECT_EQ(gpu_model.nodes.size(), 4);
  }

  ::ml_drift::TensorFloat32 src_cpu;
  src_cpu.shape = BHWC(1, 1, tokens, tc.d_model);
  src_cpu.data = tc.x_data;
  ::ml_drift::TensorFloat32 dst_cpu;
  dst_cpu.shape = BHWC(1, 1, tokens, tc.d_model);
  dst_cpu.data.resize(tc.expected.size());

  ABSL_RETURN_IF_ERROR(env.ExecuteGpuModel(
      {src_cpu}, std::vector<::ml_drift::TensorFloat32*>{&dst_cpu},
      &gpu_model));

  const float tol =
      precision == CalculationsPrecision::kF16 ? 2.0f * tc.d2_out_sc : 1e-4f;
  EXPECT_THAT(dst_cpu.data, Pointwise(FloatNear(tol), tc.expected));
  return absl::OkStatus();
}

::ml_drift::ir::IrTensorId AddIrQdq(::ml_drift::ir::IrModel* ir_model,
                                    ::ml_drift::ir::IrTensorId src_id,
                                    float scale) {
  const auto* src = ir_model->tensor(src_id);
  const auto dst_id =
      ir_model->add_tensor(src->desc.GetDataType(), src->desc.GetBHWCShape())
          ->id;
  ::ml_drift::ir::IrOp* op = ir_model->add_op();
  op->name = ToString(OperationType::kQuantizeAndDequantize);
  QuantizeAndDequantizeAttributes attr{
      .min = -128.0f * scale, .max = 127.0f * scale, .scale = scale};
  op->attr = attr;
  ir_model->AddConsumer(src_id, op->id);
  ir_model->SetProducer(dst_id, op->id);
  return dst_id;
}

::ml_drift::ir::IrTensorId AddPredefinedInt2FcIr(
    ::ml_drift::ir::IrModel* ir_model,
    absl::flat_hash_map<ValueId, TensorDescriptor>* const_tensors,
    const GpuInfo& gpu_info, DataType activation_dtype,
    ::ml_drift::ir::IrTensorId src_id, const HostInt2Weights& host_w,
    const BHWC& dst_shape) {
  const OHWI shape = host_w.values.shape;
  const auto desc =
      ::ml_drift::GetFullyConnectedInt2WeightsDesc(gpu_info, shape);
  TensorDescriptor w_desc =
      ::ml_drift::GetTensorDescriptorForWeightsLayout(host_w.values, desc);
  TensorDescriptor s_desc = ::ml_drift::ScaleOrZeroPointToTensorDesc(
      gpu_info, host_w.scales, activation_dtype);
  TensorDescriptor zp_desc = ::ml_drift::ScaleOrZeroPointToTensorDesc(
      gpu_info, host_w.zero_points, activation_dtype);

  const auto w_id =
      ir_model->add_tensor(w_desc.GetDataType(), w_desc.GetBHWCShape())->id;
  const auto s_id =
      ir_model->add_tensor(activation_dtype, s_desc.GetBHWCShape())->id;
  const auto zp_id =
      ir_model->add_tensor(activation_dtype, zp_desc.GetBHWCShape())->id;
  (*const_tensors)[w_id] = std::move(w_desc);
  (*const_tensors)[s_id] = std::move(s_desc);
  (*const_tensors)[zp_id] = std::move(zp_desc);

  const auto dst_id = ir_model->add_tensor(activation_dtype, dst_shape)->id;
  ::ml_drift::ir::IrOp* fc = ir_model->add_op();
  fc->name = ToString(OperationType::kFullyConnectedInt2);
  ::ml_drift::FullyConnectedInt2Attributes attr;
  ::ml_drift::Tensor<OHWI, DataType::kInt2> dummy_w;
  dummy_w.shape = shape;
  attr.weights = dummy_w;
  attr.scale.shape = host_w.scales.shape;
  attr.zero_point.shape = host_w.zero_points.shape;
  fc->attr = std::move(attr);
  ir_model->AddConsumer(src_id, fc->id);
  ir_model->AddConsumer(w_id, fc->id);
  ir_model->AddConsumer(s_id, fc->id);
  ir_model->AddConsumer(zp_id, fc->id);
  ir_model->SetProducer(dst_id, fc->id);
  return dst_id;
}

absl::Status RunBlockedFfwIrModelEndToEndTest(
    ::ml_drift::TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage, int tokens, bool distinct_adjacent_scales,
    int k1 = 2, int k2 = 4) {
  const DataType dtype = precision == CalculationsPrecision::kF16
                             ? DataType::kFloat16
                             : DataType::kFloat32;
  const BlockedFfwTestCase tc =
      MakeBlockedFfwTestCase(tokens, distinct_adjacent_scales, k1, k2);

  ::ml_drift::ir::IrModel ir_model;
  CreateGpuModelInfo create_info;
  create_info.precision = precision;
  create_info.storage_type = storage;
  absl::flat_hash_map<ValueId, TensorDescriptor> const_tensors;

  const GpuInfo& gpu_info = env.GetGpuInfo();
  const auto x_id =
      ir_model.add_tensor(dtype, BHWC(1, 1, tokens, tc.d_model))->id;
  const auto out_id =
      ir_model.add_tensor(dtype, BHWC(1, 1, tokens, tc.d_model))->id;
  ir_model.add_input(x_id);
  ir_model.add_output(out_id);

  ::ml_drift::ir::IrTensorId u2_in_id = 0;
  {
    std::vector<::ml_drift::ir::IrTensorId> parts(tc.k1);
    for (int k = 0; k < tc.k1; ++k) {
      const auto unp_id =
          ir_model.add_tensor(dtype, BHWC(1, 1, tokens, tc.i1))->id;
      ::ml_drift::ir::IrOp* unp = ir_model.add_op();
      unp->name = kBlockFfwUnpackPartType;
      unp->attr = BlockFfwUnpackPartAttributes{.part = k, .num_parts = tc.k1};
      ir_model.AddConsumer(x_id, unp->id);
      ir_model.SetProducer(unp_id, unp->id);

      const auto q_in_id = AddIrQdq(&ir_model, unp_id, tc.u1_in_sc);
      const auto fc_id = AddPredefinedInt2FcIr(
          &ir_model, &const_tensors, gpu_info, dtype, q_in_id, tc.u1_ws[k],
          BHWC(1, 1, tokens, tc.o1));
      parts[k] = AddIrQdq(&ir_model, fc_id, tc.u1_out_sc);
    }
    const auto pack_id =
        ir_model.add_tensor(dtype, BHWC(1, tc.k2, tokens, tc.i2))->id;
    ::ml_drift::ir::IrOp* pack = ir_model.add_op();
    pack->name = kBlockFfwPackPartsType;
    pack->attr = BlockFfwPackPartsAttributes{.num_parts = tc.k1};
    for (auto p : parts) ir_model.AddConsumer(p, pack->id);
    ir_model.SetProducer(pack_id, pack->id);
    u2_in_id = pack_id;
  }
  if (tc.distinct_adjacent_scales) {
    u2_in_id = AddIrQdq(&ir_model, u2_in_id, tc.u2_in_sc);
  }

  const auto u2_raw_id =
      AddPredefinedInt2FcIr(&ir_model, &const_tensors, gpu_info, dtype,
                            u2_in_id, tc.u2_w, BHWC(1, tc.k2, tokens, tc.o2));
  const auto u2_q_id = AddIrQdq(&ir_model, u2_raw_id, tc.u2_out_sc);
  const auto act_id =
      ir_model.add_tensor(dtype, BHWC(1, tc.k2, tokens, tc.o2))->id;
  ::ml_drift::ir::IrOp* silu = ir_model.add_op();
  silu->name = kBlockFfwSiluType;
  ir_model.AddConsumer(u2_q_id, silu->id);
  ir_model.SetProducer(act_id, silu->id);

  const auto d1_in_id = AddIrQdq(&ir_model, act_id, tc.d1_in_sc);
  const auto d1_raw_id =
      AddPredefinedInt2FcIr(&ir_model, &const_tensors, gpu_info, dtype,
                            d1_in_id, tc.d1_w, BHWC(1, tc.k2, tokens, tc.i2));
  const auto d1_q_id = AddIrQdq(&ir_model, d1_raw_id, tc.d1_out_sc);

  std::vector<::ml_drift::ir::IrTensorId> parts(tc.k1);
  for (int k = 0; k < tc.k1; ++k) {
    const auto unp_id =
        ir_model.add_tensor(dtype, BHWC(1, 1, tokens, tc.i4))->id;
    ::ml_drift::ir::IrOp* unp = ir_model.add_op();
    unp->name = kBlockFfwUnpackPartType;
    unp->attr = BlockFfwUnpackPartAttributes{.part = k, .num_parts = tc.k1};
    ir_model.AddConsumer(d1_q_id, unp->id);
    ir_model.SetProducer(unp_id, unp->id);

    const auto q_in_id = tc.distinct_adjacent_scales
                             ? AddIrQdq(&ir_model, unp_id, tc.d2_in_sc)
                             : unp_id;
    const auto fc_id = AddPredefinedInt2FcIr(
        &ir_model, &const_tensors, gpu_info, dtype, q_in_id, tc.d2_ws[k],
        BHWC(1, 1, tokens, tc.o4));
    parts[k] = AddIrQdq(&ir_model, fc_id, tc.d2_out_sc);
  }
  ::ml_drift::ir::IrOp* pack = ir_model.add_op();
  pack->name = kBlockFfwPackPartsType;
  pack->attr = BlockFfwPackPartsAttributes{.num_parts = tc.k1};
  for (auto p : parts) ir_model.AddConsumer(p, pack->id);
  ir_model.SetProducer(out_id, pack->id);

  // Verify ML Drift's standard IR transformations preserve the topology.
  ABSL_RETURN_IF_ERROR(::ml_drift::ir::TransformIrModel(&ir_model));

  absl::flat_hash_map<ValueId, TensorDescriptor> runtime_tensors;
  for (const auto& t : ir_model.tensors()) {
    if (t == nullptr || const_tensors.contains(t->id)) continue;
    const BHWC shape = t->desc.GetBHWCShape();
    TensorDescriptor desc(
        t->desc.GetDataType(), storage,
        shape.b == 1 ? ::ml_drift::Layout::kHWC : ::ml_drift::Layout::kBHWC);
    ABSL_RETURN_IF_ERROR(desc.UpdateToSupportedStorageType(gpu_info, shape));
    desc.SetBHWCShape(shape);
    runtime_tensors[t->id] = std::move(desc);
  }

  GpuModelBuilder builder(gpu_info,
                          {.storage = storage,
                           .use_f32_accum_for_f16_convolutions =
                               precision == CalculationsPrecision::kF32F16},
                          std::move(runtime_tensors), std::move(const_tensors));
  ir::LiteRtOpSelector selector(&create_info, &gpu_info);
  absl::flat_hash_set<::ml_drift::ir::IrOpId> consumed_ops;
  for (const auto& op_ptr : ir_model.ops()) {
    if (op_ptr == nullptr || consumed_ops.contains(op_ptr->id)) continue;
    absl::flat_hash_set<::ml_drift::ir::IrOpId> new_consumed;
    if (selector
            .GPUSubgraphFromIrModel(ir_model, op_ptr->id, consumed_ops,
                                    &new_consumed, &builder)
            .ok()) {
      consumed_ops.insert(new_consumed.begin(), new_consumed.end());
      continue;
    }
    std::vector<const ::ml_drift::ir::IrTensor*> ins;
    std::vector<const ::ml_drift::ir::IrTensor*> outs;
    ::ml_drift::OperationDef op_def;
    for (auto in_id : op_ptr->inputs) {
      ins.push_back(ir_model.tensor(in_id));
      ABSL_ASSIGN_OR_RETURN(auto th, builder.GetTensor(in_id));
      op_def.src_tensors.push_back(th.tensor_desc);
    }
    for (auto out_id_op : op_ptr->outputs) {
      outs.push_back(ir_model.tensor(out_id_op));
      ABSL_ASSIGN_OR_RETURN(auto th, builder.GetTensor(out_id_op));
      op_def.dst_tensors.push_back(th.tensor_desc);
    }
    ABSL_RETURN_IF_ERROR(
        selector.GPUOperationFromNode(op_def, ins, outs, *op_ptr, &builder));
    consumed_ops.insert(op_ptr->id);
  }

  GpuModel gpu_model;
  ABSL_RETURN_IF_ERROR(builder.GetGpuModel(
      std::vector<ValueId>{static_cast<ValueId>(x_id)},
      std::vector<ValueId>{static_cast<ValueId>(out_id)}, &gpu_model));

  if (tokens == 1) {
    EXPECT_EQ(gpu_model.nodes.size(), 4);
  }

  ::ml_drift::TensorFloat32 src_cpu;
  src_cpu.shape = BHWC(1, 1, tokens, tc.d_model);
  src_cpu.data = tc.x_data;
  ::ml_drift::TensorFloat32 dst_cpu;
  dst_cpu.shape = BHWC(1, 1, tokens, tc.d_model);
  dst_cpu.data.resize(tc.expected.size());

  ABSL_RETURN_IF_ERROR(env.ExecuteGpuModel(
      {src_cpu}, std::vector<::ml_drift::TensorFloat32*>{&dst_cpu},
      &gpu_model));

  const float tol =
      precision == CalculationsPrecision::kF16 ? 2.0f * tc.d2_out_sc : 1e-4f;
  EXPECT_THAT(dst_cpu.data, Pointwise(FloatNear(tol), tc.expected));
  return absl::OkStatus();
}

class BlockFfwKernelTest : public ::ml_drift::FloatTest {
 public:
  void SetUp() override {
    if (!exec_env) {
      GTEST_SKIP() << "TestExecutionEnvironment not initialized.";
    }
    const auto data_type = ::ml_drift::DeduceDataTypeFromPrecision(precision());
    if (!exec_env->IsStorageSupported(storage(), data_type)) {
      GTEST_SKIP() << "Unsupported data type: "
                   << ::ml_drift::ToString(data_type)
                   << " storage type: " << ::ml_drift::ToString(storage());
    }
  }
};

TEST_P(BlockFfwKernelTest, LargeK1ChunksPackPartsMatchesReference) {
  ASSERT_OK(RunBlockedFfwEndToEndTest(*exec_env, precision(), storage(),
                                      /*tokens=*/2,
                                      /*distinct_adjacent_scales=*/false,
                                      /*k1=*/8));
}

TEST_P(BlockFfwKernelTest, SinglePartK1K2MatchesReference) {
  ASSERT_OK(RunBlockedFfwEndToEndTest(*exec_env, precision(), storage(),
                                      /*tokens=*/2,
                                      /*distinct_adjacent_scales=*/false,
                                      /*k1=*/1, /*k2=*/1));
}

TEST_P(BlockFfwKernelTest, DecodeFusesToFourDispatches) {
  ASSERT_OK(RunBlockedFfwEndToEndTest(*exec_env, precision(), storage(),
                                      /*tokens=*/1,
                                      /*distinct_adjacent_scales=*/false));
}

TEST_P(BlockFfwKernelTest, DecodeWithDistinctScalesFusesToFourDispatches) {
  ASSERT_OK(RunBlockedFfwEndToEndTest(*exec_env, precision(), storage(),
                                      /*tokens=*/1,
                                      /*distinct_adjacent_scales=*/true));
}

TEST_P(BlockFfwKernelTest, DecodeSinglePartK1OneFusesToFourDispatches) {
  ASSERT_OK(RunBlockedFfwEndToEndTest(*exec_env, precision(), storage(),
                                      /*tokens=*/1,
                                      /*distinct_adjacent_scales=*/false,
                                      /*k1=*/1, /*k2=*/2,
                                      /*expect_four_dispatches=*/true));
}

TEST_P(BlockFfwKernelTest, DecodeLargeK1BindingLimitFallbackMatchesReference) {
  ASSERT_OK(RunBlockedFfwEndToEndTest(*exec_env, precision(), storage(),
                                      /*tokens=*/1,
                                      /*distinct_adjacent_scales=*/false,
                                      /*k1=*/8, /*k2=*/2,
                                      /*expect_four_dispatches=*/false));
}

TEST_P(BlockFfwKernelTest, PrefillMultiTokenMatchesReference) {
  ASSERT_OK(RunBlockedFfwEndToEndTest(*exec_env, precision(), storage(),
                                      /*tokens=*/5,
                                      /*distinct_adjacent_scales=*/false));
  ASSERT_OK(RunBlockedFfwEndToEndTest(*exec_env, precision(), storage(),
                                      /*tokens=*/6,
                                      /*distinct_adjacent_scales=*/true));
}

TEST_P(BlockFfwKernelTest, IrModelDecodeFusesToFourDispatches) {
  ASSERT_OK(RunBlockedFfwIrModelEndToEndTest(
      *exec_env, precision(), storage(), /*tokens=*/1,
      /*distinct_adjacent_scales=*/false));
  ASSERT_OK(RunBlockedFfwIrModelEndToEndTest(
      *exec_env, precision(), storage(), /*tokens=*/1,
      /*distinct_adjacent_scales=*/true));
}

TEST_P(BlockFfwKernelTest, IrModelPrefillMultiTokenMatchesReference) {
  ASSERT_OK(RunBlockedFfwIrModelEndToEndTest(
      *exec_env, precision(), storage(), /*tokens=*/5,
      /*distinct_adjacent_scales=*/false));
  ASSERT_OK(RunBlockedFfwIrModelEndToEndTest(
      *exec_env, precision(), storage(), /*tokens=*/6,
      /*distinct_adjacent_scales=*/true));
}

TEST_P(BlockFfwKernelTest, IrModelLargeK1ChunksPackPartsMatchesReference) {
  ASSERT_OK(RunBlockedFfwIrModelEndToEndTest(
      *exec_env, precision(), storage(), /*tokens=*/2,
      /*distinct_adjacent_scales=*/false, /*k1=*/8));
}

TEST_P(BlockFfwKernelTest, IrModelSinglePartK1K2MatchesReference) {
  ASSERT_OK(RunBlockedFfwIrModelEndToEndTest(
      *exec_env, precision(), storage(), /*tokens=*/2,
      /*distinct_adjacent_scales=*/false, /*k1=*/1, /*k2=*/1));
}

struct DirectEdgeFcConfig {
  DataType weight_type = DataType::kInt2;
  int num_parts = 2;
  int src_height = 1;
  int dst_height = 2;
  int part_i = 16;
  int part_o = 16;
  int scale_groups = 1;
  bool has_zero_point = true;
  std::vector<float> src_qdq_scales;
  std::vector<float> dst_qdq_scales;
};

absl::Status RunDirectEdgeFcTest(::ml_drift::TestExecutionEnvironment& env,
                                 CalculationsPrecision precision,
                                 TensorStorageType storage,
                                 const DirectEdgeFcConfig& cfg, uint32_t seed) {
  const GpuInfo& gpu_info = env.GetGpuInfo();
  const DataType dtype = precision == CalculationsPrecision::kF16
                             ? DataType::kFloat16
                             : DataType::kFloat32;
  const int src_row_c = cfg.part_i / cfg.src_height;
  const int dst_row_c = cfg.part_o / cfg.dst_height;
  const BHWC src_shape(1, cfg.src_height, 1, cfg.num_parts * src_row_c);
  const BHWC dst_shape(1, cfg.dst_height, 1, cfg.num_parts * dst_row_c);

  std::mt19937 rng(seed);
  int q_range = 4;
  int q_min = -2;
  float base_scale = 0.08f;
  if (cfg.weight_type == DataType::kInt4) {
    q_range = 16;
    q_min = -8;
    base_scale = 0.03f;
  } else if (cfg.weight_type == DataType::kInt8) {
    q_range = 256;
    q_min = -128;
    base_scale = 0.004f;
  }

  std::vector<float> src_data(src_shape.DimensionsProduct());
  for (size_t i = 0; i < src_data.size(); ++i) {
    src_data[i] = 0.7f * std::sin(static_cast<float>(i + 1 + seed) * 0.29f) -
                  0.35f * std::cos(static_cast<float>(i + 3 + seed) * 0.19f);
  }

  const OHWI w_shape(cfg.part_o, 1, 1, cfg.part_i);
  const OHWI scale_shape(cfg.part_o, 1, 1, cfg.scale_groups);
  ::ml_drift::WeightsDescription w_desc;
  if (cfg.weight_type == DataType::kInt2) {
    w_desc = ::ml_drift::GetFullyConnectedInt2WeightsDesc(gpu_info, w_shape);
  } else if (cfg.weight_type == DataType::kInt4) {
    w_desc = ::ml_drift::GetFullyConnectedInt4WeightsDesc(gpu_info, w_shape);
  } else {
    w_desc = ::ml_drift::GetFullyConnectedInt8WeightsDesc(gpu_info, w_shape);
  }

  GpuModelBuilder builder(gpu_info,
                          {.storage = storage,
                           .use_f32_accum_for_f16_convolutions =
                               precision == CalculationsPrecision::kF32F16});
  const GpuModelBuilder::TensorHandle src_h =
      builder.AddTensor(src_shape, dtype);
  const GpuModelBuilder::TensorHandle dst_h =
      builder.AddTensor(dst_shape, dtype);

  BlockFfwEdgeFcSpec spec;
  spec.num_parts = cfg.num_parts;
  spec.part_weights.reserve(cfg.num_parts);
  for (float sc : cfg.src_qdq_scales) {
    spec.src_qdqs.push_back(QuantizeAndDequantizeAttributes{
        .min = -128.0f * sc, .max = 127.0f * sc, .scale = sc});
  }
  for (float sc : cfg.dst_qdq_scales) {
    spec.dst_qdqs.push_back(QuantizeAndDequantizeAttributes{
        .min = -128.0f * sc, .max = 127.0f * sc, .scale = sc});
  }

  std::vector<float> expected(dst_shape.DimensionsProduct(), 0.0f);
  const int group_i = cfg.part_i / cfg.scale_groups;

  for (int part = 0; part < cfg.num_parts; ++part) {
    ::ml_drift::Tensor<OHWI, DataType::kInt8> w_vals;
    w_vals.shape = w_shape;
    w_vals.data.resize(w_shape.DimensionsProduct());
    for (int8_t& v : w_vals.data) {
      v = static_cast<int8_t>(static_cast<int>(rng() % q_range) + q_min);
    }

    ::ml_drift::Tensor<OHWI, DataType::kFloat32> scales;
    scales.shape = scale_shape;
    scales.data.resize(scale_shape.DimensionsProduct());
    for (size_t i = 0; i < scales.data.size(); ++i) {
      scales.data[i] =
          base_scale * (0.75f + 0.05f * static_cast<float>((i + part) % 7));
    }

    ::ml_drift::Tensor<OHWI, DataType::kFloat32> zps;
    if (cfg.has_zero_point) {
      zps.shape = scale_shape;
      zps.data.resize(scale_shape.DimensionsProduct());
      for (size_t i = 0; i < zps.data.size(); ++i) {
        zps.data[i] = -0.5f + 0.25f * static_cast<float>((i + part) % 5);
      }
    }

    // Compute CPU reference for this partition.
    for (int o = 0; o < cfg.part_o; ++o) {
      double acc = 0.0;
      for (int i = 0; i < cfg.part_i; ++i) {
        const int src_y = i / src_row_c;
        const int src_c = part * src_row_c + (i % src_row_c);
        float x_val = src_data[src_y * src_shape.c + src_c];
        for (float sc : cfg.src_qdq_scales) {
          x_val = ApplyHostQdq(x_val, sc);
        }
        const int g = i / group_i;
        const int param_idx = o * cfg.scale_groups + g;
        const float sc = scales.data[param_idx];
        const float zp = cfg.has_zero_point ? zps.data[param_idx] : 0.0f;
        const float w_f =
            sc * (static_cast<float>(w_vals.data[o * cfg.part_i + i]) - zp);
        acc += static_cast<double>(x_val) * static_cast<double>(w_f);
      }
      float out_val = static_cast<float>(acc);
      for (float sc : cfg.dst_qdq_scales) {
        out_val = ApplyHostQdq(out_val, sc);
      }
      const int dst_y = o / dst_row_c;
      const int dst_c = part * dst_row_c + (o % dst_row_c);
      expected[dst_y * dst_shape.c + dst_c] = out_val;
    }

    const GpuModelBuilder::TensorHandle w_h = builder.AddConstantTensor(
        ::ml_drift::GetTensorDescriptorForWeightsLayout(w_vals, w_desc));
    const GpuModelBuilder::TensorHandle s_h = builder.AddConstantTensor(
        ::ml_drift::ScaleOrZeroPointToTensorDesc(gpu_info, scales, dtype));
    GpuModelBuilder::TensorHandle zp_h;
    const GpuModelBuilder::TensorHandle* zp_ptr = nullptr;
    if (cfg.has_zero_point) {
      zp_h = builder.AddConstantTensor(
          ::ml_drift::ScaleOrZeroPointToTensorDesc(gpu_info, zps, dtype));
      zp_ptr = &zp_h;
    }
    spec.part_weights.push_back(::ml_drift::CreateExternalWeights(
        w_h, w_desc, w_shape, scale_shape, &s_h, zp_ptr, /*sum_i=*/nullptr));
  }

  ABSL_RETURN_IF_ERROR(AddBlockFfwEdgeFc(&builder, src_h, dst_h, spec));
  GpuModel gpu_model;
  ABSL_RETURN_IF_ERROR(builder.GetGpuModel(std::vector<ValueId>{src_h.id},
                                           std::vector<ValueId>{dst_h.id},
                                           &gpu_model));
  EXPECT_EQ(gpu_model.nodes.size(), 1);

  ::ml_drift::TensorFloat32 src_cpu;
  src_cpu.shape = src_shape;
  src_cpu.data = std::move(src_data);
  ::ml_drift::TensorFloat32 dst_cpu;
  dst_cpu.shape = dst_shape;
  dst_cpu.data.resize(expected.size());
  ABSL_RETURN_IF_ERROR(env.ExecuteGpuModel(
      {src_cpu}, std::vector<::ml_drift::TensorFloat32*>{&dst_cpu},
      &gpu_model));

  float tol = precision == CalculationsPrecision::kF16 ? 1e-2f : 1e-4f;
  if (!cfg.dst_qdq_scales.empty() &&
      precision == CalculationsPrecision::kF16) {
    tol = std::max(tol, 2.0f * cfg.dst_qdq_scales.back());
  }
  EXPECT_THAT(dst_cpu.data, Pointwise(FloatNear(tol), expected));
  return absl::OkStatus();
}

TEST_P(BlockFfwKernelTest, DirectEdgeFcU1LinearAndBlockwiseInt2Int4Int8) {
  // U1 layout `[1, 1, 1, K1 * I1]` -> `[1, K2, 1, K1 * S]`, linear per-channel
  // INT2 with zero_point and inline src + chained dst QDQs.
  ASSERT_OK(RunDirectEdgeFcTest(
      *exec_env, precision(), storage(),
      DirectEdgeFcConfig{.weight_type = DataType::kInt2,
                         .num_parts = 2,
                         .src_height = 1,
                         .dst_height = 2,
                         .part_i = 16,
                         .part_o = 16,
                         .scale_groups = 1,
                         .has_zero_point = true,
                         .src_qdq_scales = {0.025f},
                         .dst_qdq_scales = {0.05f, 0.045f}},
      /*seed=*/11));

  // U1 layout, blockwise (`scale_groups = 2`) INT4 with zero_point, no QDQ.
  ASSERT_OK(RunDirectEdgeFcTest(
      *exec_env, precision(), storage(),
      DirectEdgeFcConfig{.weight_type = DataType::kInt4,
                         .num_parts = 2,
                         .src_height = 1,
                         .dst_height = 2,
                         .part_i = 16,
                         .part_o = 16,
                         .scale_groups = 2,
                         .has_zero_point = true},
      /*seed=*/22));

  // U1 layout, blockwise (`scale_groups = 2`) symmetric INT8 (no zero_point),
  // with inline src QDQ only.
  ASSERT_OK(RunDirectEdgeFcTest(
      *exec_env, precision(), storage(),
      DirectEdgeFcConfig{.weight_type = DataType::kInt8,
                         .num_parts = 2,
                         .src_height = 1,
                         .dst_height = 2,
                         .part_i = 16,
                         .part_o = 16,
                         .scale_groups = 2,
                         .has_zero_point = false,
                         .src_qdq_scales = {0.02f}},
      /*seed=*/33));
}

TEST_P(BlockFfwKernelTest, DirectEdgeFcD2AndSinglePartInt2Int4Int8) {
  // D2 layout `[1, K2, 1, K1 * S]` -> `[1, 1, 1, K1 * O4]`, linear per-channel
  // symmetric INT4 (no zero_point) with inline dst QDQ.
  ASSERT_OK(RunDirectEdgeFcTest(
      *exec_env, precision(), storage(),
      DirectEdgeFcConfig{.weight_type = DataType::kInt4,
                         .num_parts = 2,
                         .src_height = 2,
                         .dst_height = 1,
                         .part_i = 16,
                         .part_o = 16,
                         .scale_groups = 1,
                         .has_zero_point = false,
                         .dst_qdq_scales = {0.04f}},
      /*seed=*/44));

  // D2 layout, blockwise (`scale_groups = 2`) INT8 with zero_point + src & dst
  // QDQs.
  ASSERT_OK(RunDirectEdgeFcTest(
      *exec_env, precision(), storage(),
      DirectEdgeFcConfig{.weight_type = DataType::kInt8,
                         .num_parts = 2,
                         .src_height = 2,
                         .dst_height = 1,
                         .part_i = 16,
                         .part_o = 16,
                         .scale_groups = 2,
                         .has_zero_point = true,
                         .src_qdq_scales = {0.025f},
                         .dst_qdq_scales = {0.05f}},
      /*seed=*/55));

  // Single-part (`num_parts == 1`) U1 blockwise INT2 with zero_point and inline
  // QDQs (exercises the wrapper-free single-part codegen path).
  ASSERT_OK(RunDirectEdgeFcTest(
      *exec_env, precision(), storage(),
      DirectEdgeFcConfig{.weight_type = DataType::kInt2,
                         .num_parts = 1,
                         .src_height = 1,
                         .dst_height = 2,
                         .part_i = 16,
                         .part_o = 16,
                         .scale_groups = 2,
                         .has_zero_point = true,
                         .src_qdq_scales = {0.025f},
                         .dst_qdq_scales = {0.04f}},
      /*seed=*/66));

  // Single-part (`num_parts == 1`) D2 linear symmetric INT8, no QDQ.
  ASSERT_OK(RunDirectEdgeFcTest(
      *exec_env, precision(), storage(),
      DirectEdgeFcConfig{.weight_type = DataType::kInt8,
                         .num_parts = 1,
                         .src_height = 2,
                         .dst_height = 1,
                         .part_i = 16,
                         .part_o = 16,
                         .scale_groups = 1,
                         .has_zero_point = false},
      /*seed=*/77));
}

TEST(BlockFfwEdgeFcValidationTest,
     RejectsInvalidInputsScaleShapesAndBindingLimits) {
  GpuInfo gpu_info;
  gpu_info.gpu_api = ::ml_drift::GpuApi::kWebGpu;
  gpu_info.webgpu_info.max_storage_buffers_per_shader_stage = 8;
  gpu_info.webgpu_info.max_storage_textures_per_shader_stage = 4;

  GpuModelBuilder::TensorHandle src;
  src.id = 1;
  src.tensor_desc = TensorDescriptor(
      DataType::kFloat32, TensorStorageType::kBuffer, ::ml_drift::Layout::kHWC);
  src.tensor_desc.SetBHWCShape(BHWC(1, 1, 1, 16));

  GpuModelBuilder::TensorHandle dst;
  dst.id = 2;
  dst.tensor_desc = TensorDescriptor(
      DataType::kFloat32, TensorStorageType::kBuffer, ::ml_drift::Layout::kHWC);
  dst.tensor_desc.SetBHWCShape(BHWC(1, 1, 1, 16));

  const OHWI w_shape(16, 1, 1, 16);
  GpuModelBuilder::Weights part_w;
  part_w.shape = w_shape;
  part_w.desc = ::ml_drift::GetFullyConnectedInt2WeightsDesc(gpu_info, w_shape);
  part_w.scale_zp_shape = OHWI(16, 1, 1, 1);
  GpuModelBuilder::TensorHandle scale_h;
  scale_h.id = 3;
  scale_h.tensor_desc = TensorDescriptor(
      DataType::kFloat32, TensorStorageType::kBuffer, ::ml_drift::Layout::kHWC);
  part_w.scale = scale_h;
  GpuModelBuilder::TensorHandle zp_h;
  zp_h.id = 4;
  zp_h.tensor_desc = TensorDescriptor(
      DataType::kFloat32, TensorStorageType::kBuffer, ::ml_drift::Layout::kHWC);
  part_w.zero_point = zp_h;

  BlockFfwEdgeFcSpec spec;
  spec.num_parts = 1;
  spec.part_weights = {part_w};

  // Null builder.
  EXPECT_EQ(AddBlockFfwEdgeFc(/*builder=*/nullptr, src, dst, spec).code(),
            absl::StatusCode::kInvalidArgument);

  GpuModelBuilder builder(gpu_info, {.storage = TensorStorageType::kBuffer});

  // Non-positive height.
  {
    GpuModelBuilder::TensorHandle bad_src = src;
    bad_src.tensor_desc.SetBHWCShape(BHWC(1, 0, 1, 16));
    EXPECT_EQ(AddBlockFfwEdgeFc(&builder, bad_src, dst, spec).code(),
              absl::StatusCode::kInvalidArgument);
  }

  // Unsupported weight storage dtype (`kInt8` / `kUnknown`).
  {
    BlockFfwEdgeFcSpec bad_spec = spec;
    bad_spec.part_weights[0].desc.type = DataType::kInt8;
    EXPECT_EQ(AddBlockFfwEdgeFc(&builder, src, dst, bad_spec).code(),
              absl::StatusCode::kUnimplemented);
    bad_spec.part_weights[0].desc.type = DataType::kUnknown;
    EXPECT_EQ(AddBlockFfwEdgeFc(&builder, src, dst, bad_spec).code(),
              absl::StatusCode::kUnimplemented);
  }

  // Unsupported I4O4-like layout (`k2DX4I4YIsSpatialIAndXIsOOGroupO4` or
  // `kCustomGroups`) or mismatched `output_group_size`.
  {
    BlockFfwEdgeFcSpec bad_spec = spec;
    bad_spec.part_weights[0].desc.layout =
        ::ml_drift::WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4;
    EXPECT_EQ(AddBlockFfwEdgeFc(&builder, src, dst, bad_spec).code(),
              absl::StatusCode::kUnimplemented);

    bad_spec.part_weights[0].desc.layout =
        ::ml_drift::WeightsLayout::kCustomGroups;
    EXPECT_EQ(AddBlockFfwEdgeFc(&builder, src, dst, bad_spec).code(),
              absl::StatusCode::kUnimplemented);

    bad_spec = spec;
    bad_spec.part_weights[0].desc.output_group_size = 2;
    EXPECT_EQ(AddBlockFfwEdgeFc(&builder, src, dst, bad_spec).code(),
              absl::StatusCode::kUnimplemented);
  }

  // Mismatched scale or zero-point data type vs activation data type.
  {
    BlockFfwEdgeFcSpec bad_spec = spec;
    bad_spec.part_weights[0].scale->tensor_desc =
        TensorDescriptor(DataType::kFloat16, TensorStorageType::kBuffer,
                         ::ml_drift::Layout::kHWC);
    EXPECT_EQ(AddBlockFfwEdgeFc(&builder, src, dst, bad_spec).code(),
              absl::StatusCode::kInvalidArgument);

    bad_spec = spec;
    bad_spec.part_weights[0].zero_point->tensor_desc =
        TensorDescriptor(DataType::kFloat16, TensorStorageType::kBuffer,
                         ::ml_drift::Layout::kHWC);
    EXPECT_EQ(AddBlockFfwEdgeFc(&builder, src, dst, bad_spec).code(),
              absl::StatusCode::kInvalidArgument);
  }

  // Invalid QDQ attributes (negative scale, NaN scale, min > max).
  {
    BlockFfwEdgeFcSpec bad_spec = spec;
    bad_spec.src_qdqs = {QuantizeAndDequantizeAttributes{
        .min = -1.0f, .max = 1.0f, .scale = -0.01f}};
    EXPECT_EQ(AddBlockFfwEdgeFc(&builder, src, dst, bad_spec).code(),
              absl::StatusCode::kInvalidArgument);

    bad_spec = spec;
    bad_spec.dst_qdqs = {QuantizeAndDequantizeAttributes{
        .min = -1.0f,
        .max = 1.0f,
        .scale = std::numeric_limits<float>::quiet_NaN()}};
    EXPECT_EQ(AddBlockFfwEdgeFc(&builder, src, dst, bad_spec).code(),
              absl::StatusCode::kInvalidArgument);

    bad_spec = spec;
    bad_spec.dst_qdqs = {QuantizeAndDequantizeAttributes{
        .min = 2.0f, .max = 1.0f, .scale = 0.01f}};
    EXPECT_EQ(AddBlockFfwEdgeFc(&builder, src, dst, bad_spec).code(),
              absl::StatusCode::kInvalidArgument);
  }

  // Batched scale_zp_shape.h != 1.
  {
    BlockFfwEdgeFcSpec bad_spec = spec;
    bad_spec.part_weights[0].scale_zp_shape = OHWI(16, 2, 1, 1);
    EXPECT_EQ(AddBlockFfwEdgeFc(&builder, src, dst, bad_spec).code(),
              absl::StatusCode::kUnimplemented);
  }

  // Per-tensor / non-per-output-channel scale_zp_shape.o == 1.
  {
    BlockFfwEdgeFcSpec bad_spec = spec;
    bad_spec.part_weights[0].scale_zp_shape = OHWI(1, 1, 1, 1);
    EXPECT_EQ(AddBlockFfwEdgeFc(&builder, src, dst, bad_spec).code(),
              absl::StatusCode::kUnimplemented);
  }

  // Exceeds WebGPU storage buffer limit (2 + 4 * 3 = 14 > 8).
  {
    GpuModelBuilder::TensorHandle wide_src = src;
    wide_src.tensor_desc.SetBHWCShape(BHWC(1, 1, 1, 64));
    GpuModelBuilder::TensorHandle wide_dst = dst;
    wide_dst.tensor_desc.SetBHWCShape(BHWC(1, 1, 1, 64));
    BlockFfwEdgeFcSpec wide_spec;
    wide_spec.num_parts = 4;
    wide_spec.part_weights = {part_w, part_w, part_w, part_w};
    EXPECT_EQ(AddBlockFfwEdgeFc(&builder, wide_src, wide_dst, wide_spec).code(),
              absl::StatusCode::kUnimplemented);
  }

  // Exceeds Metal shared read+write texture slot limit (31 src + 1 dst = 32 >
  // 31, even though src_image_args == 31 and dst_image_args == 1 individually).
  {
    GpuInfo metal_info;
    metal_info.gpu_api = ::ml_drift::GpuApi::kMetal;
    GpuModelBuilder metal_builder(metal_info,
                                  {.storage = TensorStorageType::kTexture2D});
    GpuModelBuilder::TensorHandle tex_src;
    tex_src.id = 1;
    tex_src.tensor_desc = TensorDescriptor(DataType::kFloat32,
                                           TensorStorageType::kTexture2D,
                                           ::ml_drift::Layout::kHWC);
    tex_src.tensor_desc.SetBHWCShape(BHWC(1, 1, 1, 160));
    GpuModelBuilder::TensorHandle tex_dst = tex_src;
    tex_dst.id = 2;

    GpuModelBuilder::Weights tex_part = part_w;
    tex_part.desc.layout =
        ::ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4;
    tex_part.scale->tensor_desc = TensorDescriptor(
        DataType::kFloat32, TensorStorageType::kTexture2D,
        ::ml_drift::Layout::kHWC);
    tex_part.zero_point->tensor_desc = TensorDescriptor(
        DataType::kFloat32, TensorStorageType::kTexture2D,
        ::ml_drift::Layout::kHWC);

    BlockFfwEdgeFcSpec metal_spec;
    metal_spec.num_parts = 10;
    metal_spec.part_weights.assign(10, tex_part);
    EXPECT_EQ(
        AddBlockFfwEdgeFc(&metal_builder, tex_src, tex_dst, metal_spec).code(),
        absl::StatusCode::kUnimplemented);
  }

  // Exceeds Mali 8-part cap even when OpenCL buffer count fits within 128.
  {
    GpuInfo mali_info;
    mali_info.gpu_api = ::ml_drift::GpuApi::kOpenCl;
    mali_info.vendor = ::ml_drift::GpuVendor::kMali;
    GpuModelBuilder mali_builder(mali_info,
                                 {.storage = TensorStorageType::kBuffer});
    GpuModelBuilder::TensorHandle mali_src = src;
    mali_src.tensor_desc.SetBHWCShape(BHWC(1, 1, 1, 9 * 16));
    GpuModelBuilder::TensorHandle mali_dst = dst;
    mali_dst.tensor_desc.SetBHWCShape(BHWC(1, 1, 1, 9 * 16));
    BlockFfwEdgeFcSpec mali_spec;
    mali_spec.num_parts = 9;
    mali_spec.part_weights.assign(9, part_w);
    EXPECT_EQ(
        AddBlockFfwEdgeFc(&mali_builder, mali_src, mali_dst, mali_spec).code(),
        absl::StatusCode::kUnimplemented);
  }
}

struct TryEdgeGraphOptions {
  int k1 = 2;
  int tokens = 1;
  DataType weight_type = DataType::kInt2;
  bool has_zero_point = true;
  bool mismatch_src_qdq = false;
  bool mismatch_dst_qdq = false;
  bool extra_consumer_after_unpack = false;
  bool extra_consumer_after_fc = false;
  bool pack_out_is_graph_output = false;
  bool trailing_qdq = true;
  bool trailing_qdq_extra_input = false;
  bool bias_with_data = false;
  bool bias_shape_without_data = false;
  bool produced_weight = false;
  bool swap_unpack_parts = false;
  int max_storage_buffers = 32;
};

struct TryEdgeMatchResult {
  absl::Status status;
  std::set<::ml_drift::NodeId> consumed;
  ::ml_drift::NodeId trailing_qdq_id = 0;
  size_t builder_nodes = 0;
};

TryEdgeMatchResult RunSyntheticTryBlockFfwEdgeFc(
    const TryEdgeGraphOptions& opts) {
  GpuInfo gpu_info;
  gpu_info.gpu_api = ::ml_drift::GpuApi::kWebGpu;
  gpu_info.webgpu_info.max_storage_buffers_per_shader_stage =
      opts.max_storage_buffers;
  gpu_info.webgpu_info.max_storage_textures_per_shader_stage = 16;

  GraphFloat32 graph;
  absl::flat_hash_map<ValueId, TensorDescriptor> runtime_tensors;
  absl::flat_hash_map<ValueId, TensorDescriptor> const_tensors;

  auto add_rt = [&](const BHWC& shape) {
    Value* v = AddGraphValue(&graph, DataType::kFloat32, shape);
    TensorDescriptor desc(DataType::kFloat32, TensorStorageType::kBuffer,
                          ::ml_drift::Layout::kHWC);
    desc.SetBHWCShape(shape);
    runtime_tensors[v->id] = std::move(desc);
    return v;
  };
  auto add_const = [&](const BHWC& shape, DataType dt) {
    Value* v = AddGraphValue(&graph, dt, shape);
    TensorDescriptor desc(dt, TensorStorageType::kBuffer,
                          ::ml_drift::Layout::kLinear);
    desc.SetBHWCShape(shape);
    const_tensors[v->id] = std::move(desc);
    return v;
  };

  Value* x_val = add_rt(BHWC(1, 1, opts.tokens, opts.k1 * 8));
  std::vector<Value*> parts(opts.k1);
  ::ml_drift::NodeId first_node_id = 0;

  for (int k = 0; k < opts.k1; ++k) {
    Value* unp_out = add_rt(BHWC(1, 1, opts.tokens, 8));
    Node* unp = graph.NewNode();
    if (k == 0) first_node_id = unp->id;
    unp->operation.type = kBlockFfwUnpackPartType;
    const int part_idx =
        (opts.swap_unpack_parts && opts.k1 == 2) ? (1 - k) : k;
    unp->operation.attributes =
        BlockFfwUnpackPartAttributes{.part = part_idx, .num_parts = opts.k1};
    graph.AddConsumer(unp->id, x_val->id);
    graph.SetProducer(unp->id, unp_out->id);

    if (opts.extra_consumer_after_unpack && k == 0) {
      Value* sink = add_rt(BHWC(1, 1, opts.tokens, 8));
      Node* extra = graph.NewNode();
      extra->operation.type = kBlockFfwSiluType;
      graph.AddConsumer(extra->id, unp_out->id);
      graph.SetProducer(extra->id, sink->id);
    }

    const float src_sc = (opts.mismatch_src_qdq && k == 1) ? 0.03f : 0.025f;
    Value* q_in = add_rt(BHWC(1, 1, opts.tokens, 8));
    Node* q_in_n = graph.NewNode();
    q_in_n->operation.type = ToString(OperationType::kQuantizeAndDequantize);
    q_in_n->operation.attributes = QuantizeAndDequantizeAttributes{
        .min = -128.0f * src_sc, .max = 127.0f * src_sc, .scale = src_sc};
    graph.AddConsumer(q_in_n->id, unp_out->id);
    graph.SetProducer(q_in_n->id, q_in->id);

    Value* w_val = add_const(BHWC(1, 1, 1, 16), DataType::kUint32);
    if (opts.produced_weight && k == 0) {
      Node* prod = graph.NewNode();
      prod->operation.type = ToString(OperationType::kConstant);
      graph.SetProducer(prod->id, w_val->id);
    }
    Value* s_val = add_const(BHWC(1, 1, 1, 8), DataType::kFloat32);
    Value* zp_val = opts.has_zero_point
                        ? add_const(BHWC(1, 1, 1, 8), DataType::kFloat32)
                        : nullptr;
    Value* fc_out = add_rt(BHWC(1, 1, opts.tokens, 8));
    Node* fc = graph.NewNode();
    const OHWI w_shape(8, 1, 1, 8);
    const OHWI param_shape(8, 1, 1, 1);
    if (opts.weight_type == DataType::kInt2) {
      fc->operation.type = ToString(OperationType::kFullyConnectedInt2);
      ::ml_drift::FullyConnectedInt2Attributes attr;
      ::ml_drift::Tensor<OHWI, DataType::kInt2> w;
      w.shape = w_shape;
      attr.weights = w;
      attr.scale.shape = param_shape;
      if (opts.has_zero_point) attr.zero_point.shape = param_shape;
      if (opts.bias_with_data) {
        attr.bias.shape = ::ml_drift::Linear(8);
        attr.bias.data.assign(8, 1.0f);
      } else if (opts.bias_shape_without_data) {
        attr.bias.shape = ::ml_drift::Linear(8);
      }
      fc->operation.attributes = std::move(attr);
    } else if (opts.weight_type == DataType::kInt4) {
      fc->operation.type = ToString(OperationType::kFullyConnectedInt4);
      ::ml_drift::FullyConnectedInt4Attributes attr;
      ::ml_drift::Tensor<OHWI, DataType::kInt4> w;
      w.shape = w_shape;
      attr.weights = w;
      attr.scale.shape = param_shape;
      if (opts.has_zero_point) attr.zero_point.shape = param_shape;
      if (opts.bias_with_data) {
        attr.bias.shape = ::ml_drift::Linear(8);
        attr.bias.data.assign(8, 1.0f);
      } else if (opts.bias_shape_without_data) {
        attr.bias.shape = ::ml_drift::Linear(8);
      }
      fc->operation.attributes = std::move(attr);
    } else {
      fc->operation.type = ToString(OperationType::kFullyConnectedInt8);
      ::ml_drift::FullyConnectedInt8Attributes attr;
      attr.weights.shape = w_shape;
      attr.scale.shape = param_shape;
      if (opts.has_zero_point) attr.zero_point.shape = param_shape;
      if (opts.bias_with_data) {
        attr.bias.shape = ::ml_drift::Linear(8);
        attr.bias.data.assign(8, 1.0f);
      } else if (opts.bias_shape_without_data) {
        attr.bias.shape = ::ml_drift::Linear(8);
      }
      fc->operation.attributes = std::move(attr);
    }
    graph.AddConsumer(fc->id, q_in->id);
    graph.AddConsumer(fc->id, w_val->id);
    graph.AddConsumer(fc->id, s_val->id);
    if (zp_val != nullptr) graph.AddConsumer(fc->id, zp_val->id);
    graph.SetProducer(fc->id, fc_out->id);

    if (opts.extra_consumer_after_fc && k == 0) {
      Value* sink = add_rt(BHWC(1, 1, opts.tokens, 8));
      Node* extra = graph.NewNode();
      extra->operation.type = kBlockFfwSiluType;
      graph.AddConsumer(extra->id, fc_out->id);
      graph.SetProducer(extra->id, sink->id);
    }

    const float dst_sc = (opts.mismatch_dst_qdq && k == 1) ? 0.06f : 0.05f;
    parts[k] = add_rt(BHWC(1, 1, opts.tokens, 8));
    Node* q_out_n = graph.NewNode();
    q_out_n->operation.type = ToString(OperationType::kQuantizeAndDequantize);
    q_out_n->operation.attributes = QuantizeAndDequantizeAttributes{
        .min = -128.0f * dst_sc, .max = 127.0f * dst_sc, .scale = dst_sc};
    graph.AddConsumer(q_out_n->id, fc_out->id);
    graph.SetProducer(q_out_n->id, parts[k]->id);
  }

  Value* packed = add_rt(BHWC(1, 2, opts.tokens, opts.k1 * 4));
  Node* pack = graph.NewNode();
  pack->operation.type = kBlockFfwPackPartsType;
  pack->operation.attributes =
      BlockFfwPackPartsAttributes{.num_parts = opts.k1};
  for (Value* p : parts) graph.AddConsumer(pack->id, p->id);
  graph.SetProducer(pack->id, packed->id);

  if (opts.pack_out_is_graph_output) {
    graph.AddKnownGraphOutput(packed);
  }

  Value* final_out = packed;
  ::ml_drift::NodeId trail_id = 0;
  if (opts.trailing_qdq) {
    final_out = add_rt(BHWC(1, 2, opts.tokens, opts.k1 * 4));
    Node* trail = graph.NewNode();
    trail_id = trail->id;
    trail->operation.type = ToString(OperationType::kQuantizeAndDequantize);
    trail->operation.attributes = QuantizeAndDequantizeAttributes{
        .min = -7.68f, .max = 7.62f, .scale = 0.06f};
    graph.AddConsumer(trail->id, packed->id);
    if (opts.trailing_qdq_extra_input) {
      Value* extra_in = add_rt(BHWC(1, 2, opts.tokens, opts.k1 * 4));
      graph.AddConsumer(trail->id, extra_in->id);
    }
    graph.SetProducer(trail->id, final_out->id);
  }

  GpuModelBuilder builder(gpu_info, {.storage = TensorStorageType::kBuffer},
                          std::move(runtime_tensors), std::move(const_tensors));
  CreateGpuModelInfo create_info{.precision = CalculationsPrecision::kF32,
                                 .storage_type = TensorStorageType::kBuffer};
  std::set<::ml_drift::NodeId> already_consumed;
  TryEdgeMatchResult result;
  result.trailing_qdq_id = trail_id;
  result.status = TryBlockFfwEdgeFc(create_info, gpu_info, graph, first_node_id,
                                    already_consumed, &result.consumed,
                                    &builder);
  GpuModel gpu_model;
  if (result.status.ok()) {
    const ValueId out_id =
        result.consumed.count(trail_id) != 0 ? final_out->id : packed->id;
    EXPECT_OK(builder.GetGpuModel({x_val->id}, {out_id}, &gpu_model));
    result.builder_nodes = gpu_model.nodes.size();
  }
  return result;
}

TEST(TryBlockFfwEdgeFcTest, RejectsNullArgsAndNonUnpackFirstNode) {
  GpuInfo gpu_info;
  gpu_info.gpu_api = ::ml_drift::GpuApi::kWebGpu;
  gpu_info.webgpu_info.max_storage_buffers_per_shader_stage = 32;
  CreateGpuModelInfo create_info{.precision = CalculationsPrecision::kF32,
                                 .storage_type = TensorStorageType::kBuffer};
  GraphFloat32 graph;
  Node* silu = graph.NewNode();
  silu->operation.type = kBlockFfwSiluType;
  GpuModelBuilder builder(gpu_info, {.storage = TensorStorageType::kBuffer});
  std::set<::ml_drift::NodeId> consumed;
  std::set<::ml_drift::NodeId> new_consumed;

  EXPECT_EQ(TryBlockFfwEdgeFc(create_info, gpu_info, graph, silu->id, consumed,
                              /*new_consumed_nodes=*/nullptr, &builder)
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(TryBlockFfwEdgeFc(create_info, gpu_info, graph, silu->id, consumed,
                              &new_consumed, /*model_builder=*/nullptr)
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(TryBlockFfwEdgeFc(create_info, gpu_info, graph, silu->id, consumed,
                              &new_consumed, &builder)
                .code(),
            absl::StatusCode::kUnimplemented);
  EXPECT_TRUE(new_consumed.empty());
}

TEST(TryBlockFfwEdgeFcTest, MatchesSinglePartSymmetricInt4AndInt8) {
  // Single-part (`k1 = 1`) INT2 with trailing QDQ: 4 branch nodes + Pack + QDQ
  // = 6 consumed nodes -> 1 fused dispatch.
  {
    TryEdgeGraphOptions opts;
    opts.k1 = 1;
    TryEdgeMatchResult res = RunSyntheticTryBlockFfwEdgeFc(opts);
    ASSERT_OK(res.status);
    EXPECT_EQ(res.consumed.size(), 6);
    EXPECT_EQ(res.builder_nodes, 1);
  }
  // Symmetric INT4 (no zero_point, 3 FC inputs).
  {
    TryEdgeGraphOptions opts;
    opts.weight_type = DataType::kInt4;
    opts.has_zero_point = false;
    TryEdgeMatchResult res = RunSyntheticTryBlockFfwEdgeFc(opts);
    ASSERT_OK(res.status);
    EXPECT_EQ(res.consumed.size(), 10);
    EXPECT_EQ(res.builder_nodes, 1);
  }
  // Symmetric INT8 (no zero_point, 3 FC inputs).
  {
    TryEdgeGraphOptions opts;
    opts.weight_type = DataType::kInt8;
    opts.has_zero_point = false;
    TryEdgeMatchResult res = RunSyntheticTryBlockFfwEdgeFc(opts);
    ASSERT_OK(res.status);
    EXPECT_EQ(res.consumed.size(), 10);
    EXPECT_EQ(res.builder_nodes, 1);
  }
}

TEST(TryBlockFfwEdgeFcTest, HandlesRejectionAndGraphOutputEdgeCases) {
  // Mismatched branch src QDQ scales.
  {
    TryEdgeGraphOptions opts;
    opts.mismatch_src_qdq = true;
    TryEdgeMatchResult res = RunSyntheticTryBlockFfwEdgeFc(opts);
    EXPECT_EQ(res.status.code(), absl::StatusCode::kUnimplemented);
    EXPECT_TRUE(res.consumed.empty());
  }
  // Mismatched branch dst QDQ scales.
  {
    TryEdgeGraphOptions opts;
    opts.mismatch_dst_qdq = true;
    TryEdgeMatchResult res = RunSyntheticTryBlockFfwEdgeFc(opts);
    EXPECT_EQ(res.status.code(), absl::StatusCode::kUnimplemented);
    EXPECT_TRUE(res.consumed.empty());
  }
  // Extra consumer on intermediate values (after UnpackPart and after FC).
  {
    TryEdgeGraphOptions opts;
    opts.extra_consumer_after_unpack = true;
    TryEdgeMatchResult res = RunSyntheticTryBlockFfwEdgeFc(opts);
    EXPECT_EQ(res.status.code(), absl::StatusCode::kUnimplemented);
    EXPECT_TRUE(res.consumed.empty());
  }
  {
    TryEdgeGraphOptions opts;
    opts.extra_consumer_after_fc = true;
    TryEdgeMatchResult res = RunSyntheticTryBlockFfwEdgeFc(opts);
    EXPECT_EQ(res.status.code(), absl::StatusCode::kUnimplemented);
    EXPECT_TRUE(res.consumed.empty());
  }
  // PackParts output is a known graph output before trailing QDQ: core stage
  // still fuses, while trailing QDQ is NOT absorbed.
  {
    TryEdgeGraphOptions opts;
    opts.pack_out_is_graph_output = true;
    TryEdgeMatchResult res = RunSyntheticTryBlockFfwEdgeFc(opts);
    ASSERT_OK(res.status);
    EXPECT_EQ(res.consumed.size(), 9);
    EXPECT_EQ(res.consumed.count(res.trailing_qdq_id), 0);
    EXPECT_EQ(res.builder_nodes, 1);
  }
  // Trailing QDQ with extra second input: core stage fuses, trailing QDQ is
  // NOT absorbed (`SingleInput(graph, next_qdq) != dst_val`).
  {
    TryEdgeGraphOptions opts;
    opts.trailing_qdq_extra_input = true;
    TryEdgeMatchResult res = RunSyntheticTryBlockFfwEdgeFc(opts);
    ASSERT_OK(res.status);
    EXPECT_EQ(res.consumed.size(), 9);
    EXPECT_EQ(res.consumed.count(res.trailing_qdq_id), 0);
  }
  // Non-empty bias (both populated data and shape-only without host data).
  {
    TryEdgeGraphOptions opts;
    opts.bias_with_data = true;
    EXPECT_EQ(RunSyntheticTryBlockFfwEdgeFc(opts).status.code(),
              absl::StatusCode::kUnimplemented);
  }
  {
    TryEdgeGraphOptions opts;
    opts.bias_shape_without_data = true;
    EXPECT_EQ(RunSyntheticTryBlockFfwEdgeFc(opts).status.code(),
              absl::StatusCode::kUnimplemented);
  }
  // Produced (non-constant) weight tensor.
  {
    TryEdgeGraphOptions opts;
    opts.produced_weight = true;
    EXPECT_EQ(RunSyntheticTryBlockFfwEdgeFc(opts).status.code(),
              absl::StatusCode::kUnimplemented);
  }
  // Misordered `part != k`.
  {
    TryEdgeGraphOptions opts;
    opts.swap_unpack_parts = true;
    EXPECT_EQ(RunSyntheticTryBlockFfwEdgeFc(opts).status.code(),
              absl::StatusCode::kUnimplemented);
  }
  // `k1 = 8` decode fallback when WebGPU buffer limit (8) is exceeded.
  {
    TryEdgeGraphOptions opts;
    opts.k1 = 8;
    opts.max_storage_buffers = 8;
    TryEdgeMatchResult res = RunSyntheticTryBlockFfwEdgeFc(opts);
    EXPECT_EQ(res.status.code(), absl::StatusCode::kUnimplemented);
    EXPECT_TRUE(res.consumed.empty());
  }
}

INSTANTIATE_TEST_SUITE_P(
    BlockFfwKernelTestSuite, BlockFfwKernelTest,
    Combine(
        ValuesIn({CalculationsPrecision::kF32, CalculationsPrecision::kF16}),
        ValuesIn({TensorStorageType::kBuffer, TensorStorageType::kTexture2D})),
    [](const TestParamInfo<BlockFfwKernelTest::ParamType>& info) {
      return ::ml_drift::ToString(info.param);
    });

}  // namespace
}  // namespace litert::ml_drift
