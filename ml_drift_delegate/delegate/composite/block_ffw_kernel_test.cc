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

absl::Status RunBlockedFfwEndToEndTest(
    ::ml_drift::TestExecutionEnvironment& env, CalculationsPrecision precision,
    TensorStorageType storage, int tokens, bool distinct_adjacent_scales,
    int k1 = 2) {
  const DataType dtype = precision == CalculationsPrecision::kF16
                             ? DataType::kFloat16
                             : DataType::kFloat32;

  constexpr int k2 = 4;
  constexpr int s = 8;
  constexpr int i1 = 16;
  constexpr int o1 = k2 * s;  // 32
  const int i2 = k1 * s;
  constexpr int o2 = 24;
  constexpr int i4 = k2 * s;  // 32
  constexpr int o4 = 16;
  const int d_model = k1 * i1;

  std::vector<HostInt2Weights> u1_ws;
  std::vector<HostInt2Weights> d2_ws;
  for (int k = 0; k < k1; ++k) {
    u1_ws.push_back(
        MakeRandomInt2Weights(OHWI(o1, 1, 1, i1), 101 + k, 0.08f));
    d2_ws.push_back(
        MakeRandomInt2Weights(OHWI(o4, 1, 1, i4), 404 + k, 0.06f));
  }
  const HostInt2Weights u2_w =
      MakeRandomInt2Weights(OHWI(o2, k2, 1, i2), 202, 0.07f);
  const HostInt2Weights d1_w =
      MakeRandomInt2Weights(OHWI(i2, k2, 1, o2), 303, 0.05f);

  const float u1_in_sc = 0.025f;
  const float u1_out_sc = 0.060f;
  const float u2_in_sc = distinct_adjacent_scales ? 0.055f : u1_out_sc;
  const float u2_out_sc = 0.090f;
  const float d1_in_sc = 0.045f;
  const float d1_out_sc = 0.030f;
  const float d2_in_sc = distinct_adjacent_scales ? 0.035f : d1_out_sc;
  const float d2_out_sc = 0.025f;

  std::vector<float> x_data(tokens * d_model);
  for (size_t i = 0; i < x_data.size(); ++i) {
    x_data[i] = 0.8f * std::sin(static_cast<float>(i + 1) * 0.23f) -
                0.4f * std::cos(static_cast<float>(i + 2) * 0.17f);
  }
  const std::vector<float> expected = ComputeBlockedFfwReference(
      x_data, tokens, k1, k2, s, u1_ws, u2_w, d1_w, d2_ws, u1_in_sc, u1_out_sc,
      u2_in_sc, u2_out_sc, d1_in_sc, d1_out_sc, d2_in_sc, d2_out_sc);

  GraphFloat32 graph;
  CreateGpuModelInfo create_info;
  create_info.precision = precision;
  create_info.storage_type = storage;
  absl::flat_hash_map<ValueId, TensorDescriptor> const_tensors;

  const GpuInfo& gpu_info = env.GetGpuInfo();
  Value* x_val = AddGraphValue(&graph, dtype, BHWC(1, 1, tokens, d_model));
  Value* out_val = AddGraphValue(&graph, dtype, BHWC(1, 1, tokens, d_model));

  // Build the exact GraphFloat32 topology emitted by BlockFfwOperationParser.
  Value* u2_in = nullptr;
  {
    std::vector<Value*> parts(k1);
    for (int k = 0; k < k1; ++k) {
      Value* unp = AddGraphValue(&graph, dtype, BHWC(1, 1, tokens, i1));
      Node* unp_n = graph.NewNode();
      unp_n->operation.type = kBlockFfwUnpackPartType;
      unp_n->operation.attributes =
          BlockFfwUnpackPartAttributes{.part = k, .num_parts = k1};
      graph.AddConsumer(unp_n->id, x_val->id);
      graph.SetProducer(unp_n->id, unp->id);

      Value* q_in = AddGraphQdq(&graph, unp, u1_in_sc);
      Value* fc = AddPredefinedInt2Fc(&graph, &const_tensors, gpu_info, dtype,
                                      q_in, u1_ws[k], BHWC(1, 1, tokens, o1));
      parts[k] = AddGraphQdq(&graph, fc, u1_out_sc);
    }
    Value* pack = AddGraphValue(&graph, dtype, BHWC(1, k2, tokens, i2));
    Node* pack_n = graph.NewNode();
    pack_n->operation.type = kBlockFfwPackPartsType;
    pack_n->operation.attributes = BlockFfwPackPartsAttributes{.num_parts = k1};
    for (Value* p : parts) graph.AddConsumer(pack_n->id, p->id);
    graph.SetProducer(pack_n->id, pack->id);
    u2_in = pack;
  }
  if (distinct_adjacent_scales) {
    u2_in = AddGraphQdq(&graph, u2_in, u2_in_sc);
  }

  Value* u2_raw = AddPredefinedInt2Fc(&graph, &const_tensors, gpu_info, dtype,
                                      u2_in, u2_w, BHWC(1, k2, tokens, o2));
  Value* u2_q = AddGraphQdq(&graph, u2_raw, u2_out_sc);
  Value* act = AddGraphValue(&graph, dtype, BHWC(1, k2, tokens, o2));
  Node* silu_n = graph.NewNode();
  silu_n->operation.type = kBlockFfwSiluType;
  graph.AddConsumer(silu_n->id, u2_q->id);
  graph.SetProducer(silu_n->id, act->id);

  Value* d1_in = AddGraphQdq(&graph, act, d1_in_sc);
  Value* d1_raw = AddPredefinedInt2Fc(&graph, &const_tensors, gpu_info, dtype,
                                      d1_in, d1_w, BHWC(1, k2, tokens, i2));
  Value* d1_q = AddGraphQdq(&graph, d1_raw, d1_out_sc);

  std::vector<Value*> parts(k1);
  for (int k = 0; k < k1; ++k) {
    Value* unp = AddGraphValue(&graph, dtype, BHWC(1, 1, tokens, i4));
    Node* unp_n = graph.NewNode();
    unp_n->operation.type = kBlockFfwUnpackPartType;
    unp_n->operation.attributes =
        BlockFfwUnpackPartAttributes{.part = k, .num_parts = k1};
    graph.AddConsumer(unp_n->id, d1_q->id);
    graph.SetProducer(unp_n->id, unp->id);

    Value* q_in =
        distinct_adjacent_scales ? AddGraphQdq(&graph, unp, d2_in_sc) : unp;
    Value* fc = AddPredefinedInt2Fc(&graph, &const_tensors, gpu_info, dtype,
                                    q_in, d2_ws[k], BHWC(1, 1, tokens, o4));
    parts[k] = AddGraphQdq(&graph, fc, d2_out_sc);
  }
  Node* pack_n = graph.NewNode();
  pack_n->operation.type = kBlockFfwPackPartsType;
  pack_n->operation.attributes = BlockFfwPackPartsAttributes{.num_parts = k1};
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
    if (consumed_nodes.contains(node->id)) continue;
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

  if (tokens == 1) {
    // Single-token decode must fuse U1 and D2 edge subgraphs + all QDQs + SiLU
    // down to exactly 4 GPU kernel dispatches.
    EXPECT_EQ(gpu_model.nodes.size(), 4);
  }

  ::ml_drift::TensorFloat32 src_cpu;
  src_cpu.shape = BHWC(1, 1, tokens, d_model);
  src_cpu.data = x_data;
  ::ml_drift::TensorFloat32 dst_cpu;
  dst_cpu.shape = BHWC(1, 1, tokens, d_model);
  dst_cpu.data.resize(expected.size());

  ABSL_RETURN_IF_ERROR(env.ExecuteGpuModel(
      {src_cpu}, std::vector<::ml_drift::TensorFloat32*>{&dst_cpu},
      &gpu_model));

  const float tol =
      precision == CalculationsPrecision::kF16 ? 2.0f * d2_out_sc : 1e-4f;
  EXPECT_THAT(dst_cpu.data, Pointwise(FloatNear(tol), expected));
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
    TensorStorageType storage, int tokens, bool distinct_adjacent_scales) {
  const DataType dtype = precision == CalculationsPrecision::kF16
                             ? DataType::kFloat16
                             : DataType::kFloat32;

  constexpr int k1 = 2;
  constexpr int k2 = 4;
  constexpr int s = 8;
  constexpr int i1 = 16;
  constexpr int o1 = k2 * s;
  constexpr int i2 = k1 * s;
  constexpr int o2 = 24;
  constexpr int i4 = k2 * s;
  constexpr int o4 = 16;
  constexpr int d_model = k1 * i1;

  std::vector<HostInt2Weights> u1_ws;
  std::vector<HostInt2Weights> d2_ws;
  for (int k = 0; k < k1; ++k) {
    u1_ws.push_back(MakeRandomInt2Weights(OHWI(o1, 1, 1, i1), 101 + k, 0.08f));
    d2_ws.push_back(MakeRandomInt2Weights(OHWI(o4, 1, 1, i4), 404 + k, 0.06f));
  }
  const HostInt2Weights u2_w =
      MakeRandomInt2Weights(OHWI(o2, k2, 1, i2), 202, 0.07f);
  const HostInt2Weights d1_w =
      MakeRandomInt2Weights(OHWI(i2, k2, 1, o2), 303, 0.05f);

  const float u1_in_sc = 0.025f;
  const float u1_out_sc = 0.060f;
  const float u2_in_sc = distinct_adjacent_scales ? 0.055f : u1_out_sc;
  const float u2_out_sc = 0.090f;
  const float d1_in_sc = 0.045f;
  const float d1_out_sc = 0.030f;
  const float d2_in_sc = distinct_adjacent_scales ? 0.035f : d1_out_sc;
  const float d2_out_sc = 0.025f;

  std::vector<float> x_data(tokens * d_model);
  for (size_t i = 0; i < x_data.size(); ++i) {
    x_data[i] = 0.8f * std::sin(static_cast<float>(i + 1) * 0.23f) -
                0.4f * std::cos(static_cast<float>(i + 2) * 0.17f);
  }
  const std::vector<float> expected = ComputeBlockedFfwReference(
      x_data, tokens, k1, k2, s, u1_ws, u2_w, d1_w, d2_ws, u1_in_sc, u1_out_sc,
      u2_in_sc, u2_out_sc, d1_in_sc, d1_out_sc, d2_in_sc, d2_out_sc);

  ::ml_drift::ir::IrModel ir_model;
  CreateGpuModelInfo create_info;
  create_info.precision = precision;
  create_info.storage_type = storage;
  absl::flat_hash_map<ValueId, TensorDescriptor> const_tensors;

  const GpuInfo& gpu_info = env.GetGpuInfo();
  const auto x_id = ir_model.add_tensor(dtype, BHWC(1, 1, tokens, d_model))->id;
  const auto out_id =
      ir_model.add_tensor(dtype, BHWC(1, 1, tokens, d_model))->id;
  ir_model.add_input(x_id);
  ir_model.add_output(out_id);

  ::ml_drift::ir::IrTensorId u2_in_id = 0;
  {
    std::vector<::ml_drift::ir::IrTensorId> parts(k1);
    for (int k = 0; k < k1; ++k) {
      const auto unp_id =
          ir_model.add_tensor(dtype, BHWC(1, 1, tokens, i1))->id;
      ::ml_drift::ir::IrOp* unp = ir_model.add_op();
      unp->name = kBlockFfwUnpackPartType;
      unp->attr = BlockFfwUnpackPartAttributes{.part = k, .num_parts = k1};
      ir_model.AddConsumer(x_id, unp->id);
      ir_model.SetProducer(unp_id, unp->id);

      const auto q_in_id = AddIrQdq(&ir_model, unp_id, u1_in_sc);
      const auto fc_id =
          AddPredefinedInt2FcIr(&ir_model, &const_tensors, gpu_info, dtype,
                                q_in_id, u1_ws[k], BHWC(1, 1, tokens, o1));
      parts[k] = AddIrQdq(&ir_model, fc_id, u1_out_sc);
    }
    const auto pack_id =
        ir_model.add_tensor(dtype, BHWC(1, k2, tokens, i2))->id;
    ::ml_drift::ir::IrOp* pack = ir_model.add_op();
    pack->name = kBlockFfwPackPartsType;
    pack->attr = BlockFfwPackPartsAttributes{.num_parts = k1};
    for (auto p : parts) ir_model.AddConsumer(p, pack->id);
    ir_model.SetProducer(pack_id, pack->id);
    u2_in_id = pack_id;
  }
  if (distinct_adjacent_scales) {
    u2_in_id = AddIrQdq(&ir_model, u2_in_id, u2_in_sc);
  }

  const auto u2_raw_id =
      AddPredefinedInt2FcIr(&ir_model, &const_tensors, gpu_info, dtype,
                            u2_in_id, u2_w, BHWC(1, k2, tokens, o2));
  const auto u2_q_id = AddIrQdq(&ir_model, u2_raw_id, u2_out_sc);
  const auto act_id = ir_model.add_tensor(dtype, BHWC(1, k2, tokens, o2))->id;
  ::ml_drift::ir::IrOp* silu = ir_model.add_op();
  silu->name = kBlockFfwSiluType;
  ir_model.AddConsumer(u2_q_id, silu->id);
  ir_model.SetProducer(act_id, silu->id);

  const auto d1_in_id = AddIrQdq(&ir_model, act_id, d1_in_sc);
  const auto d1_raw_id =
      AddPredefinedInt2FcIr(&ir_model, &const_tensors, gpu_info, dtype,
                            d1_in_id, d1_w, BHWC(1, k2, tokens, i2));
  const auto d1_q_id = AddIrQdq(&ir_model, d1_raw_id, d1_out_sc);

  std::vector<::ml_drift::ir::IrTensorId> parts(k1);
  for (int k = 0; k < k1; ++k) {
    const auto unp_id = ir_model.add_tensor(dtype, BHWC(1, 1, tokens, i4))->id;
    ::ml_drift::ir::IrOp* unp = ir_model.add_op();
    unp->name = kBlockFfwUnpackPartType;
    unp->attr = BlockFfwUnpackPartAttributes{.part = k, .num_parts = k1};
    ir_model.AddConsumer(d1_q_id, unp->id);
    ir_model.SetProducer(unp_id, unp->id);

    const auto q_in_id = distinct_adjacent_scales
                             ? AddIrQdq(&ir_model, unp_id, d2_in_sc)
                             : unp_id;
    const auto fc_id =
        AddPredefinedInt2FcIr(&ir_model, &const_tensors, gpu_info, dtype,
                              q_in_id, d2_ws[k], BHWC(1, 1, tokens, o4));
    parts[k] = AddIrQdq(&ir_model, fc_id, d2_out_sc);
  }
  ::ml_drift::ir::IrOp* pack = ir_model.add_op();
  pack->name = kBlockFfwPackPartsType;
  pack->attr = BlockFfwPackPartsAttributes{.num_parts = k1};
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
  src_cpu.shape = BHWC(1, 1, tokens, d_model);
  src_cpu.data = x_data;
  ::ml_drift::TensorFloat32 dst_cpu;
  dst_cpu.shape = BHWC(1, 1, tokens, d_model);
  dst_cpu.data.resize(expected.size());

  ABSL_RETURN_IF_ERROR(env.ExecuteGpuModel(
      {src_cpu}, std::vector<::ml_drift::TensorFloat32*>{&dst_cpu},
      &gpu_model));

  const float tol =
      precision == CalculationsPrecision::kF16 ? 2.0f * d2_out_sc : 1e-4f;
  EXPECT_THAT(dst_cpu.data, Pointwise(FloatNear(tol), expected));
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
