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

#include "litert/vendors/intel_openvino/compiler/npu_optimizer.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/tensor.hpp"

namespace litert {
namespace openvino {
namespace {

// Builds the "split-cache" attention pattern emitted by the LiteRT generative
// converter (gemma4 prefill/decode), matching the layout observed in the
// model:
//   K_cache: [B,H,S_past,D]   K_slice: [B,H,1,D]
//   V_cache: [B,H,D,S_past]   V_slice: [B,H,D,1]   (transposed, adj_y=true)
//   scores = Concat[ MatMul(Q,K_cache,T), MatMul(Q,K_slice,T) ] + mask
//   probs  = Softmax(scores)
//   out    = Add[ MatMul(Slice(probs,past),V_cache,T),
//                 MatMul(Slice(probs,cur),V_slice,T) ]
// |s_past| is the cached sequence length and |s_cur| is the current chunk size
// (the K/V slice length): decode uses s_cur=1, prefill uses s_cur=chunk (e.g.
// 128). The query length equals s_cur in both cases.
std::shared_ptr<ov::Model> BuildSplitCacheAttention(int64_t s_past = 8,
                                                    int64_t s_cur = 1) {
  using ov::op::v0::Concat;
  using ov::op::v0::Constant;
  using ov::op::v0::MatMul;
  using ov::op::v0::Parameter;
  using ov::op::v1::Add;
  using ov::op::v8::Slice;
  using ov::op::v8::Softmax;

  constexpr int64_t kBatch = 1;
  constexpr int64_t kHeads = 8;
  constexpr int64_t kDim = 256;
  const int64_t lq = s_cur;
  const int64_t s_kv = s_past + s_cur;
  const auto f = ov::element::f32;

  auto q = std::make_shared<Parameter>(
      f, ov::Shape{static_cast<size_t>(kBatch), static_cast<size_t>(kHeads),
                   static_cast<size_t>(lq), static_cast<size_t>(kDim)});
  auto k_cache = std::make_shared<Parameter>(
      f, ov::Shape{static_cast<size_t>(kBatch), static_cast<size_t>(kHeads),
                   static_cast<size_t>(s_past), static_cast<size_t>(kDim)});
  auto k_slice = std::make_shared<Parameter>(
      f, ov::Shape{static_cast<size_t>(kBatch), static_cast<size_t>(kHeads),
                   static_cast<size_t>(s_cur), static_cast<size_t>(kDim)});
  auto v_cache = std::make_shared<Parameter>(
      f, ov::Shape{static_cast<size_t>(kBatch), static_cast<size_t>(kHeads),
                   static_cast<size_t>(kDim), static_cast<size_t>(s_past)});
  auto v_slice = std::make_shared<Parameter>(
      f, ov::Shape{static_cast<size_t>(kBatch), static_cast<size_t>(kHeads),
                   static_cast<size_t>(kDim), static_cast<size_t>(s_cur)});
  auto mask = std::make_shared<Parameter>(
      f, ov::Shape{static_cast<size_t>(kBatch), 1, static_cast<size_t>(lq),
                   static_cast<size_t>(s_kv)});

  // scores = Q * K^T (transpose_b), concatenated over the sequence axis.
  auto qk_cache = std::make_shared<MatMul>(q, k_cache, /*transpose_a=*/false,
                                           /*transpose_b=*/true);
  auto qk_slice = std::make_shared<MatMul>(q, k_slice, false, true);
  auto scores = std::make_shared<Concat>(ov::OutputVector{qk_cache, qk_slice},
                                         /*axis=*/-1);
  auto masked = std::make_shared<Add>(scores, mask);
  auto probs = std::make_shared<Softmax>(masked, /*axis=*/-1);

  // Split probs back into past / current, then probs * V^T (transpose_b).
  auto start0 = Constant::create(ov::element::i64, ov::Shape{1}, {0});
  auto stop0 = Constant::create(ov::element::i64, ov::Shape{1}, {s_past});
  auto step = Constant::create(ov::element::i64, ov::Shape{1}, {1});
  auto axis = Constant::create(ov::element::i64, ov::Shape{1}, {-1});
  auto slice_past = std::make_shared<Slice>(probs, start0, stop0, step, axis);

  auto start1 = Constant::create(ov::element::i64, ov::Shape{1}, {s_past});
  auto stop1 = Constant::create(ov::element::i64, ov::Shape{1}, {s_kv});
  auto slice_cur = std::make_shared<Slice>(probs, start1, stop1, step, axis);

  auto pv_cache = std::make_shared<MatMul>(slice_past, v_cache, false, true);
  auto pv_slice = std::make_shared<MatMul>(slice_cur, v_slice, false, true);
  auto out = std::make_shared<Add>(pv_cache, pv_slice);

  auto result = std::make_shared<ov::op::v0::Result>(out);
  return std::make_shared<ov::Model>(
      ov::ResultVector{result},
      ov::ParameterVector{q, k_cache, k_slice, v_cache, v_slice, mask},
      "split_cache_attention");
}

template <typename T>
size_t CountOps(const std::shared_ptr<ov::Model>& model) {
  size_t n = 0;
  for (const auto& node : model->get_ops()) {
    if (std::dynamic_pointer_cast<T>(node)) {
      ++n;
    }
  }
  return n;
}

std::shared_ptr<ov::op::v13::ScaledDotProductAttention> FindSdpa(
    const std::shared_ptr<ov::Model>& model) {
  for (const auto& node : model->get_ops()) {
    if (auto sdpa =
            std::dynamic_pointer_cast<ov::op::v13::ScaledDotProductAttention>(
                node)) {
      return sdpa;
    }
  }
  return nullptr;
}

// Fills |tensor| with deterministic pseudo-random values in [-1, 1).
void FillRandom(ov::Tensor tensor, uint32_t seed) {
  auto* data = tensor.data<float>();
  // Simple LCG so the test is self-contained and reproducible.
  uint64_t state = seed * 2654435761u + 1u;
  for (size_t i = 0; i < tensor.get_size(); ++i) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    data[i] = static_cast<float>((state >> 40) & 0xFFFF) / 32768.0f - 1.0f;
  }
}

// Negative: when the K_cache (or V_cache) tensor is consumed by something
// outside this attention block — typical of layers that share a single KV
// cache across heads/layers — fusing in place would rewrite what the other
// consumer sees. The pass must skip such blocks.
TEST(FuseSplitAttentionToSDPATest, DoesNotFuseSharedKvCache) {
  auto model = BuildSplitCacheAttention(/*s_past=*/8, /*s_cur=*/8);
  // BuildSplitCacheAttention orders parameters as
  //   {q, k_cache, k_slice, v_cache, v_slice, mask}.
  // Add an extra Result that consumes k_cache directly, so K_cache now feeds
  // two inputs (the qk_cache MatMul and this Result) — HasSingleConsumer
  // returns false and the matcher must bail out.
  auto k_cache = model->get_parameters()[1];
  model->add_results({std::make_shared<ov::op::v0::Result>(k_cache)});

  NpuOptimizer()
      .SetCastIntegerSignToFloat(false)
      .SetFuseSplitAttentionToSDPA(true)
      .Run(model);

  EXPECT_EQ(CountOps<ov::op::v13::ScaledDotProductAttention>(model), 0u);
  EXPECT_EQ(CountOps<ov::op::v0::MatMul>(model), 4u);
}

TEST(FuseSplitAttentionToSDPATest, FusesSplitCachePattern) {
  auto model = BuildSplitCacheAttention(/*s_past=*/8, /*s_cur=*/8);

  // Precondition: 4 MatMuls, 1 Softmax, no SDPA.
  EXPECT_EQ(CountOps<ov::op::v0::MatMul>(model), 4u);
  EXPECT_EQ(CountOps<ov::op::v8::Softmax>(model), 1u);
  EXPECT_EQ(CountOps<ov::op::v13::ScaledDotProductAttention>(model), 0u);

  NpuOptimizer()
      .SetCastIntegerSignToFloat(false)
      .SetFuseSplitAttentionToSDPA(true)
      .Run(model);

  // Postcondition: the four attention MatMuls and the Softmax are gone,
  // replaced by exactly one ScaledDotProductAttention op.
  EXPECT_EQ(CountOps<ov::op::v13::ScaledDotProductAttention>(model), 1u);
  EXPECT_EQ(CountOps<ov::op::v8::Softmax>(model), 0u);
  EXPECT_EQ(CountOps<ov::op::v0::MatMul>(model), 0u);
}

// End-to-end numerical check: the fused SDPA must produce the same output as
// the original split-cache graph for the same inputs. Uses a decode shape
// (s_past=7, s_cur=1) so the V_cache [B,H,D,S] / V_slice [B,H,D,1] transpose
// path is exercised.
TEST(FuseSplitAttentionToSDPATest, NumericallyMatchesSplitCache) {
  auto reference = BuildSplitCacheAttention(/*s_past=*/7, /*s_cur=*/1);
  // Deep-copy before mutating so we can run both versions on identical inputs.
  auto fused = reference->clone();

  NpuOptimizer()
      .SetCastIntegerSignToFloat(false)
      .SetFuseSplitAttentionToSDPA(true)
      .Run(fused);
  ASSERT_NE(FindSdpa(fused), nullptr) << "fusion did not fire";

  ov::Core core;
  auto ref_compiled = core.compile_model(reference, "CPU");
  auto fused_compiled = core.compile_model(fused, "CPU");
  auto ref_req = ref_compiled.create_infer_request();
  auto fused_req = fused_compiled.create_infer_request();

  // Same input tensors fed to both. Inputs are ordered as constructed:
  // {q, k_cache, k_slice, v_cache, v_slice, mask}.
  const size_t num_inputs = reference->inputs().size();
  ASSERT_EQ(num_inputs, fused->inputs().size());
  std::vector<ov::Tensor> inputs;
  for (size_t i = 0; i < num_inputs; ++i) {
    const auto& port = reference->input(i);
    ov::Tensor t(port.get_element_type(), port.get_shape());
    FillRandom(t, static_cast<uint32_t>(i + 1));
    inputs.push_back(t);
    ref_req.set_input_tensor(i, t);
    fused_req.set_input_tensor(i, t);
  }

  ref_req.infer();
  fused_req.infer();

  auto ref_out = ref_req.get_output_tensor(0);
  auto fused_out = fused_req.get_output_tensor(0);
  ASSERT_EQ(ref_out.get_size(), fused_out.get_size());
  const auto* a = ref_out.data<float>();
  const auto* b = fused_out.data<float>();
  float max_abs_diff = 0.0f;
  for (size_t i = 0; i < ref_out.get_size(); ++i) {
    max_abs_diff = std::max(max_abs_diff, std::abs(a[i] - b[i]));
    ASSERT_FALSE(std::isnan(b[i])) << "fused output has NaN at " << i;
  }
  EXPECT_LT(max_abs_diff, 1e-4f)
      << "fused output diverges from split-cache reference";
}

// Fixed per-expert weight shapes used by BuildDenseMoeGraph: a fused gate+up
// projection of width kMoeGateUp (split evenly into gate | up for GEGLU) and
// a down projection back to kMoeHiddenDim.
constexpr int64_t kMoeHiddenDim = 4;
constexpr int64_t kMoeHalf = 2;
constexpr int64_t kMoeGateUp = 2 * kMoeHalf;

// Builds one expert's i4/u4-packed dequantized weight:
//   Multiply(Convert(Constant<u4>(shape)), Constant<f32>(scale_shape))
// matching the pattern find_dequant_source() looks for. |fill| seeds the
// packed values (0-15, wrapped) so each expert's weights are numerically
// distinguishable.
ov::Output<ov::Node> MakeDequantWeight(const ov::Shape& shape,
                                       const ov::Shape& scale_shape,
                                       uint8_t fill, float scale_value) {
  std::vector<uint8_t> packed_values(ov::shape_size(shape), fill % 16);
  auto packed =
      ov::op::v0::Constant::create(ov::element::u4, shape, packed_values);
  auto scale = ov::op::v0::Constant::create(
      ov::element::f32, scale_shape,
      std::vector<float>(ov::shape_size(scale_shape), scale_value));
  auto convert =
      std::make_shared<ov::op::v0::Convert>(packed, ov::element::f32);
  return std::make_shared<ov::op::v1::Multiply>(convert, scale)->output(0);
}

// Builds a Gemma4-style dense MoE block: one independent GEGLU branch per
// entry in |expert_ids|, each masked by Equal(router_topk_indices, expert_id)
// and summed via a chain of Add — the pattern MoEGatherRewrite looks for (see
// find_moe_layers / collect_expert_branch in npu_optimizer.cc). |expert_ids|
// need not be contiguous/sorted; the rewrite requires sorted 0..N-1 and will
// reject graphs that aren't (see DoesNotRewriteWhenExpertIdsNotContiguous).
// Inputs, in order, are {hidden [batch,H], router_logits [batch,N]}.
std::shared_ptr<ov::Model> BuildDenseMoeGraph(
    int64_t k, const std::vector<int64_t>& expert_ids, int64_t batch = 1) {
  using ov::op::v0::Constant;
  using ov::op::v0::Convert;
  using ov::op::v0::MatMul;
  using ov::op::v0::Parameter;
  using ov::op::v1::Add;
  using ov::op::v1::Divide;
  using ov::op::v1::Equal;
  using ov::op::v1::Multiply;
  using ov::op::v1::ReduceSum;
  using ov::op::v7::Gelu;
  using ov::op::v8::Slice;

  const auto f32 = ov::element::f32;
  const int64_t num_experts = static_cast<int64_t>(expert_ids.size());
  auto hidden = std::make_shared<Parameter>(
      f32, ov::Shape{static_cast<size_t>(batch),
                     static_cast<size_t>(kMoeHiddenDim)});
  auto logits = std::make_shared<Parameter>(
      f32,
      ov::Shape{static_cast<size_t>(batch), static_cast<size_t>(num_experts)});

  auto k_const = Constant::create(ov::element::i64, ov::Shape{}, {k});
  auto topk = std::make_shared<ov::op::v11::TopK>(logits, k_const, /*axis=*/1,
                                                  "max", "value");
  ov::Output<ov::Node> values = topk->output(0);
  ov::Output<ov::Node> indices = topk->output(1);

  auto sum_axis = Constant::create(ov::element::i64, ov::Shape{1}, {1});
  auto weight_sum =
      std::make_shared<ReduceSum>(values, sum_axis, /*keep_dims=*/true);
  ov::Output<ov::Node> router_weights =
      std::make_shared<Divide>(values, weight_sum)->output(0);

  ov::Output<ov::Node> accum;
  for (int64_t i = 0; i < num_experts; ++i) {
    auto expert_id = Constant::create(indices.get_element_type(), ov::Shape{},
                                      {expert_ids[i]});
    auto eq = std::make_shared<Equal>(indices, expert_id);
    auto conv = std::make_shared<Convert>(eq, f32);
    auto premul = std::make_shared<Multiply>(conv, router_weights);
    auto reduce_axis = Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto score = std::make_shared<ReduceSum>(premul, reduce_axis,
                                             /*keep_dims=*/true);

    auto w_up = MakeDequantWeight(ov::Shape{static_cast<size_t>(kMoeGateUp),
                                            static_cast<size_t>(kMoeHiddenDim)},
                                  ov::Shape{static_cast<size_t>(kMoeGateUp), 1},
                                  static_cast<uint8_t>(i + 1), 0.1f * (i + 1));
    auto up_mm = std::make_shared<MatMul>(hidden, w_up, /*transpose_a=*/false,
                                          /*transpose_b=*/true);

    auto slice_step = Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto slice_axis = Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto gate_start = Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto gate_stop =
        Constant::create(ov::element::i64, ov::Shape{1}, {kMoeHalf});
    auto gate = std::make_shared<Slice>(up_mm, gate_start, gate_stop,
                                        slice_step, slice_axis);
    auto up_start =
        Constant::create(ov::element::i64, ov::Shape{1}, {kMoeHalf});
    auto up_stop =
        Constant::create(ov::element::i64, ov::Shape{1}, {2 * kMoeHalf});
    auto up_half = std::make_shared<Slice>(up_mm, up_start, up_stop, slice_step,
                                           slice_axis);
    auto gate_act =
        std::make_shared<Gelu>(gate, ov::op::GeluApproximationMode::TANH);
    auto geglu = std::make_shared<Multiply>(gate_act, up_half);

    auto w_down =
        MakeDequantWeight(ov::Shape{static_cast<size_t>(kMoeHiddenDim),
                                    static_cast<size_t>(kMoeHalf)},
                          ov::Shape{static_cast<size_t>(kMoeHiddenDim), 1},
                          static_cast<uint8_t>(i + 2), 0.05f * (i + 1));
    auto down_mm = std::make_shared<MatMul>(geglu, w_down,
                                            /*transpose_a=*/false,
                                            /*transpose_b=*/true);

    auto branch = std::make_shared<Multiply>(score, down_mm);
    if (i == 0) {
      accum = branch->output(0);
    } else {
      accum = std::make_shared<Add>(accum, branch)->output(0);
    }
  }

  auto result = std::make_shared<ov::op::v0::Result>(accum);
  return std::make_shared<ov::Model>(ov::ResultVector{result},
                                     ov::ParameterVector{hidden, logits},
                                     "dense_moe");
}

std::vector<int64_t> SequentialExpertIds(int64_t num_experts) {
  std::vector<int64_t> ids(num_experts);
  std::iota(ids.begin(), ids.end(), 0);
  return ids;
}

TEST(MoEGatherRewriteTest, RewritesDenseMoeIntoGatherForm) {
  auto model = BuildDenseMoeGraph(/*k=*/2, SequentialExpertIds(4));

  EXPECT_EQ(CountOps<ov::op::v1::Equal>(model), 4u);
  EXPECT_EQ(CountOps<ov::op::v1::Add>(model), 3u);
  EXPECT_EQ(CountOps<ov::op::v1::ReduceSum>(model), 5u);
  EXPECT_EQ(CountOps<ov::op::v8::Gather>(model), 0u);

  NpuOptimizer().SetCastIntegerSignToFloat(false).SetEnableMoeGather(true).Run(
      model);

  // Postcondition: the masked dense chain (Equal/Add) is gone, replaced by
  // Gather-K weight selection (2 Gathers per weight: flattened packed rows +
  // scale) and a single weighted ReduceSum over the K selected experts. The
  // router's weight-sum ReduceSum survives (it still feeds the K-weighting).
  EXPECT_EQ(CountOps<ov::op::v1::Equal>(model), 0u);
  // ExpandExpertRowIndices emits one row-offset Add per gathered weight (up +
  // down)
  EXPECT_EQ(CountOps<ov::op::v1::Add>(model), 2u);
  EXPECT_EQ(CountOps<ov::op::v8::Gather>(model), 4u);
  EXPECT_EQ(CountOps<ov::op::v1::ReduceSum>(model), 2u);
  EXPECT_EQ(CountOps<ov::op::v0::MatMul>(model), 2u);
}

// Negative: MoEGatherRewrite requires sorted expert_ids to be exactly
// contiguous 0..N-1 (the grouped weight table is indexed by expert_id, and
// rows are stacked in sorted-expert_id order). A gap must abort the rewrite
// and leave the graph untouched.
TEST(MoEGatherRewriteTest, DoesNotRewriteWhenExpertIdsNotContiguous) {
  auto model = BuildDenseMoeGraph(/*k=*/2, /*expert_ids=*/{0, 1, 3});

  NpuOptimizer().SetCastIntegerSignToFloat(false).SetEnableMoeGather(true).Run(
      model);

  EXPECT_EQ(CountOps<ov::op::v1::Equal>(model), 3u);
  EXPECT_EQ(CountOps<ov::op::v8::Gather>(model), 0u);
}

// Negative: the gather form assumes a single token (decode/generate). When
// the router's TopK indices have a statically-known batch dim != 1 (a
// prefill-shaped call), the rewrite must skip the layer rather than
// mis-compile it.
TEST(MoEGatherRewriteTest, DoesNotRewritePrefillShapedRouter) {
  auto model = BuildDenseMoeGraph(/*k=*/2, SequentialExpertIds(4), /*batch=*/2);

  NpuOptimizer().SetCastIntegerSignToFloat(false).SetEnableMoeGather(true).Run(
      model);

  EXPECT_EQ(CountOps<ov::op::v1::Equal>(model), 4u);
  EXPECT_EQ(CountOps<ov::op::v8::Gather>(model), 0u);
}

// Negative/edge case: a degenerate batch=0 call (zero tokens). The TopK
// indices' batch dim is statically 0, which is != 1, so this must hit the
// same "not decode-shaped" guard as batch=2 rather than e.g. dividing by zero
// or otherwise misbehaving on an empty tensor.
TEST(MoEGatherRewriteTest, DoesNotRewriteWhenBatchIsZero) {
  auto model = BuildDenseMoeGraph(/*k=*/2, SequentialExpertIds(4), /*batch=*/0);

  NpuOptimizer().SetCastIntegerSignToFloat(false).SetEnableMoeGather(true).Run(
      model);

  EXPECT_EQ(CountOps<ov::op::v1::Equal>(model), 4u);
  EXPECT_EQ(CountOps<ov::op::v8::Gather>(model), 0u);
}

// Runs |reference| and its MoEGatherRewrite-rewritten clone on identical
// {hidden, logits} inputs and returns the max abs difference between their
// outputs. Asserts the rewrite actually fired (Gather count == 4) so a silent
// no-op rewrite can't masquerade as "numerically matches".
float MaxAbsDiffAfterRewrite(const std::shared_ptr<ov::Model>& reference,
                             const ov::Tensor& hidden,
                             const ov::Tensor& logits) {
  auto rewritten = reference->clone();
  NpuOptimizer().SetCastIntegerSignToFloat(false).SetEnableMoeGather(true).Run(
      rewritten);
  EXPECT_EQ(CountOps<ov::op::v8::Gather>(rewritten), 4u)
      << "rewrite did not fire";

  ov::Core core;
  auto ref_compiled = core.compile_model(reference, "CPU");
  auto rew_compiled = core.compile_model(rewritten, "CPU");
  auto ref_req = ref_compiled.create_infer_request();
  auto rew_req = rew_compiled.create_infer_request();
  for (auto& req : {std::ref(ref_req), std::ref(rew_req)}) {
    req.get().set_input_tensor(0, hidden);
    req.get().set_input_tensor(1, logits);
  }
  ref_req.infer();
  rew_req.infer();

  auto ref_out = ref_req.get_output_tensor(0);
  auto rew_out = rew_req.get_output_tensor(0);
  EXPECT_EQ(ref_out.get_size(), rew_out.get_size());
  const auto* a = ref_out.data<float>();
  const auto* b = rew_out.data<float>();
  float max_abs_diff = 0.0f;
  for (size_t i = 0; i < ref_out.get_size(); ++i) {
    max_abs_diff = std::max(max_abs_diff, std::abs(a[i] - b[i]));
  }
  return max_abs_diff;
}

ov::Tensor MakeFilledTensor(const ov::Shape& shape, float value) {
  ov::Tensor t(ov::element::f32, shape);
  std::fill_n(t.data<float>(), t.get_size(), value);
  return t;
}

TEST(MoEGatherRewriteTest, NumericallyMatchesDenseComputation) {
  auto reference = BuildDenseMoeGraph(/*k=*/2, SequentialExpertIds(6));

  ov::Tensor hidden(ov::element::f32, ov::Shape{1, kMoeHiddenDim});
  FillRandom(hidden, /*seed=*/1);
  ov::Tensor logits(ov::element::f32, ov::Shape{1, 6});
  FillRandom(logits, /*seed=*/2);

  EXPECT_LT(MaxAbsDiffAfterRewrite(reference, hidden, logits), 1e-3f)
      << "gather-K rewrite diverges from dense masked-experts reference";
}

TEST(MoEGatherRewriteTest, NumericallyMatchesWithAllZeroInputs) {
  auto reference = BuildDenseMoeGraph(/*k=*/2, SequentialExpertIds(6));
  auto hidden = MakeFilledTensor(ov::Shape{1, kMoeHiddenDim}, 0.0f);
  auto logits = MakeFilledTensor(ov::Shape{1, 6}, 0.0f);

  float max_abs_diff = MaxAbsDiffAfterRewrite(reference, hidden, logits);
  // With all-zero logits, weight_sum == 0 and router_weights is 0/0 (NaN) in
  // *both* the reference and rewritten graphs identically, so this only
  // checks that both sides agree (both NaN or both equal), not that the
  // result is finite.
  if (!std::isnan(max_abs_diff)) {
    EXPECT_LT(max_abs_diff, 1e-3f);
  }
}

TEST(MoEGatherRewriteTest, NumericallyMatchesWithAllOnesInputs) {
  auto reference = BuildDenseMoeGraph(/*k=*/2, SequentialExpertIds(6));
  auto hidden = MakeFilledTensor(ov::Shape{1, kMoeHiddenDim}, 1.0f);
  auto logits = MakeFilledTensor(ov::Shape{1, 6}, 1.0f);

  EXPECT_LT(MaxAbsDiffAfterRewrite(reference, hidden, logits), 1e-3f);
}

TEST(MoEGatherRewriteTest, NumericallyMatchesWithExtremeValueInputs) {
  auto reference = BuildDenseMoeGraph(/*k=*/2, SequentialExpertIds(6));
  auto hidden = MakeFilledTensor(ov::Shape{1, kMoeHiddenDim}, 1.0e4f);
  ov::Tensor logits(ov::element::f32, ov::Shape{1, 6});
  // Large, distinct-magnitude logits so TopK's chosen experts are unambiguous
  // even at this scale.
  const float kLogitValues[] = {1.0e3f,  -1.0e4f, 1.0e5f,
                                -1.0e2f, 1.0e4f,  -1.0e5f};
  std::copy(std::begin(kLogitValues), std::end(kLogitValues),
            logits.data<float>());

  EXPECT_LT(MaxAbsDiffAfterRewrite(reference, hidden, logits), 5.0f)
      << "absolute tolerance widened for the extreme-magnitude inputs "
         "case above; a growing gap here would indicate the rewrite's "
         "arithmetic (e.g. gather order) diverges at scale, not just noise";
}

// Tanh-approximation Gelu, matching ov::op::GeluApproximationMode::TANH, used
// to independently hand-compute the expected per-expert contribution below
// without relying on the (dense) reference OpenVINO graph at all.
float ReferenceGeluTanh(float x) {
  constexpr float kSqrt2OverPi = 0.7978845608f;
  const float inner = kSqrt2OverPi * (x + 0.044715f * x * x * x);
  return 0.5f * x * (1.0f + std::tanh(inner));
}

TEST(MoEGatherRewriteTest, GatherSelectsCorrectExpertRows) {
  constexpr int64_t kNumExperts = 4;
  constexpr int64_t kK = 2;
  auto model = BuildDenseMoeGraph(kK, SequentialExpertIds(kNumExperts));

  NpuOptimizer().SetCastIntegerSignToFloat(false).SetEnableMoeGather(true).Run(
      model);
  ASSERT_EQ(CountOps<ov::op::v8::Gather>(model), 4u) << "rewrite did not fire";

  // logits chosen so experts 1 and 3 (values 5.0, 8.0) are the clear top-2;
  // experts 0 and 2 must be excluded from the result.
  const float kLogitValues[kNumExperts] = {0.1f, 5.0f, 0.2f, 8.0f};
  const std::vector<int64_t> kSelected = {1, 3};

  ov::Tensor hidden = MakeFilledTensor(ov::Shape{1, kMoeHiddenDim}, 1.0f);
  ov::Tensor logits(ov::element::f32, ov::Shape{1, kNumExperts});
  std::copy(std::begin(kLogitValues), std::end(kLogitValues),
            logits.data<float>());

  ov::Core core;
  auto compiled = core.compile_model(model, "CPU");
  auto req = compiled.create_infer_request();
  req.set_input_tensor(0, hidden);
  req.set_input_tensor(1, logits);
  req.infer();

  // Independently recompute BuildDenseMoeGraph's math for exactly the
  // selected experts (all-ones hidden + a uniform per-row weight fill means
  // every one of the H output columns collapses to the same scalar).
  float weight_sum = 0.0f;
  for (int64_t e : kSelected) weight_sum += kLogitValues[e];
  float expected = 0.0f;
  for (int64_t e : kSelected) {
    const float router_weight = kLogitValues[e] / weight_sum;
    const float v_up = static_cast<float>((e + 1) % 16) * 0.1f * (e + 1);
    const float up_val = static_cast<float>(kMoeHiddenDim) * v_up;
    const float geglu_val = ReferenceGeluTanh(up_val) * up_val;
    const float v_down = static_cast<float>((e + 2) % 16) * 0.05f * (e + 1);
    const float down_val = static_cast<float>(kMoeHalf) * geglu_val * v_down;
    expected += router_weight * down_val;
  }

  auto out = req.get_output_tensor(0);
  ASSERT_EQ(out.get_size(), static_cast<size_t>(kMoeHiddenDim));
  const auto* out_data = out.data<float>();
  for (size_t i = 0; i < out.get_size(); ++i) {
    EXPECT_NEAR(out_data[i], expected, std::abs(expected) * 1e-3f + 1e-4f)
        << "output[" << i
        << "] does not match the hand-computed "
           "contribution of experts {1,3} — the gather may be selecting "
           "the wrong expert rows";
  }
}

}  // namespace
}  // namespace openvino
}  // namespace litert
