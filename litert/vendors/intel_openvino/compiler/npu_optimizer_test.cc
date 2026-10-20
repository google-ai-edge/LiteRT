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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/tensor.hpp"
#include <gtest/gtest.h>
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "transformations/op_conversions/scaled_dot_product_attention_decomposition.hpp"

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
// 128). The query length equals s_cur in both cases. |negative_cur_start|
// spells the current branch's start as -s_cur (counted from the end of the
// score axis) instead of +s_past; both select the same elements.
std::shared_ptr<ov::Model> BuildSplitCacheAttention(
    int64_t s_past = 8, int64_t s_cur = 1, bool reverse_v_add = false,
    bool use_strided_slice = false, bool negative_cur_start = false) {
  using ov::op::v0::Concat;
  using ov::op::v0::Constant;
  using ov::op::v0::MatMul;
  using ov::op::v0::Parameter;
  using ov::op::v1::Add;
  using ov::op::v1::StridedSlice;
  using ov::op::v8::Slice;
  using ov::op::v8::Softmax;

  constexpr int64_t kBatch = 1;
  constexpr int64_t kHeads = 8;
  constexpr int64_t kDim = 256;
  const int64_t lq = s_cur;
  const int64_t s_kv = s_past + s_cur;
  const int64_t cur_start = negative_cur_start ? -s_cur : s_past;
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
  ov::Output<ov::Node> slice_past;
  ov::Output<ov::Node> slice_cur;
  if (use_strided_slice) {
    // Exercise begin_mask[3] == 1 on the past branch and explicit
    // begin[3] == s_past on the new branch.
    auto begin_past = Constant::create(ov::element::i64, ov::Shape{4},
                                       std::vector<int64_t>{0, 0, 0, 0});
    auto end_past = Constant::create(ov::element::i64, ov::Shape{4},
                                     std::vector<int64_t>{0, 0, 0, s_past});
    auto strides = Constant::create(ov::element::i64, ov::Shape{4},
                                    std::vector<int64_t>{1, 1, 1, 1});
    slice_past = std::make_shared<StridedSlice>(
                     probs, begin_past, end_past, strides,
                     /*begin_mask=*/std::vector<int64_t>{1, 1, 1, 1},
                     /*end_mask=*/std::vector<int64_t>{1, 1, 1, 0})
                     ->output(0);

    auto begin_cur = Constant::create(ov::element::i64, ov::Shape{4},
                                      std::vector<int64_t>{0, 0, 0, cur_start});
    auto end_cur = Constant::create(ov::element::i64, ov::Shape{4},
                                    std::vector<int64_t>{0, 0, 0, s_kv});
    slice_cur = std::make_shared<StridedSlice>(
                    probs, begin_cur, end_cur, strides,
                    /*begin_mask=*/std::vector<int64_t>{1, 1, 1, 0},
                    /*end_mask=*/std::vector<int64_t>{1, 1, 1, 1})
                    ->output(0);
  } else {
    auto start0 = Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto stop0 = Constant::create(ov::element::i64, ov::Shape{1}, {s_past});
    auto step = Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto axis = Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    slice_past =
        std::make_shared<Slice>(probs, start0, stop0, step, axis)->output(0);

    auto start1 = Constant::create(ov::element::i64, ov::Shape{1}, {cur_start});
    auto stop1 = Constant::create(ov::element::i64, ov::Shape{1}, {s_kv});
    slice_cur =
        std::make_shared<Slice>(probs, start1, stop1, step, axis)->output(0);
  }

  auto pv_cache = std::make_shared<MatMul>(slice_past, v_cache, false, true);
  auto pv_slice = std::make_shared<MatMul>(slice_cur, v_slice, false, true);
  auto out = reverse_v_add ? std::make_shared<Add>(pv_slice, pv_cache)
                           : std::make_shared<Add>(pv_cache, pv_slice);

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

// Evaluates |model| on |inputs| using OpenVINO's built-in reference kernels
// (after decomposing any ScaledDotProductAttention op into primitive ops that
// implement Node::evaluate). This avoids loading the CPU plugin DSO at test
// time while exercising the exact graph topology and tensor values.
ov::Tensor EvaluateModel(const std::shared_ptr<ov::Model>& model,
                         const ov::TensorVector& inputs) {
  auto eval_model = model->clone();
  ov::pass::Manager pm;
  pm.register_pass<ov::pass::ScaledDotProductAttentionDecomposition>();
  pm.run_passes(eval_model);
  ov::TensorVector outputs;
  for (const auto& out : eval_model->outputs()) {
    outputs.emplace_back(out.get_element_type(), out.get_shape());
  }
  EXPECT_TRUE(eval_model->evaluate(outputs, inputs));
  return outputs.at(0);
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

// When the final Add has its operands reversed (`Add(pv_slice, pv_cache)`
// rather than `Add(pv_cache, pv_slice)`), FuseSplitAttentionToSDPA must
// inspect the Softmax slice offsets and swap the V branches so that K and V
// are concatenated in the same `[cache, new]` order. Without this recovery,
// every attention weight is multiplied against the wrong value vector.
TEST(FuseSplitAttentionToSDPATest, FusesAndMatchesWhenAddOperandsAreReversed) {
  auto reference = BuildSplitCacheAttention(/*s_past=*/7, /*s_cur=*/1,
                                            /*reverse_v_add=*/true);
  auto fused = reference->clone();

  NpuOptimizer()
      .SetCastIntegerSignToFloat(false)
      .SetFuseSplitAttentionToSDPA(true)
      .Run(fused);
  ASSERT_NE(FindSdpa(fused), nullptr) << "fusion did not fire";
  EXPECT_EQ(CountOps<ov::op::v0::MatMul>(fused), 0u);
  EXPECT_EQ(CountOps<ov::op::v8::Softmax>(fused), 0u);

  const size_t num_inputs = reference->inputs().size();
  ASSERT_EQ(num_inputs, fused->inputs().size());
  ov::TensorVector inputs;
  for (size_t i = 0; i < num_inputs; ++i) {
    const auto& port = reference->input(i);
    ov::Tensor t(port.get_element_type(), port.get_shape());
    FillRandom(t, static_cast<uint32_t>(i + 10));
    inputs.push_back(t);
  }

  auto ref_out = EvaluateModel(reference, inputs);
  auto fused_out = EvaluateModel(fused, inputs);
  ASSERT_EQ(ref_out.get_size(), fused_out.get_size());
  const auto* a = ref_out.data<float>();
  const auto* b = fused_out.data<float>();
  float max_abs_diff = 0.0f;
  for (size_t i = 0; i < ref_out.get_size(); ++i) {
    max_abs_diff = std::max(max_abs_diff, std::abs(a[i] - b[i]));
    ASSERT_FALSE(std::isnan(b[i])) << "fused output has NaN at " << i;
  }
  EXPECT_LT(max_abs_diff, 1e-4f)
      << "reversed-Add fused output diverges from split-cache reference";
}

// Same V-branch recovery when the Softmax is split using v1::StridedSlice
// (including begin_mask on the past branch) instead of v8::Slice.
TEST(FuseSplitAttentionToSDPATest, FusesAndMatchesWithStridedSliceVBranches) {
  auto reference = BuildSplitCacheAttention(
      /*s_past=*/7, /*s_cur=*/1, /*reverse_v_add=*/true,
      /*use_strided_slice=*/true);
  auto fused = reference->clone();

  NpuOptimizer()
      .SetCastIntegerSignToFloat(false)
      .SetFuseSplitAttentionToSDPA(true)
      .Run(fused);
  ASSERT_NE(FindSdpa(fused), nullptr) << "fusion did not fire";
  EXPECT_EQ(CountOps<ov::op::v0::MatMul>(fused), 0u);

  const size_t num_inputs = reference->inputs().size();
  ASSERT_EQ(num_inputs, fused->inputs().size());
  ov::TensorVector inputs;
  for (size_t i = 0; i < num_inputs; ++i) {
    const auto& port = reference->input(i);
    ov::Tensor t(port.get_element_type(), port.get_shape());
    FillRandom(t, static_cast<uint32_t>(i + 20));
    inputs.push_back(t);
  }

  auto ref_out = EvaluateModel(reference, inputs);
  auto fused_out = EvaluateModel(fused, inputs);
  ASSERT_EQ(ref_out.get_size(), fused_out.get_size());
  const auto* a = ref_out.data<float>();
  const auto* b = fused_out.data<float>();
  float max_abs_diff = 0.0f;
  for (size_t i = 0; i < ref_out.get_size(); ++i) {
    max_abs_diff = std::max(max_abs_diff, std::abs(a[i] - b[i]));
    ASSERT_FALSE(std::isnan(b[i])) << "fused output has NaN at " << i;
  }
  EXPECT_LT(max_abs_diff, 1e-4f)
      << "StridedSlice fused output diverges from split-cache reference";
}

// Slice and StridedSlice both accept a negative start that counts back from
// the end of the axis. The current branch written as start=-s_cur must be
// recognized as the high end of the score axis (the same as start=s_past), so
// the reversed Add is still reordered and fused correctly.
TEST(FuseSplitAttentionToSDPATest, FusesAndMatchesWithNegativeSliceStart) {
  for (const bool use_strided_slice : {false, true}) {
    SCOPED_TRACE(use_strided_slice ? "StridedSlice" : "Slice");
    auto reference = BuildSplitCacheAttention(
        /*s_past=*/7, /*s_cur=*/1, /*reverse_v_add=*/true, use_strided_slice,
        /*negative_cur_start=*/true);
    auto fused = reference->clone();

    NpuOptimizer()
        .SetCastIntegerSignToFloat(false)
        .SetFuseSplitAttentionToSDPA(true)
        .Run(fused);
    ASSERT_NE(FindSdpa(fused), nullptr) << "fusion did not fire";
    EXPECT_EQ(CountOps<ov::op::v0::MatMul>(fused), 0u);

    const size_t num_inputs = reference->inputs().size();
    ASSERT_EQ(num_inputs, fused->inputs().size());
    ov::TensorVector inputs;
    for (size_t i = 0; i < num_inputs; ++i) {
      const auto& port = reference->input(i);
      ov::Tensor t(port.get_element_type(), port.get_shape());
      FillRandom(t, static_cast<uint32_t>(i + 30));
      inputs.push_back(t);
    }

    auto ref_out = EvaluateModel(reference, inputs);
    auto fused_out = EvaluateModel(fused, inputs);
    ASSERT_EQ(ref_out.get_size(), fused_out.get_size());
    const auto* a = ref_out.data<float>();
    const auto* b = fused_out.data<float>();
    float max_abs_diff = 0.0f;
    for (size_t i = 0; i < ref_out.get_size(); ++i) {
      max_abs_diff = std::max(max_abs_diff, std::abs(a[i] - b[i]));
      ASSERT_FALSE(std::isnan(b[i])) << "fused output has NaN at " << i;
    }
    EXPECT_LT(max_abs_diff, 1e-4f)
        << "negative-start fused output diverges from split-cache reference";
  }
}

// Negative: a V-branch slice with a non-unit step does not read one forward
// range beginning at its start (a negative step reads backwards from it), so
// its start says nothing about branch order and the pass must not fuse.
TEST(FuseSplitAttentionToSDPATest, DoesNotFuseWhenVBranchSliceHasNegativeStep) {
  using ov::op::v0::Concat;
  using ov::op::v0::Constant;
  using ov::op::v0::MatMul;
  using ov::op::v0::Parameter;
  using ov::op::v1::Add;
  using ov::op::v8::Slice;
  using ov::op::v8::Softmax;

  const auto f = ov::element::f32;
  auto q = std::make_shared<Parameter>(f, ov::Shape{1, 8, 1, 256});
  auto k_cache = std::make_shared<Parameter>(f, ov::Shape{1, 8, 8, 256});
  auto k_slice = std::make_shared<Parameter>(f, ov::Shape{1, 8, 8, 256});
  auto v_cache = std::make_shared<Parameter>(f, ov::Shape{1, 8, 256, 8});
  auto v_slice = std::make_shared<Parameter>(f, ov::Shape{1, 8, 256, 8});
  auto mask = std::make_shared<Parameter>(f, ov::Shape{1, 1, 1, 16});

  auto qk_cache = std::make_shared<MatMul>(q, k_cache, false, true);
  auto qk_slice = std::make_shared<MatMul>(q, k_slice, false, true);
  auto scores =
      std::make_shared<Concat>(ov::OutputVector{qk_cache, qk_slice}, -1);
  auto masked = std::make_shared<Add>(scores, mask);
  auto probs = std::make_shared<Softmax>(masked, -1);

  auto axis = Constant::create(ov::element::i64, ov::Shape{1}, {-1});
  // Past branch: elements [0, 8) in forward order.
  auto slice_past = std::make_shared<Slice>(
      probs, Constant::create(ov::element::i64, ov::Shape{1}, {0}),
      Constant::create(ov::element::i64, ov::Shape{1}, {8}),
      Constant::create(ov::element::i64, ov::Shape{1}, {1}), axis);
  // Current branch: elements 15 down to 8, read backwards (start=15, step=-1).
  auto slice_cur = std::make_shared<Slice>(
      probs, Constant::create(ov::element::i64, ov::Shape{1}, {15}),
      Constant::create(ov::element::i64, ov::Shape{1}, {7}),
      Constant::create(ov::element::i64, ov::Shape{1}, {-1}), axis);

  auto pv_cache = std::make_shared<MatMul>(slice_past, v_cache, false, true);
  auto pv_slice = std::make_shared<MatMul>(slice_cur, v_slice, false, true);
  auto out = std::make_shared<Add>(pv_cache, pv_slice);
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{std::make_shared<ov::op::v0::Result>(out)},
      ov::ParameterVector{q, k_cache, k_slice, v_cache, v_slice, mask},
      "negative_step_split_attention");

  NpuOptimizer()
      .SetCastIntegerSignToFloat(false)
      .SetFuseSplitAttentionToSDPA(true)
      .Run(model);

  EXPECT_EQ(CountOps<ov::op::v13::ScaledDotProductAttention>(model), 0u);
  EXPECT_EQ(CountOps<ov::op::v0::MatMul>(model), 4u);
}

// Negative: a StridedSlice ellipsis mask shifts which input axis each begin[i]
// refers to. Here begin has rank + 1 entries with an ellipsis at index 0 that
// covers no axes, so begin[3] is axis 2's offset (0) while the last axis
// actually starts at begin[4] (8). The pass must not read begin[3] as the
// last-axis offset, and must not fuse.
TEST(FuseSplitAttentionToSDPATest,
     DoesNotFuseWhenVBranchStridedSliceUsesEllipsisMask) {
  using ov::op::v0::Concat;
  using ov::op::v0::Constant;
  using ov::op::v0::MatMul;
  using ov::op::v0::Parameter;
  using ov::op::v1::Add;
  using ov::op::v1::StridedSlice;
  using ov::op::v8::Slice;
  using ov::op::v8::Softmax;

  const auto f = ov::element::f32;
  auto q = std::make_shared<Parameter>(f, ov::Shape{1, 8, 1, 256});
  auto k_cache = std::make_shared<Parameter>(f, ov::Shape{1, 8, 8, 256});
  auto k_slice = std::make_shared<Parameter>(f, ov::Shape{1, 8, 8, 256});
  auto v_cache = std::make_shared<Parameter>(f, ov::Shape{1, 8, 256, 8});
  auto v_slice = std::make_shared<Parameter>(f, ov::Shape{1, 8, 256, 8});
  auto mask = std::make_shared<Parameter>(f, ov::Shape{1, 1, 1, 16});

  auto qk_cache = std::make_shared<MatMul>(q, k_cache, false, true);
  auto qk_slice = std::make_shared<MatMul>(q, k_slice, false, true);
  auto scores =
      std::make_shared<Concat>(ov::OutputVector{qk_cache, qk_slice}, -1);
  auto masked = std::make_shared<Add>(scores, mask);
  auto probs = std::make_shared<Softmax>(masked, -1);

  auto slice_past = std::make_shared<Slice>(
      probs, Constant::create(ov::element::i64, ov::Shape{1}, {0}),
      Constant::create(ov::element::i64, ov::Shape{1}, {8}),
      Constant::create(ov::element::i64, ov::Shape{1}, {1}),
      Constant::create(ov::element::i64, ov::Shape{1}, {-1}));
  auto slice_cur = std::make_shared<StridedSlice>(
      probs,
      Constant::create(ov::element::i64, ov::Shape{5},
                       std::vector<int64_t>{0, 0, 0, 0, 8}),
      Constant::create(ov::element::i64, ov::Shape{5},
                       std::vector<int64_t>{0, 0, 0, 0, 16}),
      Constant::create(ov::element::i64, ov::Shape{5},
                       std::vector<int64_t>{1, 1, 1, 1, 1}),
      /*begin_mask=*/std::vector<int64_t>{0, 1, 1, 1, 0},
      /*end_mask=*/std::vector<int64_t>{0, 1, 1, 1, 0},
      /*new_axis_mask=*/std::vector<int64_t>{},
      /*shrink_axis_mask=*/std::vector<int64_t>{},
      /*ellipsis_mask=*/std::vector<int64_t>{1, 0, 0, 0, 0});

  auto pv_cache = std::make_shared<MatMul>(slice_past, v_cache, false, true);
  auto pv_slice = std::make_shared<MatMul>(slice_cur, v_slice, false, true);
  auto out = std::make_shared<Add>(pv_cache, pv_slice);
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{std::make_shared<ov::op::v0::Result>(out)},
      ov::ParameterVector{q, k_cache, k_slice, v_cache, v_slice, mask},
      "ellipsis_mask_split_attention");
  ASSERT_EQ(slice_cur->get_output_partial_shape(0),
            ov::PartialShape({1, 8, 1, 8}));

  NpuOptimizer()
      .SetCastIntegerSignToFloat(false)
      .SetFuseSplitAttentionToSDPA(true)
      .Run(model);

  EXPECT_EQ(CountOps<ov::op::v13::ScaledDotProductAttention>(model), 0u);
  EXPECT_EQ(CountOps<ov::op::v0::MatMul>(model), 4u);
}

// Negative: when the V-branch slice offsets do not match the score Concat
// split point (e.g. K splits 16 into 8 + 8 while V splits 16 into 4 + 12),
// concatenating K and V would misalign keys and values, so the pass must
// reject the fusion.
TEST(FuseSplitAttentionToSDPATest,
     DoesNotFuseWhenVBranchSliceOffsetsMisalignedWithConcatSplit) {
  using ov::op::v0::Concat;
  using ov::op::v0::Constant;
  using ov::op::v0::MatMul;
  using ov::op::v0::Parameter;
  using ov::op::v1::Add;
  using ov::op::v8::Slice;
  using ov::op::v8::Softmax;

  const auto f = ov::element::f32;
  auto q = std::make_shared<Parameter>(f, ov::Shape{1, 8, 1, 256});
  auto k_cache = std::make_shared<Parameter>(f, ov::Shape{1, 8, 8, 256});
  auto k_slice = std::make_shared<Parameter>(f, ov::Shape{1, 8, 8, 256});
  auto v_cache = std::make_shared<Parameter>(f, ov::Shape{1, 8, 256, 4});
  auto v_slice = std::make_shared<Parameter>(f, ov::Shape{1, 8, 256, 12});
  auto mask = std::make_shared<Parameter>(f, ov::Shape{1, 1, 1, 16});

  auto qk_cache = std::make_shared<MatMul>(q, k_cache, false, true);
  auto qk_slice = std::make_shared<MatMul>(q, k_slice, false, true);
  auto scores =
      std::make_shared<Concat>(ov::OutputVector{qk_cache, qk_slice}, -1);
  auto masked = std::make_shared<Add>(scores, mask);
  auto probs = std::make_shared<Softmax>(masked, -1);

  auto step = Constant::create(ov::element::i64, ov::Shape{1}, {1});
  auto axis = Constant::create(ov::element::i64, ov::Shape{1}, {-1});
  auto slice_past = std::make_shared<Slice>(
      probs, Constant::create(ov::element::i64, ov::Shape{1}, {0}),
      Constant::create(ov::element::i64, ov::Shape{1}, {4}), step, axis);
  auto slice_cur = std::make_shared<Slice>(
      probs, Constant::create(ov::element::i64, ov::Shape{1}, {4}),
      Constant::create(ov::element::i64, ov::Shape{1}, {16}), step, axis);

  auto pv_cache = std::make_shared<MatMul>(slice_past, v_cache, false, true);
  auto pv_slice = std::make_shared<MatMul>(slice_cur, v_slice, false, true);
  auto out = std::make_shared<Add>(pv_cache, pv_slice);
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{std::make_shared<ov::op::v0::Result>(out)},
      ov::ParameterVector{q, k_cache, k_slice, v_cache, v_slice, mask},
      "misaligned_split_attention");

  NpuOptimizer()
      .SetCastIntegerSignToFloat(false)
      .SetFuseSplitAttentionToSDPA(true)
      .Run(model);

  EXPECT_EQ(CountOps<ov::op::v13::ScaledDotProductAttention>(model), 0u);
  EXPECT_EQ(CountOps<ov::op::v0::MatMul>(model), 4u);
}

// Clamps -inf entries in the attention mask to a finite floor (-3e4) and uses
// the same -3e4 bias on padded key positions so flash-attention tiles whose
// positions are all masked do not produce exp(-inf - (-inf)) = NaN, while
// still underflowing exp(-3e4) to 0 and matching the split-cache reference.
TEST(FuseSplitAttentionToSDPATest,
     ClampsNegativeInfinityMaskToFiniteFloorAndPadsWithSameFloor) {
  auto reference = BuildSplitCacheAttention(/*s_past=*/7, /*s_cur=*/1);
  auto fused = reference->clone();

  NpuOptimizer()
      .SetCastIntegerSignToFloat(false)
      .SetFuseSplitAttentionToSDPA(true)
      .Run(fused);
  auto sdpa = FindSdpa(fused);
  ASSERT_NE(sdpa, nullptr) << "fusion did not fire";
  EXPECT_EQ(CountOps<ov::op::v1::Maximum>(fused), 1u);
  EXPECT_EQ(CountOps<ov::op::v1::Pad>(fused), 3u);

  const size_t num_inputs = reference->inputs().size();
  ASSERT_EQ(num_inputs, 6u);
  ov::TensorVector inputs;
  for (size_t i = 0; i < num_inputs; ++i) {
    const auto& port = reference->input(i);
    ov::Tensor t(port.get_element_type(), port.get_shape());
    FillRandom(t, static_cast<uint32_t>(i + 30));
    if (i == 5) {
      // Mask shape is [1, 1, 1, 8]: leave positions 0, 1, and 7 unmasked (0.0)
      // and mask positions 2..6 with -inf.
      auto* mask_data = t.data<float>();
      for (size_t k = 0; k < t.get_size(); ++k) {
        mask_data[k] = (k == 0 || k == 1 || k == 7)
                           ? 0.0f
                           : -std::numeric_limits<float>::infinity();
      }
    }
    inputs.push_back(t);
  }

  auto ref_out = EvaluateModel(reference, inputs);
  auto fused_out = EvaluateModel(fused, inputs);
  ASSERT_EQ(ref_out.get_size(), fused_out.get_size());
  const auto* a = ref_out.data<float>();
  const auto* b = fused_out.data<float>();
  float max_abs_diff = 0.0f;
  for (size_t i = 0; i < ref_out.get_size(); ++i) {
    max_abs_diff = std::max(max_abs_diff, std::abs(a[i] - b[i]));
    ASSERT_TRUE(std::isfinite(b[i])) << "fused output not finite at " << i;
  }
  EXPECT_LT(max_abs_diff, 1e-4f);

  // Also inspect the clamped+padded mask tensor fed into SDPA input 3: all 16
  // entries (8 original + 8 padded) must be finite, with -inf replaced by
  // -3e4f and padded positions filled with -3e4f.
  auto mask_probe = fused->clone();
  auto probe_sdpa = FindSdpa(mask_probe);
  ASSERT_NE(probe_sdpa, nullptr);
  mask_probe->add_results(
      {std::make_shared<ov::op::v0::Result>(probe_sdpa->input_value(3))});
  ov::TensorVector probe_outputs;
  for (const auto& out : mask_probe->outputs()) {
    probe_outputs.emplace_back(out.get_element_type(), out.get_shape());
  }
  ov::pass::Manager pm;
  pm.register_pass<ov::pass::ScaledDotProductAttentionDecomposition>();
  pm.run_passes(mask_probe);
  ASSERT_TRUE(mask_probe->evaluate(probe_outputs, inputs));
  const auto& sdpa_mask = probe_outputs.at(1);
  ASSERT_EQ(sdpa_mask.get_size(), 16u);
  const auto* m = sdpa_mask.data<float>();
  for (size_t k = 0; k < 16u; ++k) {
    EXPECT_TRUE(std::isfinite(m[k])) << "mask entry " << k << " is not finite";
    if (k == 0 || k == 1 || k == 7) {
      EXPECT_FLOAT_EQ(m[k], 0.0f);
    } else {
      EXPECT_FLOAT_EQ(m[k], -3e4f) << "at index " << k;
    }
  }
}

// Exercises the operand-shape inspection path when pad_kv_to_alignment is
// disabled: an unaligned merged KV length (7 + 1 = 8, not a multiple of 16)
// must be rejected after logging the candidate shapes at LITERT_DEBUG, whereas
// an already 16-aligned merged KV length (8 + 8 = 16) logs its shapes and fuses
// without inserting Pad ops.
TEST(FuseSplitAttentionToSDPATest,
     RespectsSdpaPadKvToAlignmentKnobOnUnalignedAndAlignedShapes) {
  ASSERT_EQ(LiteRtUseSinkLogger(), kLiteRtStatusOk);
  LiteRtLogger logger = LiteRtGetDefaultLogger();
  LiteRtLogSeverity prev_severity = kLiteRtLogSeverityInfo;
  ASSERT_EQ(LiteRtGetMinLoggerSeverity(logger, &prev_severity),
            kLiteRtStatusOk);
  ASSERT_EQ(LiteRtSetMinLoggerSeverity(logger, kLiteRtLogSeverityDebug),
            kLiteRtStatusOk);
  ASSERT_EQ(LiteRtClearSinkLogger(logger), kLiteRtStatusOk);

  auto unaligned = BuildSplitCacheAttention(/*s_past=*/7, /*s_cur=*/1);
  NpuOptimizer()
      .SetCastIntegerSignToFloat(false)
      .SetFuseSplitAttentionToSDPA(true)
      .SetSdpaPadKvToAlignment(false)
      .Run(unaligned);
  EXPECT_EQ(CountOps<ov::op::v13::ScaledDotProductAttention>(unaligned), 0u);
  EXPECT_EQ(CountOps<ov::op::v0::MatMul>(unaligned), 4u);

  auto aligned = BuildSplitCacheAttention(/*s_past=*/8, /*s_cur=*/8);
  NpuOptimizer()
      .SetCastIntegerSignToFloat(false)
      .SetFuseSplitAttentionToSDPA(true)
      .SetSdpaPadKvToAlignment(false)
      .Run(aligned);
  EXPECT_EQ(CountOps<ov::op::v13::ScaledDotProductAttention>(aligned), 1u);
  EXPECT_EQ(CountOps<ov::op::v1::Pad>(aligned), 0u);

  size_t log_count = 0;
  ASSERT_EQ(LiteRtGetSinkLoggerSize(logger, &log_count), kLiteRtStatusOk);
  bool saw_unaligned_shapes = false;
  bool saw_aligned_shapes = false;
  for (size_t i = 0; i < log_count; ++i) {
    const char* msg = nullptr;
    ASSERT_EQ(LiteRtGetSinkLoggerMessage(logger, i, &msg), kLiteRtStatusOk);
    ASSERT_NE(msg, nullptr);
    const std::string line(msg);
    if (line.find("shapes Q=[1,8,1,256] K_cache=[1,8,7,256] K_new=[1,8,1,256] "
                  "V_cache=[1,8,256,7] V_new=[1,8,256,1] mask=[1,1,1,8]") !=
        std::string::npos) {
      saw_unaligned_shapes = true;
    }
    if (line.find("shapes Q=[1,8,8,256] K_cache=[1,8,8,256] K_new=[1,8,8,256] "
                  "V_cache=[1,8,256,8] V_new=[1,8,256,8] mask=[1,1,8,16]") !=
        std::string::npos) {
      saw_aligned_shapes = true;
    }
  }
  EXPECT_EQ(LiteRtSetMinLoggerSeverity(logger, prev_severity), kLiteRtStatusOk);
  EXPECT_EQ(LiteRtUseStandardLogger(), kLiteRtStatusOk);

  EXPECT_TRUE(saw_unaligned_shapes) << "expected FuseSplitAttentionToSDPA "
                                       "DEBUG shape log for s_past=7,s_cur=1";
  EXPECT_TRUE(saw_aligned_shapes) << "expected FuseSplitAttentionToSDPA DEBUG "
                                     "shape log for s_past=8,s_cur=8";
}

// DecomposeBooleanConstant rewrites sizable boolean Constants into
// Constant(u8) -> Convert(boolean), preserving the Constant's friendly_name and
// backing buffer pointer (zero-copy) for WeightBank lookup, and leaves tiny
// ones alone.
TEST(DecomposeBooleanConstantTest, RewritesOnlySizableBooleanConstants) {
  const auto f = ov::element::f32;
  auto data_big = std::make_shared<ov::op::v0::Parameter>(f, ov::Shape{32});
  auto data_small = std::make_shared<ov::op::v0::Parameter>(f, ov::Shape{4});
  std::vector<char> big_bits(32);
  for (size_t i = 0; i < big_bits.size(); ++i) big_bits[i] = (i % 3 == 0);
  auto big_cond = ov::op::v0::Constant::create(ov::element::boolean,
                                               ov::Shape{32}, big_bits);
  big_cond->set_friendly_name("big_bool_mask");
  const void* original_big_ptr = big_cond->get_data_ptr();
  auto small_cond = ov::op::v0::Constant::create(
      ov::element::boolean, ov::Shape{4}, std::vector<char>{1, 0, 1, 0});
  auto zero = ov::op::v0::Constant::create(f, ov::Shape{}, {0.0f});
  auto select_big =
      std::make_shared<ov::op::v1::Select>(big_cond, data_big, zero);
  auto select_small =
      std::make_shared<ov::op::v1::Select>(small_cond, data_small, zero);
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{std::make_shared<ov::op::v0::Result>(select_big),
                       std::make_shared<ov::op::v0::Result>(select_small)},
      ov::ParameterVector{data_big, data_small}, "boolean_constants");

  NpuOptimizer()
      .SetCastIntegerSignToFloat(false)
      .SetDecomposeBooleanConstants(true)
      .Run(model);

  size_t large_boolean_constants = 0;
  size_t small_boolean_constants = 0;
  std::shared_ptr<ov::op::v0::Constant> rewritten_u8_const;
  std::shared_ptr<ov::op::v0::Convert> convert_node;
  for (const auto& node : model->get_ops()) {
    if (auto c = std::dynamic_pointer_cast<ov::op::v0::Constant>(node)) {
      if (c->get_element_type() == ov::element::boolean) {
        if (c->get_byte_size() >= 16) {
          ++large_boolean_constants;
        } else {
          ++small_boolean_constants;
        }
      } else if (c->get_element_type() == ov::element::u8) {
        rewritten_u8_const = c;
      }
    } else if (auto conv =
                   std::dynamic_pointer_cast<ov::op::v0::Convert>(node)) {
      convert_node = conv;
    }
  }
  EXPECT_EQ(large_boolean_constants, 0u);
  EXPECT_EQ(small_boolean_constants, 1u);
  EXPECT_EQ(CountOps<ov::op::v0::Convert>(model), 1u);
  ASSERT_NE(rewritten_u8_const, nullptr);
  EXPECT_EQ(rewritten_u8_const->get_friendly_name(), "big_bool_mask");
  EXPECT_EQ(rewritten_u8_const->get_data_ptr(), original_big_ptr);
  ASSERT_NE(convert_node, nullptr);
  EXPECT_EQ(convert_node->get_friendly_name(), "big_bool_mask_to_bool");
}

// Verifies that DecomposeBooleanConstant runs after ConstantFolding so the
// `Constant(u8) -> Convert(boolean)` pair is not folded back to a boolean
// Constant, that output tensor names are transferred to the Convert output, and
// that the rewritten graph evaluates to identical values.
TEST(DecomposeBooleanConstantTest,
     SurvivesConstantFoldingAndPreservesSemanticsAndTensorNames) {
  const auto f = ov::element::f32;
  auto x = std::make_shared<ov::op::v0::Parameter>(f, ov::Shape{32});
  auto y = std::make_shared<ov::op::v0::Parameter>(f, ov::Shape{32});
  std::vector<char> bits(32);
  for (size_t i = 0; i < bits.size(); ++i) bits[i] = (i % 2 == 0);
  auto cond =
      ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{32}, bits);
  cond->set_friendly_name("shared_attn_mask");
  cond->output(0).set_names({"shared_attn_mask_tensor"});
  auto sel = std::make_shared<ov::op::v1::Select>(cond, x, y);
  auto reference = std::make_shared<ov::Model>(
      ov::ResultVector{std::make_shared<ov::op::v0::Result>(sel)},
      ov::ParameterVector{x, y}, "select_with_bool_const");
  auto rewritten = reference->clone();

  NpuOptimizer()
      .SetCastIntegerSignToFloat(false)
      .SetConstantFold(true)
      .SetDecomposeBooleanConstants(true)
      .Run(rewritten);

  EXPECT_EQ(CountOps<ov::op::v0::Convert>(rewritten), 1u);
  std::shared_ptr<ov::op::v0::Convert> convert_node;
  for (const auto& node : rewritten->get_ops()) {
    if (auto c = std::dynamic_pointer_cast<ov::op::v0::Constant>(node)) {
      EXPECT_EQ(c->get_element_type(), ov::element::u8);
      EXPECT_EQ(c->get_friendly_name(), "shared_attn_mask");
    } else if (auto conv =
                   std::dynamic_pointer_cast<ov::op::v0::Convert>(node)) {
      convert_node = conv;
    }
  }
  ASSERT_NE(convert_node, nullptr);
  EXPECT_EQ(
      convert_node->output(0).get_names().count("shared_attn_mask_tensor"),
      1u);

  ov::Tensor tx(f, ov::Shape{32});
  ov::Tensor ty(f, ov::Shape{32});
  FillRandom(tx, /*seed=*/41);
  FillRandom(ty, /*seed=*/42);
  auto ref_out = EvaluateModel(reference, {tx, ty});
  auto rew_out = EvaluateModel(rewritten, {tx, ty});
  ASSERT_EQ(ref_out.get_size(), rew_out.get_size());
  const auto* a = ref_out.data<float>();
  const auto* b = rew_out.data<float>();
  for (size_t i = 0; i < ref_out.get_size(); ++i) {
    EXPECT_FLOAT_EQ(a[i], b[i]) << "at index " << i;
  }
}

TEST(DecomposeBooleanConstantTest,
     DisabledByDefaultLeavesBooleanConstantsIntact) {
  const auto f = ov::element::f32;
  auto x = std::make_shared<ov::op::v0::Parameter>(f, ov::Shape{32});
  std::vector<char> bits(32, 1);
  auto cond =
      ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{32}, bits);
  auto zero = ov::op::v0::Constant::create(f, ov::Shape{}, {0.0f});
  auto sel = std::make_shared<ov::op::v1::Select>(cond, x, zero);
  auto model = std::make_shared<ov::Model>(
      ov::ResultVector{std::make_shared<ov::op::v0::Result>(sel)},
      ov::ParameterVector{x}, "bool_default_off");

  NpuOptimizer().SetCastIntegerSignToFloat(false).Run(model);

  EXPECT_EQ(CountOps<ov::op::v0::Convert>(model), 0u);
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
  // Gather-K weight selection (gate, up, and down are each gathered
  // independently: 2 Gathers per weight -- flattened packed rows + scale --
  // times 3 weights) and a single weighted ReduceSum over the K selected
  // experts. The router's weight-sum ReduceSum survives (it still feeds the
  // K-weighting).
  EXPECT_EQ(CountOps<ov::op::v1::Equal>(model), 0u);
  // ExpandExpertRowIndices emits one row-offset Add per gathered weight
  // (gate + up + down).
  EXPECT_EQ(CountOps<ov::op::v1::Add>(model), 3u);
  EXPECT_EQ(CountOps<ov::op::v8::Gather>(model), 6u);
  EXPECT_EQ(CountOps<ov::op::v1::ReduceSum>(model), 2u);
  EXPECT_EQ(CountOps<ov::op::v0::MatMul>(model), 3u);
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

// When the router's TopK indices have a statically-known batch dim != 1 (a
// prefill-shaped call), the single-token Gather-K form doesn't apply (no
// single K-of-N subset is selected across all tokens), so this must dispatch
// to the multi-token chunk path (RegroupAndRewriteChunk) instead of the
// single-token Gather-K one: the masked Equal chain is still eliminated, but
// via a dense N-expert computation, and the only Gather introduced is the
// chunk-size extraction from ShapeOf(topk_indices).
TEST(MoEGatherRewriteTest, RewritesPrefillShapedRouterViaChunkPath) {
  auto model = BuildDenseMoeGraph(/*k=*/2, SequentialExpertIds(4), /*batch=*/2);

  NpuOptimizer().SetCastIntegerSignToFloat(false).SetEnableMoeGather(true).Run(
      model);

  EXPECT_EQ(CountOps<ov::op::v1::Equal>(model), 0u);
  EXPECT_EQ(CountOps<ov::op::v8::Gather>(model), 1u);
}

// Edge case: a degenerate batch=0 call (zero tokens). The chunk path reads
// the chunk dim from ShapeOf rather than assuming a static size, so a
// statically-zero batch is just another chunk size -- it must still take the
// chunk path rather than mis-dispatching to the single-token Gather-K one.
TEST(MoEGatherRewriteTest, RewritesZeroBatchViaChunkPath) {
  auto model = BuildDenseMoeGraph(/*k=*/2, SequentialExpertIds(4), /*batch=*/0);

  NpuOptimizer().SetCastIntegerSignToFloat(false).SetEnableMoeGather(true).Run(
      model);

  EXPECT_EQ(CountOps<ov::op::v1::Equal>(model), 0u);
  EXPECT_EQ(CountOps<ov::op::v8::Gather>(model), 1u);
}

// Mixed shapes: the NPUW config picked from the classification is
// compile-wide, so rewriting only the layers of one shape would leave that
// config wrong for the rest. Both layers must be left dense instead.
TEST(MoEGatherRewriteTest, LeavesGraphDenseWhenLayersDisagreeOnChunkShape) {
  auto single = BuildDenseMoeGraph(/*k=*/2, SequentialExpertIds(4),
                                   /*batch=*/1);
  auto multi = BuildDenseMoeGraph(/*k=*/2, SequentialExpertIds(4),
                                  /*batch=*/2);
  ov::ResultVector results = single->get_results();
  for (const auto& r : multi->get_results()) results.push_back(r);
  ov::ParameterVector params = single->get_parameters();
  for (const auto& p : multi->get_parameters()) params.push_back(p);
  auto mixed = std::make_shared<ov::Model>(results, params, "mixed_moe");

  const auto result = NpuOptimizer()
                          .SetCastIntegerSignToFloat(false)
                          .SetEnableMoeGather(true)
                          .Run(mixed);

  // Neither layer was touched, and the verdict falls back to single-token.
  EXPECT_EQ(CountOps<ov::op::v1::Equal>(mixed), 8u);
  EXPECT_EQ(CountOps<ov::op::v8::Gather>(mixed), 0u);
  EXPECT_FALSE(result.moe_is_multi_token_chunk);
}

// Regression test for the Run() Result plumbing: NpuOptimizer::Run() must
// surface MoEGatherRewrite's is_multi_token_chunk() classification regardless
// of which strategy actually rewrote the layer (see
// RewritesPrefillShapedRouterViaChunkPath) -- classification happens before
// the per-layer rewrite dispatch, not as a side effect of a successful
// rewrite.
TEST(NpuOptimizerRunResultTest, ReportsMultiTokenChunkForBatchGreaterThanOne) {
  auto model = BuildDenseMoeGraph(/*k=*/2, SequentialExpertIds(4), /*batch=*/2);

  auto result = NpuOptimizer()
                    .SetCastIntegerSignToFloat(false)
                    .SetEnableMoeGather(true)
                    .Run(model);

  EXPECT_TRUE(result.moe_is_multi_token_chunk);
}

TEST(NpuOptimizerRunResultTest, ReportsSingleTokenChunkForBatchOne) {
  auto model = BuildDenseMoeGraph(/*k=*/2, SequentialExpertIds(4), /*batch=*/1);

  auto result = NpuOptimizer()
                    .SetCastIntegerSignToFloat(false)
                    .SetEnableMoeGather(true)
                    .Run(model);

  EXPECT_FALSE(result.moe_is_multi_token_chunk);
}

TEST(NpuOptimizerRunResultTest, DefaultsToSingleTokenWhenMoeGatherDisabled) {
  auto model = BuildDenseMoeGraph(/*k=*/2, SequentialExpertIds(4), /*batch=*/2);

  auto result = NpuOptimizer().SetCastIntegerSignToFloat(false).Run(model);

  EXPECT_FALSE(result.moe_is_multi_token_chunk);
}

// Runs |reference| and its MoEGatherRewrite-rewritten clone on identical
// {hidden, logits} inputs and returns the max abs difference between their
// outputs. Asserts the rewrite actually fired (|expected_gathers|: 6 for the
// single-token gather path, 1 for the chunk path's ShapeOf extraction) so a
// silent no-op rewrite can't masquerade as "numerically matches".
float MaxAbsDiffAfterRewrite(const std::shared_ptr<ov::Model>& reference,
                             const ov::Tensor& hidden, const ov::Tensor& logits,
                             size_t expected_gathers = 6u) {
  auto rewritten = reference->clone();
  NpuOptimizer().SetCastIntegerSignToFloat(false).SetEnableMoeGather(true).Run(
      rewritten);
  EXPECT_EQ(CountOps<ov::op::v8::Gather>(rewritten), expected_gathers)
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

// The multi-token chunk rewrite (dense N-expert batched MatMul + scattered
// [chunk,N] routing matrix) must reproduce the dense masked reference just as
// the single-token gather path does. Op counts alone can't show this: the
// chunk path introduces a whole new Tile/ScatterElementsUpdate/Transpose
// chain whose only correctness evidence is the numbers.
TEST(MoEGatherRewriteTest, ChunkPathNumericallyMatchesDenseComputation) {
  auto reference =
      BuildDenseMoeGraph(/*k=*/2, SequentialExpertIds(6), /*batch=*/3);

  ov::Tensor hidden(ov::element::f32, ov::Shape{3, kMoeHiddenDim});
  FillRandom(hidden, /*seed=*/3);
  ov::Tensor logits(ov::element::f32, ov::Shape{3, 6});
  FillRandom(logits, /*seed=*/4);

  EXPECT_LT(MaxAbsDiffAfterRewrite(reference, hidden, logits,
                                   /*expected_gathers=*/1u),
            1e-3f)
      << "multi-token chunk rewrite diverges from dense masked-experts "
         "reference";
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
  ASSERT_EQ(CountOps<ov::op::v8::Gather>(model), 6u) << "rewrite did not fire";

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

// Builds Add(Multiply(x, shared), shared): one |num_elems|-element Constant
// feeding two consumers, which is what SplitSharedConstants targets.
std::shared_ptr<ov::Model> BuildTwoConsumerConstantModel(size_t num_elems) {
  auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                   ov::Shape{num_elems});
  std::vector<float> values(num_elems, 2.0f);
  auto shared = ov::op::v0::Constant::create(ov::element::f32,
                                             ov::Shape{num_elems}, values);
  auto mul = std::make_shared<ov::op::v1::Multiply>(x, shared);
  auto add = std::make_shared<ov::op::v1::Add>(mul, shared);
  return std::make_shared<ov::Model>(ov::OutputVector{add->output(0)},
                                     ov::ParameterVector{x}, "two_consumer");
}

// Each consumer ends up on its own Constant node, so downstream passes (and
// NPUW's per-subgraph weight handling) see independent weights rather than
// one node shared across unrelated consumers.
TEST(SplitSharedConstantsTest, GivesEachConsumerItsOwnConstant) {
  auto model = BuildTwoConsumerConstantModel(256);
  ASSERT_EQ(CountOps<ov::op::v0::Constant>(model), 1u);

  NpuOptimizer()
      .SetCastIntegerSignToFloat(false)
      .SetSplitSharedConstants(true)
      .Run(model);

  EXPECT_EQ(CountOps<ov::op::v0::Constant>(model), 2u);
}

// Disabled by default: without the opt-in the graph must be left alone.
TEST(SplitSharedConstantsTest, DoesNothingWhenDisabled) {
  auto model = BuildTwoConsumerConstantModel(256);

  NpuOptimizer().SetCastIntegerSignToFloat(false).Run(model);

  EXPECT_EQ(CountOps<ov::op::v0::Constant>(model), 1u);
}

// Only 256- and 512-element Constants are split today (see
// SplitSharedConstants); any other size is left shared.
TEST(SplitSharedConstantsTest, LeavesOtherSizesShared) {
  auto model = BuildTwoConsumerConstantModel(128);

  NpuOptimizer()
      .SetCastIntegerSignToFloat(false)
      .SetSplitSharedConstants(true)
      .Run(model);

  EXPECT_EQ(CountOps<ov::op::v0::Constant>(model), 1u);
}

// A single-consumer Constant is already private, so it must not be cloned.
TEST(SplitSharedConstantsTest, LeavesSingleConsumerConstantAlone) {
  auto x =
      std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{256});
  std::vector<float> values(256, 2.0f);
  auto only =
      ov::op::v0::Constant::create(ov::element::f32, ov::Shape{256}, values);
  auto mul = std::make_shared<ov::op::v1::Multiply>(x, only);
  auto model = std::make_shared<ov::Model>(
      ov::OutputVector{mul->output(0)}, ov::ParameterVector{x}, "one_consumer");

  NpuOptimizer()
      .SetCastIntegerSignToFloat(false)
      .SetSplitSharedConstants(true)
      .Run(model);

  EXPECT_EQ(CountOps<ov::op::v0::Constant>(model), 1u);
}

// Builds the "GQA ratio folded into the sequence axis" attention the LiteRT
// generative export emits for a |hkv|-KV-head / |hkv * r|-query-head layer:
// Q is reshaped so the r query-head groups stack along the sequence axis, and
// the head-invariant additive mask is Concat-replicated r times to line back
// up with it. This is what UnfoldGqaSequenceFold matches.
std::shared_ptr<ov::Model> BuildGqaSequenceFoldAttention(int64_t hkv, int64_t r,
                                                         int64_t s,
                                                         int64_t kv_len,
                                                         int64_t head_dim) {
  using ov::op::v0::Constant;
  using ov::op::v0::Parameter;
  const int64_t hq = hkv * r;
  const auto f = ov::element::f32;
  auto dim = [](int64_t v) { return static_cast<size_t>(v); };

  auto q = std::make_shared<Parameter>(
      f, ov::Shape{1, dim(hq), dim(s), dim(head_dim)});
  auto k = std::make_shared<Parameter>(
      f, ov::Shape{1, dim(hkv), dim(kv_len), dim(head_dim)});
  auto v = std::make_shared<Parameter>(
      f, ov::Shape{1, dim(hkv), dim(kv_len), dim(head_dim)});
  auto mask =
      std::make_shared<Parameter>(f, ov::Shape{1, 1, dim(s), dim(kv_len)});

  auto fold_shape =
      Constant::create(ov::element::i64, ov::Shape{4},
                       std::vector<int64_t>{1, hkv, s * r, head_dim});
  auto q_folded = std::make_shared<ov::op::v1::Reshape>(q, fold_shape,
                                                        /*special_zero=*/false);

  ov::OutputVector mask_copies(static_cast<size_t>(r), mask->output(0));
  auto mask_tiled =
      std::make_shared<ov::op::v0::Concat>(mask_copies, /*axis=*/2);

  auto scale = Constant::create(f, ov::Shape{}, std::vector<float>{1.0f});
  auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(
      ov::OutputVector{q_folded, k, v, mask_tiled, scale}, /*causal=*/false);

  auto unfold_shape = Constant::create(
      ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, hq, s, head_dim});
  auto unfolded = std::make_shared<ov::op::v1::Reshape>(sdpa, unfold_shape,
                                                        /*special_zero=*/false);

  return std::make_shared<ov::Model>(ov::OutputVector{unfolded->output(0)},
                                     ov::ParameterVector{q, k, v, mask},
                                     "gqa_sequence_fold");
}

// The rewrite is only worth anything if it is numerically a no-op. Both forms
// must map query head h to KV head h / r; this pins that down end to end,
// along with the head-parallel shapes the rewrite is supposed to produce.
TEST(UnfoldGqaSequenceFoldTest, NumericallyMatchesSequenceFoldedForm) {
  for (const int64_t r : {2, 4}) {
    auto reference = BuildGqaSequenceFoldAttention(/*hkv=*/2, r, /*s=*/3,
                                                   /*kv_len=*/6,
                                                   /*head_dim=*/4);
    auto rewritten = reference->clone();

    NpuOptimizer()
        .SetCastIntegerSignToFloat(false)
        .SetUnfoldGqaSequenceFold(true)
        .Run(rewritten);
    ASSERT_EQ(CountOps<ov::op::v3::Broadcast>(rewritten), 2u)
        << "rewrite did not fire for r=" << r;

    // The mask tile is gone and SDPA now runs head-parallel: Q keeps all
    // hq = hkv * r heads at the original sequence length, K/V are broadcast
    // up to match, and the mask is back to its untiled length.
    EXPECT_EQ(CountOps<ov::op::v0::Concat>(rewritten), 0u);
    const size_t hq = 2 * static_cast<size_t>(r);
    auto sdpa = FindSdpa(rewritten);
    ASSERT_NE(sdpa, nullptr);
    EXPECT_EQ(sdpa->get_input_shape(0), (ov::Shape{1, hq, 3, 4}));
    EXPECT_EQ(sdpa->get_input_shape(1), (ov::Shape{1, hq, 6, 4}));
    EXPECT_EQ(sdpa->get_input_shape(2), (ov::Shape{1, hq, 6, 4}));
    EXPECT_EQ(sdpa->get_input_shape(3), (ov::Shape{1, 1, 3, 6}));

    ov::Core core;
    auto ref_compiled = core.compile_model(reference, "CPU");
    auto new_compiled = core.compile_model(rewritten, "CPU");
    auto ref_req = ref_compiled.create_infer_request();
    auto new_req = new_compiled.create_infer_request();

    // Inputs are ordered as constructed: {q, k, v, mask}.
    const size_t num_inputs = reference->inputs().size();
    ASSERT_EQ(num_inputs, rewritten->inputs().size());
    for (size_t i = 0; i < num_inputs; ++i) {
      const auto& port = reference->input(i);
      ov::Tensor t(port.get_element_type(), port.get_shape());
      FillRandom(t, static_cast<uint32_t>(i + 1));
      ref_req.set_input_tensor(i, t);
      new_req.set_input_tensor(i, t);
    }

    ref_req.infer();
    new_req.infer();

    auto ref_out = ref_req.get_output_tensor(0);
    auto new_out = new_req.get_output_tensor(0);
    ASSERT_EQ(ref_out.get_shape(), new_out.get_shape());
    const auto* ref_data = ref_out.data<float>();
    const auto* new_data = new_out.data<float>();
    for (size_t i = 0; i < ref_out.get_size(); ++i) {
      EXPECT_NEAR(new_data[i], ref_data[i],
                  std::abs(ref_data[i]) * 1e-5f + 1e-6f)
          << "output[" << i << "] diverges for r=" << r
          << " -- the two forms disagree on which KV head a query head reads";
    }
  }
}

// Disabled by default: without the opt-in the graph must be left alone.
TEST(UnfoldGqaSequenceFoldTest, DoesNothingWhenDisabled) {
  auto model = BuildGqaSequenceFoldAttention(/*hkv=*/2, /*r=*/4, /*s=*/3,
                                             /*kv_len=*/6, /*head_dim=*/4);

  NpuOptimizer().SetCastIntegerSignToFloat(false).Run(model);

  EXPECT_EQ(CountOps<ov::op::v0::Concat>(model), 1u);
  EXPECT_EQ(CountOps<ov::op::v3::Broadcast>(model), 0u);
}

// A mask built from genuinely different per-group slices is NOT a replicated
// head-invariant mask, so folding it back would change the math. Leave it.
TEST(UnfoldGqaSequenceFoldTest, LeavesNonReplicatedMaskAlone) {
  using ov::op::v0::Constant;
  using ov::op::v0::Parameter;
  const auto f = ov::element::f32;
  auto q = std::make_shared<Parameter>(f, ov::Shape{1, 4, 3, 4});
  auto k = std::make_shared<Parameter>(f, ov::Shape{1, 2, 6, 4});
  auto v = std::make_shared<Parameter>(f, ov::Shape{1, 2, 6, 4});
  // Two independent mask halves rather than one tensor repeated twice.
  auto mask_a = std::make_shared<Parameter>(f, ov::Shape{1, 1, 3, 6});
  auto mask_b = std::make_shared<Parameter>(f, ov::Shape{1, 1, 3, 6});

  auto fold_shape = Constant::create(ov::element::i64, ov::Shape{4},
                                     std::vector<int64_t>{1, 2, 6, 4});
  auto q_folded = std::make_shared<ov::op::v1::Reshape>(q, fold_shape, false);
  auto mask = std::make_shared<ov::op::v0::Concat>(
      ov::OutputVector{mask_a, mask_b}, /*axis=*/2);
  auto scale = Constant::create(f, ov::Shape{}, std::vector<float>{1.0f});
  auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(
      ov::OutputVector{q_folded, k, v, mask, scale}, false);
  auto unfold_shape = Constant::create(ov::element::i64, ov::Shape{4},
                                       std::vector<int64_t>{1, 4, 3, 4});
  auto unfolded =
      std::make_shared<ov::op::v1::Reshape>(sdpa, unfold_shape, false);
  auto model = std::make_shared<ov::Model>(
      ov::OutputVector{unfolded->output(0)},
      ov::ParameterVector{q, k, v, mask_a, mask_b}, "distinct_mask_halves");

  NpuOptimizer()
      .SetCastIntegerSignToFloat(false)
      .SetUnfoldGqaSequenceFold(true)
      .Run(model);

  EXPECT_EQ(CountOps<ov::op::v0::Concat>(model), 1u);
  EXPECT_EQ(CountOps<ov::op::v3::Broadcast>(model), 0u);
}

}  // namespace
}  // namespace openvino
}  // namespace litert
