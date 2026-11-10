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

#include "litert/vendors/mediatek/compiler/transformations/attention_chunk_transformation.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/compiler/cc/litert_builder.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/compiler/cc/litert_op_options.h"
#include "litert/core/model/model.h"
#include "litert/vendors/mediatek/compiler/transformations/orphan_cleanup_transformation.h"

namespace litert::mediatek {
namespace {

using ::litert::ElementType;
using ::litert::compiler::BatchMatmulOptions;
using ::litert::compiler::Builder;
using ::litert::compiler::ConcatenationOptions;
using ::litert::compiler::GetOptionsAs;
using ::litert::compiler::Op;
using ::litert::compiler::RankedTensorSpecBuilder;
using ::litert::compiler::ReshapeOptions;
using ::litert::compiler::SoftmaxOptions;
using ::litert::compiler::Tensor;

using Dims = std::vector<int32_t>;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

class GraphBuilder {
 public:
  GraphBuilder() : ctx_(LrtGetCompilerContext()), builder_(ctx_, &impl_) {}

  Tensor Typed(ElementType type, const Dims& dims) {
    auto t = builder_.BuildTensor(
        RankedTensorSpecBuilder(
            RankedTensorType(
                type,
                Layout(BuildLayout(dims.data(), dims.data() + dims.size()))))
            .Build());
    return *t;
  }

  Tensor Quant(ElementType type, const Dims& dims, float scale,
               int64_t zero_point) {
    auto t = builder_.BuildTensor(
        RankedTensorSpecBuilder(
            RankedTensorType(
                type,
                Layout(BuildLayout(dims.data(), dims.data() + dims.size()))))
            .WithPerTensorQuantization(
                LiteRtQuantizationPerTensor{scale, zero_point})
            .Build());
    return *t;
  }

  Tensor ConstI32(const std::vector<int32_t>& data) {
    Tensor t = Typed(ElementType::Int32, {static_cast<int32_t>(data.size())});
    EXPECT_TRUE(builder_.BuildWeights<int32_t>(absl::MakeConstSpan(data), t)
                    .HasValue());
    return t;
  }

  Op AddOp(LiteRtOpCode code, const std::vector<Tensor>& ins,
           const std::vector<Tensor>& outs) {
    auto op = builder_.BuildOp(code, ins, outs);
    return *op;
  }

  template <typename OptionsT>
  void SetOptions(const Op& op, OptionsT options) {
    EXPECT_TRUE(builder_.SetOpOptions(op, std::move(options)).HasValue());
  }

  void ApplyTo(LiteRtSubgraphT* subgraph) { impl_.ApplyChanges(subgraph); }

 private:
  const LiteRtCompilerContext* ctx_;
  LiteRtBuilderT impl_;
  Builder builder_;
};

const LiteRtCompilerContext* Ctx() { return LrtGetCompilerContext(); }

int CountOps(const LiteRtSubgraphT& subgraph, LiteRtOpCode code) {
  int count = 0;
  for (const auto* op : subgraph.Ops()) {
    if (op->OpCode() == code) ++count;
  }
  return count;
}

std::vector<Op> FindOps(const LiteRtSubgraphT& subgraph, LiteRtOpCode code) {
  std::vector<Op> res;
  for (auto* op : subgraph.Ops()) {
    if (op->OpCode() == code) res.emplace_back(Ctx(), op);
  }
  return res;
}

bool IsTopologicallySorted(const LiteRtSubgraphT& subgraph) {
  absl::flat_hash_set<LiteRtOp> seen;
  for (auto* op : subgraph.Ops()) {
    for (auto* in : op->Inputs()) {
      if (in == nullptr) continue;
      LiteRtOp def = in->DefiningOp();
      if (def != nullptr && !seen.contains(def)) return false;
    }
    seen.insert(op);
  }
  return true;
}

Dims TensorDims(const Tensor& t) {
  auto type = t.RankedTensorType();
  if (!type) return {};
  auto dims = type->Layout().Dimensions();
  return Dims(dims.begin(), dims.end());
}

void RunOrphanCleanup(LiteRtSubgraphT& subgraph) {
  bool changed = true;
  while (changed) {
    changed = false;
    std::vector<LiteRtOp> ops(subgraph.Ops().begin(), subgraph.Ops().end());
    for (LiteRtOp op : ops) {
      LiteRtBuilderT builder;
      if (OrphanCleanupTransformation(Ctx(), &builder, op) == kLiteRtStatusOk) {
        builder.ApplyChanges(&subgraph);
        changed = true;
        break;
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Test graph
//===----------------------------------------------------------------------===//

constexpr int32_t kH = 4;
constexpr int32_t kS = 6;
constexpr int32_t kD = 2;
constexpr int32_t kK = 6;
constexpr int32_t kDv = 3;
constexpr float kBeta = 1.0f;

// Distinct per-tensor scales so that quantization propagation is observable.
constexpr float kQScale = 0.1f;
constexpr float kKtScale = 0.2f;
constexpr float kVScale = 0.3f;
constexpr float kLogitScale = 0.4f;
constexpr float kR1Scale = 0.5f;
constexpr float kSmScale = 0.6f;
constexpr float kR2Scale = 0.7f;
constexpr float kOutScale = 0.8f;

struct AttnConfig {
  bool quantized = true;
  bool adj_y = false;
  bool extra_logit_user = false;
  bool float_v = false;          // Mixed types -> no match.
  bool v_after_bmm1 = false;     // V produced by an op placed after BMM1.
  bool with_tail = false;        // out -> Reshape -> Transpose -> Quantize.
  int32_t h = kH, s = kS, k = kK;  // Heads, queries, keys.
};

struct AttnGraph {
  Tensor q, kt, v, logits, out, q8;
  Op bmm1, bmm2;
};

AttnGraph BuildAttnGraph(LiteRtSubgraphT& subgraph, const AttnConfig& cfg) {
  GraphBuilder g;
  auto t = [&](const Dims& dims, float scale) {
    return cfg.quantized ? g.Quant(ElementType::Int16, dims, scale, 0)
                         : g.Typed(ElementType::Float32, dims);
  };
  const int32_t H = cfg.h, S = cfg.s, K = cfg.k;
  AttnGraph ag;
  ag.q = t({H, S, kD}, kQScale);
  ag.kt = t({H, kD, K}, kKtScale);
  Tensor v_src = cfg.float_v ? g.Typed(ElementType::Float32, {H, K, kDv})
                             : t({H, K, kDv}, kVScale);
  ag.logits = t({H, S, K}, kLogitScale);
  ag.bmm1 = g.AddOp(kLiteRtOpCodeTflBatchMatmul, {ag.q, ag.kt}, {ag.logits});
  BatchMatmulOptions bmm1_opts;
  bmm1_opts.adj_x = false;
  bmm1_opts.adj_y = cfg.adj_y;
  bmm1_opts.asymmetric_quantize_input = false;
  g.SetOptions(ag.bmm1, std::move(bmm1_opts));

  if (cfg.extra_logit_user) {
    Tensor side = t({H, S, K}, kLogitScale);
    g.AddOp(kLiteRtOpCodeTflAbs, {ag.logits}, {side});
  }

  Tensor r1 = t({1, H, S, K}, kR1Scale);
  Op reshape1 = g.AddOp(kLiteRtOpCodeTflReshape,
                        {ag.logits, g.ConstI32({1, H, S, K})}, {r1});
  ReshapeOptions r1_opts;
  r1_opts.new_shape = {1, H, S, K};
  g.SetOptions(reshape1, std::move(r1_opts));

  Tensor sm = t({1, H, S, K}, kSmScale);
  Op softmax = g.AddOp(kLiteRtOpCodeTflSoftmax, {r1}, {sm});
  SoftmaxOptions sm_opts;
  sm_opts.beta = kBeta;
  g.SetOptions(softmax, std::move(sm_opts));

  Tensor p = t({H, S, K}, kR2Scale);
  Op reshape2 =
      g.AddOp(kLiteRtOpCodeTflReshape, {sm, g.ConstI32({H, S, K})}, {p});
  ReshapeOptions r2_opts;
  r2_opts.new_shape = {H, S, K};
  g.SetOptions(reshape2, std::move(r2_opts));

  ag.v = v_src;
  if (cfg.v_after_bmm1) {
    ag.v = t({H, K, kDv}, kVScale);
    Op v_op = g.AddOp(kLiteRtOpCodeTflReshape,
                      {v_src, g.ConstI32({H, K, kDv})}, {ag.v});
    ReshapeOptions v_opts;
    v_opts.new_shape = {H, K, kDv};
    g.SetOptions(v_op, std::move(v_opts));
  }

  ag.out = t({H, S, kDv}, kOutScale);
  ag.bmm2 = g.AddOp(kLiteRtOpCodeTflBatchMatmul, {p, ag.v}, {ag.out});
  BatchMatmulOptions bmm2_opts;
  bmm2_opts.adj_x = false;
  bmm2_opts.adj_y = false;
  bmm2_opts.asymmetric_quantize_input = false;
  g.SetOptions(ag.bmm2, std::move(bmm2_opts));

  // Consumer of the attention output.
  if (cfg.with_tail) {
    Tensor r = t({1, H, S, kDv}, kOutScale);
    Op r_op = g.AddOp(kLiteRtOpCodeTflReshape,
                      {ag.out, g.ConstI32({1, H, S, kDv})}, {r});
    ReshapeOptions r_opts;
    r_opts.new_shape = {1, H, S, kDv};
    g.SetOptions(r_op, std::move(r_opts));
    Tensor tr = t({1, S, H, kDv}, kOutScale);
    g.AddOp(kLiteRtOpCodeTflTranspose, {r, g.ConstI32({0, 2, 1, 3})}, {tr});
    ag.q8 = g.Quant(ElementType::Int8, {1, S, H, kDv}, 0.05f, 0);
    g.AddOp(kLiteRtOpCodeTflQuantize, {tr}, {ag.q8});
    Tensor final_out = g.Quant(ElementType::Int8, {1, S, H, kDv}, 0.05f, 0);
    g.AddOp(kLiteRtOpCodeTflAbs, {ag.q8}, {final_out});
  } else {
    Tensor final_out = t({H, S, kDv}, kOutScale);
    g.AddOp(kLiteRtOpCodeTflAbs, {ag.out}, {final_out});
  }

  g.ApplyTo(&subgraph);
  return ag;
}

float ScaleOf(const Tensor& t) { return t.PerTensorQuantization().scale; }

// Runs the rewrite with a head_chunks x query_chunks grid and checks the
// resulting graph.
void RunPositive(int head_chunks, int query_chunks, const AttnConfig& cfg) {
  ResetOrphanRegistry();
  SetAttentionChunkConfig(head_chunks, query_chunks);
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  AttnGraph ag = BuildAttnGraph(subgraph, cfg);
  ASSERT_TRUE(IsTopologicallySorted(subgraph));

  LiteRtBuilderT builder;
  ASSERT_EQ(AttentionChunkTransformation(Ctx(), &builder, ag.bmm1.Get()),
            kLiteRtStatusOk);
  builder.ApplyChanges(&subgraph);
  EXPECT_TRUE(IsTopologicallySorted(subgraph));

  const int chunks = head_chunks * query_chunks;
  // The old head of the chain is still there until orphan cleanup.
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflBatchMatmul), 2 * chunks + 1);
  RunOrphanCleanup(subgraph);
  EXPECT_TRUE(IsTopologicallySorted(subgraph));

  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflBatchMatmul), 2 * chunks);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflSoftmax), chunks);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflReshape),
            2 * chunks + (cfg.v_after_bmm1 ? 1 : 0));
  const int kv_slices = head_chunks > 1 ? 2 * head_chunks : 0;
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflSlice), chunks + kv_slices);
  const int concats =
      (query_chunks > 1 ? head_chunks : 0) + (head_chunks > 1 ? 1 : 0);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflConcatenation), concats);
  EXPECT_FALSE(ag.logits.GetDefiningOp().HasValue());

  // The original output is now written by the final Concat.
  auto concat = ag.out.GetDefiningOp();
  ASSERT_TRUE(concat.HasValue());
  ASSERT_EQ(concat->Code(), kLiteRtOpCodeTflConcatenation);
  auto concat_opts = GetOptionsAs<ConcatenationOptions>(Ctx(), concat->Get());
  ASSERT_TRUE(concat_opts.HasValue());
  EXPECT_EQ(concat_opts->axis, head_chunks > 1 ? 0 : 1);
  EXPECT_EQ(concat->Inputs().size(),
            static_cast<size_t>(head_chunks > 1 ? head_chunks : query_chunks));
  ASSERT_EQ(ag.out.Uses().size(), 1);

  const int32_t hc = kH / head_chunks;
  const int32_t qc = kS / query_chunks;
  for (const auto& sm : FindOps(subgraph, kLiteRtOpCodeTflSoftmax)) {
    EXPECT_EQ(TensorDims(sm.Outputs()[0]), (Dims{1, hc, qc, kK}));
    auto opts = GetOptionsAs<SoftmaxOptions>(Ctx(), sm.Get());
    ASSERT_TRUE(opts.HasValue());
    EXPECT_FLOAT_EQ(opts->beta, kBeta);
    if (cfg.quantized) {
      EXPECT_FLOAT_EQ(ScaleOf(sm.Inputs()[0]), kR1Scale);
      EXPECT_FLOAT_EQ(ScaleOf(sm.Outputs()[0]), kSmScale);
    }
  }
  int num_qk = 0;
  for (const auto& bmm : FindOps(subgraph, kLiteRtOpCodeTflBatchMatmul)) {
    auto opts = GetOptionsAs<BatchMatmulOptions>(Ctx(), bmm.Get());
    ASSERT_TRUE(opts.HasValue());
    EXPECT_FALSE(opts->adj_x);
    EXPECT_FALSE(opts->adj_y);
    Tensor lhs = bmm.Inputs()[0];
    Tensor rhs = bmm.Inputs()[1];
    Tensor res = bmm.Outputs()[0];
    auto lhs_def = lhs.GetDefiningOp();
    ASSERT_TRUE(lhs_def.HasValue());
    if (lhs_def->Code() == kLiteRtOpCodeTflSlice &&
        TensorDims(res) == (Dims{hc, qc, kK})) {
      // Q chunk x Kt.
      ++num_qk;
      EXPECT_EQ(lhs_def->Inputs()[0], ag.q);
      EXPECT_EQ(TensorDims(lhs), (Dims{hc, qc, kD}));
      if (head_chunks == 1) EXPECT_EQ(rhs, ag.kt);
      if (cfg.quantized) {
        EXPECT_FLOAT_EQ(ScaleOf(lhs), kQScale);
        EXPECT_FLOAT_EQ(ScaleOf(res), kLogitScale);
      }
      // A second invocation on a produced chunk must not re-chunk it.
      LiteRtBuilderT builder2;
      EXPECT_EQ(AttentionChunkTransformation(Ctx(), &builder2, bmm.Get()),
                kLiteRtStatusPatternNoMatch);
    } else {
      // P chunk x V.
      ASSERT_EQ(lhs_def->Code(), kLiteRtOpCodeTflReshape);
      EXPECT_EQ(TensorDims(res), (Dims{hc, qc, kDv}));
      if (head_chunks == 1) EXPECT_EQ(rhs, ag.v);
      if (cfg.quantized) {
        EXPECT_FLOAT_EQ(ScaleOf(lhs), kR2Scale);
        EXPECT_FLOAT_EQ(ScaleOf(res), kOutScale);
        EXPECT_FLOAT_EQ(ScaleOf(rhs), kVScale);
      }
    }
  }
  EXPECT_EQ(num_qk, chunks);

  // Slice begins cover the whole Q tensor without overlap.
  std::vector<std::vector<int32_t>> q_begins;
  for (const auto& slice : FindOps(subgraph, kLiteRtOpCodeTflSlice)) {
    if (slice.Inputs()[0] != ag.q) continue;
    auto begin = slice.Inputs()[1].WeightsData<int32_t>();
    auto size = slice.Inputs()[2].WeightsData<int32_t>();
    ASSERT_TRUE(begin.HasValue());
    ASSERT_TRUE(size.HasValue());
    EXPECT_EQ(std::vector<int32_t>(size->begin(), size->end()),
              (Dims{hc, qc, kD}));
    q_begins.emplace_back(begin->begin(), begin->end());
  }
  ASSERT_EQ(q_begins.size(), static_cast<size_t>(chunks));
  int i = 0;
  for (int h = 0; h < head_chunks; ++h) {
    for (int q = 0; q < query_chunks; ++q, ++i) {
      EXPECT_EQ(q_begins[i], (Dims{h * hc, q * qc, 0}));
    }
  }
}

TEST(AttentionChunkTransformationTest, QueryChunksQuantized) {
  RunPositive(1, 3, AttnConfig{});
}

TEST(AttentionChunkTransformationTest, QueryChunksFloat) {
  AttnConfig cfg;
  cfg.quantized = false;
  RunPositive(1, 2, cfg);
}

TEST(AttentionChunkTransformationTest, HeadChunks) {
  RunPositive(2, 1, AttnConfig{});
}

TEST(AttentionChunkTransformationTest, OneHeadPerChunk) {
  RunPositive(kH, 1, AttnConfig{});
}

TEST(AttentionChunkTransformationTest, HeadByQueryGrid) {
  RunPositive(2, 3, AttnConfig{});
}

TEST(AttentionChunkTransformationTest, VDefinedAfterFirstBatchMatmul) {
  AttnConfig cfg;
  cfg.v_after_bmm1 = true;
  RunPositive(1, 2, cfg);
}

// The out -> Reshape -> Transpose -> Quantize epilogue is absorbed per chunk;
// Concats run on the requantized chunks (query axis 1, head axis 2).
TEST(AttentionChunkTransformationTest, AbsorbTailGrid) {
  ResetOrphanRegistry();
  SetAttentionChunkConfig(2, 3);
  SetAttentionChunkAbsorbTail(true);
  AttnConfig cfg;
  cfg.with_tail = true;
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  AttnGraph ag = BuildAttnGraph(subgraph, cfg);
  LiteRtBuilderT builder;
  ASSERT_EQ(AttentionChunkTransformation(Ctx(), &builder, ag.bmm1.Get()),
            kLiteRtStatusOk);
  builder.ApplyChanges(&subgraph);
  EXPECT_TRUE(IsTopologicallySorted(subgraph));
  // The old chain is still connected (only the Quantize was erased) but must
  // not be chunked a second time.
  {
    LiteRtBuilderT builder2;
    EXPECT_EQ(AttentionChunkTransformation(Ctx(), &builder2, ag.bmm1.Get()),
              kLiteRtStatusPatternNoMatch);
  }
  RunOrphanCleanup(subgraph);
  EXPECT_TRUE(IsTopologicallySorted(subgraph));
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflBatchMatmul), 12);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflQuantize), 6);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflTranspose), 6);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflReshape), 3 * 6);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflConcatenation), 2 + 1);
  EXPECT_FALSE(ag.out.GetDefiningOp().HasValue());
  auto concat = ag.q8.GetDefiningOp();
  ASSERT_TRUE(concat.HasValue());
  ASSERT_EQ(concat->Code(), kLiteRtOpCodeTflConcatenation);
  auto opts = GetOptionsAs<ConcatenationOptions>(Ctx(), concat->Get());
  ASSERT_TRUE(opts.HasValue());
  EXPECT_EQ(opts->axis, 2);
  ASSERT_EQ(concat->Inputs().size(), 2);
  EXPECT_EQ(TensorDims(concat->Inputs()[0]), (Dims{1, kS, kH / 2, kDv}));
  for (const auto& qz : FindOps(subgraph, kLiteRtOpCodeTflQuantize)) {
    EXPECT_EQ(TensorDims(qz.Outputs()[0]), (Dims{1, kS / 3, kH / 2, kDv}));
    EXPECT_EQ(qz.Outputs()[0].ElementType(), ElementType::Int8);
    EXPECT_FLOAT_EQ(ScaleOf(qz.Outputs()[0]), 0.05f);
    EXPECT_FLOAT_EQ(ScaleOf(qz.Inputs()[0]), kOutScale);
    auto tr = qz.Inputs()[0].GetDefiningOp();
    ASSERT_TRUE(tr.HasValue());
    EXPECT_EQ(tr->Code(), kLiteRtOpCodeTflTranspose);
  }
}

// Without the flag, the epilogue is left alone.
TEST(AttentionChunkTransformationTest, TailKeptWhenNotAbsorbed) {
  ResetOrphanRegistry();
  SetAttentionChunkConfig(2, 1);
  SetAttentionChunkAbsorbTail(false);
  AttnConfig cfg;
  cfg.with_tail = true;
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  AttnGraph ag = BuildAttnGraph(subgraph, cfg);
  LiteRtBuilderT builder;
  ASSERT_EQ(AttentionChunkTransformation(Ctx(), &builder, ag.bmm1.Get()),
            kLiteRtStatusOk);
  builder.ApplyChanges(&subgraph);
  RunOrphanCleanup(subgraph);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflQuantize), 1);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflTranspose), 1);
  auto concat = ag.out.GetDefiningOp();
  ASSERT_TRUE(concat.HasValue());
  EXPECT_EQ(concat->Code(), kLiteRtOpCodeTflConcatenation);
}

void ExpectNoMatch(int head_chunks, int query_chunks, const AttnConfig& cfg) {
  ResetOrphanRegistry();
  SetAttentionChunkConfig(head_chunks, query_chunks);
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  AttnGraph ag = BuildAttnGraph(subgraph, cfg);
  const size_t num_ops = subgraph.Ops().size();
  LiteRtBuilderT builder;
  EXPECT_EQ(AttentionChunkTransformation(Ctx(), &builder, ag.bmm1.Get()),
            kLiteRtStatusPatternNoMatch);
  // The second BatchMatmul is never a root.
  EXPECT_EQ(AttentionChunkTransformation(Ctx(), &builder, ag.bmm2.Get()),
            kLiteRtStatusPatternNoMatch);
  builder.ApplyChanges(&subgraph);
  EXPECT_EQ(subgraph.Ops().size(), num_ops);
}

TEST(AttentionChunkTransformationTest, DisabledByDefault) {
  ExpectNoMatch(1, 1, AttnConfig{});
}

TEST(AttentionChunkTransformationTest, NonDivisibleQueryChunksRejected) {
  ExpectNoMatch(1, 4, AttnConfig{});
}

TEST(AttentionChunkTransformationTest, NonDivisibleHeadChunksRejected) {
  ExpectNoMatch(3, 1, AttnConfig{});
}

TEST(AttentionChunkTransformationTest, AdjYRejected) {
  AttnConfig cfg;
  cfg.adj_y = true;
  ExpectNoMatch(1, 2, cfg);
}

TEST(AttentionChunkTransformationTest, ExtraLogitUserRejected) {
  AttnConfig cfg;
  cfg.extra_logit_user = true;
  ExpectNoMatch(1, 2, cfg);
}

TEST(AttentionChunkTransformationTest, MixedTypesRejected) {
  AttnConfig cfg;
  cfg.float_v = true;
  ExpectNoMatch(1, 2, cfg);
}

TEST(AttentionChunkTransformationTest, ConfigFromEnvironment) {
  setenv("LITERT_MEDIATEK_ATTN_CHUNKS", "2", 1);
  setenv("LITERT_MEDIATEK_ATTN_CHUNK_AXIS", "head", 1);
  ResetAttentionChunkTransformationState();
  {
    ResetOrphanRegistry();
    LiteRtModelT model;
    auto& subgraph = model.EmplaceSubgraph();
    AttnGraph ag = BuildAttnGraph(subgraph, AttnConfig{});
    LiteRtBuilderT builder;
    ASSERT_EQ(AttentionChunkTransformation(Ctx(), &builder, ag.bmm1.Get()),
              kLiteRtStatusOk);
    builder.ApplyChanges(&subgraph);
    auto concat = ag.out.GetDefiningOp();
    ASSERT_TRUE(concat.HasValue());
    auto opts = GetOptionsAs<ConcatenationOptions>(Ctx(), concat->Get());
    ASSERT_TRUE(opts.HasValue());
    EXPECT_EQ(opts->axis, 0);
  }
  unsetenv("LITERT_MEDIATEK_ATTN_CHUNKS");
  unsetenv("LITERT_MEDIATEK_ATTN_CHUNK_AXIS");
  ResetAttentionChunkTransformationState();
  {
    LiteRtModelT model;
    auto& subgraph = model.EmplaceSubgraph();
    AttnGraph ag = BuildAttnGraph(subgraph, AttnConfig{});
    LiteRtBuilderT builder;
    EXPECT_EQ(AttentionChunkTransformation(Ctx(), &builder, ag.bmm1.Get()),
              kLiteRtStatusPatternNoMatch);
  }
}

// Runs the rewrite with the configuration derived from the environment
// (LITERT_MEDIATEK_ATTN_CHUNKS unset unless `chunks_env` is given).
struct EnvRun {
  LiteRtStatus status;
  int softmaxes = 0;
  int root_concat_axis = -1;  // Axis of the Concat producing the output.
};

EnvRun RunFromEnv(const AttnConfig& cfg, const char* chunks_env = nullptr) {
  unsetenv("LITERT_MEDIATEK_ATTN_CHUNK_AXIS");
  if (chunks_env != nullptr) {
    setenv("LITERT_MEDIATEK_ATTN_CHUNKS", chunks_env, 1);
  } else {
    unsetenv("LITERT_MEDIATEK_ATTN_CHUNKS");
  }
  ResetAttentionChunkTransformationState();
  ResetOrphanRegistry();
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  AttnGraph ag = BuildAttnGraph(subgraph, cfg);
  LiteRtBuilderT builder;
  EnvRun res;
  res.status = AttentionChunkTransformation(Ctx(), &builder, ag.bmm1.Get());
  builder.ApplyChanges(&subgraph);
  RunOrphanCleanup(subgraph);
  EXPECT_TRUE(IsTopologicallySorted(subgraph));
  res.softmaxes = CountOps(subgraph, kLiteRtOpCodeTflSoftmax);
  auto concat = (cfg.with_tail ? ag.q8 : ag.out).GetDefiningOp();
  if (concat.HasValue() && concat->Code() == kLiteRtOpCodeTflConcatenation) {
    auto opts = GetOptionsAs<ConcatenationOptions>(Ctx(), concat->Get());
    if (opts.HasValue()) res.root_concat_axis = opts->axis;
  }
  unsetenv("LITERT_MEDIATEK_ATTN_CHUNKS");
  ResetAttentionChunkTransformationState();
  return res;
}

// 4 x 1024 x 1024 int16 logits = 8 MiB: exactly at the auto threshold.
AttnConfig LargeAttn() {
  AttnConfig cfg;
  cfg.h = 4;
  cfg.s = 1024;
  cfg.k = 1024;
  return cfg;
}

TEST(AttentionChunkTransformationTest, AutoChunksLargeAttentionPerHead) {
  const EnvRun res = RunFromEnv(LargeAttn());
  ASSERT_EQ(res.status, kLiteRtStatusOk);
  EXPECT_EQ(res.softmaxes, 4);
  EXPECT_EQ(res.root_concat_axis, 0);
}

TEST(AttentionChunkTransformationTest, AutoChunksLargeAttentionWithTail) {
  AttnConfig cfg = LargeAttn();
  cfg.with_tail = true;
  const EnvRun res = RunFromEnv(cfg);
  ASSERT_EQ(res.status, kLiteRtStatusOk);
  EXPECT_EQ(res.softmaxes, 4);
  // Heads are concatenated on the int8 [1, S, H, Dv] output.
  EXPECT_EQ(res.root_concat_axis, 2);
}

TEST(AttentionChunkTransformationTest, AutoSplitsSingleHeadAlongQueries) {
  AttnConfig cfg;
  cfg.h = 1;
  cfg.s = 4096;
  cfg.k = 4096;  // 32 MiB of logits -> 8 chunks of 4 MiB.
  const EnvRun res = RunFromEnv(cfg);
  ASSERT_EQ(res.status, kLiteRtStatusOk);
  EXPECT_EQ(res.softmaxes, 8);
  EXPECT_EQ(res.root_concat_axis, 1);
}

TEST(AttentionChunkTransformationTest, AutoSkipsSmallLogits) {
  const EnvRun res = RunFromEnv(AttnConfig{});
  EXPECT_EQ(res.status, kLiteRtStatusPatternNoMatch);
  EXPECT_EQ(res.softmaxes, 1);
}

TEST(AttentionChunkTransformationTest, AutoSkipsJustBelowThreshold) {
  AttnConfig cfg = LargeAttn();
  cfg.k = 1023;
  EXPECT_EQ(RunFromEnv(cfg).status, kLiteRtStatusPatternNoMatch);
}

TEST(AttentionChunkTransformationTest, AutoSkipsSingleQueryDecode) {
  AttnConfig cfg;
  cfg.h = 8;
  cfg.s = 1;
  cfg.k = 1 << 19;  // 8 MiB of logits, but q_len = 1 (LLM decode step).
  EXPECT_EQ(RunFromEnv(cfg).status, kLiteRtStatusPatternNoMatch);
}

TEST(AttentionChunkTransformationTest, ChunksZeroDisablesAuto) {
  const EnvRun res = RunFromEnv(LargeAttn(), "0");
  EXPECT_EQ(res.status, kLiteRtStatusPatternNoMatch);
  EXPECT_EQ(res.softmaxes, 1);
}

TEST(AttentionChunkTransformationTest, ExplicitChunksOverrideAuto) {
  const EnvRun res = RunFromEnv(LargeAttn(), "2");  // Query axis by default.
  ASSERT_EQ(res.status, kLiteRtStatusOk);
  EXPECT_EQ(res.softmaxes, 2);
  EXPECT_EQ(res.root_concat_axis, 1);
}

}  // namespace
}  // namespace litert::mediatek
