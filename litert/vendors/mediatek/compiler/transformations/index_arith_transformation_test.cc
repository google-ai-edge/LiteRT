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

#include "litert/vendors/mediatek/compiler/transformations/index_arith_transformation.h"

#include <cstddef>
#include <cstdint>
#include <optional>
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
using ::litert::compiler::AddOptions;
using ::litert::compiler::Builder;
using ::litert::compiler::DivOptions;
using ::litert::compiler::GetOptionsAs;
using ::litert::compiler::MulOptions;
using ::litert::compiler::OneHotOptions;
using ::litert::compiler::Op;
using ::litert::compiler::RankedTensorSpecBuilder;
using ::litert::compiler::ReduceAllOptions;
using ::litert::compiler::ReduceMaxOptions;
using ::litert::compiler::ReshapeOptions;
using ::litert::compiler::SubOptions;
using ::litert::compiler::Tensor;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

using Dims = std::vector<int32_t>;

// Thin wrapper around the compiler builder used to construct test graphs.
// All tensors/ops are created in a scratch builder and moved into the target
// subgraph with ApplyTo().
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

  Tensor F32(const Dims& dims) { return Typed(ElementType::Float32, dims); }
  Tensor I32(const Dims& dims) { return Typed(ElementType::Int32, dims); }
  Tensor Bool(const Dims& dims) { return Typed(ElementType::Bool, dims); }

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

  template <typename T>
  Tensor Const(const Dims& dims, const std::vector<T>& data) {
    Tensor t = Typed(GetElementType<T>(), dims);
    auto w = builder_.BuildWeights<T>(absl::MakeConstSpan(data), t);
    EXPECT_TRUE(w.HasValue());
    return t;
  }

  template <typename T>
  Tensor QuantConst(const Dims& dims, float scale, int64_t zero_point,
                    const std::vector<T>& data) {
    Tensor t = Quant(GetElementType<T>(), dims, scale, zero_point);
    auto w = builder_.BuildWeights<T>(absl::MakeConstSpan(data), t);
    EXPECT_TRUE(w.HasValue());
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

// Every input produced inside the subgraph must be produced by an earlier op.
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

// Runs OrphanCleanupTransformation on every op of the subgraph (one builder
// per invocation, like the plugin does) until nothing changes.
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

Dims TensorDims(const Tensor& t) {
  auto type = t.RankedTensorType();
  if (!type) return {};
  auto dims = type->Layout().Dimensions();
  return Dims(dims.begin(), dims.end());
}

template <typename T>
std::vector<T> ConstData(const Tensor& t) {
  auto data = t.WeightsData<T>();
  if (!data) return {};
  return std::vector<T>(data->begin(), data->end());
}

// Returns the defining op of `t`, asserting it has the given code.
#define ASSERT_DEF(var, tensor, code)          \
  auto var = (tensor).GetDefiningOp();         \
  ASSERT_TRUE(var.HasValue()) << #tensor;      \
  ASSERT_EQ(var->Code(), code) << #tensor

//===----------------------------------------------------------------------===//
// FloorDivTransformation
//===----------------------------------------------------------------------===//

constexpr int32_t kLen = 6;

struct FloorDivConfig {
  bool select_v2 = false;
  Dims k_dims = {};
  int32_t k = 4;
  // FloorMod divisor; defaults to `k`.
  std::optional<int32_t> mod_k;
  int32_t ne_sign_value = 1;
  int32_t ne_mod_value = 0;
  int32_t sub_value = 1;
  int32_t sub_activation = 0;
  bool swap_and_operands = false;
  // Adds a third use of q = Div(a, k).
  bool extra_q_use = false;
};

struct FloorDivGraph {
  Tensor a;
  Tensor k;
  Op select;
  Tensor out;
  Op consumer;
};

// q = Div(a, k); Select(Sign(a) != 1 && FloorMod(a, k) != 0, q - 1, q)
// -> Add(out, a).
FloorDivGraph BuildFloorDivGraph(LiteRtSubgraphT& subgraph,
                                 const FloorDivConfig& cfg) {
  GraphBuilder g;
  FloorDivGraph fg;
  const Dims dims = {1, kLen};
  auto splat = [&](int32_t value) {
    int64_t n = 1;
    for (int32_t d : cfg.k_dims) n *= d;
    return g.Const<int32_t>(cfg.k_dims, std::vector<int32_t>(n, value));
  };
  fg.a = g.I32(dims);
  fg.k = splat(cfg.k);

  Tensor q = g.I32(dims);
  Op div = g.AddOp(kLiteRtOpCodeTflDiv, {fg.a, fg.k}, {q});
  DivOptions div_opts;
  div_opts.fused_activation_function = 0;
  g.SetOptions(div, std::move(div_opts));

  Tensor sign_out = g.I32(dims);
  g.AddOp(kLiteRtOpCodeTflSign, {fg.a}, {sign_out});
  Tensor ne_sign_out = g.Bool(dims);
  g.AddOp(kLiteRtOpCodeTflNotEqual,
          {sign_out, g.Const<int32_t>({}, {cfg.ne_sign_value})}, {ne_sign_out});

  Tensor mod_out = g.I32(dims);
  g.AddOp(kLiteRtOpCodeTflFloorMod, {fg.a, splat(cfg.mod_k.value_or(cfg.k))},
          {mod_out});
  Tensor ne_mod_out = g.Bool(dims);
  g.AddOp(kLiteRtOpCodeTflNotEqual,
          {mod_out, g.Const<int32_t>({}, {cfg.ne_mod_value})}, {ne_mod_out});

  Tensor cond = g.Bool(dims);
  if (cfg.swap_and_operands) {
    g.AddOp(kLiteRtOpCodeTflLogicalAnd, {ne_mod_out, ne_sign_out}, {cond});
  } else {
    g.AddOp(kLiteRtOpCodeTflLogicalAnd, {ne_sign_out, ne_mod_out}, {cond});
  }

  Tensor q_minus_1 = g.I32(dims);
  Op sub = g.AddOp(kLiteRtOpCodeTflSub,
                   {q, g.Const<int32_t>({}, {cfg.sub_value})}, {q_minus_1});
  SubOptions sub_opts;
  sub_opts.fused_activation_function = cfg.sub_activation;
  g.SetOptions(sub, std::move(sub_opts));

  fg.out = g.I32(dims);
  fg.select = g.AddOp(
      cfg.select_v2 ? kLiteRtOpCodeTflSelectV2 : kLiteRtOpCodeTflSelect,
      {cond, q_minus_1, q}, {fg.out});

  Tensor consumer_out = g.I32(dims);
  fg.consumer = g.AddOp(kLiteRtOpCodeTflAdd, {fg.out, fg.a}, {consumer_out});
  AddOptions add_opts;
  add_opts.fused_activation_function = 0;
  g.SetOptions(fg.consumer, std::move(add_opts));

  if (cfg.extra_q_use) {
    Tensor extra_out = g.I32(dims);
    Op extra = g.AddOp(kLiteRtOpCodeTflAdd, {q, fg.a}, {extra_out});
    AddOptions extra_opts;
    extra_opts.fused_activation_function = 0;
    g.SetOptions(extra, std::move(extra_opts));
  }

  g.ApplyTo(&subgraph);
  return fg;
}

void RunFloorDivPositive(const FloorDivConfig& cfg) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  FloorDivGraph fg = BuildFloorDivGraph(subgraph, cfg);
  ASSERT_TRUE(IsTopologicallySorted(subgraph));
  ASSERT_EQ(subgraph.Ops().size(), 9);

  LiteRtBuilderT builder;
  ASSERT_EQ(FloorDivTransformation(Ctx(), &builder, fg.select.Get()),
            kLiteRtStatusOk);
  builder.ApplyChanges(&subgraph);

  // Only FloorDiv + the downstream consumer remain.
  EXPECT_EQ(subgraph.Ops().size(), 2);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflFloorDiv), 1);
  for (LiteRtOpCode code :
       {kLiteRtOpCodeTflSelect, kLiteRtOpCodeTflSelectV2, kLiteRtOpCodeTflDiv,
        kLiteRtOpCodeTflSub, kLiteRtOpCodeTflLogicalAnd,
        kLiteRtOpCodeTflNotEqual, kLiteRtOpCodeTflSign,
        kLiteRtOpCodeTflFloorMod}) {
    EXPECT_EQ(CountOps(subgraph, code), 0) << "op code " << code;
  }
  EXPECT_TRUE(IsTopologicallySorted(subgraph));

  // FloorDiv(a, k) writes the original select output.
  ASSERT_DEF(fd, fg.out, kLiteRtOpCodeTflFloorDiv);
  auto fd_ins = fd->Inputs();
  ASSERT_EQ(fd_ins.size(), 2);
  EXPECT_EQ(fd_ins[0], fg.a);
  EXPECT_EQ(fd_ins[1], fg.k);
  EXPECT_EQ(TensorDims(fg.out), (Dims{1, kLen}));
  EXPECT_EQ(fg.out.ElementType(), ElementType::Int32);
  ASSERT_EQ(fg.out.Uses().size(), 1);
  EXPECT_EQ(fg.out.Uses()[0].user.Get(), fg.consumer.Get());
}

void ExpectFloorDivNoMatch(const FloorDivConfig& cfg) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  FloorDivGraph fg = BuildFloorDivGraph(subgraph, cfg);
  const size_t num_ops = subgraph.Ops().size();

  LiteRtBuilderT builder;
  EXPECT_EQ(FloorDivTransformation(Ctx(), &builder, fg.select.Get()),
            kLiteRtStatusPatternNoMatch);
  builder.ApplyChanges(&subgraph);
  EXPECT_EQ(subgraph.Ops().size(), num_ops);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflFloorDiv), 0);
}

TEST(FloorDivTransformationTest, SelectScalarDivisorRewritten) {
  RunFloorDivPositive(FloorDivConfig{});
}

TEST(FloorDivTransformationTest, SelectV2ScalarDivisorRewritten) {
  FloorDivConfig cfg;
  cfg.select_v2 = true;
  RunFloorDivPositive(cfg);
}

TEST(FloorDivTransformationTest, SplatDivisor1x1Rewritten) {
  FloorDivConfig cfg;
  cfg.select_v2 = true;
  cfg.k_dims = {1, 1};
  cfg.k = 7;
  RunFloorDivPositive(cfg);
}

TEST(FloorDivTransformationTest, SwappedLogicalAndOperandsRewritten) {
  FloorDivConfig cfg;
  cfg.swap_and_operands = true;
  RunFloorDivPositive(cfg);
}

TEST(FloorDivTransformationTest, RootNotSelectRejected) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  FloorDivGraph fg = BuildFloorDivGraph(subgraph, FloorDivConfig{});
  LiteRtBuilderT builder;
  EXPECT_EQ(FloorDivTransformation(Ctx(), &builder, fg.consumer.Get()),
            kLiteRtStatusPatternNoMatch);
}

TEST(FloorDivTransformationTest, ZeroDivisorRejected) {
  FloorDivConfig cfg;
  cfg.k = 0;
  ExpectFloorDivNoMatch(cfg);
}

TEST(FloorDivTransformationTest, NegativeDivisorRejected) {
  FloorDivConfig cfg;
  cfg.k = -3;
  ExpectFloorDivNoMatch(cfg);
}

TEST(FloorDivTransformationTest, FloorModDivisorMismatchRejected) {
  FloorDivConfig cfg;
  cfg.mod_k = 5;
  ExpectFloorDivNoMatch(cfg);
}

TEST(FloorDivTransformationTest, SignComparedWithZeroRejected) {
  FloorDivConfig cfg;
  cfg.ne_sign_value = 0;
  ExpectFloorDivNoMatch(cfg);
}

TEST(FloorDivTransformationTest, ModComparedWithOneRejected) {
  FloorDivConfig cfg;
  cfg.ne_mod_value = 1;
  ExpectFloorDivNoMatch(cfg);
}

TEST(FloorDivTransformationTest, SubtractTwoRejected) {
  FloorDivConfig cfg;
  cfg.sub_value = 2;
  ExpectFloorDivNoMatch(cfg);
}

TEST(FloorDivTransformationTest, SubWithFusedActivationRejected) {
  FloorDivConfig cfg;
  cfg.sub_activation = 1;  // RELU
  ExpectFloorDivNoMatch(cfg);
}

TEST(FloorDivTransformationTest, QuotientWithExtraUseRejected) {
  FloorDivConfig cfg;
  cfg.extra_q_use = true;
  ExpectFloorDivNoMatch(cfg);
}

//===----------------------------------------------------------------------===//
// OneHotArithTransformation
//===----------------------------------------------------------------------===//

constexpr int32_t kSeq = 4;
constexpr int32_t kDepth = 6;
constexpr float kQuantScale = 1.0f / 32767.0f;

struct OneHotConfig {
  int32_t depth = kDepth;
  float on = 1.0f;
  float off = 0.0f;
  int32_t axis = -1;
  // Adds a Relu user of the one-hot next to the Quantize.
  bool non_quantize_user = false;
  // Adds a Dequantize user of the Quantize (keeps it alive after the mask
  // chain rewrite).
  bool dequantize_user = true;
  // Quantized type (int16 or int8) and scale of Quantize/Mul/Equal tensors.
  bool int8 = false;
  float quant_scale = kQuantScale;
  // Builds Quantize -> Mul(c) -> Equal(0) -> ReduceAll -> LogicalNot.
  bool mask = false;
  bool keep_dims = false;
  int16_t equal_const = 0;
  float mul_out_scale = 3.39e-6f;
};

struct OneHotGraph {
  Tensor idx;
  Op one_hot;
  Tensor out;
  Op quantize;
  Tensor mul_out;
  Tensor axes;
  Tensor mask_out;
};

// idx [1, S] -> OneHot(depth) [1, S, D] f32 -> Quantize [-> Dequantize]
// [-> Mul(c) -> Equal(0) -> ReduceAll(axis 2) -> LogicalNot -> mask].
OneHotGraph BuildOneHotGraph(LiteRtSubgraphT& subgraph,
                             const OneHotConfig& cfg) {
  GraphBuilder g;
  OneHotGraph og;
  const Dims out_dims = {1, kSeq, cfg.depth};
  const ElementType q_type = cfg.int8 ? ElementType::Int8 : ElementType::Int16;
  const int32_t q_max = cfg.int8 ? 127 : 32767;
  auto quant_const = [&](float scale, int32_t value) {
    if (cfg.int8) {
      return g.QuantConst<int8_t>({}, scale, 0, {static_cast<int8_t>(value)});
    }
    return g.QuantConst<int16_t>({}, scale, 0, {static_cast<int16_t>(value)});
  };

  og.idx = g.I32({1, kSeq});
  og.out = g.F32(out_dims);
  og.one_hot = g.AddOp(kLiteRtOpCodeTflOneHot,
                       {og.idx, g.Const<int32_t>({}, {cfg.depth}),
                        g.Const<float>({}, {cfg.on}),
                        g.Const<float>({}, {cfg.off})},
                       {og.out});
  OneHotOptions one_hot_opts;
  one_hot_opts.axis = cfg.axis;
  g.SetOptions(og.one_hot, std::move(one_hot_opts));

  Tensor q_out = g.Quant(q_type, out_dims, cfg.quant_scale, 0);
  og.quantize = g.AddOp(kLiteRtOpCodeTflQuantize, {og.out}, {q_out});
  if (cfg.dequantize_user) {
    g.AddOp(kLiteRtOpCodeTflDequantize, {q_out}, {g.F32(out_dims)});
  }

  if (cfg.non_quantize_user) {
    g.AddOp(kLiteRtOpCodeTflRelu, {og.out}, {g.F32(out_dims)});
  }

  if (cfg.mask) {
    // c_real = q_max * (1 / (9 * q_max)) = 1 / 9.
    Tensor c = quant_const(1.0f / (9.0f * q_max), q_max);
    og.mul_out = g.Quant(q_type, out_dims, cfg.mul_out_scale, 0);
    Op mul = g.AddOp(kLiteRtOpCodeTflMul, {q_out, c}, {og.mul_out});
    MulOptions mul_opts;
    mul_opts.fused_activation_function = 0;
    g.SetOptions(mul, std::move(mul_opts));

    Tensor zero = quant_const(cfg.mul_out_scale, cfg.equal_const);
    Tensor eq_out = g.Bool(out_dims);
    g.AddOp(kLiteRtOpCodeTflEqual, {og.mul_out, zero}, {eq_out});

    og.axes = g.Const<int32_t>({1}, {2});
    const Dims mask_dims =
        cfg.keep_dims ? Dims{1, kSeq, 1} : Dims{1, kSeq};
    Tensor all_out = g.Bool(mask_dims);
    Op reduce_all =
        g.AddOp(kLiteRtOpCodeTflReduceAll, {eq_out, og.axes}, {all_out});
    ReduceAllOptions ra_opts;
    ra_opts.keep_dims = cfg.keep_dims;
    g.SetOptions(reduce_all, std::move(ra_opts));

    og.mask_out = g.Bool(mask_dims);
    g.AddOp(kLiteRtOpCodeTflLogicalNot, {all_out}, {og.mask_out});
  }

  g.ApplyTo(&subgraph);
  return og;
}

// Checks out = Relu(1 - |Tile(Reshape(Cast(idx)), [1, 1, D]) - iota|).
void ExpectArithOneHot(const OneHotGraph& og, int32_t depth,
                       size_t expected_out_uses) {
  ASSERT_DEF(relu, og.out, kLiteRtOpCodeTflRelu);
  const Tensor lin = relu->Inputs()[0];
  EXPECT_EQ(TensorDims(lin), (Dims{1, kSeq, depth}));

  ASSERT_DEF(sub1, lin, kLiteRtOpCodeTflSub);
  auto sub1_opts = GetOptionsAs<SubOptions>(Ctx(), sub1->Get());
  ASSERT_TRUE(sub1_opts.HasValue());
  EXPECT_EQ(sub1_opts->fused_activation_function, 0);
  ASSERT_EQ(sub1->Inputs().size(), 2);
  EXPECT_TRUE(TensorDims(sub1->Inputs()[0]).empty());
  EXPECT_EQ(ConstData<float>(sub1->Inputs()[0]), (std::vector<float>{1.0f}));

  ASSERT_DEF(abs, sub1->Inputs()[1], kLiteRtOpCodeTflAbs);
  const Tensor diff = abs->Inputs()[0];
  ASSERT_DEF(sub0, diff, kLiteRtOpCodeTflSub);
  auto sub0_opts = GetOptionsAs<SubOptions>(Ctx(), sub0->Get());
  ASSERT_TRUE(sub0_opts.HasValue());
  EXPECT_EQ(sub0_opts->fused_activation_function, 0);
  ASSERT_EQ(sub0->Inputs().size(), 2);
  const Tensor iota = sub0->Inputs()[1];
  EXPECT_TRUE(iota.IsConstant());
  EXPECT_EQ(TensorDims(iota), (Dims{1, 1, depth}));
  std::vector<float> expected_iota(depth);
  for (int32_t j = 0; j < depth; ++j) expected_iota[j] = j;
  EXPECT_EQ(ConstData<float>(iota), expected_iota);

  const Tensor tiled = sub0->Inputs()[0];
  EXPECT_EQ(TensorDims(tiled), (Dims{1, kSeq, depth}));
  ASSERT_DEF(tile, tiled, kLiteRtOpCodeTflTile);
  ASSERT_EQ(tile->Inputs().size(), 2);
  EXPECT_EQ(ConstData<int32_t>(tile->Inputs()[1]), (Dims{1, 1, depth}));

  const Tensor col = tile->Inputs()[0];
  EXPECT_EQ(TensorDims(col), (Dims{1, kSeq, 1}));
  EXPECT_EQ(col.ElementType(), ElementType::Float32);
  ASSERT_DEF(reshape, col, kLiteRtOpCodeTflReshape);
  ASSERT_EQ(reshape->Inputs().size(), 2);
  EXPECT_EQ(ConstData<int32_t>(reshape->Inputs()[1]), (Dims{1, kSeq, 1}));
  auto reshape_opts = GetOptionsAs<ReshapeOptions>(Ctx(), reshape->Get());
  ASSERT_TRUE(reshape_opts.HasValue());
  EXPECT_EQ(reshape_opts->new_shape, (std::vector<int32_t>{1, kSeq, 1}));

  const Tensor idx_f = reshape->Inputs()[0];
  EXPECT_EQ(TensorDims(idx_f), (Dims{1, kSeq}));
  EXPECT_EQ(idx_f.ElementType(), ElementType::Float32);
  ASSERT_DEF(cast, idx_f, kLiteRtOpCodeTflCast);
  EXPECT_EQ(cast->Inputs()[0], og.idx);

  // The Quantize still consumes the (now arithmetic) one-hot.
  ASSERT_EQ(og.out.Uses().size(), expected_out_uses);
  bool quantize_user = false;
  for (const auto& use : og.out.Uses()) {
    if (use.user.Get() == og.quantize.Get()) quantize_user = true;
  }
  EXPECT_TRUE(quantize_user);
}

void RunOneHotPositive(const OneHotConfig& cfg, bool expect_mask_rewrite) {
  ResetOrphanRegistry();
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  OneHotGraph og = BuildOneHotGraph(subgraph, cfg);
  ASSERT_TRUE(IsTopologicallySorted(subgraph));

  LiteRtBuilderT builder;
  ASSERT_EQ(OneHotArithTransformation(Ctx(), &builder, og.one_hot.Get()),
            kLiteRtStatusOk);
  builder.ApplyChanges(&subgraph);

  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflOneHot), 0);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflCast), 1);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflTile), 1);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflSub), 2);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflAbs), 1);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflRelu), 1);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflQuantize), 1);
  EXPECT_TRUE(IsTopologicallySorted(subgraph));

  if (!cfg.mask) {
    EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflReduceMax), 0);
    ExpectArithOneHot(og, cfg.depth, /*expected_out_uses=*/1);
    return;
  }

  if (!expect_mask_rewrite) {
    EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflReduceMax), 0);
    EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflEqual), 1);
    EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflReduceAll), 1);
    EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflLogicalNot), 1);
    ASSERT_DEF(lnot, og.mask_out, kLiteRtOpCodeTflLogicalNot);
    // The one-hot itself is still rewritten; only Quantize consumes it.
    ASSERT_DEF(relu, og.out, kLiteRtOpCodeTflRelu);
    ASSERT_EQ(og.out.Uses().size(), 1);
    // Nothing is registered for cleanup: the Mul still feeds the Equal.
    const size_t num_ops = subgraph.Ops().size();
    RunOrphanCleanup(subgraph);
    EXPECT_EQ(subgraph.Ops().size(), num_ops);
    EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflMul), 1);
    return;
  }

  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflEqual), 0);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflReduceAll), 0);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflLogicalNot), 0);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflReduceMax), 1);
  ExpectArithOneHot(og, cfg.depth, /*expected_out_uses=*/2);

  // mask = NotEqual(ReduceMax(one_hot, axes), 0.0f) writes the original
  // LogicalNot output.
  ASSERT_DEF(ne, og.mask_out, kLiteRtOpCodeTflNotEqual);
  ASSERT_EQ(ne->Inputs().size(), 2);
  EXPECT_TRUE(TensorDims(ne->Inputs()[1]).empty());
  EXPECT_EQ(ConstData<float>(ne->Inputs()[1]), (std::vector<float>{0.0f}));
  const Tensor max_t = ne->Inputs()[0];
  EXPECT_EQ(max_t.ElementType(), ElementType::Float32);
  EXPECT_EQ(TensorDims(max_t), TensorDims(og.mask_out));
  ASSERT_DEF(rmax, max_t, kLiteRtOpCodeTflReduceMax);
  ASSERT_EQ(rmax->Inputs().size(), 2);
  EXPECT_EQ(rmax->Inputs()[0], og.out);
  EXPECT_EQ(rmax->Inputs()[1], og.axes);
  auto rmax_opts = GetOptionsAs<ReduceMaxOptions>(Ctx(), rmax->Get());
  ASSERT_TRUE(rmax_opts.HasValue());
  EXPECT_EQ(rmax_opts->keep_dims, cfg.keep_dims);

  // The rescale Mul lost its only user (Equal) and is removed by orphan
  // cleanup; the Quantize goes too unless it still feeds the Dequantize.
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflMul), 1);
  EXPECT_TRUE(og.mul_out.Uses().empty());
  RunOrphanCleanup(subgraph);
  EXPECT_TRUE(IsTopologicallySorted(subgraph));
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflMul), 0);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflQuantize),
            cfg.dequantize_user ? 1 : 0);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflDequantize),
            cfg.dequantize_user ? 1 : 0);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflRelu), 1);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflReduceMax), 1);
  ASSERT_DEF(ne_after, og.mask_out, kLiteRtOpCodeTflNotEqual);
  EXPECT_EQ(og.out.Uses().size(), cfg.dequantize_user ? 2 : 1);
}

void ExpectOneHotNoMatch(const OneHotConfig& cfg) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  OneHotGraph og = BuildOneHotGraph(subgraph, cfg);
  const size_t num_ops = subgraph.Ops().size();

  LiteRtBuilderT builder;
  EXPECT_EQ(OneHotArithTransformation(Ctx(), &builder, og.one_hot.Get()),
            kLiteRtStatusPatternNoMatch);
  builder.ApplyChanges(&subgraph);
  EXPECT_EQ(subgraph.Ops().size(), num_ops);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflOneHot), 1);
}

TEST(OneHotArithTransformationTest, QuantizedOneHotRewritten) {
  RunOneHotPositive(OneHotConfig{}, /*expect_mask_rewrite=*/false);
}

TEST(OneHotArithTransformationTest, ExplicitLastAxisRewritten) {
  OneHotConfig cfg;
  cfg.axis = 2;
  RunOneHotPositive(cfg, /*expect_mask_rewrite=*/false);
}

TEST(OneHotArithTransformationTest, MaxDepthRewritten) {
  OneHotConfig cfg;
  cfg.depth = 1024;
  RunOneHotPositive(cfg, /*expect_mask_rewrite=*/false);
}

TEST(OneHotArithTransformationTest, MaskChainRewritten) {
  OneHotConfig cfg;
  cfg.mask = true;
  RunOneHotPositive(cfg, /*expect_mask_rewrite=*/true);
}

TEST(OneHotArithTransformationTest, MaskChainKeepDimsRewritten) {
  OneHotConfig cfg;
  cfg.mask = true;
  cfg.keep_dims = true;
  RunOneHotPositive(cfg, /*expect_mask_rewrite=*/true);
}

TEST(OneHotArithTransformationTest, MaskEqualNonZeroConstNotRewritten) {
  OneHotConfig cfg;
  cfg.mask = true;
  cfg.equal_const = 3;
  RunOneHotPositive(cfg, /*expect_mask_rewrite=*/false);
}

TEST(OneHotArithTransformationTest, MaskRescaleToZeroNotRewritten) {
  // |1 * (1/9) / 1.0| < 1: the rescaled one could round to 0.
  OneHotConfig cfg;
  cfg.mask = true;
  cfg.mul_out_scale = 1.0f;
  RunOneHotPositive(cfg, /*expect_mask_rewrite=*/false);
}

TEST(OneHotArithTransformationTest, MaskChainDeadRescaleMulCleanedUp) {
  // Quantize keeps its Dequantize user; only the rescale Mul dies.
  OneHotConfig cfg;
  cfg.mask = true;
  cfg.dequantize_user = true;
  RunOneHotPositive(cfg, /*expect_mask_rewrite=*/true);
}

TEST(OneHotArithTransformationTest, MaskChainDeadQuantizeAndMulCleanedUp) {
  // The Mul was the Quantize's only user: both are removed.
  OneHotConfig cfg;
  cfg.mask = true;
  cfg.dequantize_user = false;
  RunOneHotPositive(cfg, /*expect_mask_rewrite=*/true);
}

TEST(OneHotArithTransformationTest, Int8MaskChainRewritten) {
  // one = 127 * (1/127) = 1; |1 * (1/9) / 0.005| = 22.2 >= 1.
  OneHotConfig cfg;
  cfg.mask = true;
  cfg.int8 = true;
  cfg.quant_scale = 1.0f / 127.0f;
  cfg.mul_out_scale = 0.005f;
  RunOneHotPositive(cfg, /*expect_mask_rewrite=*/true);
}

TEST(OneHotArithTransformationTest, Int8SaturatedQuantizedOneIsClamped) {
  // round(1 / 0.001) = 1000 saturates to 127, so the quantized one is 0.127:
  // |0.127 * (1/9) / 0.05| = 0.28 < 1 (unclamped it would be 2.2 >= 1).
  OneHotConfig cfg;
  cfg.mask = true;
  cfg.int8 = true;
  cfg.quant_scale = 0.001f;
  cfg.mul_out_scale = 0.05f;
  RunOneHotPositive(cfg, /*expect_mask_rewrite=*/false);
}

TEST(OneHotArithTransformationTest, Int16SaturatedQuantizedOneIsClamped) {
  // round(1 / 1e-5) = 100000 saturates to 32767 (one = 0.32767):
  // |0.32767 * (1/9) / 0.1| = 0.36 < 1 (unclamped it would be 1.11 >= 1).
  OneHotConfig cfg;
  cfg.mask = true;
  cfg.quant_scale = 1e-5f;
  cfg.mul_out_scale = 0.1f;
  RunOneHotPositive(cfg, /*expect_mask_rewrite=*/false);
}

TEST(OneHotArithTransformationTest, NonQuantizeUserRejected) {
  OneHotConfig cfg;
  cfg.non_quantize_user = true;
  ExpectOneHotNoMatch(cfg);
}

TEST(OneHotArithTransformationTest, DepthAboveLimitRejected) {
  OneHotConfig cfg;
  cfg.depth = 1025;
  ExpectOneHotNoMatch(cfg);
}

TEST(OneHotArithTransformationTest, OnValueNotOneRejected) {
  OneHotConfig cfg;
  cfg.on = 2.0f;
  ExpectOneHotNoMatch(cfg);
}

TEST(OneHotArithTransformationTest, OffValueNotZeroRejected) {
  OneHotConfig cfg;
  cfg.off = -1.0f;
  ExpectOneHotNoMatch(cfg);
}

TEST(OneHotArithTransformationTest, NonLastAxisRejected) {
  OneHotConfig cfg;
  cfg.axis = 0;
  ExpectOneHotNoMatch(cfg);
}

TEST(OneHotArithTransformationTest, RootNotOneHotRejected) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  OneHotGraph og = BuildOneHotGraph(subgraph, OneHotConfig{});
  LiteRtBuilderT builder;
  EXPECT_EQ(OneHotArithTransformation(Ctx(), &builder, og.quantize.Get()),
            kLiteRtStatusPatternNoMatch);
}

}  // namespace
}  // namespace litert::mediatek
