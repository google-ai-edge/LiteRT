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

#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
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
#include "litert/vendors/mediatek/compiler/transformations/attention_mask_transformation.h"
#include "litert/vendors/mediatek/compiler/transformations/entry_embedding_transformation.h"
#include "litert/vendors/mediatek/compiler/transformations/orphan_cleanup_transformation.h"
#include "litert/vendors/mediatek/compiler/transformations/rope_transformation.h"

namespace litert::mediatek {
namespace {

using ::litert::ElementType;
using ::litert::compiler::AddOptions;
using ::litert::compiler::Builder;
using ::litert::compiler::ConcatenationOptions;
using ::litert::compiler::FullyConnectedOptions;
using ::litert::compiler::GatherOptions;
using ::litert::compiler::GetOptionsAs;
using ::litert::compiler::MulOptions;
using ::litert::compiler::OneHotOptions;
using ::litert::compiler::Op;
using ::litert::compiler::RankedTensorSpecBuilder;
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

bool ContainsOp(const LiteRtSubgraphT& subgraph, LiteRtOp op) {
  for (const auto* o : subgraph.Ops()) {
    if (o == op) return true;
  }
  return false;
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

Dims TensorDims(const Tensor& t) {
  auto type = t.RankedTensorType();
  if (!type) return {};
  auto dims = type->Layout().Dimensions();
  return Dims(dims.begin(), dims.end());
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

//===----------------------------------------------------------------------===//
// EntryEmbeddingTransformation
//===----------------------------------------------------------------------===//

constexpr int32_t kSeq = 4;
constexpr int32_t kEmb = 3;
constexpr int32_t kDepth = 5;
constexpr int32_t kFreqs = 2;

constexpr char kEmbeddingLegacyEnv[] = "LITERT_MEDIATEK_EMBEDDING_LEGACY";
constexpr char kEmbeddingNoTrigFoldEnv[] =
    "LITERT_MEDIATEK_EMBEDDING_NO_TRIG_FOLD";

// Relative op order of the coordinate Reshape(src) producers and the
// Cast -> Mul -> Reshape -> {Sin, Cos} chains of the two branches.
enum class TrigOrder {
  kCoordsFirst,  // coord0, coord1, trig0, trig1
  kInterleaved,  // coord0, trig0, coord1, trig1
  kTrigFirst,    // trig0, trig1, coord0, coord1
};

struct EmbeddingConfig {
  bool with_select = false;
  // Builds mask = Reshape((c < 0 || c >= depth) && c != -1) from the coord
  // instead of using a plain bool graph input.
  bool real_mask = false;
  // Constant of the NotEqual in branch 1's mask (only -1 matches).
  int32_t branch1_mask_ne_value = -1;
  // By default both masks are computed before either lookup branch. When
  // set, each mask is computed right before its own OneHot, i.e. branch 1's
  // mask producers come after branch 0's FullyConnected.
  bool interleave_masks = false;
  // coord = Reshape(src [1, S, 1]) with a RoPE Cast -> Mul -> Reshape ->
  // {Sin, Cos} chain hanging off src.
  bool trig = false;
  TrigOrder trig_order = TrigOrder::kCoordsFirst;
  float fill = 0.0f;
  float on_value = 1.0f;
  float off_value = 0.0f;
  Dims weight_dims = {kEmb, kDepth};
};

struct EmbeddingGraph {
  Tensor coord[2];
  Tensor mask[2];
  Tensor fill[2];
  std::vector<float> weights[2];
  // Trig chain (only with cfg.trig).
  std::vector<float> freqs[2];
  Tensor sin_out[2];
  Tensor cos_out[2];
  Tensor src[2];
  Op trig_consumer[2];
  Op sum;
  Tensor sum_out;
};

void SetMulNoActivation(GraphBuilder& g, const Op& op) {
  MulOptions opts;
  opts.fused_activation_function = 0;
  g.SetOptions(op, std::move(opts));
}

void SetReshapeShape(GraphBuilder& g, const Op& op, const Dims& dims) {
  ReshapeOptions opts;
  opts.new_shape = dims;
  g.SetOptions(op, std::move(opts));
}

// [src [1, S, 1] -> Reshape ->] coord [1, S]
//   [-> (c < 0 || c >= depth) && c != -1 -> Reshape -> mask [1, S, 1]]
//   -> OneHot [-> SelectV2(mask, fill, .)] -> FC(W) -> Reshape [1, 1, S, D]
// (x2) -> Concat(axis 0) [2, 1, S, D] -> Sum(axis 0) [1, S, D].
// With cfg.trig: src -> Cast -> Mul(freq [1, 1, F]) -> Reshape [1, S, 1, F]
//   -> {Sin, Cos} -> Mul(sin, cos) (downstream RoPE consumer).
EmbeddingGraph BuildEmbeddingGraph(LiteRtSubgraphT& subgraph,
                                   const EmbeddingConfig& cfg) {
  GraphBuilder g;
  EmbeddingGraph eg;
  for (int br = 0; br < 2; ++br) {
    eg.coord[br] = g.I32({1, kSeq});
    if (cfg.trig) {
      eg.src[br] = g.I32({1, kSeq, 1});
      eg.freqs[br] = {0.5f + br, 0.125f + 0.25f * br};
    }
  }

  auto build_coord_reshape = [&](int br) {
    Tensor shape = g.Const<int32_t>({2}, {1, kSeq});
    Op op =
        g.AddOp(kLiteRtOpCodeTflReshape, {eg.src[br], shape}, {eg.coord[br]});
    SetReshapeShape(g, op, {1, kSeq});
  };
  auto build_trig = [&](int br) {
    Tensor cast_out = g.F32({1, kSeq, 1});
    g.AddOp(kLiteRtOpCodeTflCast, {eg.src[br]}, {cast_out});
    Tensor freq = g.Const<float>({1, 1, kFreqs}, eg.freqs[br]);
    Tensor mul_out = g.F32({1, kSeq, kFreqs});
    Op mul = g.AddOp(kLiteRtOpCodeTflMul, {cast_out, freq}, {mul_out});
    SetMulNoActivation(g, mul);
    Tensor shape = g.Const<int32_t>({4}, {1, kSeq, 1, kFreqs});
    Tensor angle = g.F32({1, kSeq, 1, kFreqs});
    Op reshape = g.AddOp(kLiteRtOpCodeTflReshape, {mul_out, shape}, {angle});
    SetReshapeShape(g, reshape, {1, kSeq, 1, kFreqs});
    eg.sin_out[br] = g.F32({1, kSeq, 1, kFreqs});
    eg.cos_out[br] = g.F32({1, kSeq, 1, kFreqs});
    g.AddOp(kLiteRtOpCodeTflSin, {angle}, {eg.sin_out[br]});
    g.AddOp(kLiteRtOpCodeTflCos, {angle}, {eg.cos_out[br]});
  };
  if (cfg.trig) {
    switch (cfg.trig_order) {
      case TrigOrder::kCoordsFirst:
        build_coord_reshape(0);
        build_coord_reshape(1);
        build_trig(0);
        build_trig(1);
        break;
      case TrigOrder::kInterleaved:
        build_coord_reshape(0);
        build_trig(0);
        build_coord_reshape(1);
        build_trig(1);
        break;
      case TrigOrder::kTrigFirst:
        build_trig(0);
        build_trig(1);
        build_coord_reshape(0);
        build_coord_reshape(1);
        break;
    }
  }

  auto build_mask = [&](int br) {
    if (!cfg.with_select) return;
    if (!cfg.real_mask) {
      eg.mask[br] = g.Bool({1, kSeq, 1});
      return;
    }
    const Tensor& coord = eg.coord[br];
    auto compare = [&](LiteRtOpCode code, int32_t value) {
      Tensor k = g.Const<int32_t>({}, {value});
      Tensor out = g.Bool({1, kSeq});
      g.AddOp(code, {coord, k}, {out});
      return out;
    };
    Tensor lt = compare(kLiteRtOpCodeTflLess, 0);
    Tensor ge = compare(kLiteRtOpCodeTflGreaterEqual, kDepth);
    Tensor or_out = g.Bool({1, kSeq});
    g.AddOp(kLiteRtOpCodeTflLogicalOr, {lt, ge}, {or_out});
    Tensor ne = compare(kLiteRtOpCodeTflNotEqual,
                        br == 1 ? cfg.branch1_mask_ne_value : -1);
    Tensor and_out = g.Bool({1, kSeq});
    // Operands swapped on purpose: the matcher accepts either order.
    g.AddOp(kLiteRtOpCodeTflLogicalAnd, {ne, or_out}, {and_out});
    Tensor shape = g.Const<int32_t>({3}, {1, kSeq, 1});
    eg.mask[br] = g.Bool({1, kSeq, 1});
    Op reshape =
        g.AddOp(kLiteRtOpCodeTflReshape, {and_out, shape}, {eg.mask[br]});
    SetReshapeShape(g, reshape, {1, kSeq, 1});
  };
  if (!cfg.interleave_masks) {
    build_mask(0);
    build_mask(1);
  }

  std::vector<Tensor> reshape_outs;
  for (int br = 0; br < 2; ++br) {
    const Tensor& coord = eg.coord[br];
    if (cfg.interleave_masks) build_mask(br);

    Tensor depth = g.Const<int32_t>({}, {kDepth});
    Tensor on = g.Const<float>({}, {cfg.on_value});
    Tensor off = g.Const<float>({}, {cfg.off_value});
    Tensor one_hot_out = g.F32({1, kSeq, kDepth});
    Op one_hot =
        g.AddOp(kLiteRtOpCodeTflOneHot, {coord, depth, on, off}, {one_hot_out});
    OneHotOptions one_hot_opts;
    one_hot_opts.axis = -1;
    g.SetOptions(one_hot, std::move(one_hot_opts));

    Tensor fc_in = one_hot_out;
    if (cfg.with_select) {
      eg.fill[br] = g.Const<float>({}, {cfg.fill});
      Tensor sel_out = g.F32({1, kSeq, kDepth});
      g.AddOp(kLiteRtOpCodeTflSelectV2, {eg.mask[br], eg.fill[br], one_hot_out},
              {sel_out});
      fc_in = sel_out;
    }

    const int32_t rows = cfg.weight_dims[0];
    const int32_t cols = cfg.weight_dims[1];
    std::vector<float> w(rows * cols);
    for (int32_t r = 0; r < rows; ++r) {
      for (int32_t c = 0; c < cols; ++c) {
        w[r * cols + c] = 1.0f + r * 10.0f + c + br * 100.0f;
      }
    }
    eg.weights[br] = w;
    Tensor w_t = g.Const<float>(cfg.weight_dims, w);

    Tensor fc_out = g.F32({1, kSeq, kEmb});
    Op fc = g.AddOp(kLiteRtOpCodeTflFullyConnected, {fc_in, w_t}, {fc_out});
    FullyConnectedOptions fc_opts;
    fc_opts.fused_activation_function = 0;
    fc_opts.weights_format = 0;
    fc_opts.keep_num_dims = true;
    fc_opts.quantized_bias_type = kLiteRtElementTypeNone;
    fc_opts.asymmetric_quantize_input = false;
    g.SetOptions(fc, std::move(fc_opts));

    Tensor shape = g.Const<int32_t>({4}, {1, 1, kSeq, kEmb});
    Tensor reshape_out = g.F32({1, 1, kSeq, kEmb});
    Op reshape =
        g.AddOp(kLiteRtOpCodeTflReshape, {fc_out, shape}, {reshape_out});
    SetReshapeShape(g, reshape, {1, 1, kSeq, kEmb});
    reshape_outs.push_back(reshape_out);
  }

  Tensor concat_out = g.F32({2, 1, kSeq, kEmb});
  Op concat =
      g.AddOp(kLiteRtOpCodeTflConcatenation, reshape_outs, {concat_out});
  ConcatenationOptions concat_opts;
  concat_opts.axis = 0;
  concat_opts.fused_activation_function = 0;
  g.SetOptions(concat, std::move(concat_opts));

  Tensor axis = g.Const<int32_t>({1}, {0});
  eg.sum_out = g.F32({1, kSeq, kEmb});
  eg.sum = g.AddOp(kLiteRtOpCodeTflSum, {concat_out, axis}, {eg.sum_out});

  if (cfg.trig) {
    for (int br = 0; br < 2; ++br) {
      Tensor out = g.F32({1, kSeq, 1, kFreqs});
      eg.trig_consumer[br] =
          g.AddOp(kLiteRtOpCodeTflMul, {eg.sin_out[br], eg.cos_out[br]}, {out});
      SetMulNoActivation(g, eg.trig_consumer[br]);
    }
  }

  g.ApplyTo(&subgraph);
  return eg;
}

// Checks the legacy rewritten x/y lookup branch feeding `add_in`.
void ExpectLookupBranch(const Tensor& add_in, const EmbeddingGraph& eg, int br,
                        const EmbeddingConfig& cfg) {
  Tensor gather_out = add_in;
  if (cfg.with_select) {
    // SelectV2(mask, fill, gather_out) re-applied on the [1, S, D] rows.
    auto sel = add_in.GetDefiningOp();
    ASSERT_TRUE(sel.HasValue());
    ASSERT_EQ(sel->Code(), kLiteRtOpCodeTflSelectV2);
    auto sel_ins = sel->Inputs();
    ASSERT_EQ(sel_ins.size(), 3);
    EXPECT_EQ(sel_ins[0], eg.mask[br]);
    EXPECT_EQ(sel_ins[1], eg.fill[br]);
    EXPECT_EQ(TensorDims(add_in), (Dims{1, kSeq, kEmb}));
    gather_out = sel_ins[2];
  }

  auto gather = gather_out.GetDefiningOp();
  ASSERT_TRUE(gather.HasValue());
  ASSERT_EQ(gather->Code(), kLiteRtOpCodeTflGather);
  EXPECT_EQ(TensorDims(gather_out), (Dims{1, kSeq, kEmb}));
  EXPECT_EQ(gather_out.ElementType(), ElementType::Float32);
  auto gather_opts = GetOptionsAs<GatherOptions>(Ctx(), gather->Get());
  ASSERT_TRUE(gather_opts.HasValue());
  EXPECT_EQ(gather_opts->axis, 0);
  EXPECT_EQ(gather_opts->batch_dims, 0);

  auto gather_ins = gather->Inputs();
  ASSERT_EQ(gather_ins.size(), 2);

  // Table: [depth + 1, D] f32 constant; rows 0..depth-1 = W^T, last row = 0.
  const Tensor& table = gather_ins[0];
  EXPECT_TRUE(table.IsConstant());
  EXPECT_EQ(table.ElementType(), ElementType::Float32);
  EXPECT_EQ(TensorDims(table), (Dims{kDepth + 1, kEmb}));
  auto table_data = table.WeightsData<float>();
  ASSERT_TRUE(table_data.HasValue());
  ASSERT_EQ(table_data->size(), (kDepth + 1) * kEmb);
  for (int32_t c = 0; c < kDepth; ++c) {
    for (int32_t r = 0; r < kEmb; ++r) {
      EXPECT_EQ((*table_data)[c * kEmb + r], eg.weights[br][r * kDepth + c])
          << "branch " << br << " row " << c << " col " << r;
    }
  }
  for (int32_t r = 0; r < kEmb; ++r) {
    EXPECT_EQ((*table_data)[kDepth * kEmb + r], 0.0f);
  }

  // safe = SelectV2(LogicalAnd(coord >= 0, coord < depth), coord, depth).
  const Tensor& safe = gather_ins[1];
  EXPECT_EQ(safe.ElementType(), ElementType::Int32);
  EXPECT_EQ(TensorDims(safe), (Dims{1, kSeq}));
  auto safe_sel = safe.GetDefiningOp();
  ASSERT_TRUE(safe_sel.HasValue());
  ASSERT_EQ(safe_sel->Code(), kLiteRtOpCodeTflSelectV2);
  auto safe_ins = safe_sel->Inputs();
  ASSERT_EQ(safe_ins.size(), 3);
  EXPECT_EQ(safe_ins[1], eg.coord[br]);
  auto depth_val = safe_ins[2].WeightsData<int32_t>();
  ASSERT_TRUE(depth_val.HasValue());
  ASSERT_EQ(depth_val->size(), 1);
  EXPECT_EQ((*depth_val)[0], kDepth);

  auto logical_and = safe_ins[0].GetDefiningOp();
  ASSERT_TRUE(logical_and.HasValue());
  ASSERT_EQ(logical_and->Code(), kLiteRtOpCodeTflLogicalAnd);
  auto and_ins = logical_and->Inputs();
  ASSERT_EQ(and_ins.size(), 2);

  auto ge = and_ins[0].GetDefiningOp();
  ASSERT_TRUE(ge.HasValue());
  EXPECT_EQ(ge->Code(), kLiteRtOpCodeTflGreaterEqual);
  auto ge_ins = ge->Inputs();
  ASSERT_EQ(ge_ins.size(), 2);
  EXPECT_EQ(ge_ins[0], eg.coord[br]);
  auto zero_val = ge_ins[1].WeightsData<int32_t>();
  ASSERT_TRUE(zero_val.HasValue());
  EXPECT_EQ((*zero_val)[0], 0);

  auto lt = and_ins[1].GetDefiningOp();
  ASSERT_TRUE(lt.HasValue());
  EXPECT_EQ(lt->Code(), kLiteRtOpCodeTflLess);
  auto lt_ins = lt->Inputs();
  ASSERT_EQ(lt_ins.size(), 2);
  EXPECT_EQ(lt_ins[0], eg.coord[br]);
  auto lt_depth = lt_ins[1].WeightsData<int32_t>();
  ASSERT_TRUE(lt_depth.HasValue());
  EXPECT_EQ((*lt_depth)[0], kDepth);
}

// Reads a scalar int32 constant (or returns INT_MIN).
int32_t ScalarI32(const Tensor& t) {
  auto data = t.WeightsData<int32_t>();
  if (!data || data->size() != 1) return INT_MIN;
  return (*data)[0];
}

std::vector<int32_t> I32Data(const Tensor& t) {
  auto data = t.WeightsData<int32_t>();
  if (!data) return {};
  return std::vector<int32_t>(data->begin(), data->end());
}

void ExpectFillValue(float actual, float fill) {
  if (std::isnan(fill)) {
    EXPECT_TRUE(std::isnan(actual)) << actual;
  } else {
    EXPECT_EQ(actual, fill);
  }
}

// Checks Slice(rows, begin, size) producing `t`.
void ExpectSliceOf(const Tensor& t, const Tensor& rows, const Dims& begin,
                   const Dims& size) {
  auto slice = t.GetDefiningOp();
  ASSERT_TRUE(slice.HasValue());
  ASSERT_EQ(slice->Code(), kLiteRtOpCodeTflSlice);
  auto ins = slice->Inputs();
  ASSERT_EQ(ins.size(), 3);
  EXPECT_EQ(ins[0], rows);
  EXPECT_EQ(I32Data(ins[1]), begin);
  EXPECT_EQ(I32Data(ins[2]), size);
  EXPECT_EQ(TensorDims(t), size);
}

// Checks idx = Minimum(Maximum(Add(c, 2), 0), depth + 2) with idx.dims ==
// c.dims.
void ExpectClampedIndex(const Tensor& idx, const Tensor& c) {
  EXPECT_EQ(idx.ElementType(), ElementType::Int32);
  EXPECT_EQ(TensorDims(idx), TensorDims(c));
  auto min_op = idx.GetDefiningOp();
  ASSERT_TRUE(min_op.HasValue());
  ASSERT_EQ(min_op->Code(), kLiteRtOpCodeTflMinimum);
  ASSERT_EQ(min_op->Inputs().size(), 2);
  EXPECT_EQ(ScalarI32(min_op->Inputs()[1]), kDepth + 2);
  auto max_op = min_op->Inputs()[0].GetDefiningOp();
  ASSERT_TRUE(max_op.HasValue());
  ASSERT_EQ(max_op->Code(), kLiteRtOpCodeTflMaximum);
  ASSERT_EQ(max_op->Inputs().size(), 2);
  EXPECT_EQ(ScalarI32(max_op->Inputs()[1]), 0);
  auto add_op = max_op->Inputs()[0].GetDefiningOp();
  ASSERT_TRUE(add_op.HasValue());
  ASSERT_EQ(add_op->Code(), kLiteRtOpCodeTflAdd);
  ASSERT_EQ(add_op->Inputs().size(), 2);
  EXPECT_EQ(add_op->Inputs()[0], c);
  EXPECT_EQ(ScalarI32(add_op->Inputs()[1]), 2);
  auto add_opts = GetOptionsAs<AddOptions>(Ctx(), add_op->Get());
  ASSERT_TRUE(add_opts.HasValue());
  EXPECT_EQ(add_opts->fused_activation_function, 0);
}

// Checks Gather(table, idx) (axis 0) producing `rows` and returns the table
// and index tensors through the out-params.
void ExpectGather(const Tensor& rows, const Dims& rows_dims, Tensor* table,
                  Tensor* idx) {
  auto gather = rows.GetDefiningOp();
  ASSERT_TRUE(gather.HasValue());
  ASSERT_EQ(gather->Code(), kLiteRtOpCodeTflGather);
  EXPECT_EQ(TensorDims(rows), rows_dims);
  EXPECT_EQ(rows.ElementType(), ElementType::Float32);
  auto gather_opts = GetOptionsAs<GatherOptions>(Ctx(), gather->Get());
  ASSERT_TRUE(gather_opts.HasValue());
  EXPECT_EQ(gather_opts->axis, 0);
  EXPECT_EQ(gather_opts->batch_dims, 0);
  auto gather_ins = gather->Inputs();
  ASSERT_EQ(gather_ins.size(), 2);
  *table = gather_ins[0];
  *idx = gather_ins[1];
}

// Checks the bool-free padded lookup branch feeding `add_in`:
//   idx  = Minimum(Maximum(coord + 2, 0), depth + 2)
//   rows = Gather(table [depth + 3, D], idx)
void ExpectPaddedBranch(const Tensor& add_in, const EmbeddingGraph& eg, int br,
                        const EmbeddingConfig& cfg) {
  const float fill = cfg.with_select ? cfg.fill : 0.0f;
  const int32_t rows = kDepth + 3;
  Tensor table = add_in;
  Tensor idx = add_in;
  ExpectGather(add_in, {1, kSeq, kEmb}, &table, &idx);
  if (::testing::Test::HasFatalFailure()) return;

  // Table rows: 0 <-> c <= -2, 1 <-> c == -1, 2 + k <-> c == k,
  // depth + 2 <-> c >= depth.
  EXPECT_TRUE(table.IsConstant());
  EXPECT_EQ(table.ElementType(), ElementType::Float32);
  ASSERT_EQ(TensorDims(table), (Dims{rows, kEmb}));
  auto table_data = table.WeightsData<float>();
  ASSERT_TRUE(table_data.HasValue());
  ASSERT_EQ(table_data->size(), rows * kEmb);
  auto at = [&](int32_t r, int32_t c) { return (*table_data)[r * kEmb + c]; };
  for (int32_t d = 0; d < kEmb; ++d) {
    ExpectFillValue(at(0, d), fill);
    ExpectFillValue(at(rows - 1, d), fill);
    EXPECT_EQ(at(1, d), 0.0f) << "branch " << br << " col " << d;
    for (int32_t k = 0; k < kDepth; ++k) {
      EXPECT_EQ(at(k + 2, d), eg.weights[br][d * kDepth + k])
          << "branch " << br << " row " << k + 2 << " col " << d;
    }
  }
  ExpectClampedIndex(idx, eg.coord[br]);
}

// Checks the folded RoPE chain of branch `br`:
//   idx  = Minimum(Maximum(src + 2, 0), depth + 2)            [1, S, 1]
//   rows = Gather(table [depth + 3, 2F], idx)                  [1, S, 1, 2F]
//   sin  = Slice(rows, [0, 0, 0, 0], [1, S, 1, F])
//   cos  = Slice(rows, [0, 0, 0, F], [1, S, 1, F])
void ExpectTrigFolded(const EmbeddingGraph& eg, int br) {
  const int32_t rows = kDepth + 3;
  const int32_t cols = 2 * kFreqs;
  auto sin_slice = eg.sin_out[br].GetDefiningOp();
  ASSERT_TRUE(sin_slice.HasValue());
  ASSERT_EQ(sin_slice->Code(), kLiteRtOpCodeTflSlice);
  const Tensor rows_t = sin_slice->Inputs()[0];
  ExpectSliceOf(eg.sin_out[br], rows_t, {0, 0, 0, 0}, {1, kSeq, 1, kFreqs});
  ExpectSliceOf(eg.cos_out[br], rows_t, {0, 0, 0, kFreqs},
                {1, kSeq, 1, kFreqs});

  Tensor table = rows_t;
  Tensor idx = rows_t;
  ExpectGather(rows_t, {1, kSeq, 1, cols}, &table, &idx);
  if (::testing::Test::HasFatalFailure()) return;
  EXPECT_TRUE(table.IsConstant());
  ASSERT_EQ(TensorDims(table), (Dims{rows, cols}));
  auto table_data = table.WeightsData<float>();
  ASSERT_TRUE(table_data.HasValue());
  ASSERT_EQ(table_data->size(), rows * cols);
  auto at = [&](int32_t r, int32_t c) { return (*table_data)[r * cols + c]; };
  for (int32_t c = 0; c < cols; ++c) {
    // Out-of-range coordinates (NaN embeddings) map to zero rows.
    EXPECT_EQ(at(0, c), 0.0f);
    EXPECT_EQ(at(rows - 1, c), 0.0f);
  }
  for (int32_t r = 1; r + 1 < rows; ++r) {
    for (int32_t j = 0; j < kFreqs; ++j) {
      const float angle = static_cast<float>(r - 2) * eg.freqs[br][j];
      EXPECT_FLOAT_EQ(at(r, j), std::sin(angle))
          << "branch " << br << " row " << r << " freq " << j;
      EXPECT_FLOAT_EQ(at(r, kFreqs + j), std::cos(angle))
          << "branch " << br << " row " << r << " freq " << j;
    }
  }
  ExpectClampedIndex(idx, eg.src[br]);

  // The downstream consumer still reads the original Sin/Cos tensors.
  auto consumer_ins = eg.trig_consumer[br].Inputs();
  ASSERT_EQ(consumer_ins.size(), 2);
  EXPECT_EQ(consumer_ins[0], eg.sin_out[br]);
  EXPECT_EQ(consumer_ins[1], eg.cos_out[br]);
}

// Applies EntryEmbeddingTrigFoldTransformation to every Cast (one builder per
// op, like the plugin driver) and returns the number of successful folds.
int RunTrigFold(LiteRtSubgraphT& subgraph) {
  int folds = 0;
  std::vector<LiteRtOp> casts;
  for (LiteRtOp op : subgraph.Ops()) {
    if (op->OpCode() == kLiteRtOpCodeTflCast) casts.push_back(op);
  }
  for (LiteRtOp cast : casts) {
    LiteRtBuilderT builder;
    if (EntryEmbeddingTrigFoldTransformation(Ctx(), &builder, cast) ==
        kLiteRtStatusOk) {
      builder.ApplyChanges(&subgraph);
      ++folds;
      EXPECT_TRUE(IsTopologicallySorted(subgraph)) << "after fold " << folds;
    }
  }
  return folds;
}

enum class EmbeddingPath { kLegacy, kPadded, kPaddedTrigFolded };

// Runs the plugin pipeline: EntryEmbeddingTransformation on the Sum, then
// EntryEmbeddingTrigFoldTransformation on every Cast, then OrphanCleanup to a
// fixpoint, checking the graph after each step.
void RunEmbeddingPositive(const EmbeddingConfig& cfg, EmbeddingPath path) {
  const bool legacy = path == EmbeddingPath::kLegacy;
  const bool folded = path == EmbeddingPath::kPaddedTrigFolded;
  const bool real_mask = cfg.with_select && cfg.real_mask;
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  EmbeddingGraph eg = BuildEmbeddingGraph(subgraph, cfg);
  ASSERT_TRUE(IsTopologicallySorted(subgraph));

  // Step 1: EntryEmbeddingTransformation. Only Sum and Concat are erased.
  {
    LiteRtBuilderT builder;
    ASSERT_EQ(EntryEmbeddingTransformation(Ctx(), &builder, eg.sum.Get()),
              kLiteRtStatusOk);
    builder.ApplyChanges(&subgraph);
  }
  EXPECT_TRUE(IsTopologicallySorted(subgraph));
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflSum), 0);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflConcatenation), 0);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflFullyConnected), 2);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflOneHot), 2);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflGather), 2);
  // Legacy: final Add only. Padded: + one Add(coord, 2) per branch.
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflAdd), legacy ? 1 : 3);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflMaximum), legacy ? 0 : 2);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflMinimum), legacy ? 0 : 2);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflSlice), 0);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflSin), cfg.trig ? 2 : 0);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflCos), cfg.trig ? 2 : 0);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflLogicalAnd),
            (legacy ? 2 : 0) + (real_mask ? 2 : 0));

  // The Add writes the original Sum output tensor.
  auto add = eg.sum_out.GetDefiningOp();
  ASSERT_TRUE(add.HasValue());
  ASSERT_EQ(add->Code(), kLiteRtOpCodeTflAdd);
  auto add_opts = GetOptionsAs<AddOptions>(Ctx(), add->Get());
  ASSERT_TRUE(add_opts.HasValue());
  EXPECT_EQ(add_opts->fused_activation_function, 0);
  auto add_ins = add->Inputs();
  ASSERT_EQ(add_ins.size(), 2);
  for (int br = 0; br < 2; ++br) {
    if (legacy) {
      ExpectLookupBranch(add_ins[br], eg, br, cfg);
    } else {
      ExpectPaddedBranch(add_ins[br], eg, br, cfg);
    }
  }

  // Step 2: EntryEmbeddingTrigFoldTransformation on every Cast.
  EXPECT_EQ(RunTrigFold(subgraph), folded ? 2 : 0);
  EXPECT_TRUE(IsTopologicallySorted(subgraph));
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflCast),
            cfg.trig && !folded ? 2 : 0);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflSin),
            cfg.trig && !folded ? 2 : 0);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflCos),
            cfg.trig && !folded ? 2 : 0);
  // Mul: RoPE angle Mul (unfolded) + downstream consumer, per branch.
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflMul),
            !cfg.trig ? 0 : (folded ? 2 : 4));
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflSlice), folded ? 4 : 0);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflGather), folded ? 4 : 2);
  if (folded) {
    for (int br = 0; br < 2; ++br) ExpectTrigFolded(eg, br);
  }

  // Step 3: OrphanCleanup removes the replaced sub-DAG.
  RunOrphanCleanup(subgraph);
  EXPECT_TRUE(IsTopologicallySorted(subgraph));
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflOneHot), 0);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflFullyConnected), 0);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflGather), folded ? 4 : 2);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflAdd),
            (legacy ? 1 : 3) + (folded ? 2 : 0));
  // Remaining Reshapes: coord Reshape(src) and unfolded RoPE angle Reshapes,
  // plus (legacy only) the still-used mask Reshapes.
  const int coord_reshapes = cfg.trig ? 2 : 0;
  const int angle_reshapes = cfg.trig && !folded ? 2 : 0;
  const int mask_reshapes = legacy && real_mask ? 2 : 0;
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflReshape),
            coord_reshapes + angle_reshapes + mask_reshapes);
  if (legacy) {
    // Remaining SelectV2: 2 safe-index selects (+ 2 re-applied mask selects).
    EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflSelectV2),
              cfg.with_select ? 4 : 2);
    for (const auto& sel : FindOps(subgraph, kLiteRtOpCodeTflSelectV2)) {
      // None of the original [1, S, depth] selects survive.
      EXPECT_NE(TensorDims(sel.Outputs()[0]), (Dims{1, kSeq, kDepth}));
    }
  } else {
    // Bool-free: no select / logical / comparison ops remain.
    for (LiteRtOpCode code :
         {kLiteRtOpCodeTflSelectV2, kLiteRtOpCodeTflLogicalAnd,
          kLiteRtOpCodeTflLogicalOr, kLiteRtOpCodeTflNotEqual,
          kLiteRtOpCodeTflLess, kLiteRtOpCodeTflGreaterEqual}) {
      EXPECT_EQ(CountOps(subgraph, code), 0) << "op code " << code;
    }
  }
  if (folded) {
    for (int br = 0; br < 2; ++br) ExpectTrigFolded(eg, br);
  }
}

class EntryEmbeddingTransformationTest : public ::testing::Test {
 protected:
  void SetUp() override { Reset(); }
  void TearDown() override { Reset(); }

 private:
  static void Reset() {
    unsetenv(kEmbeddingLegacyEnv);
    unsetenv(kEmbeddingNoTrigFoldEnv);
    ResetOrphanRegistry();
    ResetEntryEmbeddingTransformationState();
  }
};

TEST_F(EntryEmbeddingTransformationTest, PlainOneHotBranchesUsePaddedTable) {
  RunEmbeddingPositive(EmbeddingConfig{}, EmbeddingPath::kPadded);
}

TEST_F(EntryEmbeddingTransformationTest,
       SelectZeroFillPlainMaskUsesLegacyLookup) {
  EmbeddingConfig cfg;
  cfg.with_select = true;
  cfg.fill = 0.0f;
  RunEmbeddingPositive(cfg, EmbeddingPath::kLegacy);
}

TEST_F(EntryEmbeddingTransformationTest,
       SelectNanFillPlainMaskUsesLegacyLookup) {
  EmbeddingConfig cfg;
  cfg.with_select = true;
  cfg.fill = std::numeric_limits<float>::quiet_NaN();
  RunEmbeddingPositive(cfg, EmbeddingPath::kLegacy);
}

TEST_F(EntryEmbeddingTransformationTest, RealMaskNanFillUsesPaddedTable) {
  EmbeddingConfig cfg;
  cfg.with_select = true;
  cfg.real_mask = true;
  cfg.fill = std::numeric_limits<float>::quiet_NaN();
  RunEmbeddingPositive(cfg, EmbeddingPath::kPadded);
}

TEST_F(EntryEmbeddingTransformationTest, RealMaskZeroFillUsesPaddedTable) {
  EmbeddingConfig cfg;
  cfg.with_select = true;
  cfg.real_mask = true;
  cfg.fill = 0.0f;
  RunEmbeddingPositive(cfg, EmbeddingPath::kPadded);
}

TEST_F(EntryEmbeddingTransformationTest,
       MaskMismatchInOneBranchFallsBackToLegacy) {
  EmbeddingConfig cfg;
  cfg.with_select = true;
  cfg.real_mask = true;
  cfg.branch1_mask_ne_value = -2;
  cfg.fill = std::numeric_limits<float>::quiet_NaN();
  RunEmbeddingPositive(cfg, EmbeddingPath::kLegacy);
}

TEST_F(EntryEmbeddingTransformationTest, RealMaskNanFillTrigFolded) {
  EmbeddingConfig cfg;
  cfg.with_select = true;
  cfg.real_mask = true;
  cfg.trig = true;
  cfg.fill = std::numeric_limits<float>::quiet_NaN();
  RunEmbeddingPositive(cfg, EmbeddingPath::kPaddedTrigFolded);
}

TEST_F(EntryEmbeddingTransformationTest, ZeroFillDoesNotFoldTrig) {
  EmbeddingConfig cfg;
  cfg.with_select = true;
  cfg.real_mask = true;
  cfg.trig = true;
  cfg.fill = 0.0f;
  RunEmbeddingPositive(cfg, EmbeddingPath::kPadded);
}

TEST_F(EntryEmbeddingTransformationTest, NoTrigFoldEnvKeepsTrigChain) {
  setenv(kEmbeddingNoTrigFoldEnv, "1", /*overwrite=*/1);
  EmbeddingConfig cfg;
  cfg.with_select = true;
  cfg.real_mask = true;
  cfg.trig = true;
  cfg.fill = std::numeric_limits<float>::quiet_NaN();
  RunEmbeddingPositive(cfg, EmbeddingPath::kPadded);
}

TEST_F(EntryEmbeddingTransformationTest, LegacyEnvForcesLegacyLookup) {
  setenv(kEmbeddingLegacyEnv, "1", /*overwrite=*/1);
  EmbeddingConfig cfg;
  cfg.with_select = true;
  cfg.real_mask = true;
  cfg.trig = true;
  cfg.fill = std::numeric_limits<float>::quiet_NaN();
  RunEmbeddingPositive(cfg, EmbeddingPath::kLegacy);
}

TEST_F(EntryEmbeddingTransformationTest, LegacyEnvForcesLegacyForPlainOneHot) {
  setenv(kEmbeddingLegacyEnv, "1", /*overwrite=*/1);
  RunEmbeddingPositive(EmbeddingConfig{}, EmbeddingPath::kLegacy);
}

TEST_F(EntryEmbeddingTransformationTest, LegacyEnvZeroIsIgnored) {
  setenv(kEmbeddingLegacyEnv, "0", /*overwrite=*/1);
  RunEmbeddingPositive(EmbeddingConfig{}, EmbeddingPath::kPadded);
}

// The padded path never reads the bool masks, so the splice point is valid
// even when branch 1's mask is computed after branch 0's FullyConnected.
TEST_F(EntryEmbeddingTransformationTest,
       PaddedInterleavedMasksStayTopologicallySorted) {
  EmbeddingConfig cfg;
  cfg.with_select = true;
  cfg.real_mask = true;
  cfg.interleave_masks = true;
  cfg.fill = std::numeric_limits<float>::quiet_NaN();
  RunEmbeddingPositive(cfg, EmbeddingPath::kPadded);
}

// Order coord0, trig0, coord1, trig1: branch 1's coord Reshape comes after
// branch 0's Cast.
TEST_F(EntryEmbeddingTransformationTest,
       TrigFoldInterleavedOrderStaysTopologicallySorted) {
  EmbeddingConfig cfg;
  cfg.with_select = true;
  cfg.real_mask = true;
  cfg.trig = true;
  cfg.trig_order = TrigOrder::kInterleaved;
  cfg.fill = std::numeric_limits<float>::quiet_NaN();
  RunEmbeddingPositive(cfg, EmbeddingPath::kPaddedTrigFolded);
}

// Both trig chains ahead of both coord Reshapes.
TEST_F(EntryEmbeddingTransformationTest,
       TrigFoldTrigBeforeCoordStaysTopologicallySorted) {
  EmbeddingConfig cfg;
  cfg.with_select = true;
  cfg.real_mask = true;
  cfg.trig = true;
  cfg.trig_order = TrigOrder::kTrigFirst;
  cfg.fill = std::numeric_limits<float>::quiet_NaN();
  RunEmbeddingPositive(cfg, EmbeddingPath::kPaddedTrigFolded);
}

TEST_F(EntryEmbeddingTransformationTest,
       TrigFoldInterleavedMasksAndTrigStayTopologicallySorted) {
  EmbeddingConfig cfg;
  cfg.with_select = true;
  cfg.real_mask = true;
  cfg.interleave_masks = true;
  cfg.trig = true;
  cfg.trig_order = TrigOrder::kInterleaved;
  cfg.fill = std::numeric_limits<float>::quiet_NaN();
  RunEmbeddingPositive(cfg, EmbeddingPath::kPaddedTrigFolded);
}

// The legacy path re-applies SelectV2(mask1, ...) at the splice point; branch
// 1's mask producers come after branch 0's FullyConnected.
TEST_F(EntryEmbeddingTransformationTest,
       LegacyInterleavedMasksStayTopologicallySorted) {
  EmbeddingConfig cfg;
  cfg.with_select = true;
  cfg.real_mask = true;
  cfg.interleave_masks = true;
  cfg.branch1_mask_ne_value = -2;
  cfg.fill = std::numeric_limits<float>::quiet_NaN();
  RunEmbeddingPositive(cfg, EmbeddingPath::kLegacy);
}

// Without a prior EntryEmbeddingTransformation the Cast is not registered.
TEST_F(EntryEmbeddingTransformationTest, TrigFoldRequiresRegisteredCast) {
  EmbeddingConfig cfg;
  cfg.with_select = true;
  cfg.real_mask = true;
  cfg.trig = true;
  cfg.fill = std::numeric_limits<float>::quiet_NaN();
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  BuildEmbeddingGraph(subgraph, cfg);
  const size_t num_ops = subgraph.Ops().size();
  EXPECT_EQ(RunTrigFold(subgraph), 0);
  EXPECT_EQ(subgraph.Ops().size(), num_ops);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflCast), 2);
}

// ResetEntryEmbeddingTransformationState() drops pending trig folds.
TEST_F(EntryEmbeddingTransformationTest, ResetStateDropsPendingTrigFolds) {
  EmbeddingConfig cfg;
  cfg.with_select = true;
  cfg.real_mask = true;
  cfg.trig = true;
  cfg.fill = std::numeric_limits<float>::quiet_NaN();
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  EmbeddingGraph eg = BuildEmbeddingGraph(subgraph, cfg);
  LiteRtBuilderT builder;
  ASSERT_EQ(EntryEmbeddingTransformation(Ctx(), &builder, eg.sum.Get()),
            kLiteRtStatusOk);
  builder.ApplyChanges(&subgraph);
  ResetEntryEmbeddingTransformationState();
  EXPECT_EQ(RunTrigFold(subgraph), 0);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflSin), 2);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflCos), 2);
}

void ExpectEmbeddingNoMatch(const EmbeddingConfig& cfg) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  EmbeddingGraph eg = BuildEmbeddingGraph(subgraph, cfg);
  const size_t num_ops = subgraph.Ops().size();

  LiteRtBuilderT builder;
  EXPECT_EQ(EntryEmbeddingTransformation(Ctx(), &builder, eg.sum.Get()),
            kLiteRtStatusPatternNoMatch);
  builder.ApplyChanges(&subgraph);
  EXPECT_EQ(subgraph.Ops().size(), num_ops);
  // Nothing registered for cleanup.
  RunOrphanCleanup(subgraph);
  EXPECT_EQ(subgraph.Ops().size(), num_ops);
}

TEST_F(EntryEmbeddingTransformationTest, RootNotSumRejected) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  GraphBuilder g;
  Tensor a = g.F32({1, kSeq, kEmb});
  Tensor b = g.F32({1, kSeq, kEmb});
  Tensor out = g.F32({1, kSeq, kEmb});
  Op add = g.AddOp(kLiteRtOpCodeTflAdd, {a, b}, {out});
  g.ApplyTo(&subgraph);

  LiteRtBuilderT builder;
  EXPECT_EQ(EntryEmbeddingTransformation(Ctx(), &builder, add.Get()),
            kLiteRtStatusPatternNoMatch);
}

TEST_F(EntryEmbeddingTransformationTest, WrongWeightRowsRejected) {
  EmbeddingConfig cfg;
  cfg.weight_dims = {kEmb + 1, kDepth};
  ExpectEmbeddingNoMatch(cfg);
}

TEST_F(EntryEmbeddingTransformationTest, TransposedWeightRejected) {
  EmbeddingConfig cfg;
  cfg.weight_dims = {kDepth, kEmb};
  ExpectEmbeddingNoMatch(cfg);
}

TEST_F(EntryEmbeddingTransformationTest, OneHotOnValueNotOneRejected) {
  EmbeddingConfig cfg;
  cfg.on_value = 2.0f;
  ExpectEmbeddingNoMatch(cfg);
}

TEST_F(EntryEmbeddingTransformationTest, OneHotOffValueNotZeroRejected) {
  EmbeddingConfig cfg;
  cfg.off_value = -1.0f;
  ExpectEmbeddingNoMatch(cfg);
}

TEST_F(EntryEmbeddingTransformationTest, SelectFillNotZeroOrNanRejected) {
  EmbeddingConfig cfg;
  cfg.with_select = true;
  cfg.fill = 1.0f;
  ExpectEmbeddingNoMatch(cfg);
}

TEST_F(EntryEmbeddingTransformationTest, RealMaskFillNotZeroOrNanRejected) {
  EmbeddingConfig cfg;
  cfg.with_select = true;
  cfg.real_mask = true;
  cfg.trig = true;
  cfg.fill = 1.0f;
  ExpectEmbeddingNoMatch(cfg);
}

//===----------------------------------------------------------------------===//
// RopeTransformation
//===----------------------------------------------------------------------===//

constexpr int32_t kRopeB = 1;
constexpr int32_t kRopeS = 3;
constexpr int32_t kRopeH = 2;
constexpr int32_t kRopeD = 8;

// Functions rather than globals: `Dims` is not trivially destructible.
Dims RopeFullDims() { return {kRopeB, kRopeS, kRopeH, kRopeD}; }
Dims RopeHalfDims() { return {kRopeB, kRopeS, kRopeH, kRopeD / 2}; }
Dims RopeQuarterDims() { return {kRopeB, kRopeS, kRopeH, kRopeD / 4}; }
Dims RopeTrigDims() { return {kRopeB, kRopeS, 1, kRopeD / 4}; }
Dims RopeTableDims() { return {kRopeB, kRopeS, 1, kRopeD}; }

struct RopeConfig {
  bool wrong_inner_op = false;           // Mul instead of Sub in the lo half.
  bool swapped_quarter_offsets = false;  // s0/s1 slice offsets swapped.
};

struct RopeTrig {
  Tensor cos_lo, sin_lo, cos_hi, sin_hi;
};

struct RopeBlock {
  Tensor in;
  Tensor out;
  Tensor s[4];
  Op root;
};

void SetConcatOptions(GraphBuilder& g, const Op& op, int32_t axis) {
  ConcatenationOptions opts;
  opts.axis = axis;
  opts.fused_activation_function = 0;
  g.SetOptions(op, std::move(opts));
}

RopeTrig BuildRopeTrig(GraphBuilder& g) {
  const Dims dims = RopeTrigDims();
  return RopeTrig{g.F32(dims), g.F32(dims), g.F32(dims), g.F32(dims)};
}

// Builds the documented 4 Slice + 8 Mul + 2 Sub + 2 Add + 3 Concat block.
RopeBlock BuildRopeBlock(GraphBuilder& g, const RopeTrig& trig,
                         const RopeConfig& cfg) {
  RopeBlock blk;
  blk.in = g.F32(RopeFullDims());

  auto slice = [&](const Tensor& src, int32_t begin_last, const Dims& dims) {
    Tensor begin = g.Const<int32_t>({4}, {0, 0, 0, begin_last});
    Tensor size = g.Const<int32_t>({4}, dims);
    Tensor out = g.F32(dims);
    g.AddOp(kLiteRtOpCodeTflSlice, {src, begin, size}, {out});
    return out;
  };
  auto mul = [&](const Tensor& a, const Tensor& b) {
    Tensor out = g.F32(RopeQuarterDims());
    Op op = g.AddOp(kLiteRtOpCodeTflMul, {a, b}, {out});
    MulOptions opts;
    opts.fused_activation_function = 0;
    g.SetOptions(op, std::move(opts));
    return out;
  };
  // Concat(Sub(a * cos, b * sin), Add(b * cos, a * sin)).
  auto rotate = [&](const Tensor& a, const Tensor& b, const Tensor& cos,
                    const Tensor& sin, bool wrong_inner_op) {
    Tensor m0 = mul(a, cos);
    Tensor m1 = mul(b, sin);
    Tensor m2 = mul(b, cos);
    Tensor m3 = mul(a, sin);
    Tensor sub_out = g.F32(RopeQuarterDims());
    if (wrong_inner_op) {
      Op op = g.AddOp(kLiteRtOpCodeTflMul, {m0, m1}, {sub_out});
      MulOptions opts;
      opts.fused_activation_function = 0;
      g.SetOptions(op, std::move(opts));
    } else {
      Op op = g.AddOp(kLiteRtOpCodeTflSub, {m0, m1}, {sub_out});
      SubOptions opts;
      opts.fused_activation_function = 0;
      g.SetOptions(op, std::move(opts));
    }
    Tensor add_out = g.F32(RopeQuarterDims());
    Op add = g.AddOp(kLiteRtOpCodeTflAdd, {m2, m3}, {add_out});
    AddOptions add_opts;
    add_opts.fused_activation_function = 0;
    g.SetOptions(add, std::move(add_opts));
    Tensor half_out = g.F32(RopeHalfDims());
    Op concat =
        g.AddOp(kLiteRtOpCodeTflConcatenation, {sub_out, add_out}, {half_out});
    SetConcatOptions(g, concat, 3);
    return half_out;
  };

  Tensor half_lo = slice(blk.in, 0, RopeHalfDims());
  Tensor half_hi = slice(blk.in, kRopeD / 2, RopeHalfDims());
  const int32_t q0 = cfg.swapped_quarter_offsets ? kRopeD / 4 : 0;
  const int32_t q1 = cfg.swapped_quarter_offsets ? 0 : kRopeD / 4;
  blk.s[0] = slice(half_lo, q0, RopeQuarterDims());
  blk.s[1] = slice(half_lo, q1, RopeQuarterDims());
  blk.s[2] = slice(half_hi, 0, RopeQuarterDims());
  blk.s[3] = slice(half_hi, kRopeD / 4, RopeQuarterDims());

  Tensor lo =
      rotate(blk.s[0], blk.s[1], trig.cos_lo, trig.sin_lo, cfg.wrong_inner_op);
  Tensor hi = rotate(blk.s[2], blk.s[3], trig.cos_hi, trig.sin_hi,
                     /*wrong_inner_op=*/false);
  blk.out = g.F32(RopeFullDims());
  blk.root = g.AddOp(kLiteRtOpCodeTflConcatenation, {lo, hi}, {blk.out});
  SetConcatOptions(g, blk.root, 3);
  return blk;
}

void ExpectConcatInputs(const Tensor& t, const std::vector<Tensor>& expected) {
  auto concat = t.GetDefiningOp();
  ASSERT_TRUE(concat.HasValue());
  ASSERT_EQ(concat->Code(), kLiteRtOpCodeTflConcatenation);
  auto opts = GetOptionsAs<ConcatenationOptions>(Ctx(), concat->Get());
  ASSERT_TRUE(opts.HasValue());
  EXPECT_EQ(opts->axis, 3);
  auto ins = concat->Inputs();
  ASSERT_EQ(ins.size(), expected.size());
  for (size_t i = 0; i < ins.size(); ++i) {
    EXPECT_EQ(ins[i], expected[i]) << "concat input " << i;
  }
}

// Checks `neg` == Mul(sin, -1).
void ExpectNegated(const Tensor& neg, const Tensor& sin) {
  auto mul = neg.GetDefiningOp();
  ASSERT_TRUE(mul.HasValue());
  ASSERT_EQ(mul->Code(), kLiteRtOpCodeTflMul);
  auto ins = mul->Inputs();
  ASSERT_EQ(ins.size(), 2);
  EXPECT_EQ(ins[0], sin);
  auto minus_one = ins[1].WeightsData<float>();
  ASSERT_TRUE(minus_one.HasValue());
  ASSERT_EQ(minus_one->size(), 1);
  EXPECT_EQ((*minus_one)[0], -1.0f);
  EXPECT_EQ(TensorDims(neg), RopeTrigDims());
}

// Checks the rewritten block and returns the (cos_tab, sin_tab) tensors.
void ExpectRopeRewritten(const RopeBlock& blk, const RopeTrig& trig,
                         Tensor* cos_tab_out, Tensor* sin_tab_out) {
  auto add = blk.out.GetDefiningOp();
  ASSERT_TRUE(add.HasValue());
  ASSERT_EQ(add->Code(), kLiteRtOpCodeTflAdd);
  auto add_ins = add->Inputs();
  ASSERT_EQ(add_ins.size(), 2);

  // term1 = Mul(in, cos_tab).
  auto term1 = add_ins[0].GetDefiningOp();
  ASSERT_TRUE(term1.HasValue());
  ASSERT_EQ(term1->Code(), kLiteRtOpCodeTflMul);
  auto t1_ins = term1->Inputs();
  ASSERT_EQ(t1_ins.size(), 2);
  EXPECT_EQ(t1_ins[0], blk.in);
  const Tensor cos_tab = t1_ins[1];
  EXPECT_EQ(TensorDims(cos_tab), RopeTableDims());
  ExpectConcatInputs(cos_tab,
                     {trig.cos_lo, trig.cos_lo, trig.cos_hi, trig.cos_hi});

  // term2 = Mul(Concat(s1, s0, s3, s2), sin_tab).
  auto term2 = add_ins[1].GetDefiningOp();
  ASSERT_TRUE(term2.HasValue());
  ASSERT_EQ(term2->Code(), kLiteRtOpCodeTflMul);
  auto t2_ins = term2->Inputs();
  ASSERT_EQ(t2_ins.size(), 2);
  EXPECT_EQ(TensorDims(t2_ins[0]), RopeFullDims());
  ExpectConcatInputs(t2_ins[0], {blk.s[1], blk.s[0], blk.s[3], blk.s[2]});
  const Tensor sin_tab = t2_ins[1];
  EXPECT_EQ(TensorDims(sin_tab), RopeTableDims());
  auto sin_concat = sin_tab.GetDefiningOp();
  ASSERT_TRUE(sin_concat.HasValue());
  ASSERT_EQ(sin_concat->Code(), kLiteRtOpCodeTflConcatenation);
  auto sin_ins = sin_concat->Inputs();
  ASSERT_EQ(sin_ins.size(), 4);
  ExpectNegated(sin_ins[0], trig.sin_lo);
  EXPECT_EQ(sin_ins[1], trig.sin_lo);
  ExpectNegated(sin_ins[2], trig.sin_hi);
  EXPECT_EQ(sin_ins[3], trig.sin_hi);

  if (cos_tab_out) *cos_tab_out = cos_tab;
  if (sin_tab_out) *sin_tab_out = sin_tab;
}

TEST(RopeTransformationTest, PositiveMatchAndRewrite) {
  ResetRopeTransformationState();
  ResetOrphanRegistry();
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  GraphBuilder g;
  RopeTrig trig = BuildRopeTrig(g);
  RopeBlock blk = BuildRopeBlock(g, trig, RopeConfig{});
  g.ApplyTo(&subgraph);
  // 6 Slice + 8 Mul + 2 Sub + 2 Add + 3 Concat.
  ASSERT_EQ(subgraph.Ops().size(), 21);

  LiteRtBuilderT builder;
  ASSERT_EQ(RopeTransformation(Ctx(), &builder, blk.root.Get()),
            kLiteRtStatusOk);
  builder.ApplyChanges(&subgraph);

  EXPECT_FALSE(ContainsOp(subgraph, blk.root.Get()));
  EXPECT_TRUE(IsTopologicallySorted(subgraph));
  ExpectRopeRewritten(blk, trig, nullptr, nullptr);

  // The old lo/hi halves are dead and removed by orphan cleanup, leaving
  // 6 Slice + (2 Mul(-1) + 2 table Concat + 1 Concat + 2 Mul + 1 Add).
  RunOrphanCleanup(subgraph);
  EXPECT_TRUE(IsTopologicallySorted(subgraph));
  EXPECT_EQ(subgraph.Ops().size(), 14);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflSlice), 6);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflSub), 0);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflAdd), 1);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflMul), 4);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflConcatenation), 3);
  ExpectRopeRewritten(blk, trig, nullptr, nullptr);
  ResetRopeTransformationState();
}

TEST(RopeTransformationTest, TrigTablesSharedAcrossBlocks) {
  ResetRopeTransformationState();
  ResetOrphanRegistry();
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  GraphBuilder g;
  RopeTrig trig = BuildRopeTrig(g);
  RopeBlock q = BuildRopeBlock(g, trig, RopeConfig{});
  RopeBlock k = BuildRopeBlock(g, trig, RopeConfig{});
  g.ApplyTo(&subgraph);

  for (const RopeBlock* blk : {&q, &k}) {
    LiteRtBuilderT builder;
    ASSERT_EQ(RopeTransformation(Ctx(), &builder, blk->root.Get()),
              kLiteRtStatusOk);
    builder.ApplyChanges(&subgraph);
  }
  RunOrphanCleanup(subgraph);
  EXPECT_TRUE(IsTopologicallySorted(subgraph));

  Tensor q_cos, q_sin, k_cos, k_sin;
  ExpectRopeRewritten(q, trig, &q_cos, &q_sin);
  ExpectRopeRewritten(k, trig, &k_cos, &k_sin);
  EXPECT_EQ(q_cos, k_cos);
  EXPECT_EQ(q_sin, k_sin);
  // 2 shared table Concats + 1 rotate Concat per block.
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflConcatenation), 4);
  ResetRopeTransformationState();
}

TEST(RopeTransformationTest, NonConcatRootRejected) {
  ResetRopeTransformationState();
  ResetOrphanRegistry();
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  GraphBuilder g;
  Tensor a = g.F32(RopeFullDims());
  Tensor b = g.F32(RopeFullDims());
  Tensor out = g.F32(RopeFullDims());
  Op add = g.AddOp(kLiteRtOpCodeTflAdd, {a, b}, {out});
  g.ApplyTo(&subgraph);

  LiteRtBuilderT builder;
  EXPECT_EQ(RopeTransformation(Ctx(), &builder, add.Get()),
            kLiteRtStatusPatternNoMatch);
}

void ExpectRopeNoMatch(const RopeConfig& cfg) {
  ResetRopeTransformationState();
  ResetOrphanRegistry();
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  GraphBuilder g;
  RopeTrig trig = BuildRopeTrig(g);
  RopeBlock blk = BuildRopeBlock(g, trig, cfg);
  g.ApplyTo(&subgraph);
  const size_t num_ops = subgraph.Ops().size();

  LiteRtBuilderT builder;
  EXPECT_EQ(RopeTransformation(Ctx(), &builder, blk.root.Get()),
            kLiteRtStatusPatternNoMatch);
  builder.ApplyChanges(&subgraph);
  RunOrphanCleanup(subgraph);
  EXPECT_EQ(subgraph.Ops().size(), num_ops);
  EXPECT_TRUE(ContainsOp(subgraph, blk.root.Get()));
}

TEST(RopeTransformationTest, WrongInnerOpRejected) {
  RopeConfig cfg;
  cfg.wrong_inner_op = true;
  ExpectRopeNoMatch(cfg);
}

TEST(RopeTransformationTest, WrongSliceOffsetsRejected) {
  RopeConfig cfg;
  cfg.swapped_quarter_offsets = true;
  ExpectRopeNoMatch(cfg);
}

//===----------------------------------------------------------------------===//
// OrphanCleanupTransformation
//===----------------------------------------------------------------------===//

struct OrphanGraph {
  Op producer;  // Unregistered producer of `a`.
  Op mul0;
  Op mul1;
  Op add;   // Consumes mul0 and mul1; output has no users.
  Op live;  // Optional extra live user of mul0's output.
};

OrphanGraph BuildOrphanGraph(LiteRtSubgraphT& subgraph, bool extra_live_user) {
  GraphBuilder g;
  OrphanGraph og;
  const Dims dims = {1, 4};
  Tensor x = g.F32(dims);
  Tensor a = g.F32(dims);
  og.producer = g.AddOp(kLiteRtOpCodeTflAbs, {x}, {a});
  Tensor b = g.F32(dims);
  Tensor m0 = g.F32(dims);
  og.mul0 = g.AddOp(kLiteRtOpCodeTflMul, {a, b}, {m0});
  Tensor m1 = g.F32(dims);
  og.mul1 = g.AddOp(kLiteRtOpCodeTflMul, {b, b}, {m1});
  Tensor add_out = g.F32(dims);
  og.add = g.AddOp(kLiteRtOpCodeTflAdd, {m0, m1}, {add_out});
  if (extra_live_user) {
    Tensor live_out = g.F32(dims);
    og.live = g.AddOp(kLiteRtOpCodeTflAbs, {m0}, {live_out});
  }
  g.ApplyTo(&subgraph);
  return og;
}

TEST(OrphanCleanupTransformationTest, DeadRegisteredGroupErased) {
  ResetOrphanRegistry();
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  OrphanGraph og = BuildOrphanGraph(subgraph, /*extra_live_user=*/false);
  RegisterOrphanGroup({og.mul0.Get(), og.mul1.Get(), og.add.Get()});

  LiteRtBuilderT builder;
  ASSERT_EQ(OrphanCleanupTransformation(Ctx(), &builder, og.add.Get()),
            kLiteRtStatusOk);
  builder.ApplyChanges(&subgraph);

  // Only the unregistered producer survives.
  ASSERT_EQ(subgraph.Ops().size(), 1);
  EXPECT_EQ(subgraph.Ops()[0], og.producer.Get());

  // Erased ops are removed from the registry.
  LiteRtBuilderT builder2;
  EXPECT_EQ(OrphanCleanupTransformation(Ctx(), &builder2, og.producer.Get()),
            kLiteRtStatusPatternNoMatch);
}

TEST(OrphanCleanupTransformationTest, RegisteredOpWithUsersNotErased) {
  ResetOrphanRegistry();
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  OrphanGraph og = BuildOrphanGraph(subgraph, /*extra_live_user=*/false);
  // mul0's output is still consumed by the (unregistered) add.
  RegisterOrphanGroup({og.mul0.Get()});

  LiteRtBuilderT builder;
  EXPECT_EQ(OrphanCleanupTransformation(Ctx(), &builder, og.mul0.Get()),
            kLiteRtStatusPatternNoMatch);
  builder.ApplyChanges(&subgraph);
  EXPECT_EQ(subgraph.Ops().size(), 4);
}

TEST(OrphanCleanupTransformationTest, UnregisteredDeadOpNotErased) {
  ResetOrphanRegistry();
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  OrphanGraph og = BuildOrphanGraph(subgraph, /*extra_live_user=*/false);

  LiteRtBuilderT builder;
  EXPECT_EQ(OrphanCleanupTransformation(Ctx(), &builder, og.add.Get()),
            kLiteRtStatusPatternNoMatch);
  builder.ApplyChanges(&subgraph);
  EXPECT_EQ(subgraph.Ops().size(), 4);
}

TEST(OrphanCleanupTransformationTest, RegisteredProducerWithLiveUserKept) {
  ResetOrphanRegistry();
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  OrphanGraph og = BuildOrphanGraph(subgraph, /*extra_live_user=*/true);
  RegisterOrphanGroup({og.mul0.Get(), og.mul1.Get(), og.add.Get()});

  LiteRtBuilderT builder;
  ASSERT_EQ(OrphanCleanupTransformation(Ctx(), &builder, og.add.Get()),
            kLiteRtStatusOk);
  builder.ApplyChanges(&subgraph);

  // add and mul1 are erased; mul0 still feeds the live Abs.
  EXPECT_EQ(subgraph.Ops().size(), 3);
  EXPECT_TRUE(ContainsOp(subgraph, og.producer.Get()));
  EXPECT_TRUE(ContainsOp(subgraph, og.mul0.Get()));
  EXPECT_TRUE(ContainsOp(subgraph, og.live.Get()));
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflAdd), 0);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflMul), 1);
}

TEST(OrphanCleanupTransformationTest, ResetClearsRegistry) {
  ResetOrphanRegistry();
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  OrphanGraph og = BuildOrphanGraph(subgraph, /*extra_live_user=*/false);
  RegisterOrphanGroup({og.mul0.Get(), og.mul1.Get(), og.add.Get()});
  ResetOrphanRegistry();

  LiteRtBuilderT builder;
  EXPECT_EQ(OrphanCleanupTransformation(Ctx(), &builder, og.add.Get()),
            kLiteRtStatusPatternNoMatch);
}

//===----------------------------------------------------------------------===//
// AttentionMaskTransformation
//===----------------------------------------------------------------------===//

TEST(AttentionMaskTransformationTest, PositiveMatch) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  GraphBuilder g;
  Tensor x = g.F32({1, 4, 1, 4});
  Tensor zero = g.Const<float>({}, {0.0f});
  Tensor ne_out = g.Bool({1, 4, 1, 4});
  Tensor lnot_out = g.Bool({1, 4, 1, 4});
  Tensor cast_out = g.F32({1, 4, 1, 4});

  Op ne = g.AddOp(kLiteRtOpCodeTflNotEqual, {x, zero}, {ne_out});
  Op lnot = g.AddOp(kLiteRtOpCodeTflLogicalNot, {ne_out}, {lnot_out});
  Op cast = g.AddOp(kLiteRtOpCodeTflCast, {lnot_out}, {cast_out});
  g.ApplyTo(&subgraph);

  LiteRtBuilderT builder;
  EXPECT_EQ(AttentionMaskTransformation(Ctx(), &builder, cast.Get()),
            kLiteRtStatusOk);
  builder.ApplyChanges(&subgraph);

  EXPECT_FALSE(ContainsOp(subgraph, cast.Get()));
  EXPECT_FALSE(ContainsOp(subgraph, lnot.Get()));
  EXPECT_FALSE(ContainsOp(subgraph, ne.Get()));
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflSub), 1);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflLogicalNot), 0);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflNotEqual), 0);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflCast), 0);
}

TEST(AttentionMaskTransformationTest, NegativeNotZeroConst) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  GraphBuilder g;
  Tensor x = g.F32({1, 4, 1, 4});
  Tensor one = g.Const<float>({}, {1.0f});
  Tensor ne_out = g.Bool({1, 4, 1, 4});
  Tensor lnot_out = g.Bool({1, 4, 1, 4});
  Tensor cast_out = g.F32({1, 4, 1, 4});

  g.AddOp(kLiteRtOpCodeTflNotEqual, {x, one}, {ne_out});
  g.AddOp(kLiteRtOpCodeTflLogicalNot, {ne_out}, {lnot_out});
  Op cast = g.AddOp(kLiteRtOpCodeTflCast, {lnot_out}, {cast_out});
  g.ApplyTo(&subgraph);

  LiteRtBuilderT builder;
  EXPECT_EQ(AttentionMaskTransformation(Ctx(), &builder, cast.Get()),
            kLiteRtStatusPatternNoMatch);
}

TEST(AttentionMaskTransformationTest, PositiveMatchOuterProductMul) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  GraphBuilder g;
  const int32_t S = 4;
  Tensor mask = g.F32({1, S});
  Tensor shape_col = g.Const<int32_t>({3}, {1, S, 1});
  Tensor col_mask = g.F32({1, S, 1});
  g.AddOp(kLiteRtOpCodeTflReshape, {mask, shape_col}, {col_mask});

  Tensor mul_out = g.F32({1, S, S});
  Op mul = g.AddOp(kLiteRtOpCodeTflMul, {mask, col_mask}, {mul_out});
  g.ApplyTo(&subgraph);

  LiteRtBuilderT builder;
  EXPECT_EQ(AttentionMaskTransformation(Ctx(), &builder, mul.Get()),
            kLiteRtStatusOk);
  builder.ApplyChanges(&subgraph);

  EXPECT_FALSE(ContainsOp(subgraph, mul.Get()));
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflMul), 0);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflBatchMatmul), 1);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflReshape), 2);
}

TEST(AttentionMaskTransformationTest, EndToEndAttentionMaskTransformation) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  GraphBuilder g;
  const int32_t S = 4;
  Tensor mask = g.F32({1, S});
  Tensor shape_col = g.Const<int32_t>({3}, {1, S, 1});
  Tensor col_mask = g.F32({1, S, 1});
  g.AddOp(kLiteRtOpCodeTflReshape, {mask, shape_col}, {col_mask});

  Tensor mul_out = g.F32({1, S, S});
  Op mul = g.AddOp(kLiteRtOpCodeTflMul, {mask, col_mask}, {mul_out});

  Tensor shape_4d = g.Const<int32_t>({4}, {1, S, 1, S});
  Tensor x = g.F32({1, S, 1, S});
  g.AddOp(kLiteRtOpCodeTflReshape, {mul_out, shape_4d}, {x});

  Tensor zero = g.Const<float>({}, {0.0f});
  Tensor ne_out = g.Bool({1, S, 1, S});
  Tensor lnot_out = g.Bool({1, S, 1, S});
  Tensor cast_out = g.F32({1, S, 1, S});

  Op ne = g.AddOp(kLiteRtOpCodeTflNotEqual, {x, zero}, {ne_out});
  Op lnot = g.AddOp(kLiteRtOpCodeTflLogicalNot, {ne_out}, {lnot_out});
  Op cast = g.AddOp(kLiteRtOpCodeTflCast, {lnot_out}, {cast_out});
  g.ApplyTo(&subgraph);

  // Apply Pattern 1 to Mul
  {
    LiteRtBuilderT builder;
    EXPECT_EQ(AttentionMaskTransformation(Ctx(), &builder, mul.Get()),
              kLiteRtStatusOk);
    builder.ApplyChanges(&subgraph);
  }

  // Apply Pattern 2 to Cast
  {
    LiteRtBuilderT builder;
    EXPECT_EQ(AttentionMaskTransformation(Ctx(), &builder, cast.Get()),
              kLiteRtStatusOk);
    builder.ApplyChanges(&subgraph);
  }

  EXPECT_FALSE(ContainsOp(subgraph, cast.Get()));
  EXPECT_FALSE(ContainsOp(subgraph, lnot.Get()));
  EXPECT_FALSE(ContainsOp(subgraph, ne.Get()));
  EXPECT_FALSE(ContainsOp(subgraph, mul.Get()));
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflMul), 0);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflBatchMatmul), 1);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflSub), 1);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflCast), 0);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflLogicalNot), 0);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflNotEqual), 0);
}

}  // namespace
}  // namespace litert::mediatek
