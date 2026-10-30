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

#include "litert/vendors/qualcomm/compiler/transformations/attention_chunk_transformation.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/compiler/cc/litert_builder.h"
#include "litert/compiler/cc/litert_matchers.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/compiler/cc/litert_op_options.h"
#include "litert/vendors/qualcomm/compiler/transformations/orphan_cleanup_transformation.h"

namespace {

using litert::BuildLayout;
using litert::ElementType;
using litert::Expected;
using litert::Layout;
using litert::RankedTensorType;
using litert::compiler::BatchMatmulOptions;
using litert::compiler::Builder;
using litert::compiler::ConcatenationOptions;
using litert::compiler::GetOptionsAs;
using litert::compiler::m_AllOf;
using litert::compiler::m_Any;
using litert::compiler::m_CaptureOrSameAs;
using litert::compiler::m_Custom;
using litert::compiler::m_HasOneUse;
using litert::compiler::m_Op;
using litert::compiler::m_Shape;
using litert::compiler::Op;
using litert::compiler::RankedTensorSpecBuilder;
using litert::compiler::ReshapeOptions;
using litert::compiler::SoftmaxOptions;
using litert::compiler::SplitOptions;
using litert::compiler::Tensor;

// Default minimum logits tensor size in bytes to trigger auto-chunking.
constexpr int64_t kDefaultAutoMinLogitsBytes = 8 << 20;
// Maximum per-chunk logits size in bytes when splitting along queries.
constexpr int64_t kAutoMaxChunkBytes = 4 << 20;
// Minimum sequence length to fold SHA Q/Softmax/Out into 4D Crouton tiles.
constexpr int32_t kMinSequenceFor4DSha = 600;

struct Config {
  int head_chunks = 1;
  int query_chunks = 1;
  bool absorb_tail = true;
  bool auto_select = false;
  int64_t min_logits_bytes = kDefaultAutoMinLogitsBytes;
  bool Enabled() const {
    return auto_select || (head_chunks >= 1 && query_chunks >= 1 &&
                           head_chunks * query_chunks > 1);
  }
};

Config& GetConfig() {
  static auto* config = new Config();
  return *config;
}

// First BatchMatmuls of the chunks built by this rewrite; they have the same
// shape of chain and must not be chunked again.
absl::flat_hash_set<LiteRtOp>& Produced() {
  static auto* produced = new absl::flat_hash_set<LiteRtOp>();
  return *produced;
}

std::vector<int32_t> Dims(const Tensor& t) {
  auto type = t.RankedTensorType();
  if (!type) return {};
  auto dims = type->Layout().Dimensions();
  return std::vector<int32_t>(dims.begin(), dims.end());
}

enum class Kind { kFloat, kQuant, kUnsupported };

Kind KindOf(const Tensor& t) {
  const auto et = t.ElementType();
  if (et == ElementType::Float32 && !t.HasQuantization()) return Kind::kFloat;
  if ((et == ElementType::Int16 || et == ElementType::Int8) &&
      t.QTypeId() == kLiteRtQuantizationPerTensor) {
    return Kind::kQuant;
  }
  return Kind::kUnsupported;
}

bool HasPerm(const Op& transpose, const std::vector<int32_t>& expected) {
  if (transpose.Code() != kLiteRtOpCodeTflTranspose ||
      transpose.Inputs().size() != 2) {
    return false;
  }
  Tensor perm = transpose.Inputs()[1];
  if (!perm.IsConstant()) return false;
  auto perm_data = perm.WeightsData<int32_t>();
  return perm_data &&
         std::vector<int32_t>(perm_data->begin(), perm_data->end()) == expected;
}

// Returns the single user of `t` if it has exactly one use, with opcode
// `code`, consuming `t` as its `arg`-th input.
std::optional<Op> SingleUser(const Tensor& t, LiteRtOpCode code,
                             LiteRtParamIndex arg) {
  auto uses = t.Uses();
  if (uses.size() != 1 || uses[0].user.Code() != code ||
      uses[0].user_arg_ind != arg) {
    return std::nullopt;
  }
  return uses[0].user;
}

std::optional<Tensor> SingleOutput(const Op& op) {
  auto outs = op.Outputs();
  if (outs.size() != 1) return std::nullopt;
  return outs[0];
}

bool IsPlainBatchMatmul(const Op& op) {
  if (op.Code() != kLiteRtOpCodeTflBatchMatmul || op.Inputs().size() != 2 ||
      op.Outputs().size() != 1) {
    return false;
  }
  auto opts = GetOptionsAs<BatchMatmulOptions>(op.ctx(), op.Get());
  return opts && !opts->adj_x && !opts->adj_y;
}

// The matched attention core.
struct Match {
  Op bmm1, reshape1, softmax, reshape2, bmm2;
  Tensor q, kt, v, logits, r1_out, sm_out, r2_out, out;
  int32_t h, s, k, d, dv;
  float beta;
  bool asymmetric_quantize_input;
};

std::optional<Match> MatchAttention(const LiteRtCompilerContext* ctx,
                                    LiteRtOp root) {
  Op bmm1(ctx, root);
  if (!IsPlainBatchMatmul(bmm1)) return std::nullopt;
  Tensor q = bmm1.Inputs()[0];
  Tensor kt = bmm1.Inputs()[1];
  Tensor logits = bmm1.Outputs()[0];
  const auto qd = Dims(q);
  const auto ktd = Dims(kt);
  const auto ld = Dims(logits);
  if (qd.size() != 3 || ktd.size() != 3 || ld.size() != 3) return std::nullopt;
  const int32_t h = qd[0], s = qd[1], d = qd[2], k = ktd[2];
  if (ktd[0] != h || ktd[1] != d || ld != std::vector<int32_t>{h, s, k}) {
    return std::nullopt;
  }

  auto reshape1 = SingleUser(logits, kLiteRtOpCodeTflReshape, 0);
  if (!reshape1) return std::nullopt;
  auto r1_out = SingleOutput(*reshape1);
  if (!r1_out || Dims(*r1_out) != std::vector<int32_t>{1, h, s, k}) {
    return std::nullopt;
  }
  auto softmax = SingleUser(*r1_out, kLiteRtOpCodeTflSoftmax, 0);
  if (!softmax || softmax->Inputs().size() != 1) return std::nullopt;
  auto sm_out = SingleOutput(*softmax);
  if (!sm_out || Dims(*sm_out) != std::vector<int32_t>{1, h, s, k}) {
    return std::nullopt;
  }
  auto sm_opts = GetOptionsAs<SoftmaxOptions>(ctx, softmax->Get());
  if (!sm_opts) return std::nullopt;
  auto reshape2 = SingleUser(*sm_out, kLiteRtOpCodeTflReshape, 0);
  if (!reshape2) return std::nullopt;
  auto r2_out = SingleOutput(*reshape2);
  if (!r2_out || Dims(*r2_out) != std::vector<int32_t>{h, s, k}) {
    return std::nullopt;
  }
  // P must be the LHS of the second BatchMatmul.
  auto bmm2 = SingleUser(*r2_out, kLiteRtOpCodeTflBatchMatmul, 0);
  if (!bmm2 || !IsPlainBatchMatmul(*bmm2)) return std::nullopt;
  Tensor v = bmm2->Inputs()[1];
  Tensor out = bmm2->Outputs()[0];
  const auto vd = Dims(v);
  if (vd.size() != 3 || vd[0] != h || vd[1] != k) return std::nullopt;
  const int32_t dv = vd[2];
  if (Dims(out) != std::vector<int32_t>{h, s, dv}) return std::nullopt;

  // Both BatchMatmuls must use the same input quantization mode.
  auto bmm1_opts = GetOptionsAs<BatchMatmulOptions>(ctx, bmm1.Get());
  auto bmm2_opts = GetOptionsAs<BatchMatmulOptions>(ctx, bmm2->Get());
  if (!bmm1_opts || !bmm2_opts ||
      bmm1_opts->asymmetric_quantize_input !=
          bmm2_opts->asymmetric_quantize_input) {
    return std::nullopt;
  }

  const Kind kind = KindOf(q);
  if (kind == Kind::kUnsupported) return std::nullopt;
  for (const Tensor* t :
       {&kt, &logits, &*r1_out, &*sm_out, &*r2_out, &v, &out}) {
    if (KindOf(*t) != kind) return std::nullopt;
  }
  return Match{bmm1,
               *reshape1,
               *softmax,
               *reshape2,
               *bmm2,
               q,
               kt,
               v,
               logits,
               *r1_out,
               *sm_out,
               *r2_out,
               out,
               h,
               s,
               k,
               d,
               dv,
               sm_opts->beta,
               bmm1_opts->asymmetric_quantize_input};
}

// Pre-attention 4D -> 3D prologue:
//   q_4d [1, S, H, D]  -> Transpose(0,2,1,3) [1, H, S, D]  -> Reshape [H, S, D]
//   k_4d [1, K, H, D]  -> Transpose(0,2,3,1) [1, H, D, K]  -> Reshape [H, D, K]
//   v_4d [1, K, H, Dv] -> Transpose(0,2,1,3) [1, H, K, Dv] -> Reshape [H, K,
//   Dv]
struct PreHead {
  Op q_transpose, q_reshape;
  Op k_transpose, k_reshape;
  Op v_transpose, v_reshape;
  Tensor q_4d, k_4d, v_4d;
};

std::optional<PreHead> MatchPreHead(const Match& m) {
  PreHead p;
  const Kind kind = KindOf(m.q);
  auto match_head = [&](const Tensor& t_3d, const std::vector<int32_t>& t_dims,
                        const std::vector<int32_t>& perm,
                        const std::vector<int32_t>& in_dims, Op* out_reshape,
                        Op* out_transpose, Tensor* out_4d) {
    auto in_m = m_CaptureOrSameAs(
        out_4d, m_AllOf(m_Shape(in_dims), m_Custom([&](const Tensor& t) {
                          return KindOf(t) == kind;
                        })));
    auto tr_m =
        m_AllOf(m_HasOneUse(), m_Shape(t_dims),
                m_CaptureOrSameAs(
                    out_transpose,
                    m_AllOf(m_Custom([&](const Tensor& t) {
                              auto op = t.GetDefiningOp();
                              return op && HasPerm(*op, perm);
                            }),
                            m_Op<kLiteRtOpCodeTflTranspose>(in_m, m_Any()))));
    auto r_m =
        m_AllOf(m_HasOneUse(),
                m_CaptureOrSameAs(
                    out_reshape, m_Op<kLiteRtOpCodeTflReshape>(tr_m, m_Any())));
    return litert::compiler::Match(t_3d, r_m);
  };
  if (!match_head(m.q, {1, m.h, m.s, m.d}, {0, 2, 1, 3}, {1, m.s, m.h, m.d},
                  &p.q_reshape, &p.q_transpose, &p.q_4d) ||
      !match_head(m.kt, {1, m.h, m.d, m.k}, {0, 2, 3, 1}, {1, m.k, m.h, m.d},
                  &p.k_reshape, &p.k_transpose, &p.k_4d) ||
      !match_head(m.v, {1, m.h, m.k, m.dv}, {0, 2, 1, 3}, {1, m.k, m.h, m.dv},
                  &p.v_reshape, &p.v_transpose, &p.v_4d)) {
    return std::nullopt;
  }
  return p;
}

// Output epilogue out [H, S, Dv] -> Reshape [1, H, S, Dv] -> Transpose
// (0, 2, 1, 3) -> [1, S, H, Dv] -> Quantize (-> optional Reshape [1, S, H*Dv]).
struct Tail {
  Op reshape, transpose, quantize;
  Tensor r_out, t_out, q_out;
  bool has_flat_reshape = false;
  Op flat_reshape;
  Tensor flat_out;
};

std::optional<Tail> MatchTail(const Match& m) {
  auto reshape = SingleUser(m.out, kLiteRtOpCodeTflReshape, 0);
  if (!reshape) return std::nullopt;
  auto r_out = SingleOutput(*reshape);
  if (!r_out || Dims(*r_out) != std::vector<int32_t>{1, m.h, m.s, m.dv}) {
    return std::nullopt;
  }
  auto transpose = SingleUser(*r_out, kLiteRtOpCodeTflTranspose, 0);
  if (!transpose || !HasPerm(*transpose, {0, 2, 1, 3})) return std::nullopt;
  auto t_out = SingleOutput(*transpose);
  if (!t_out || Dims(*t_out) != std::vector<int32_t>{1, m.s, m.h, m.dv}) {
    return std::nullopt;
  }
  auto quantize = SingleUser(*t_out, kLiteRtOpCodeTflQuantize, 0);
  if (!quantize || quantize->Inputs().size() != 1) return std::nullopt;
  auto q_out = SingleOutput(*quantize);
  if (!q_out || Dims(*q_out) != std::vector<int32_t>{1, m.s, m.h, m.dv}) {
    return std::nullopt;
  }
  const Kind kind = KindOf(m.out);
  if (kind != Kind::kQuant || KindOf(*r_out) != kind ||
      KindOf(*t_out) != kind || KindOf(*q_out) != Kind::kQuant) {
    return std::nullopt;
  }
  Tail tail{*reshape, *transpose, *quantize, *r_out, *t_out,
            *q_out,   false,      *reshape,  *q_out};
  if (auto flat_r = SingleUser(*q_out, kLiteRtOpCodeTflReshape, 0)) {
    if (auto f_out = SingleOutput(*flat_r)) {
      if (Dims(*f_out) == std::vector<int32_t>{1, m.s, m.h * m.dv} &&
          KindOf(*f_out) == Kind::kQuant) {
        tail.has_flat_reshape = true;
        tail.flat_reshape = *flat_r;
        tail.flat_out = *f_out;
      }
    }
  }
  return tail;
}

int64_t ElementBytes(const Tensor& t) {
  switch (t.ElementType()) {
    case ElementType::Int8:
      return 1;
    case ElementType::Int16:
      return 2;
    default:
      return 4;
  }
}

// Returns {head_chunks, query_chunks} for auto mode, or nullopt if the
// attention is too small to benefit.
std::optional<std::pair<int, int>> AutoSelect(const Match& m,
                                              int64_t min_logits_bytes) {
  const int64_t bytes = ElementBytes(m.logits);
  const int64_t logits_bytes = static_cast<int64_t>(m.h) * m.s * m.k * bytes;
  if (m.s < 2 || logits_bytes < min_logits_bytes) return std::nullopt;
  if (m.h > 1) return std::make_pair(static_cast<int>(m.h), 1);
  for (int32_t n = 2; n <= m.s; ++n) {
    if (m.s % n == 0 &&
        static_cast<int64_t>(m.s / n) * m.k * bytes <= kAutoMaxChunkBytes) {
      return std::make_pair(1, static_cast<int>(n));
    }
  }
  return std::nullopt;
}

Expected<Tensor> NewLike(Builder& builder, const Tensor& like,
                         const std::vector<int32_t>& dims) {
  RankedTensorType type(
      like.ElementType(),
      Layout(BuildLayout(dims.data(), dims.data() + dims.size())));
  auto spec = RankedTensorSpecBuilder(type);
  if (like.QTypeId() == kLiteRtQuantizationPerTensor) {
    spec =
        std::move(spec).WithPerTensorQuantization(like.PerTensorQuantization());
  }
  return builder.BuildTensor(std::move(spec).Build());
}

Expected<Tensor> ConstI32(Builder& builder, const std::vector<int32_t>& data) {
  const std::vector<int32_t> dims = {static_cast<int32_t>(data.size())};
  RankedTensorType type(ElementType::Int32,
                        Layout(BuildLayout(dims.data(), dims.data() + 1)));
  LITERT_ASSIGN_OR_RETURN(
      auto t, builder.BuildTensor(RankedTensorSpecBuilder(type).Build()));
  LITERT_RETURN_IF_ERROR(
      builder.BuildWeights<int32_t>(absl::MakeConstSpan(data), t));
  return t;
}

Expected<Tensor> BuildSlice(Builder& builder, const Tensor& in,
                            const std::vector<int32_t>& begin,
                            const std::vector<int32_t>& size) {
  LITERT_ASSIGN_OR_RETURN(auto out, NewLike(builder, in, size));
  LITERT_ASSIGN_OR_RETURN(auto b, ConstI32(builder, begin));
  LITERT_ASSIGN_OR_RETURN(auto s, ConstI32(builder, size));
  LITERT_RETURN_IF_ERROR(
      builder.BuildOp(kLiteRtOpCodeTflSlice, {in, b, s}, {out}));
  return out;
}

Expected<void> BuildBatchMatmul(Builder& builder, const Tensor& a,
                                const Tensor& b, const Tensor& out,
                                bool asymmetric_quantize_input,
                                bool adj_y = false, LiteRtOp* built = nullptr) {
  LITERT_ASSIGN_OR_RETURN(
      auto op, builder.BuildOp(kLiteRtOpCodeTflBatchMatmul, {a, b}, {out}));
  BatchMatmulOptions opts;
  opts.adj_x = false;
  opts.adj_y = adj_y;
  opts.asymmetric_quantize_input = asymmetric_quantize_input;
  LITERT_RETURN_IF_ERROR(builder.SetOpOptions(op, std::move(opts)));
  if (built != nullptr) *built = op.Get();
  return {};
}

Expected<void> BuildReshapeInto(Builder& builder, const Tensor& in,
                                const std::vector<int32_t>& dims,
                                const Tensor& out) {
  LITERT_ASSIGN_OR_RETURN(auto shape, ConstI32(builder, dims));
  LITERT_ASSIGN_OR_RETURN(
      auto op, builder.BuildOp(kLiteRtOpCodeTflReshape, {in, shape}, {out}));
  ReshapeOptions opts;
  opts.new_shape = dims;
  LITERT_RETURN_IF_ERROR(builder.SetOpOptions(op, std::move(opts)));
  return {};
}

Expected<Tensor> BuildReshape(Builder& builder, const Tensor& in,
                              const Tensor& like,
                              const std::vector<int32_t>& dims) {
  LITERT_ASSIGN_OR_RETURN(auto out, NewLike(builder, like, dims));
  LITERT_RETURN_IF_ERROR(BuildReshapeInto(builder, in, dims, out));
  return out;
}

Expected<void> BuildConcat(Builder& builder, const std::vector<Tensor>& ins,
                           int32_t axis, const Tensor& out) {
  LITERT_ASSIGN_OR_RETURN(
      auto op, builder.BuildOp(kLiteRtOpCodeTflConcatenation, ins, {out}));
  ConcatenationOptions opts;
  opts.axis = axis;
  opts.fused_activation_function = 0;
  LITERT_RETURN_IF_ERROR(builder.SetOpOptions(op, std::move(opts)));
  return {};
}

Expected<std::vector<Tensor>> BuildSplit3D(Builder& builder,
                                           const Tensor& in_4d, int32_t s,
                                           int32_t h, int32_t d) {
  LITERT_ASSIGN_OR_RETURN(auto in_3d,
                          BuildReshape(builder, in_4d, in_4d, {1, s, h * d}));
  LITERT_ASSIGN_OR_RETURN(auto axis_t, ConstI32(builder, {2}));
  std::vector<Tensor> outs;
  outs.reserve(h);
  for (int32_t i = 0; i < h; ++i) {
    LITERT_ASSIGN_OR_RETURN(auto t, NewLike(builder, in_4d, {1, s, d}));
    outs.push_back(t);
  }
  LITERT_ASSIGN_OR_RETURN(
      auto op, builder.BuildOp(kLiteRtOpCodeTflSplit, {axis_t, in_3d}, outs));
  SplitOptions opts;
  opts.num_splits = h;
  LITERT_RETURN_IF_ERROR(builder.SetOpOptions(op, std::move(opts)));
  return outs;
}

Expected<std::vector<Tensor>> BuildSplit4D(Builder& builder,
                                           const Tensor& in_4d,
                                           const std::vector<int32_t>& flat_4d,
                                           int32_t h, int32_t d) {
  LITERT_ASSIGN_OR_RETURN(auto reshaped,
                          BuildReshape(builder, in_4d, in_4d, flat_4d));
  LITERT_ASSIGN_OR_RETURN(auto axis_t, ConstI32(builder, {3}));
  std::vector<Tensor> outs;
  outs.reserve(h);
  const std::vector<int32_t> head_dims = {flat_4d[0], flat_4d[1], flat_4d[2],
                                          d};
  for (int32_t i = 0; i < h; ++i) {
    LITERT_ASSIGN_OR_RETURN(auto t, NewLike(builder, in_4d, head_dims));
    outs.push_back(t);
  }
  LITERT_ASSIGN_OR_RETURN(auto op, builder.BuildOp(kLiteRtOpCodeTflSplit,
                                                   {axis_t, reshaped}, outs));
  SplitOptions opts;
  opts.num_splits = h;
  LITERT_RETURN_IF_ERROR(builder.SetOpOptions(op, std::move(opts)));
  return outs;
}

std::optional<std::pair<int32_t, int32_t>> BestCroutonSpatialDims(int32_t s) {
  auto ceil8 = [](int32_t x) { return (x + 7) / 8; };
  int32_t best_tiles = ceil8(1) * ceil8(s);
  std::optional<std::pair<int32_t, int32_t>> best;
  for (int32_t h_s = 2; h_s * h_s <= s; ++h_s) {
    if (s % h_s != 0) continue;
    const int32_t w_s = s / h_s;
    const int32_t tiles = ceil8(h_s) * ceil8(w_s);
    if (tiles < best_tiles) {
      best_tiles = tiles;
      best = std::make_pair(h_s, w_s);
    }
  }
  return best;
}

Expected<std::vector<Tensor>> BuildSplitAxis1(Builder& builder,
                                              const Tensor& in_3d,
                                              int32_t q_chunks, int32_t qc,
                                              int32_t d) {
  LITERT_ASSIGN_OR_RETURN(auto axis_t, ConstI32(builder, {1}));
  std::vector<Tensor> outs;
  outs.reserve(q_chunks);
  for (int32_t i = 0; i < q_chunks; ++i) {
    LITERT_ASSIGN_OR_RETURN(auto t, NewLike(builder, in_3d, {1, qc, d}));
    outs.push_back(t);
  }
  LITERT_ASSIGN_OR_RETURN(
      auto op, builder.BuildOp(kLiteRtOpCodeTflSplit, {axis_t, in_3d}, outs));
  SplitOptions opts;
  opts.num_splits = q_chunks;
  LITERT_RETURN_IF_ERROR(builder.SetOpOptions(op, std::move(opts)));
  return outs;
}

Expected<std::vector<Tensor>> BuildSplit4DAxis2(Builder& builder,
                                                const Tensor& in_4d,
                                                int32_t h_s, int32_t q_chunks,
                                                int32_t wc, int32_t d) {
  LITERT_ASSIGN_OR_RETURN(auto axis_t, ConstI32(builder, {2}));
  std::vector<Tensor> outs;
  outs.reserve(q_chunks);
  for (int32_t i = 0; i < q_chunks; ++i) {
    LITERT_ASSIGN_OR_RETURN(auto t, NewLike(builder, in_4d, {1, h_s, wc, d}));
    outs.push_back(t);
  }
  LITERT_ASSIGN_OR_RETURN(
      auto op, builder.BuildOp(kLiteRtOpCodeTflSplit, {axis_t, in_4d}, outs));
  SplitOptions opts;
  opts.num_splits = q_chunks;
  LITERT_RETURN_IF_ERROR(builder.SetOpOptions(op, std::move(opts)));
  return outs;
}

// Qualcomm MHA -> SHA rewrite (`mha_to_sha.cc` style) when splitting into
// single heads (`head_chunks == m.h && query_chunks >= 1` with `PreHead`).
// When `m.s >= kMinSequenceFor4DSha` and `m.s` factors into `(H_s, W_s)` with
// fewer 8x8 Crouton tiles than `(1, S)`, uses 4D `[1, H_s, W_s, D]` for Q,
// Softmax, and Out while broadcasting K/V from `[1, 1, K, D]`.
Expected<void> BuildQualcommMhaToSha(Builder& builder, const Match& m,
                                     const PreHead& pre, int query_chunks,
                                     const Tail* tail,
                                     std::vector<LiteRtOp>& produced) {
  const bool use_4d_sha = m.s >= kMinSequenceFor4DSha;
  const auto sha_4d_dims =
      use_4d_sha ? BestCroutonSpatialDims(m.s) : std::nullopt;
  if (sha_4d_dims && sha_4d_dims->second % query_chunks == 0) {
    const int32_t h_s = sha_4d_dims->first;
    const int32_t w_s = sha_4d_dims->second;
    const int32_t wc = w_s / query_chunks;
    LITERT_ASSIGN_OR_RETURN(
        auto q_heads,
        BuildSplit4D(builder, pre.q_4d, {1, h_s, w_s, m.h * m.d}, m.h, m.d));
    LITERT_ASSIGN_OR_RETURN(
        auto k_heads,
        BuildSplit4D(builder, pre.k_4d, {1, 1, m.k, m.h * m.d}, m.h, m.d));
    LITERT_ASSIGN_OR_RETURN(
        auto v_heads,
        BuildSplit4D(builder, pre.v_4d, {1, 1, m.k, m.h * m.dv}, m.h, m.dv));

    std::vector<Tensor> sha_outs;
    sha_outs.reserve(m.h);
    for (int32_t hi = 0; hi < m.h; ++hi) {
      std::vector<Tensor> q_chunks_vec;
      if (query_chunks > 1) {
        LITERT_ASSIGN_OR_RETURN(q_chunks_vec,
                                BuildSplit4DAxis2(builder, q_heads[hi], h_s,
                                                  query_chunks, wc, m.d));
      } else {
        q_chunks_vec.push_back(q_heads[hi]);
      }

      std::vector<Tensor> q_outs;
      q_outs.reserve(query_chunks);
      for (int32_t qi = 0; qi < query_chunks; ++qi) {
        LITERT_ASSIGN_OR_RETURN(auto logits,
                                NewLike(builder, m.logits, {1, h_s, wc, m.k}));
        LiteRtOp bmm1 = nullptr;
        LITERT_RETURN_IF_ERROR(BuildBatchMatmul(
            builder, q_chunks_vec[qi], k_heads[hi], logits,
            m.asymmetric_quantize_input, /*adj_y=*/true, &bmm1));
        produced.push_back(bmm1);

        LITERT_ASSIGN_OR_RETURN(auto sm,
                                NewLike(builder, m.sm_out, {1, h_s, wc, m.k}));
        LITERT_ASSIGN_OR_RETURN(
            auto sm_op,
            builder.BuildOp(kLiteRtOpCodeTflSoftmax, {logits}, {sm}));
        SoftmaxOptions sm_opts;
        sm_opts.beta = m.beta;
        LITERT_RETURN_IF_ERROR(builder.SetOpOptions(sm_op, std::move(sm_opts)));

        LITERT_ASSIGN_OR_RETURN(auto o,
                                NewLike(builder, m.out, {1, h_s, wc, m.dv}));
        LITERT_RETURN_IF_ERROR(BuildBatchMatmul(builder, sm, v_heads[hi], o,
                                                m.asymmetric_quantize_input,
                                                /*adj_y=*/false));
        if (tail != nullptr) {
          LITERT_ASSIGN_OR_RETURN(
              auto o_q, NewLike(builder, tail->q_out, {1, h_s, wc, m.dv}));
          LITERT_RETURN_IF_ERROR(
              builder.BuildOp(kLiteRtOpCodeTflQuantize, {o}, {o_q}));
          q_outs.push_back(o_q);
        } else {
          q_outs.push_back(o);
        }
      }

      if (query_chunks == 1) {
        sha_outs.push_back(q_outs[0]);
      } else {
        const Tensor& like_t = tail != nullptr ? tail->q_out : m.out;
        LITERT_ASSIGN_OR_RETURN(auto head_out,
                                NewLike(builder, like_t, {1, h_s, w_s, m.dv}));
        LITERT_RETURN_IF_ERROR(
            BuildConcat(builder, q_outs, /*axis=*/2, head_out));
        sha_outs.push_back(head_out);
      }
    }

    if (tail != nullptr) {
      LITERT_ASSIGN_OR_RETURN(
          auto concat_out,
          NewLike(builder, tail->q_out, {1, h_s, w_s, m.h * m.dv}));
      LITERT_RETURN_IF_ERROR(
          BuildConcat(builder, sha_outs, /*axis=*/3, concat_out));
      if (tail->has_flat_reshape) {
        LITERT_RETURN_IF_ERROR(BuildReshapeInto(
            builder, concat_out, {1, m.s, m.h * m.dv}, tail->flat_out));
      } else {
        LITERT_RETURN_IF_ERROR(BuildReshapeInto(
            builder, concat_out, {1, m.s, m.h, m.dv}, tail->q_out));
      }
    } else {
      LITERT_ASSIGN_OR_RETURN(
          auto concat_out, NewLike(builder, m.out, {1, h_s, w_s, m.h * m.dv}));
      LITERT_RETURN_IF_ERROR(
          BuildConcat(builder, sha_outs, /*axis=*/3, concat_out));
      LITERT_RETURN_IF_ERROR(
          BuildReshapeInto(builder, concat_out, {m.h, m.s, m.dv}, m.out));
    }
    return {};
  }

  LITERT_ASSIGN_OR_RETURN(auto q_heads,
                          BuildSplit3D(builder, pre.q_4d, m.s, m.h, m.d));
  LITERT_ASSIGN_OR_RETURN(auto k_heads,
                          BuildSplit3D(builder, pre.k_4d, m.k, m.h, m.d));
  LITERT_ASSIGN_OR_RETURN(auto v_heads,
                          BuildSplit3D(builder, pre.v_4d, m.k, m.h, m.dv));

  const int32_t qc = m.s / query_chunks;
  std::vector<Tensor> sha_outs;
  sha_outs.reserve(m.h);
  for (int32_t hi = 0; hi < m.h; ++hi) {
    std::vector<Tensor> q_chunks_vec;
    if (query_chunks > 1) {
      LITERT_ASSIGN_OR_RETURN(
          q_chunks_vec,
          BuildSplitAxis1(builder, q_heads[hi], query_chunks, qc, m.d));
    } else {
      q_chunks_vec.push_back(q_heads[hi]);
    }

    std::vector<Tensor> q_outs;
    q_outs.reserve(query_chunks);
    for (int32_t qi = 0; qi < query_chunks; ++qi) {
      LITERT_ASSIGN_OR_RETURN(auto logits,
                              NewLike(builder, m.logits, {1, qc, m.k}));
      LiteRtOp bmm1 = nullptr;
      LITERT_RETURN_IF_ERROR(
          BuildBatchMatmul(builder, q_chunks_vec[qi], k_heads[hi], logits,
                           m.asymmetric_quantize_input, /*adj_y=*/true, &bmm1));
      produced.push_back(bmm1);

      LITERT_ASSIGN_OR_RETURN(auto sm,
                              NewLike(builder, m.sm_out, {1, qc, m.k}));
      LITERT_ASSIGN_OR_RETURN(
          auto sm_op, builder.BuildOp(kLiteRtOpCodeTflSoftmax, {logits}, {sm}));
      SoftmaxOptions sm_opts;
      sm_opts.beta = m.beta;
      LITERT_RETURN_IF_ERROR(builder.SetOpOptions(sm_op, std::move(sm_opts)));

      LITERT_ASSIGN_OR_RETURN(auto o, NewLike(builder, m.out, {1, qc, m.dv}));
      LITERT_RETURN_IF_ERROR(BuildBatchMatmul(builder, sm, v_heads[hi], o,
                                              m.asymmetric_quantize_input,
                                              /*adj_y=*/false));
      if (tail != nullptr) {
        LITERT_ASSIGN_OR_RETURN(auto o_q,
                                NewLike(builder, tail->q_out, {1, qc, m.dv}));
        LITERT_RETURN_IF_ERROR(
            builder.BuildOp(kLiteRtOpCodeTflQuantize, {o}, {o_q}));
        q_outs.push_back(o_q);
      } else {
        q_outs.push_back(o);
      }
    }

    if (query_chunks == 1) {
      sha_outs.push_back(q_outs[0]);
    } else {
      const Tensor& like_t = tail != nullptr ? tail->q_out : m.out;
      LITERT_ASSIGN_OR_RETURN(auto head_out,
                              NewLike(builder, like_t, {1, m.s, m.dv}));
      LITERT_RETURN_IF_ERROR(
          BuildConcat(builder, q_outs, /*axis=*/1, head_out));
      sha_outs.push_back(head_out);
    }
  }

  if (tail != nullptr) {
    if (tail->has_flat_reshape) {
      LITERT_RETURN_IF_ERROR(
          BuildConcat(builder, sha_outs, /*axis=*/2, tail->flat_out));
    } else {
      LITERT_ASSIGN_OR_RETURN(
          auto concat_out, NewLike(builder, tail->q_out, {1, m.s, m.h * m.dv}));
      LITERT_RETURN_IF_ERROR(
          BuildConcat(builder, sha_outs, /*axis=*/2, concat_out));
      LITERT_RETURN_IF_ERROR(BuildReshapeInto(
          builder, concat_out, {1, m.s, m.h, m.dv}, tail->q_out));
    }
  } else {
    LITERT_RETURN_IF_ERROR(BuildConcat(builder, sha_outs, /*axis=*/0, m.out));
  }
  return {};
}

// Builds the chunked attention core.
Expected<void> BuildChunks(Builder& builder, const Match& m, int head_chunks,
                           int query_chunks, const Tail* tail,
                           std::vector<LiteRtOp>& produced) {
  const int32_t hc = m.h / head_chunks;
  const int32_t qc = m.s / query_chunks;
  const Tensor& target = tail != nullptr ? tail->q_out : m.out;
  std::vector<Tensor> head_outs;
  for (int hi = 0; hi < head_chunks; ++hi) {
    const int32_t h0 = hi * hc;
    Tensor kt = m.kt;
    Tensor v = m.v;
    if (head_chunks > 1) {
      LITERT_ASSIGN_OR_RETURN(
          kt, BuildSlice(builder, m.kt, {h0, 0, 0}, {hc, m.d, m.k}));
      LITERT_ASSIGN_OR_RETURN(
          v, BuildSlice(builder, m.v, {h0, 0, 0}, {hc, m.k, m.dv}));
    }
    std::vector<Tensor> query_outs;
    for (int qi = 0; qi < query_chunks; ++qi) {
      const int32_t q0 = qi * qc;
      LITERT_ASSIGN_OR_RETURN(
          auto q, BuildSlice(builder, m.q, {h0, q0, 0}, {hc, qc, m.d}));
      LITERT_ASSIGN_OR_RETURN(auto logits,
                              NewLike(builder, m.logits, {hc, qc, m.k}));
      LiteRtOp bmm1 = nullptr;
      LITERT_RETURN_IF_ERROR(BuildBatchMatmul(builder, q, kt, logits,
                                              m.asymmetric_quantize_input,
                                              /*adj_y=*/false, &bmm1));
      produced.push_back(bmm1);
      LITERT_ASSIGN_OR_RETURN(
          auto r1, BuildReshape(builder, logits, m.r1_out, {1, hc, qc, m.k}));
      LITERT_ASSIGN_OR_RETURN(auto sm,
                              NewLike(builder, m.sm_out, {1, hc, qc, m.k}));
      LITERT_ASSIGN_OR_RETURN(
          auto sm_op, builder.BuildOp(kLiteRtOpCodeTflSoftmax, {r1}, {sm}));
      SoftmaxOptions sm_opts;
      sm_opts.beta = m.beta;
      LITERT_RETURN_IF_ERROR(builder.SetOpOptions(sm_op, std::move(sm_opts)));
      LITERT_ASSIGN_OR_RETURN(
          auto p, BuildReshape(builder, sm, m.r2_out, {hc, qc, m.k}));
      LITERT_ASSIGN_OR_RETURN(auto o, NewLike(builder, m.out, {hc, qc, m.dv}));
      LITERT_RETURN_IF_ERROR(
          BuildBatchMatmul(builder, p, v, o, m.asymmetric_quantize_input));
      if (tail != nullptr) {
        LITERT_ASSIGN_OR_RETURN(
            auto r, BuildReshape(builder, o, tail->r_out, {1, hc, qc, m.dv}));
        LITERT_ASSIGN_OR_RETURN(
            auto t, NewLike(builder, tail->t_out, {1, qc, hc, m.dv}));
        LITERT_ASSIGN_OR_RETURN(auto perm, ConstI32(builder, {0, 2, 1, 3}));
        LITERT_RETURN_IF_ERROR(
            builder.BuildOp(kLiteRtOpCodeTflTranspose, {r, perm}, {t}));
        LITERT_ASSIGN_OR_RETURN(
            o, NewLike(builder, tail->q_out, {1, qc, hc, m.dv}));
        LITERT_RETURN_IF_ERROR(
            builder.BuildOp(kLiteRtOpCodeTflQuantize, {t}, {o}));
      }
      query_outs.push_back(o);
    }
    if (query_chunks == 1) {
      head_outs.push_back(query_outs[0]);
      continue;
    }
    Tensor head_out = target;
    if (head_chunks > 1) {
      const std::vector<int32_t> dims =
          tail != nullptr ? std::vector<int32_t>{1, m.s, hc, m.dv}
                          : std::vector<int32_t>{hc, m.s, m.dv};
      LITERT_ASSIGN_OR_RETURN(head_out, NewLike(builder, target, dims));
    }
    LITERT_RETURN_IF_ERROR(BuildConcat(builder, query_outs, 1, head_out));
    head_outs.push_back(head_out);
  }
  if (head_chunks > 1) {
    LITERT_RETURN_IF_ERROR(
        BuildConcat(builder, head_outs, tail != nullptr ? 2 : 0, target));
  }
  return {};
}

}  // namespace

extern "C" {

void ResetAttentionChunkTransformationState() {
  Produced().clear();
  Config c;
  c.auto_select = true;
  GetConfig() = c;
}

void SetAttentionChunkConfig(int head_chunks, int query_chunks) {
  Produced().clear();
  GetConfig() = Config{head_chunks, query_chunks};
}

void SetAttentionChunkAbsorbTail(bool absorb_tail) {
  GetConfig().absorb_tail = absorb_tail;
}

LiteRtStatus AttentionChunkTransformation(const LiteRtCompilerContext* context,
                                          LiteRtBuilder builder_ptr,
                                          LiteRtOp op) {
  const Config cfg = GetConfig();
  if (!cfg.Enabled() || Produced().contains(op)) {
    return kLiteRtStatusPatternNoMatch;
  }
  auto m = MatchAttention(context, op);
  if (!m) return kLiteRtStatusPatternNoMatch;
  int head_chunks = cfg.head_chunks;
  int query_chunks = cfg.query_chunks;
  if (cfg.auto_select) {
    const auto picked = AutoSelect(*m, cfg.min_logits_bytes);
    if (!picked) return kLiteRtStatusPatternNoMatch;
    head_chunks = picked->first;
    query_chunks = picked->second;
  }
  if (head_chunks * query_chunks < 2 || m->h % head_chunks != 0 ||
      m->s % query_chunks != 0) {
    return kLiteRtStatusPatternNoMatch;
  }
  std::optional<Tail> tail;
  if (cfg.absorb_tail) {
    tail = MatchTail(*m);
  }
  std::optional<PreHead> pre;
  const bool use_mha_to_sha = head_chunks == m->h && query_chunks >= 1;
  if (use_mha_to_sha) {
    pre = MatchPreHead(*m);
  }

  Builder builder(context, builder_ptr);
  std::vector<LiteRtOp> produced;
  if (pre) {
    auto built = BuildQualcommMhaToSha(builder, *m, *pre, query_chunks,
                                       tail ? &*tail : nullptr, produced);
    if (!built) return built.Error().Status();
  } else {
    auto built = BuildChunks(builder, *m, head_chunks, query_chunks,
                             tail ? &*tail : nullptr, produced);
    if (!built) return built.Error().Status();
  }

  std::vector<LiteRtOp> orphans;
  if (pre && tail && tail->has_flat_reshape) {
    builder.EraseOp(tail->flat_reshape);
    orphans = {tail->quantize.Get(), tail->transpose.Get(), tail->reshape.Get(),
               m->bmm2.Get(),        m->reshape2.Get(),     m->softmax.Get(),
               m->reshape1.Get(),    m->bmm1.Get()};
  } else if (tail) {
    builder.EraseOp(tail->quantize);
    orphans = {tail->transpose.Get(), tail->reshape.Get(), m->bmm2.Get(),
               m->reshape2.Get(),     m->softmax.Get(),    m->reshape1.Get(),
               m->bmm1.Get()};
  } else {
    builder.EraseOp(m->bmm2);
    orphans = {m->reshape2.Get(), m->softmax.Get(), m->reshape1.Get(),
               m->bmm1.Get()};
  }
  if (pre) {
    orphans.push_back(pre->q_reshape.Get());
    orphans.push_back(pre->q_transpose.Get());
    orphans.push_back(pre->k_reshape.Get());
    orphans.push_back(pre->k_transpose.Get());
    orphans.push_back(pre->v_reshape.Get());
    orphans.push_back(pre->v_transpose.Get());
  }
  RegisterOrphanGroup(orphans);
  Produced().insert(produced.begin(), produced.end());
  Produced().insert(op);
  return kLiteRtStatusOk;
}

}  // extern "C"
