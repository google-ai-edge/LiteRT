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

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
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
#include "litert/vendors/mediatek/compiler/transformations/orphan_cleanup_transformation.h"

using litert::BuildLayout;
using litert::ElementType;
using litert::Expected;
using litert::Layout;
using litert::RankedTensorType;
using litert::compiler::BatchMatmulOptions;
using litert::compiler::Builder;
using litert::compiler::ConcatenationOptions;
using litert::compiler::m_AllOf;
using litert::compiler::m_Any;
using litert::compiler::m_AnyOf;
using litert::compiler::m_CaptureOrSameAs;
using litert::compiler::m_ElementType;
using litert::compiler::m_HasOneUse;
using litert::compiler::m_IsConstant;
using litert::compiler::m_IsQuantized;
using litert::compiler::m_Not;
using litert::compiler::m_Op;
using litert::compiler::m_Options;
using litert::compiler::m_OpVariadic;
using litert::compiler::m_Predicate;
using litert::compiler::m_QType;
using litert::compiler::m_Rank;
using litert::compiler::Match;
using litert::compiler::Op;
using litert::compiler::RankedTensorSpecBuilder;
using litert::compiler::ReshapeOptions;
using litert::compiler::SoftmaxOptions;
using litert::compiler::Tensor;

namespace {

struct Config {
  int head_chunks = 1;
  int query_chunks = 1;
  bool absorb_tail = true;
  bool auto_select = false;
  bool Enabled() const {
    return auto_select || (head_chunks >= 1 && query_chunks >= 1 &&
                           head_chunks * query_chunks > 1);
  }
};

// Minimum logits tensor size in bytes to trigger auto-chunking.
constexpr int64_t kAutoMinLogitsBytes = 8 << 20;
// Maximum per-chunk logits size in bytes when splitting along queries.
constexpr int64_t kAutoMaxChunkBytes = 4 << 20;

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

int EnvInt(const char* name, int default_value) {
  const char* env = std::getenv(name);
  if (env == nullptr || *env == '\0') return default_value;
  int v = 0;
  if (!absl::SimpleAtoi(env, &v) || v < 0 || v > 4096) return default_value;
  return v;
}

std::vector<int32_t> Dims(const Tensor& t) {
  auto type = t.RankedTensorType();
  if (!type) return {};
  auto dims = type->Layout().Dimensions();
  return std::vector<int32_t>(dims.begin(), dims.end());
}

enum class Kind { kFloat, kQuant, kUnsupported };

// Float32 without quantization.
auto IsFloat() {
  return m_AllOf(m_ElementType(kLiteRtElementTypeFloat32),
                 m_Not(m_IsQuantized()), "float");
}

// Int8 / int16 with per-tensor quantization.
auto IsPerTensorQuant() {
  return m_AllOf(m_AnyOf(m_ElementType(kLiteRtElementTypeInt16),
                         m_ElementType(kLiteRtElementTypeInt8)),
                 m_QType(kLiteRtQuantizationPerTensor), "per_tensor_quant");
}

Kind KindOf(const Tensor& t) {
  if (Match(t, IsFloat())) return Kind::kFloat;
  if (Match(t, IsPerTensorQuant())) return Kind::kQuant;
  return Kind::kUnsupported;
}

auto IsKind(Kind kind) {
  return m_Predicate<Tensor>(
      [kind](const Tensor& t) { return KindOf(t) == kind; }, "kind");
}

// Exact shape. Unlike m_Shape, -1 is not a wildcard: the expected dims are
// derived from the matched graph and may themselves be dynamic.
auto HasDims(std::vector<int32_t> dims, absl::string_view label) {
  return m_Predicate<Tensor>(
      [dims = std::move(dims)](const Tensor& t) { return Dims(t) == dims; },
      label);
}

// An intermediate tensor of the chain: a single use, the exact `dims` and the
// given kind.
auto Intermediate(std::vector<int32_t> dims, Kind kind,
                  absl::string_view label) {
  return m_AllOf(m_HasOneUse(), HasDims(std::move(dims), label), IsKind(kind),
                 label);
}

// Matches a tensor that is the only output of its defining op, and that op
// against the op matcher `op_m` (which, unlike a tensor matcher, may be an
// m_OpVariadic, e.g. for a Reshape with or without its shape operand).
template <typename OpMatcher>
auto SoleOutputOf(OpMatcher op_m, absl::string_view label) {
  return m_Predicate<Tensor>(
      [op_m = std::move(op_m)](const Tensor& t) {
        auto op = t.GetDefiningOp();
        return op && op->Outputs().size() == 1 && Match(*op, op_m);
      },
      label);
}

// Matches an op with exactly one output, which matches `out_m`.
template <typename OutMatcher>
auto HasSingleOutput(OutMatcher out_m) {
  return m_Predicate<Op>(
      [out_m = std::move(out_m)](const Op& op) {
        auto outs = op.Outputs();
        return outs.size() == 1 && Match(outs[0], out_m);
      },
      "single_output");
}

// Follows the chain of sole uses `hops` ops down from `t` (through the single
// output of every intermediate op) and returns the last user. The matchers
// then verify the chain bottom-up from there.
std::optional<Op> SoleUserChainEnd(Tensor t, int hops) {
  for (int i = 1;; ++i) {
    auto uses = t.Uses();
    if (uses.size() != 1) return std::nullopt;
    if (i == hops) return uses[0].user;
    auto outs = uses[0].user.Outputs();
    if (outs.size() != 1) return std::nullopt;
    t = outs[0];
  }
}

// The matched attention core.
struct AttnMatch {
  Op bmm1, reshape1, softmax, reshape2, bmm2;
  Tensor q, kt, v, logits, r1_out, sm_out, r2_out, out;
  int32_t h, s, k, d, dv;
  float beta;
  bool asymmetric_quantize_input;
};

std::optional<AttnMatch> MatchAttention(const LiteRtCompilerContext* ctx,
                                        LiteRtOp root) {
  // Q [H, S, D] x Kt [H, D, K] -> logits [H, S, K] at the root.
  Op bmm1(ctx, root);
  Tensor q(ctx, nullptr);
  Tensor kt(ctx, nullptr);
  Tensor logits(ctx, nullptr);
  bool asym = false;
  auto bmm1_pattern =
      m_AllOf(m_Op<kLiteRtOpCodeTflBatchMatmul>(
                  m_CaptureOrSameAs(&q, m_Rank(3)),
                  m_CaptureOrSameAs(&kt, m_Rank(3)), "bmm1"),
              HasSingleOutput(m_CaptureOrSameAs(&logits, m_Any())),
              m_Options<BatchMatmulOptions>(
                  [&asym](const BatchMatmulOptions& opts) {
                    asym = opts.asymmetric_quantize_input;
                    return !opts.adj_x && !opts.adj_y;
                  },
                  "bmm1_options"));
  if (!Match(bmm1, bmm1_pattern)) return std::nullopt;
  const auto qd = Dims(q);
  const int32_t h = qd[0], s = qd[1], d = qd[2], k = Dims(kt)[2];
  const Kind kind = KindOf(q);
  if (kind == Kind::kUnsupported) return std::nullopt;
  if (!Match(kt, m_AllOf(HasDims({h, d, k}, "kt"), IsKind(kind))) ||
      !Match(logits, m_AllOf(HasDims({h, s, k}, "logits"), IsKind(kind)))) {
    return std::nullopt;
  }

  // logits -> Reshape -> Softmax -> Reshape -> BatchMatmul, every link being
  // the sole use of its tensor; matched bottom-up from the second
  // BatchMatmul down to the root's `logits`.
  auto bmm2 = SoleUserChainEnd(logits, 4);
  if (!bmm2) return std::nullopt;
  Op reshape1(ctx, nullptr);
  Op softmax(ctx, nullptr);
  Op reshape2(ctx, nullptr);
  Tensor r1_out(ctx, nullptr);
  Tensor sm_out(ctx, nullptr);
  Tensor r2_out(ctx, nullptr);
  Tensor v(ctx, nullptr);
  Tensor out(ctx, nullptr);
  float beta = 0.0f;
  // logits [H, S, K] -> Reshape -> r1_out [1, H, S, K].
  auto logits_m = m_AllOf(m_HasOneUse(), m_CaptureOrSameAs(&logits, m_Any()));
  auto reshape1_m = m_CaptureOrSameAs(
      &reshape1, m_OpVariadic<kLiteRtOpCodeTflReshape>(logits_m, "reshape1"));
  auto r1_out_m = m_CaptureOrSameAs(
      &r1_out, m_AllOf(Intermediate({1, h, s, k}, kind, "r1_out"),
                       SoleOutputOf(reshape1_m, "reshape1_out")));
  // r1_out -> Softmax -> sm_out [1, H, S, K].
  auto softmax_options_m = m_Options<SoftmaxOptions>(
      [&beta](const SoftmaxOptions& opts) {
        beta = opts.beta;
        return true;
      },
      "softmax_options");
  auto softmax_m = m_CaptureOrSameAs(
      &softmax, m_AllOf(m_Op<kLiteRtOpCodeTflSoftmax>(r1_out_m, "softmax"),
                        softmax_options_m));
  auto sm_out_m = m_CaptureOrSameAs(
      &sm_out, m_AllOf(Intermediate({1, h, s, k}, kind, "sm_out"),
                       SoleOutputOf(softmax_m, "softmax_out")));
  // sm_out -> Reshape -> P = r2_out [H, S, K].
  auto reshape2_m = m_CaptureOrSameAs(
      &reshape2, m_OpVariadic<kLiteRtOpCodeTflReshape>(sm_out_m, "reshape2"));
  auto r2_out_m = m_CaptureOrSameAs(
      &r2_out, m_AllOf(Intermediate({h, s, k}, kind, "r2_out"),
                       SoleOutputOf(reshape2_m, "reshape2_out")));
  // V [H, K, Dv].
  auto v_dims_m = m_Predicate<Tensor>(
      [h, k](const Tensor& t) {
        const auto vd = Dims(t);
        return vd[0] == h && vd[1] == k;
      },
      "v_dims");
  auto v_m = m_CaptureOrSameAs(&v, m_AllOf(m_Rank(3), v_dims_m, IsKind(kind)));
  // P must be the LHS of the second BatchMatmul, which must use the same
  // input quantization mode as the first one.
  auto bmm2_pattern =
      m_AllOf(m_Op<kLiteRtOpCodeTflBatchMatmul>(r2_out_m, v_m, "bmm2"),
              HasSingleOutput(m_CaptureOrSameAs(&out, m_Any())),
              m_Options<BatchMatmulOptions>(
                  [asym](const BatchMatmulOptions& opts) {
                    return !opts.adj_x && !opts.adj_y &&
                           opts.asymmetric_quantize_input == asym;
                  },
                  "bmm2_options"));
  if (!Match(*bmm2, bmm2_pattern)) return std::nullopt;
  const int32_t dv = Dims(v)[2];
  if (!Match(out, m_AllOf(HasDims({h, s, dv}, "out"), IsKind(kind)))) {
    return std::nullopt;
  }
  return AttnMatch{bmm1, reshape1, softmax, reshape2, *bmm2,  q,   kt,
                   v,    logits,   r1_out,  sm_out,   r2_out, out, h,
                   s,    k,        d,       dv,       beta,   asym};
}

// Output epilogue out [H, S, Dv] -> Reshape [1, H, S, Dv] -> Transpose
// (0, 2, 1, 3) -> [1, S, H, Dv] -> Quantize.
struct Tail {
  Op reshape, transpose, quantize;
  Tensor r_out, t_out, q_out;
};

std::optional<Tail> MatchTail(const LiteRtCompilerContext* ctx,
                              const AttnMatch& m) {
  // Matched bottom-up from the Quantize down to `out`.
  auto quantize = SoleUserChainEnd(m.out, 3);
  if (!quantize) return std::nullopt;
  Tensor out = m.out;
  Tail tail{Op(ctx, nullptr),     Op(ctx, nullptr),     *quantize,
            Tensor(ctx, nullptr), Tensor(ctx, nullptr), Tensor(ctx, nullptr)};
  // out [H, S, Dv] -> Reshape -> r_out [1, H, S, Dv].
  auto out_m = m_AllOf(m_HasOneUse(), m_CaptureOrSameAs(&out, m_Any()),
                       IsKind(Kind::kQuant));
  auto reshape_m = m_CaptureOrSameAs(
      &tail.reshape,
      m_OpVariadic<kLiteRtOpCodeTflReshape>(out_m, "tail_reshape"));
  auto r_out_m = m_CaptureOrSameAs(
      &tail.r_out,
      m_AllOf(Intermediate({1, m.h, m.s, m.dv}, Kind::kQuant, "tail_r_out"),
              SoleOutputOf(reshape_m, "tail_reshape_out")));
  // r_out -> Transpose(0, 2, 1, 3) -> t_out [1, S, H, Dv].
  auto perm_values_m = m_Predicate<Tensor>(
      [](const Tensor& t) {
        auto data = t.WeightsData<int32_t>();
        return data && std::vector<int32_t>(data->begin(), data->end()) ==
                           std::vector<int32_t>{0, 2, 1, 3};
      },
      "perm_0213");
  auto transpose_m = m_CaptureOrSameAs(
      &tail.transpose,
      m_Op<kLiteRtOpCodeTflTranspose>(
          r_out_m, m_AllOf(m_IsConstant(), perm_values_m), "tail_transpose"));
  auto t_out_m = m_CaptureOrSameAs(
      &tail.t_out,
      m_AllOf(Intermediate({1, m.s, m.h, m.dv}, Kind::kQuant, "tail_t_out"),
              SoleOutputOf(transpose_m, "tail_transpose_out")));
  // t_out -> Quantize -> q_out [1, S, H, Dv].
  auto quantize_pattern = m_AllOf(
      m_Op<kLiteRtOpCodeTflQuantize>(t_out_m, "tail_quantize"),
      HasSingleOutput(m_CaptureOrSameAs(
          &tail.q_out, m_AllOf(HasDims({1, m.s, m.h, m.dv}, "tail_q_out"),
                               IsKind(Kind::kQuant)))));
  if (!Match(*quantize, quantize_pattern)) return std::nullopt;
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
std::optional<std::pair<int, int>> AutoSelect(const AttnMatch& m) {
  const int64_t bytes = ElementBytes(m.logits);
  const int64_t logits_bytes = static_cast<int64_t>(m.h) * m.s * m.k * bytes;
  if (m.s < 2 || logits_bytes < kAutoMinLogitsBytes) return std::nullopt;
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
                                LiteRtOp* built = nullptr) {
  LITERT_ASSIGN_OR_RETURN(
      auto op, builder.BuildOp(kLiteRtOpCodeTflBatchMatmul, {a, b}, {out}));
  BatchMatmulOptions opts;
  opts.adj_x = false;
  opts.adj_y = false;
  opts.asymmetric_quantize_input = asymmetric_quantize_input;
  LITERT_RETURN_IF_ERROR(builder.SetOpOptions(op, std::move(opts)));
  if (built != nullptr) *built = op.Get();
  return {};
}

Expected<Tensor> BuildReshape(Builder& builder, const Tensor& in,
                              const Tensor& like,
                              const std::vector<int32_t>& dims) {
  LITERT_ASSIGN_OR_RETURN(auto out, NewLike(builder, like, dims));
  LITERT_ASSIGN_OR_RETURN(auto shape, ConstI32(builder, dims));
  LITERT_ASSIGN_OR_RETURN(
      auto op, builder.BuildOp(kLiteRtOpCodeTflReshape, {in, shape}, {out}));
  ReshapeOptions opts;
  opts.new_shape = dims;
  LITERT_RETURN_IF_ERROR(builder.SetOpOptions(op, std::move(opts)));
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

// Builds the chunked attention core.
Expected<void> BuildChunks(Builder& builder, const AttnMatch& m,
                           int head_chunks, int query_chunks, const Tail* tail,
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
      LITERT_RETURN_IF_ERROR(BuildBatchMatmul(
          builder, q, kt, logits, m.asymmetric_quantize_input, &bmm1));
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
  const char* chunks = std::getenv("LITERT_MEDIATEK_ATTN_CHUNKS");
  const int n = EnvInt("LITERT_MEDIATEK_ATTN_CHUNKS", 0);
  const char* axis = std::getenv("LITERT_MEDIATEK_ATTN_CHUNK_AXIS");
  const bool head_axis = axis != nullptr && std::strcmp(axis, "head") == 0;
  if (chunks == nullptr || chunks[0] == '\0') {
    c.auto_select = true;
  } else if (n > 1) {
    if (head_axis) {
      c.head_chunks = n;
    } else {
      c.query_chunks = n;
      c.head_chunks =
          std::max(1, EnvInt("LITERT_MEDIATEK_ATTN_HEAD_CHUNKS", 1));
    }
  }
  c.absorb_tail = EnvInt("LITERT_MEDIATEK_ATTN_CHUNK_TAIL", 1) != 0;
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
  if (!cfg.Enabled() || Produced().count(op) != 0) {
    return kLiteRtStatusPatternNoMatch;
  }
  auto m = MatchAttention(context, op);
  if (!m) return kLiteRtStatusPatternNoMatch;
  int head_chunks = cfg.head_chunks;
  int query_chunks = cfg.query_chunks;
  if (cfg.auto_select) {
    const auto picked = AutoSelect(*m);
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
    tail = MatchTail(context, *m);
  }

  Builder builder(context, builder_ptr);
  std::vector<LiteRtOp> produced;
  auto built = BuildChunks(builder, *m, head_chunks, query_chunks,
                           tail ? &*tail : nullptr, produced);
  if (!built) return built.Error().Status();

  std::vector<LiteRtOp> orphans;
  if (tail) {
    builder.EraseOp(tail->quantize);
    orphans = {tail->transpose.Get(), tail->reshape.Get(), m->bmm2.Get(),
               m->reshape2.Get(),     m->softmax.Get(),    m->reshape1.Get(),
               m->bmm1.Get()};
  } else {
    builder.EraseOp(m->bmm2);
    orphans = {m->reshape2.Get(), m->softmax.Get(), m->reshape1.Get(),
               m->bmm1.Get()};
  }
  RegisterOrphanGroup(orphans);
  Produced().insert(produced.begin(), produced.end());
  Produced().insert(op);
  return kLiteRtStatusOk;
}

}  // extern "C"
