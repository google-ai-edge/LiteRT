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

#include "litert/vendors/mediatek/compiler/transformations/bool_mask_transformation.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
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

namespace litert::mediatek {
namespace {

using ::litert::ElementType;
using ::litert::compiler::AddOptions;
using ::litert::compiler::Builder;
using ::litert::compiler::ConcatenationOptions;
using ::litert::compiler::DepthwiseConv2dOptions;
using ::litert::compiler::GetOptionsAs;
using ::litert::compiler::MulOptions;
using ::litert::compiler::Op;
using ::litert::compiler::RankedTensorSpecBuilder;
using ::litert::compiler::ReshapeOptions;
using ::litert::compiler::SubOptions;
using ::litert::compiler::Tensor;

using Dims = std::vector<int32_t>;

struct PatternEntry {
  LiteRtStatus (*fn)(const LiteRtCompilerContext*, LiteRtBuilder, LiteRtOp);
  const char* name;
};

const LiteRtCompilerContext* Ctx() { return LrtGetCompilerContext(); }

class GraphBuilder {
 public:
  GraphBuilder() : ctx_(Ctx()), builder_(ctx_, &impl_) {}

  Tensor Typed(ElementType type, const Dims& dims) {
    auto t = builder_.BuildTensor(
        RankedTensorSpecBuilder(
            RankedTensorType(
                type,
                Layout(BuildLayout(dims.data(), dims.data() + dims.size()))))
            .Build());
    EXPECT_TRUE(t.HasValue());
    return *t;
  }

  Tensor F32(const Dims& dims) { return Typed(ElementType::Float32, dims); }
  Tensor I32(const Dims& dims) { return Typed(ElementType::Int32, dims); }
  Tensor Bool(const Dims& dims) { return Typed(ElementType::Bool, dims); }

  Tensor Quant(ElementType type, const Dims& dims, float scale, int64_t zp) {
    auto t = builder_.BuildTensor(
        RankedTensorSpecBuilder(
            RankedTensorType(
                type,
                Layout(BuildLayout(dims.data(), dims.data() + dims.size()))))
            .WithPerTensorQuantization(LiteRtQuantizationPerTensor{scale, zp})
            .Build());
    EXPECT_TRUE(t.HasValue());
    return *t;
  }

  template <typename T>
  Tensor Const(const Dims& dims, const std::vector<T>& data) {
    Tensor t = Typed(GetElementType<T>(), dims);
    EXPECT_TRUE(builder_.BuildWeights<T>(absl::MakeConstSpan(data), t));
    return t;
  }

  Tensor BoolConst(const Dims& dims, absl::Span<const uint8_t> data) {
    Tensor t = Typed(ElementType::Bool, dims);
    auto buf = std::make_unique<bool[]>(data.size());
    for (size_t i = 0; i < data.size(); ++i) buf[i] = data[i] != 0;
    EXPECT_TRUE(builder_.BuildWeights<bool>(
        absl::Span<const bool>(buf.get(), data.size()), t));
    return t;
  }

  Op AddOp(LiteRtOpCode code, const std::vector<Tensor>& ins,
           const std::vector<Tensor>& outs) {
    auto op = builder_.BuildOp(code, ins, outs);
    EXPECT_TRUE(op.HasValue());
    return *op;
  }

  template <typename OptionsT>
  void SetOptions(const Op& op, OptionsT options) {
    EXPECT_TRUE(builder_.SetOpOptions(op, std::move(options)).HasValue());
  }

  Op AddReshape(const Tensor& in, const Dims& dims, const Tensor& out) {
    Op op = AddOp(
        kLiteRtOpCodeTflReshape,
        {in, Const<int32_t>({static_cast<int32_t>(dims.size())}, dims)}, {out});
    ReshapeOptions opts;
    opts.new_shape = dims;
    SetOptions(op, std::move(opts));
    return op;
  }

  Op AddConcat(const std::vector<Tensor>& ins, const Tensor& out, int32_t axis,
               uint32_t act = 0) {
    Op op = AddOp(kLiteRtOpCodeTflConcatenation, ins, {out});
    ConcatenationOptions opts;
    opts.axis = axis;
    opts.fused_activation_function = act;
    SetOptions(op, std::move(opts));
    return op;
  }

  Op AddSlice(const Tensor& in, const Dims& beg, const Dims& sz,
              const Tensor& out) {
    const int32_t r = static_cast<int32_t>(beg.size());
    return AddOp(kLiteRtOpCodeTflSlice,
                 {in, Const<int32_t>({r}, beg), Const<int32_t>({r}, sz)},
                 {out});
  }

  void AddTileWrap(const Tensor& in_b, const std::vector<int64_t>& reps,
                   const Dims& in_dims, const Dims& out_dims,
                   const Tensor& out_b) {
    Tensor i_in = I32(in_dims);
    Tensor i_out = I32(out_dims);
    AddOp(kLiteRtOpCodeTflCast, {in_b}, {i_in});
    AddOp(kLiteRtOpCodeTflTile,
          {i_in, Const<int64_t>({static_cast<int32_t>(reps.size())}, reps)},
          {i_out});
    AddOp(kLiteRtOpCodeTflCast, {i_out}, {out_b});
  }

  void MarkInput(const Tensor& t) { inputs_.push_back(t.Get()); }
  void MarkOutput(const Tensor& t) { outputs_.push_back(t.Get()); }

  void ApplyTo(LiteRtSubgraphT* subgraph) {
    impl_.ApplyChanges(subgraph);
    const auto& live = subgraph->Tensors();
    for (LiteRtTensor in : inputs_) {
      if (absl::c_find(live, in) != live.end()) {
        subgraph->Inputs().push_back(in);
      }
    }
    for (LiteRtTensor out : outputs_) {
      if (absl::c_find(live, out) != live.end()) {
        subgraph->Outputs().push_back(out);
      }
    }
  }

 private:
  const LiteRtCompilerContext* ctx_;
  LiteRtBuilderT impl_;
  Builder builder_;
  std::vector<LiteRtTensor> inputs_;
  std::vector<LiteRtTensor> outputs_;
};

void SetDwConv2dOptions(GraphBuilder& g, const Op& op, uint32_t act = 0) {
  DepthwiseConv2dOptions opts;
  opts.padding = 1;  // VALID
  opts.stride_w = 1;
  opts.stride_h = 2;
  opts.depth_multiplier = 1;
  opts.fused_activation_function = act;
  opts.dilation_w_factor = 1;
  opts.dilation_h_factor = 1;
  g.SetOptions(op, std::move(opts));
}

struct EvalVal {
  LiteRtElementType dtype = kLiteRtElementTypeNone;
  Dims dims;
  std::vector<float> f32;
  std::vector<int64_t> i64;
  std::vector<uint8_t> b;
};

size_t FlatSize(const Dims& dims) {
  size_t n = 1;
  for (int32_t d : dims) n *= static_cast<size_t>(d);
  return n;
}

Dims TensorDims(const LiteRtTensorT* t) {
  const auto& layout = t->Type().second.ranked_tensor_type.layout;
  return Dims(layout.dimensions, layout.dimensions + layout.rank);
}

std::vector<size_t> Unflatten(size_t linear, const Dims& dims) {
  std::vector<size_t> coords(dims.size(), 0);
  for (size_t i = dims.size(); i > 0; --i) {
    const size_t d = static_cast<size_t>(dims[i - 1]);
    coords[i - 1] = linear % d;
    linear /= d;
  }
  return coords;
}

size_t BroadcastIndex(size_t out_linear, const Dims& out_dims,
                      const Dims& in_dims) {
  if (in_dims.empty()) return 0;
  const std::vector<size_t> coords = Unflatten(out_linear, out_dims);
  const size_t offset = out_dims.size() - in_dims.size();
  size_t idx = 0;
  for (size_t i = 0; i < in_dims.size(); ++i) {
    const size_t c = (in_dims[i] == 1) ? 0 : coords[offset + i];
    idx = idx * static_cast<size_t>(in_dims[i]) + c;
  }
  return idx;
}

float ApplyActivation(float v, uint32_t act) {
  if (act == 1) return (std::max)(0.0f, v);
  if (act == 3) return (std::min)(6.0f, (std::max)(0.0f, v));
  if (act == 2) return std::tanh(v);
  return v;
}

EvalVal LoadConstVal(const LiteRtTensorT* t) {
  EvalVal v;
  v.dtype = t->Type().second.ranked_tensor_type.element_type;
  v.dims = TensorDims(t);
  const size_t n = FlatSize(v.dims);
  const uint8_t* raw = t->Weights().Buffer().Data();
  if (v.dtype == kLiteRtElementTypeFloat32) {
    v.f32.resize(n);
    if (n > 0) std::memcpy(v.f32.data(), raw, n * sizeof(float));
  } else if (v.dtype == kLiteRtElementTypeInt32) {
    v.i64.resize(n);
    for (size_t i = 0; i < n; ++i) {
      int32_t x = 0;
      std::memcpy(&x, raw + i * sizeof(int32_t), sizeof(int32_t));
      v.i64[i] = x;
    }
  } else if (v.dtype == kLiteRtElementTypeInt64) {
    v.i64.resize(n);
    if (n > 0) std::memcpy(v.i64.data(), raw, n * sizeof(int64_t));
  } else if (v.dtype == kLiteRtElementTypeBool) {
    v.b.resize(n);
    for (size_t i = 0; i < n; ++i) v.b[i] = raw[i] != 0 ? 1 : 0;
  }
  return v;
}

absl::flat_hash_map<const LiteRtTensorT*, EvalVal> EvalSubgraph(
    const LiteRtSubgraphT& subgraph,
    const absl::flat_hash_map<const LiteRtTensorT*, EvalVal>& feed) {
  absl::flat_hash_map<const LiteRtTensorT*, EvalVal> env;
  const EvalVal empty_val{};
  for (const LiteRtTensorT* in : subgraph.Inputs()) {
    if (auto it = feed.find(in); it != feed.end()) env[in] = it->second;
  }
  for (const LiteRtTensorT* t : subgraph.Tensors()) {
    if (t->Weights().Buffer().Size() > 0) env[t] = LoadConstVal(t);
  }
  for (LiteRtOp op : subgraph.Ops()) {
    const LiteRtOpCode code = op->OpCode();
    const LiteRtTensorT* out_t = op->Outputs()[0];
    EvalVal out;
    out.dtype = out_t->Type().second.ranked_tensor_type.element_type;
    out.dims = TensorDims(out_t);
    const size_t n = FlatSize(out.dims);
    auto in_val = [&](size_t idx) -> const EvalVal& {
      auto it = env.find(op->Inputs()[idx]);
      if (it == env.end()) {
        ADD_FAILURE() << "Missing input " << idx << " on op " << code;
        return empty_val;
      }
      return it->second;
    };

    if (code == kLiteRtOpCodeTflCast) {
      const EvalVal& a = in_val(0);
      if (out.dtype == kLiteRtElementTypeFloat32) {
        out.f32.resize(n);
        for (size_t i = 0; i < n; ++i) {
          out.f32[i] = (a.dtype == kLiteRtElementTypeBool)
                           ? (a.b[i] ? 1.0f : 0.0f)
                           : static_cast<float>(a.i64[i]);
        }
      } else if (out.dtype == kLiteRtElementTypeBool) {
        out.b.resize(n);
        for (size_t i = 0; i < n; ++i) {
          out.b[i] = (a.dtype == kLiteRtElementTypeFloat32)
                         ? (a.f32[i] != 0.0f ? 1 : 0)
                         : (a.i64[i] != 0 ? 1 : 0);
        }
      } else if (out.dtype == kLiteRtElementTypeInt32) {
        out.i64.resize(n);
        for (size_t i = 0; i < n; ++i) {
          out.i64[i] = (a.dtype == kLiteRtElementTypeBool)
                           ? (a.b[i] ? 1 : 0)
                           : static_cast<int64_t>(a.f32[i]);
        }
      }
    } else if (code == kLiteRtOpCodeTflReshape ||
               code == kLiteRtOpCodeTflQuantize) {
      const EvalVal& a = in_val(0);
      out.f32 = a.f32;
      out.i64 = a.i64;
      out.b = a.b;
    } else if (code == kLiteRtOpCodeTflAbs) {
      const EvalVal& a = in_val(0);
      out.f32.resize(n);
      for (size_t i = 0; i < n; ++i) out.f32[i] = std::abs(a.f32[i]);
    } else if (code == kLiteRtOpCodeTflLogicalNot) {
      const EvalVal& a = in_val(0);
      out.b.resize(n);
      for (size_t i = 0; i < n; ++i) out.b[i] = a.b[i] ? 0 : 1;
    } else if (code == kLiteRtOpCodeTflLogicalAnd ||
               code == kLiteRtOpCodeTflLogicalOr) {
      const EvalVal &a = in_val(0), &b = in_val(1);
      out.b.resize(n);
      for (size_t i = 0; i < n; ++i) {
        const uint8_t va = a.b[BroadcastIndex(i, out.dims, a.dims)];
        const uint8_t vb = b.b[BroadcastIndex(i, out.dims, b.dims)];
        out.b[i] = (code == kLiteRtOpCodeTflLogicalAnd) ? ((va && vb) ? 1 : 0)
                                                        : ((va || vb) ? 1 : 0);
      }
    } else if (code == kLiteRtOpCodeTflAdd || code == kLiteRtOpCodeTflSub ||
               code == kLiteRtOpCodeTflMul || code == kLiteRtOpCodeTflMinimum ||
               code == kLiteRtOpCodeTflMaximum) {
      const EvalVal &a = in_val(0), &b = in_val(1);
      uint32_t act = 0;
      if (code == kLiteRtOpCodeTflAdd) {
        if (auto o = GetOptionsAs<AddOptions>(Ctx(), op)) {
          act = o->fused_activation_function;
        }
      } else if (code == kLiteRtOpCodeTflSub) {
        if (auto o = GetOptionsAs<SubOptions>(Ctx(), op)) {
          act = o->fused_activation_function;
        }
      } else if (code == kLiteRtOpCodeTflMul) {
        if (auto o = GetOptionsAs<MulOptions>(Ctx(), op)) {
          act = o->fused_activation_function;
        }
      }
      out.f32.resize(n);
      for (size_t i = 0; i < n; ++i) {
        const float va = a.f32[BroadcastIndex(i, out.dims, a.dims)];
        const float vb = b.f32[BroadcastIndex(i, out.dims, b.dims)];
        float r = (code == kLiteRtOpCodeTflAdd)       ? (va + vb)
                  : (code == kLiteRtOpCodeTflSub)     ? (va - vb)
                  : (code == kLiteRtOpCodeTflMul)     ? (va * vb)
                  : (code == kLiteRtOpCodeTflMinimum) ? (std::min)(va, vb)
                                                      : (std::max)(va, vb);
        out.f32[i] = ApplyActivation(r, act);
      }
    } else if (code == kLiteRtOpCodeTflNotEqual ||
               code == kLiteRtOpCodeTflEqual) {
      const EvalVal &a = in_val(0), &b = in_val(1);
      const bool eq = (code == kLiteRtOpCodeTflEqual);
      out.b.resize(n);
      for (size_t i = 0; i < n; ++i) {
        const size_t ia = BroadcastIndex(i, out.dims, a.dims);
        const size_t ib = BroadcastIndex(i, out.dims, b.dims);
        const bool match =
            (a.dtype == kLiteRtElementTypeFloat32)
                ? (eq ? (a.f32[ia] == b.f32[ib]) : (a.f32[ia] != b.f32[ib]))
                : (eq ? (a.b[ia] == b.b[ib]) : (a.b[ia] != b.b[ib]));
        out.b[i] = match ? 1 : 0;
      }
    } else if (code == kLiteRtOpCodeTflSelect ||
               code == kLiteRtOpCodeTflSelectV2) {
      const EvalVal &c = in_val(0), &x = in_val(1), &y = in_val(2);
      Dims eff_c = c.dims;
      if (code == kLiteRtOpCodeTflSelect && c.dims.size() == 1 &&
          out.dims.size() > 1 && c.dims[0] == out.dims[0]) {
        eff_c.assign(out.dims.size(), 1);
        eff_c[0] = c.dims[0];
      }
      if (out.dtype == kLiteRtElementTypeFloat32) out.f32.resize(n);
      if (out.dtype == kLiteRtElementTypeBool) out.b.resize(n);
      if (out.dtype == kLiteRtElementTypeInt32) out.i64.resize(n);
      for (size_t i = 0; i < n; ++i) {
        const bool pick_x = c.b[BroadcastIndex(i, out.dims, eff_c)] != 0;
        const size_t ix = BroadcastIndex(i, out.dims, x.dims);
        const size_t iy = BroadcastIndex(i, out.dims, y.dims);
        if (out.dtype == kLiteRtElementTypeFloat32) {
          out.f32[i] = pick_x ? x.f32[ix] : y.f32[iy];
        } else if (out.dtype == kLiteRtElementTypeBool) {
          out.b[i] = pick_x ? x.b[ix] : y.b[iy];
        } else {
          out.i64[i] = pick_x ? x.i64[ix] : y.i64[iy];
        }
      }
    } else if (code == kLiteRtOpCodeTflSlice || code == kLiteRtOpCodeTflTile) {
      const EvalVal& a = in_val(0);
      if (out.dtype == kLiteRtElementTypeFloat32) out.f32.resize(n);
      if (out.dtype == kLiteRtElementTypeBool) out.b.resize(n);
      if (out.dtype == kLiteRtElementTypeInt32) out.i64.resize(n);
      for (size_t i = 0; i < n; ++i) {
        const std::vector<size_t> c = Unflatten(i, out.dims);
        size_t src = 0;
        for (size_t d = 0; d < out.dims.size(); ++d) {
          const size_t sc = (code == kLiteRtOpCodeTflSlice)
                                ? (c[d] + static_cast<size_t>(in_val(1).i64[d]))
                                : (c[d] % static_cast<size_t>(a.dims[d]));
          src = src * static_cast<size_t>(a.dims[d]) + sc;
        }
        if (out.dtype == kLiteRtElementTypeFloat32) out.f32[i] = a.f32[src];
        if (out.dtype == kLiteRtElementTypeBool) out.b[i] = a.b[src];
        if (out.dtype == kLiteRtElementTypeInt32) out.i64[i] = a.i64[src];
      }
    } else if (code == kLiteRtOpCodeTflConcatenation) {
      auto opts = GetOptionsAs<ConcatenationOptions>(Ctx(), op);
      int32_t axis = opts->axis;
      if (axis < 0) axis += static_cast<int32_t>(out.dims.size());
      size_t outer = 1, inner = 1;
      for (int32_t d = 0; d < axis; ++d) outer *= out.dims[d];
      for (size_t d = axis + 1; d < out.dims.size(); ++d) inner *= out.dims[d];
      if (out.dtype == kLiteRtElementTypeFloat32) out.f32.resize(n);
      if (out.dtype == kLiteRtElementTypeBool) out.b.resize(n);
      for (size_t o = 0; o < outer; ++o) {
        size_t dst_off = o * static_cast<size_t>(out.dims[axis]) * inner;
        for (size_t k = 0; k < op->Inputs().size(); ++k) {
          const EvalVal& a = in_val(k);
          const size_t chunk = static_cast<size_t>(a.dims[axis]) * inner;
          for (size_t j = 0; j < chunk; ++j) {
            if (out.dtype == kLiteRtElementTypeFloat32) {
              out.f32[dst_off + j] = ApplyActivation(
                  a.f32[o * chunk + j], opts->fused_activation_function);
            } else {
              out.b[dst_off + j] = a.b[o * chunk + j];
            }
          }
          dst_off += chunk;
        }
      }
    } else if (code == kLiteRtOpCodeTflPad) {
      const EvalVal &a = in_val(0), &pad = in_val(1);
      out.f32.assign(n, 0.0f);
      for (size_t i = 0; i < FlatSize(a.dims); ++i) {
        const std::vector<size_t> c = Unflatten(i, a.dims);
        size_t dst = 0;
        for (size_t d = 0; d < out.dims.size(); ++d) {
          dst = dst * static_cast<size_t>(out.dims[d]) + c[d] +
                static_cast<size_t>(pad.i64[2 * d]);
        }
        out.f32[dst] = a.f32[i];
      }
    } else if (code == kLiteRtOpCodeTflDepthwiseConv2d) {
      const EvalVal &x = in_val(0), &w = in_val(1), &bias = in_val(2);
      auto opts = GetOptionsAs<DepthwiseConv2dOptions>(Ctx(), op);
      out.f32.resize(n);
      const int32_t b_dim = out.dims[0], oh = out.dims[1], ow = out.dims[2],
                    oc = out.dims[3], kh = w.dims[1], kw = w.dims[2];
      for (int32_t bi = 0; bi < b_dim; ++bi) {
        for (int32_t hi = 0; hi < oh; ++hi) {
          for (int32_t wi = 0; wi < ow; ++wi) {
            for (int32_t ci = 0; ci < oc; ++ci) {
              float acc = bias.f32[ci];
              for (int32_t fh = 0; fh < kh; ++fh) {
                for (int32_t fw = 0; fw < kw; ++fw) {
                  const int32_t ih = hi * opts->stride_h + fh;
                  const int32_t iw = wi * opts->stride_w + fw;
                  if (ih >= 0 && ih < x.dims[1] && iw >= 0 && iw < x.dims[2]) {
                    const size_t x_idx =
                        (((bi * x.dims[1] + ih) * x.dims[2] + iw) * x.dims[3]) +
                        (ci / opts->depth_multiplier);
                    acc += x.f32[x_idx] * w.f32[((fh * kw + fw) * oc) + ci];
                  }
                }
              }
              out.f32[(((bi * oh + hi) * ow + wi) * oc) + ci] =
                  ApplyActivation(acc, opts->fused_activation_function);
            }
          }
        }
      }
    } else {
      ADD_FAILURE() << "Unhandled op in test evaluator: " << code;
    }
    env[out_t] = std::move(out);
  }
  return env;
}

void ExpectSameOutputs(
    const LiteRtSubgraphT& subgraph,
    const absl::flat_hash_map<const LiteRtTensorT*, EvalVal>& before,
    const absl::flat_hash_map<const LiteRtTensorT*, EvalVal>& after) {
  for (const LiteRtTensorT* out : subgraph.Outputs()) {
    auto it_b = before.find(out), it_a = after.find(out);
    ASSERT_NE(it_b, before.end());
    ASSERT_NE(it_a, after.end());
    EXPECT_EQ(it_b->second.dtype, it_a->second.dtype);
    EXPECT_EQ(it_b->second.dims, it_a->second.dims);
    if (it_b->second.dtype == kLiteRtElementTypeBool) {
      EXPECT_EQ(it_b->second.b, it_a->second.b);
    } else if (it_b->second.dtype == kLiteRtElementTypeInt32) {
      EXPECT_EQ(it_b->second.i64, it_a->second.i64);
    } else {
      ASSERT_EQ(it_b->second.f32.size(), it_a->second.f32.size());
      for (size_t i = 0; i < it_b->second.f32.size(); ++i) {
        const float vb = it_b->second.f32[i], va = it_a->second.f32[i];
        if (std::isnan(vb)) {
          EXPECT_TRUE(std::isnan(va)) << "index " << i;
        } else {
          EXPECT_FLOAT_EQ(vb, va) << "index " << i;
        }
      }
    }
  }
}

std::vector<PatternEntry> AllPatterns(bool include_output_not_equal) {
  std::vector<PatternEntry> patterns = {
      {&BoolMaskBypassCastTransformation, "BoolMaskBypassCast"},
      {&BoolMaskCleanupCastTransformation, "BoolMaskCleanupCast"},
      {&BoolMaskNotEqualTransformation, "BoolMaskNotEqual"},
      {&BoolMaskTileWrapTransformation, "BoolMaskTileWrap"},
      {&BoolMaskOpToFloatTransformation, "BoolMaskOpToFloat"},
  };
  if (include_output_not_equal) {
    patterns.push_back(
        {&BoolMaskOutputNotEqualTransformation, "BoolMaskOutputNotEqual"});
  }
  return patterns;
}

int RunPatterns(LiteRtSubgraphT* subgraph,
                absl::Span<const PatternEntry> patterns, int max_iters = 500) {
  const LiteRtCompilerContext* ctx = Ctx();
  int iterations = 0;
  for (bool modified = true; modified && iterations < max_iters; ++iterations) {
    modified = false;
    std::queue<LiteRtOp> worklist;
    for (LiteRtOp op : subgraph->Ops()) worklist.push(op);
    while (!worklist.empty() && !modified) {
      LiteRtOp op = worklist.front();
      worklist.pop();
      for (const auto& entry : patterns) {
        LiteRtBuilderT builder;
        if (entry.fn(ctx, &builder, op) == kLiteRtStatusOk) {
          builder.ApplyChanges(subgraph);
          modified = true;
          break;
        }
      }
    }
  }
  EXPECT_LT(iterations, max_iters);
  return iterations;
}

void ExpectTopologicalOrder(const LiteRtSubgraphT& subgraph) {
  absl::flat_hash_map<LiteRtOp, size_t> op_index;
  for (size_t i = 0; i < subgraph.Ops().size(); ++i) {
    op_index[subgraph.Ops()[i]] = i;
  }
  for (size_t i = 0; i < subgraph.Ops().size(); ++i) {
    for (const LiteRtTensor in : subgraph.Ops()[i]->Inputs()) {
      if (in->DefiningOp() != nullptr) {
        auto it = op_index.find(in->DefiningOp());
        ASSERT_NE(it, op_index.end()) << "Dangling defining op at index " << i;
        EXPECT_LT(it->second, i);
      }
    }
  }
}

int CountOps(const LiteRtSubgraphT& subgraph, LiteRtOpCode code) {
  int count = 0;
  for (const auto* op : subgraph.Ops()) {
    if (op->OpCode() == code) ++count;
  }
  return count;
}

std::string DumpOps(const LiteRtSubgraphT& subgraph) {
  std::ostringstream os;
  for (size_t i = 0; i < subgraph.Ops().size(); ++i) {
    os << "[" << i << "]c=" << subgraph.Ops()[i]->OpCode() << "\n";
  }
  return os.str();
}

int CountBoolOps(const LiteRtSubgraphT& subgraph) {
  auto is_bool = [](const LiteRtTensor t) {
    return t->Type().first == kLiteRtRankedTensorType &&
           t->Type().second.ranked_tensor_type.element_type ==
               kLiteRtElementTypeBool;
  };
  int count = 0;
  for (const auto* op : subgraph.Ops()) {
    if (op->OpCode() == kLiteRtOpCodeTflCast ||
        op->OpCode() == kLiteRtOpCodeTflNotEqual) {
      continue;
    }
    if (std::any_of(op->Inputs().begin(), op->Inputs().end(), is_bool) ||
        std::any_of(op->Outputs().begin(), op->Outputs().end(), is_bool)) {
      ++count;
    }
  }
  return count;
}

void BuildAudioMaskSubgraph(LiteRtSubgraphT* subgraph) {
  GraphBuilder g;
  Tensor seg_mask = g.Bool({1, 9}), prev_mask = g.Bool({1, 4});
  Tensor logits = g.F32({2, 2, 2, 4}), proj_in = g.F32({1, 2, 8});
  for (const Tensor& in : {seg_mask, prev_mask, logits, proj_in}) {
    g.MarkInput(in);
  }
  Tensor dw_filter = g.Const<float>({1, 3, 1, 1}, {0.0f, 1.0f, 0.0f});
  Tensor dw_bias = g.Const<float>({1}, {0.0f});
  Tensor zero = g.Const<float>({}, {0.0f});
  Tensor neg_inf = g.Const<float>({}, {-58982.398438f});

  auto add_subsampler = [&](const Tensor& mask_in, int32_t in_len,
                            int32_t out_len) {
    Tensor c = g.F32({1, in_len}), r0 = g.F32({1, in_len, 1, 1});
    Tensor dw = g.F32({1, out_len, 1, 1}), r1 = g.F32({1, out_len});
    Tensor out = g.Bool({1, out_len});
    g.AddOp(kLiteRtOpCodeTflCast, {mask_in}, {c});
    g.AddReshape(c, {1, in_len, 1, 1}, r0);
    SetDwConv2dOptions(g, g.AddOp(kLiteRtOpCodeTflDepthwiseConv2d,
                                  {r0, dw_filter, dw_bias}, {dw}));
    g.AddReshape(dw, {1, out_len}, r1);
    g.AddOp(kLiteRtOpCodeTflNotEqual, {r1, zero}, {out});
    return out;
  };
  Tensor mid_mask = add_subsampler(seg_mask, 9, 4);
  Tensor out_down_mask = add_subsampler(mid_mask, 4, 2);
  g.MarkOutput(out_down_mask);

  Tensor cond_proj = g.Bool({1, 2, 1}), out_proj = g.F32({1, 2, 8});
  g.AddReshape(out_down_mask, {1, 2, 1}, cond_proj);
  g.AddOp(kLiteRtOpCodeTflSelectV2, {cond_proj, proj_in, zero}, {out_proj});
  g.MarkOutput(out_proj);

  Tensor out_slice = g.Bool({1, 1});
  g.AddSlice(out_down_mask, {0, 1}, {1, 1}, out_slice);
  g.MarkOutput(out_slice);

  Tensor cat_2d = g.Bool({1, 6}), cat_3d = g.Bool({1, 1, 6});
  Tensor pad_in = g.F32({1, 1, 6}), pad_out = g.F32({1, 1, 8});
  Tensor pad_bool = g.Bool({1, 1, 8}), r_412 = g.Bool({4, 1, 2});
  g.AddConcat({prev_mask, out_down_mask}, cat_2d, 1);
  g.AddReshape(cat_2d, {1, 1, 6}, cat_3d);
  g.AddOp(kLiteRtOpCodeTflCast, {cat_3d}, {pad_in});
  g.AddOp(kLiteRtOpCodeTflPad,
          {pad_in, g.Const<int32_t>({3, 2}, {0, 0, 0, 0, 0, 2})}, {pad_out});
  g.AddOp(kLiteRtOpCodeTflCast, {pad_out}, {pad_bool});
  g.AddReshape(pad_bool, {4, 1, 2}, r_412);

  Tensor s0 = g.Bool({2, 1, 2}), s1 = g.Bool({2, 1, 2});
  Tensor c0 = g.Bool({1, 2, 1, 2}), c1 = g.Bool({1, 2, 1, 2});
  Tensor cat4d = g.Bool({1, 2, 1, 4}), tiled_b = g.Bool({1, 2, 2, 4});
  g.AddSlice(r_412, {0, 0, 0}, {2, 1, 2}, s0);
  g.AddSlice(r_412, {1, 0, 0}, {2, 1, 2}, s1);
  g.AddReshape(s0, {1, 2, 1, 2}, c0);
  g.AddReshape(s1, {1, 2, 1, 2}, c1);
  g.AddConcat({c0, c1}, cat4d, 3);
  g.AddTileWrap(cat4d, {1, 1, 2, 1}, {1, 2, 1, 4}, {1, 2, 2, 4}, tiled_b);

  std::vector<uint8_t> window(16, 1);
  window[0] = 0;
  Tensor t_and = g.Bool({1, 2, 2, 4}), t_not = g.Bool({1, 2, 2, 4});
  Tensor t_masked = g.F32({2, 2, 2, 4});
  Tensor out_attn = g.Quant(ElementType::Int16, {2, 2, 2, 4}, 1.8f, 0);
  g.AddOp(kLiteRtOpCodeTflLogicalAnd,
          {tiled_b, g.BoolConst({1, 2, 2, 4}, window)}, {t_and});
  g.AddOp(kLiteRtOpCodeTflLogicalNot, {t_and}, {t_not});
  g.AddOp(kLiteRtOpCodeTflSelectV2, {t_not, neg_inf, logits}, {t_masked});
  g.AddOp(kLiteRtOpCodeTflQuantize, {t_masked}, {out_attn});
  g.MarkOutput(out_attn);

  Tensor prev_tail = g.Bool({1, 2}), out_prev = g.Bool({1, 4});
  g.AddSlice(prev_mask, {0, 2}, {1, 2}, prev_tail);
  g.AddConcat({prev_tail, out_down_mask}, out_prev, 1);
  g.MarkOutput(out_prev);
  g.ApplyTo(subgraph);
}

TEST(BoolMaskTransformationTest, EndToEndAudioMaskPipeline) {
  for (bool with_output_ne : {true, false}) {
    LiteRtModelT model;
    LiteRtSubgraphT& subgraph = model.EmplaceSubgraph();
    BuildAudioMaskSubgraph(&subgraph);

    absl::flat_hash_map<const LiteRtTensorT*, EvalVal> feed;
    feed[subgraph.Inputs()[0]] = {
        kLiteRtElementTypeBool, {1, 9}, {}, {}, {1, 1, 1, 1, 1, 0, 0, 0, 0}};
    feed[subgraph.Inputs()[1]] = {
        kLiteRtElementTypeBool, {1, 4}, {}, {}, {0, 1, 1, 0}};
    std::vector<float> logits_v(32), proj_v(16);
    for (size_t i = 0; i < logits_v.size(); ++i) logits_v[i] = 0.25f * i - 3.0f;
    for (size_t i = 0; i < proj_v.size(); ++i) proj_v[i] = 0.5f * i + 1.0f;
    feed[subgraph.Inputs()[2]] = {
        kLiteRtElementTypeFloat32, {2, 2, 2, 4}, logits_v, {}, {}};
    feed[subgraph.Inputs()[3]] = {
        kLiteRtElementTypeFloat32, {1, 2, 8}, proj_v, {}, {}};

    const auto expected = EvalSubgraph(subgraph, feed);
    RunPatterns(&subgraph, AllPatterns(with_output_ne));
    ExpectTopologicalOrder(subgraph);
    ExpectSameOutputs(subgraph, expected, EvalSubgraph(subgraph, feed));

    EXPECT_EQ(CountBoolOps(subgraph), 0);
    EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflLogicalAnd), 0);
    EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflLogicalNot), 0);
    EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflSelectV2), 0);
    EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflTile), 1);
    EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflNotEqual),
              with_output_ne ? 3 : 0);
    EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflCast), with_output_ne ? 3 : 6)
        << DumpOps(subgraph);
  }
}

TEST(BoolMaskTransformationTest,
     ArbitraryFloatsAndNonFiniteConstantsPreserveExactSemantics) {
  LiteRtModelT model;
  LiteRtSubgraphT& subgraph = model.EmplaceSubgraph();
  GraphBuilder g;
  const float inf = std::numeric_limits<float>::infinity();
  const float nan_val = std::numeric_limits<float>::quiet_NaN();

  Tensor cond = g.Bool({1, 8}), x = g.F32({1, 8});
  Tensor y = g.F32({1, 8}), x_fin = g.F32({1, 8});
  Tensor zero = g.Const<float>({}, {0.0f});
  for (const Tensor& in : {cond, x, y, x_fin}) g.MarkInput(in);

  Tensor ne0 = g.Bool({1, 8}), inv0 = g.Bool({1, 8}), out_inv0 = g.F32({1, 8});
  g.AddOp(kLiteRtOpCodeTflNotEqual, {x, zero}, {ne0});
  g.AddOp(kLiteRtOpCodeTflLogicalNot, {ne0}, {inv0});
  g.AddOp(kLiteRtOpCodeTflCast, {inv0}, {out_inv0});

  Tensor eq_xy = g.Bool({1, 8});
  g.AddOp(kLiteRtOpCodeTflEqual, {x, y}, {eq_xy});

  Tensor b_cast = g.Bool({1, 8}), f_cast = g.F32({1, 8}),
         out_add = g.F32({1, 8});
  g.AddOp(kLiteRtOpCodeTflCast, {x}, {b_cast});
  g.AddOp(kLiteRtOpCodeTflCast, {b_cast}, {f_cast});
  AddOptions add_opts;
  add_opts.fused_activation_function = 0;
  g.SetOptions(g.AddOp(kLiteRtOpCodeTflAdd, {f_cast, zero}, {out_add}),
               std::move(add_opts));

  Tensor mid_nan = g.F32({1, 8}), out_nan_sel = g.F32({1, 8});
  Tensor out_inf_sel = g.F32({1, 8}), out_neg_zero_sel = g.F32({1, 8});
  g.AddOp(kLiteRtOpCodeTflSelectV2, {cond, g.Const<float>({}, {nan_val}), x},
          {mid_nan});
  g.AddOp(kLiteRtOpCodeTflSelectV2, {cond, y, mid_nan}, {out_nan_sel});
  g.AddOp(kLiteRtOpCodeTflSelectV2, {cond, g.Const<float>({}, {-inf}), x},
          {out_inf_sel});
  g.AddOp(kLiteRtOpCodeTflSelectV2, {cond, x_fin, g.Const<float>({}, {-0.0f})},
          {out_neg_zero_sel});

  for (const Tensor& o :
       {out_inv0, eq_xy, out_add, out_nan_sel, out_inf_sel, out_neg_zero_sel}) {
    g.MarkOutput(o);
  }
  g.ApplyTo(&subgraph);

  absl::flat_hash_map<const LiteRtTensorT*, EvalVal> feed;
  feed[cond.Get()] = {
      kLiteRtElementTypeBool, {1, 8}, {}, {}, {0, 1, 0, 1, 0, 1, 0, 1}};
  feed[x.Get()] = {kLiteRtElementTypeFloat32,
                   {1, 8},
                   {1e-12f, -1e-12f, 0.0f, -0.0f, inf, -inf, nan_val, 2e38f},
                   {},
                   {}};
  feed[y.Get()] = {kLiteRtElementTypeFloat32,
                   {1, 8},
                   {0.0f, 1e-12f, -0.0f, 0.0f, inf, -inf, nan_val, -2e38f},
                   {},
                   {}};
  feed[x_fin.Get()] = {
      kLiteRtElementTypeFloat32,
      {1, 8},
      {1e-12f, -1e-12f, 0.0f, -0.0f, 2.5f, -3.0f, 5e-7f, 2e38f},
      {},
      {}};
  const auto expected = EvalSubgraph(subgraph, feed);

  RunPatterns(&subgraph, AllPatterns(/*include_output_not_equal=*/true));
  ExpectTopologicalOrder(subgraph);
  const auto actual = EvalSubgraph(subgraph, feed);
  ExpectSameOutputs(subgraph, expected, actual);

  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflSelectV2), 3);
  EXPECT_FLOAT_EQ(actual.at(out_inv0.Get()).f32[0], 0.0f);
  EXPECT_EQ(actual.at(eq_xy.Get()).b[4], 1);  // +Inf == +Inf
  EXPECT_EQ(actual.at(eq_xy.Get()).b[6], 0);  // NaN != NaN
  EXPECT_FALSE(std::signbit(actual.at(out_add.Get()).f32[3]));
  EXPECT_FLOAT_EQ(actual.at(out_inf_sel.Get()).f32[0], 1e-12f);
}

TEST(BoolMaskTransformationTest,
     SubgraphOutputsAndOutputNotEqualPreserveInvariants) {
  LiteRtModelT model;
  LiteRtSubgraphT& subgraph = model.EmplaceSubgraph();
  GraphBuilder g;

  Tensor b_in = g.Bool({1, 2}), x_f = g.F32({1, 4}), y_f = g.F32({1, 4});
  Tensor idx_i32 = g.I32({1, 4}), b_21 = g.Bool({2, 1});
  for (const Tensor& in : {b_in, x_f, y_f, idx_i32, b_21}) g.MarkInput(in);

  Tensor dead_b = g.Bool({1, 4}), native_ne = g.Bool({1, 4});
  g.AddOp(kLiteRtOpCodeTflCast, {x_f}, {dead_b});
  g.AddOp(kLiteRtOpCodeTflNotEqual, {x_f, y_f}, {native_ne});
  g.MarkOutput(native_ne);

  Tensor x_21 = g.F32({2, 1}), bcast_ne = g.Bool({2, 4});
  g.AddOp(kLiteRtOpCodeTflCast, {b_21}, {x_21});
  g.AddOp(kLiteRtOpCodeTflNotEqual,
          {x_21, g.Const<float>({1, 4}, {1.0f, 0.0f, 1.0f, 0.0f})}, {bcast_ne});
  g.MarkOutput(bcast_ne);

  Tensor not_b = g.Bool({1, 2}), tiled_b = g.Bool({1, 4});
  g.AddOp(kLiteRtOpCodeTflLogicalNot, {b_in}, {not_b});
  g.MarkOutput(not_b);
  g.AddTileWrap(not_b, {1, 2}, {1, 2}, {1, 4}, tiled_b);
  g.MarkOutput(tiled_b);

  Tensor sel_f = g.F32({1, 4}), sel_i32 = g.I32({1, 4});
  g.AddOp(kLiteRtOpCodeTflSelectV2, {tiled_b, x_f, g.Const<float>({}, {0.0f})},
          {sel_f});
  g.AddOp(kLiteRtOpCodeTflSelectV2,
          {tiled_b, idx_i32, g.Const<int32_t>({1, 4}, {0, 0, 0, 0})},
          {sel_i32});
  g.MarkOutput(sel_f);
  g.MarkOutput(sel_i32);
  g.ApplyTo(&subgraph);

  absl::flat_hash_map<const LiteRtTensorT*, EvalVal> feed;
  feed[b_in.Get()] = {kLiteRtElementTypeBool, {1, 2}, {}, {}, {0, 1}};
  feed[x_f.Get()] = {
      kLiteRtElementTypeFloat32, {1, 4}, {0.5f, 0.0f, -1.0f, 2.0f}, {}, {}};
  feed[y_f.Get()] = {
      kLiteRtElementTypeFloat32, {1, 4}, {0.5f, 1.0f, -1.0f, 0.0f}, {}, {}};
  feed[idx_i32.Get()] = {
      kLiteRtElementTypeInt32, {1, 4}, {}, {10, 20, 30, 40}, {}};
  feed[b_21.Get()] = {kLiteRtElementTypeBool, {2, 1}, {}, {}, {0, 1}};
  const auto expected = EvalSubgraph(subgraph, feed);

  RunPatterns(&subgraph, AllPatterns(/*include_output_not_equal=*/true));
  ExpectTopologicalOrder(subgraph);
  ExpectSameOutputs(subgraph, expected, EvalSubgraph(subgraph, feed));

  EXPECT_EQ(not_b.Get()->NumUses(), 0);
  EXPECT_EQ(not_b.Get()->DefiningOp()->OpCode(), kLiteRtOpCodeTflNotEqual);
  EXPECT_EQ(tiled_b.Get()->DefiningOp()->OpCode(), kLiteRtOpCodeTflCast);
  EXPECT_EQ(tiled_b.Get()->NumUses(), 1);
  EXPECT_EQ(native_ne.Get()->DefiningOp()->Inputs()[0], x_f.Get());
  EXPECT_EQ(native_ne.Get()->DefiningOp()->Inputs()[1], y_f.Get());
}

TEST(BoolMaskTransformationTest,
     SelectVsSelectV2BroadcastingRulesAndLogicalOr) {
  LiteRtModelT model;
  LiteRtSubgraphT& subgraph = model.EmplaceSubgraph();
  GraphBuilder g;

  Tensor cond_batch = g.Bool({2}), cond_trailing = g.Bool({4});
  Tensor cond_rank2 = g.Bool({2, 1, 1}), cond_scalar = g.Bool({});
  Tensor x_234 = g.F32({2, 3, 4}), y_234 = g.F32({2, 3, 4});
  Tensor x_232 = g.F32({2, 3, 2}), y_232 = g.F32({2, 3, 2});
  Tensor y_134 = g.F32({1, 3, 4});
  for (const Tensor& in : {cond_batch, cond_trailing, cond_rank2, cond_scalar,
                           x_234, y_234, x_232, y_232, y_134}) {
    g.MarkInput(in);
  }

  Tensor lor = g.Bool({2});
  g.AddOp(kLiteRtOpCodeTflLogicalOr, {cond_batch, cond_batch}, {lor});

  Tensor o1 = g.F32({2, 3, 4}), o2 = g.F32({2, 3, 2}), o3 = g.F32({2, 3, 4});
  Tensor o4 = g.F32({2, 3, 4}), o5 = g.F32({2, 3, 4});
  g.AddOp(kLiteRtOpCodeTflSelect, {lor, x_234, y_234}, {o1});
  g.AddOp(kLiteRtOpCodeTflSelect, {cond_batch, x_232, y_232}, {o2});
  g.AddOp(kLiteRtOpCodeTflSelect, {cond_scalar, x_234, y_234}, {o3});
  g.AddOp(kLiteRtOpCodeTflSelectV2, {cond_trailing, x_234, y_134}, {o4});
  g.AddOp(kLiteRtOpCodeTflSelectV2,
          {cond_trailing, g.Const<float>({}, {0.0f}), y_234}, {o5});

  Op op_bad1 = g.AddOp(kLiteRtOpCodeTflSelect, {cond_trailing, x_234, y_234},
                       {g.F32({2, 3, 4})});
  Op op_bad2 = g.AddOp(kLiteRtOpCodeTflSelect, {cond_rank2, x_234, y_234},
                       {g.F32({2, 3, 4})});
  Op op_bad3 = g.AddOp(kLiteRtOpCodeTflSelect, {cond_batch, x_234, y_134},
                       {g.F32({2, 3, 4})});
  for (const Tensor& o : {o1, o2, o3, o4, o5}) g.MarkOutput(o);
  g.ApplyTo(&subgraph);

  for (const Op& bad : {op_bad1, op_bad2, op_bad3}) {
    LiteRtBuilderT b;
    EXPECT_EQ(BoolMaskOpToFloatTransformation(Ctx(), &b, bad.Get()),
              kLiteRtStatusPatternNoMatch);
  }

  absl::flat_hash_map<const LiteRtTensorT*, EvalVal> feed;
  feed[cond_batch.Get()] = {kLiteRtElementTypeBool, {2}, {}, {}, {1, 0}};
  feed[cond_trailing.Get()] = {
      kLiteRtElementTypeBool, {4}, {}, {}, {1, 0, 1, 0}};
  feed[cond_rank2.Get()] = {kLiteRtElementTypeBool, {2, 1, 1}, {}, {}, {0, 1}};
  feed[cond_scalar.Get()] = {kLiteRtElementTypeBool, {}, {}, {}, {1}};
  std::vector<float> vx234(24), vy234(24), vx232(12), vy232(12), vy134(12);
  for (size_t i = 0; i < 24; ++i) {
    vx234[i] = 10.0f + i;
    vy234[i] = -10.0f - i;
  }
  for (size_t i = 0; i < 12; ++i) {
    vx232[i] = 100.0f + i;
    vy232[i] = -100.0f - i;
    vy134[i] = 50.0f + i;
  }
  feed[x_234.Get()] = {kLiteRtElementTypeFloat32, {2, 3, 4}, vx234, {}, {}};
  feed[y_234.Get()] = {kLiteRtElementTypeFloat32, {2, 3, 4}, vy234, {}, {}};
  feed[x_232.Get()] = {kLiteRtElementTypeFloat32, {2, 3, 2}, vx232, {}, {}};
  feed[y_232.Get()] = {kLiteRtElementTypeFloat32, {2, 3, 2}, vy232, {}, {}};
  feed[y_134.Get()] = {kLiteRtElementTypeFloat32, {1, 3, 4}, vy134, {}, {}};

  const auto expected = EvalSubgraph(subgraph, feed);
  RunPatterns(&subgraph, AllPatterns(/*include_output_not_equal=*/true));
  ExpectTopologicalOrder(subgraph);
  ExpectSameOutputs(subgraph, expected, EvalSubgraph(subgraph, feed));
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflSelect), 3);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflSelectV2), 0);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflLogicalOr), 0);
}

TEST(BoolMaskTransformationTest, ZeroSizedDynamicAndRankZeroGuards) {
  LiteRtModelT model;
  LiteRtSubgraphT& subgraph = model.EmplaceSubgraph();
  GraphBuilder g;

  Tensor zero_b = g.Bool({0, 4}), zero_b_out = g.Bool({0, 4});
  Tensor dyn_b = g.Bool({-1, 4}), dyn_b_out = g.Bool({-1, 4});
  Tensor b_in = g.Bool({1, 2}), scalar_f = g.F32({}), scalar_b_out = g.Bool({});
  for (const Tensor& in : {zero_b, dyn_b, b_in, scalar_f}) g.MarkInput(in);

  g.AddOp(kLiteRtOpCodeTflLogicalNot, {zero_b}, {zero_b_out});
  g.AddOp(kLiteRtOpCodeTflLogicalNot, {dyn_b}, {dyn_b_out});
  Tensor tb0 = g.Bool({1, 0}), bad_slice = g.Bool({1, 2});
  g.AddTileWrap(b_in, {1, 0}, {1, 2}, {1, 0}, tb0);
  g.AddSlice(b_in, {0, 1}, {1, 2}, bad_slice);
  g.AddOp(kLiteRtOpCodeTflCast, {scalar_f}, {scalar_b_out});

  for (const Tensor& o :
       {zero_b_out, dyn_b_out, tb0, bad_slice, scalar_b_out}) {
    g.MarkOutput(o);
  }
  g.ApplyTo(&subgraph);

  const size_t ops_before = subgraph.Ops().size();
  RunPatterns(&subgraph, AllPatterns(/*include_output_not_equal=*/true));
  ExpectTopologicalOrder(subgraph);
  EXPECT_EQ(subgraph.Ops().size(), ops_before) << DumpOps(subgraph);

  LiteRtBuilderT null_check_builder;
  for (const auto& entry : AllPatterns(/*include_output_not_equal=*/true)) {
    EXPECT_EQ(entry.fn(nullptr, &null_check_builder, subgraph.Ops()[0]),
              kLiteRtStatusPatternNoMatch);
    EXPECT_EQ(entry.fn(Ctx(), nullptr, subgraph.Ops()[0]),
              kLiteRtStatusPatternNoMatch);
    EXPECT_EQ(entry.fn(Ctx(), &null_check_builder, nullptr),
              kLiteRtStatusPatternNoMatch);
  }
}

TEST(BoolMaskTransformationTest,
     SelfAliasingEqualActivationsQuantizedAndMemoization) {
  LiteRtModelT model;
  LiteRtSubgraphT& subgraph = model.EmplaceSubgraph();
  GraphBuilder g;

  Tensor b1 = g.Bool({1, 4}), b2 = g.Bool({1, 4});
  Tensor q_bool = g.Quant(ElementType::Bool, {1, 4}, 1.0f, 0);
  Tensor q_f32 = g.Quant(ElementType::Float32, {1, 4}, 0.5f, 0);
  for (const Tensor& in : {b1, b2, q_bool, q_f32}) g.MarkInput(in);

  Tensor and_self = g.Bool({1, 4}), cat_self = g.Bool({1, 8});
  g.AddOp(kLiteRtOpCodeTflLogicalAnd, {b1, b1}, {and_self});
  g.AddConcat({and_self, and_self}, cat_self, 1);
  g.MarkOutput(cat_self);

  Tensor sel_self = g.Bool({1, 4}), ne_xor = g.Bool({1, 4}),
         eq_xnor = g.Bool({1, 4});
  g.AddOp(kLiteRtOpCodeTflSelect, {b1, b1, b2}, {sel_self});
  g.AddOp(kLiteRtOpCodeTflNotEqual, {sel_self, b2}, {ne_xor});
  g.AddOp(kLiteRtOpCodeTflEqual, {ne_xor, b1}, {eq_xnor});
  g.MarkOutput(eq_xnor);

  Tensor f01 = g.F32({1, 4}), b_mid = g.Bool({1, 4}), f_mid = g.F32({1, 4});
  g.AddOp(kLiteRtOpCodeTflCast, {b1}, {f01});
  g.AddOp(kLiteRtOpCodeTflCast, {f01}, {b_mid});
  g.AddOp(kLiteRtOpCodeTflCast, {b_mid}, {f_mid});
  Tensor tanh_out = g.F32({1, 4}), eq_tanh = g.Bool({1, 4});
  MulOptions tanh_opts;
  tanh_opts.fused_activation_function = 2;
  g.SetOptions(g.AddOp(kLiteRtOpCodeTflMul, {f_mid, f_mid}, {tanh_out}),
               tanh_opts);
  g.AddOp(kLiteRtOpCodeTflEqual, {tanh_out, g.Const<float>({}, {1.0f})},
          {eq_tanh});
  g.MarkOutput(eq_tanh);

  Tensor q_bool_out = g.Quant(ElementType::Bool, {1, 4}, 1.0f, 0);
  Tensor q_ne_out = g.Bool({1, 4});
  g.AddOp(kLiteRtOpCodeTflLogicalNot, {q_bool}, {q_bool_out});
  g.AddOp(kLiteRtOpCodeTflNotEqual, {q_f32, g.Const<float>({}, {0.0f})},
          {q_ne_out});
  g.MarkOutput(q_bool_out);
  g.MarkOutput(q_ne_out);

  Tensor t = f01;
  for (int i = 0; i < 23; ++i) {
    Tensor next = g.F32({1, 4});
    MulOptions opts;
    opts.fused_activation_function = 0;
    g.SetOptions(g.AddOp(kLiteRtOpCodeTflMul, {t, t}, {next}), opts);
    t = next;
  }
  Tensor deep_eq = g.Bool({1, 4});
  g.AddOp(kLiteRtOpCodeTflEqual, {t, g.Const<float>({}, {1.0f})}, {deep_eq});
  g.MarkOutput(deep_eq);
  g.ApplyTo(&subgraph);

  absl::flat_hash_map<const LiteRtTensorT*, EvalVal> feed;
  feed[b1.Get()] = {kLiteRtElementTypeBool, {1, 4}, {}, {}, {0, 1, 0, 1}};
  feed[b2.Get()] = {kLiteRtElementTypeBool, {1, 4}, {}, {}, {0, 0, 1, 1}};
  feed[q_bool.Get()] = {kLiteRtElementTypeBool, {1, 4}, {}, {}, {1, 0, 1, 0}};
  feed[q_f32.Get()] = {
      kLiteRtElementTypeFloat32, {1, 4}, {0.0f, 1.0f, 0.5f, 0.0f}, {}, {}};
  const auto expected = EvalSubgraph(subgraph, feed);

  RunPatterns(&subgraph, AllPatterns(/*include_output_not_equal=*/true));
  ExpectTopologicalOrder(subgraph);
  const auto actual = EvalSubgraph(subgraph, feed);
  ExpectSameOutputs(subgraph, expected, actual);
  EXPECT_EQ(actual.at(eq_tanh.Get()).b[1], 0);  // tanh(1.0) != 1.0
}

TEST(BoolMaskTransformationTest, UnrewrittenProducerBoolChainsStayIntact) {
  LiteRtModelT model;
  LiteRtSubgraphT& subgraph = model.EmplaceSubgraph();
  GraphBuilder g;

  Tensor coord = g.I32({1, 4}), table = g.F32({1, 4, 8});
  Tensor arb_mask_f = g.F32({1, 4, 1, 4});
  for (const Tensor& in : {coord, table, arb_mask_f}) g.MarkInput(in);

  Tensor lt = g.Bool({1, 4}), ge = g.Bool({1, 4}), lor = g.Bool({1, 4});
  Tensor ne_i32 = g.Bool({1, 4}), land = g.Bool({1, 4});
  Tensor r_cond = g.Bool({1, 4, 1}), sel_out = g.F32({1, 4, 8});
  g.AddOp(kLiteRtOpCodeTflLess, {coord, g.Const<int32_t>({}, {0})}, {lt});
  g.AddOp(kLiteRtOpCodeTflGreaterEqual, {coord, g.Const<int32_t>({}, {16})},
          {ge});
  g.AddOp(kLiteRtOpCodeTflLogicalOr, {lt, ge}, {lor});
  g.AddOp(kLiteRtOpCodeTflNotEqual, {coord, g.Const<int32_t>({}, {-1})},
          {ne_i32});
  g.AddOp(kLiteRtOpCodeTflLogicalAnd, {lor, ne_i32}, {land});
  g.AddReshape(land, {1, 4, 1}, r_cond);
  g.AddOp(kLiteRtOpCodeTflSelectV2, {r_cond, g.Const<float>({}, {0.0f}), table},
          {sel_out});
  g.MarkOutput(sel_out);

  Tensor ne_f = g.Bool({1, 4, 1, 4}), lnot_f = g.Bool({1, 4, 1, 4}),
         cast_f = g.F32({1, 4, 1, 4});
  g.AddOp(kLiteRtOpCodeTflNotEqual, {arb_mask_f, g.Const<float>({}, {0.0f})},
          {ne_f});
  g.AddOp(kLiteRtOpCodeTflLogicalNot, {ne_f}, {lnot_f});
  g.AddOp(kLiteRtOpCodeTflCast, {lnot_f}, {cast_f});
  g.MarkOutput(cast_f);
  g.ApplyTo(&subgraph);

  const size_t ops_before = subgraph.Ops().size();
  RunPatterns(&subgraph, AllPatterns(/*include_output_not_equal=*/true));
  ExpectTopologicalOrder(subgraph);
  EXPECT_EQ(subgraph.Ops().size(), ops_before) << DumpOps(subgraph);
}

TEST(BoolMaskTransformationTest, RandomSeededMaskDagParityStress) {
  const float inf = std::numeric_limits<float>::infinity();
  const float nan_val = std::numeric_limits<float>::quiet_NaN();
  for (uint32_t trial = 0; trial < 200; ++trial) {
    std::mt19937 rng(20261120u + trial);
    LiteRtModelT model;
    LiteRtSubgraphT& subgraph = model.EmplaceSubgraph();
    GraphBuilder g;

    Tensor b0 = g.Bool({2, 3}), b1 = g.Bool({2, 3}), b_batch = g.Bool({2});
    Tensor f_clean = g.F32({2, 3}), f_tiny = g.F32({2, 3}),
           f_nonfinite = g.F32({2, 3});
    for (const Tensor& in : {b0, b1, b_batch, f_clean, f_tiny, f_nonfinite}) {
      g.MarkInput(in);
    }

    std::vector<Tensor> bool_pool = {b0, b1}, f01_pool;
    std::vector<Tensor> farbit_pool = {f_clean, f_tiny};
    const int num_steps = 8 + static_cast<int>(rng() % 7);
    for (int step = 0; step < num_steps; ++step) {
      const uint32_t kind = rng() % 10;
      auto pick_bool = [&] { return bool_pool[rng() % bool_pool.size()]; };
      auto pick_f32 = [&] {
        return (!f01_pool.empty() && (rng() % 3 == 0))
                   ? f01_pool[rng() % f01_pool.size()]
                   : farbit_pool[rng() % farbit_pool.size()];
      };
      if (kind == 0) {
        Tensor out = g.Bool({2, 3});
        g.AddOp(kLiteRtOpCodeTflLogicalNot, {pick_bool()}, {out});
        bool_pool.push_back(out);
      } else if (kind == 1 || kind == 2) {
        Tensor lhs = pick_bool();
        Tensor rhs = (rng() % 4 == 0) ? lhs : pick_bool();
        Tensor out = g.Bool({2, 3});
        g.AddOp(
            kind == 1 ? kLiteRtOpCodeTflLogicalAnd : kLiteRtOpCodeTflLogicalOr,
            {lhs, rhs}, {out});
        bool_pool.push_back(out);
      } else if (kind == 3) {
        Tensor lhs = pick_bool();
        Tensor rhs = (rng() % 4 == 0) ? lhs : pick_bool();
        Tensor out = g.Bool({2, 3});
        g.AddOp(
            (rng() % 2 == 0) ? kLiteRtOpCodeTflNotEqual : kLiteRtOpCodeTflEqual,
            {lhs, rhs}, {out});
        bool_pool.push_back(out);
      } else if (kind == 4) {
        Tensor f01 = g.F32({2, 3}), out = g.Bool({2, 3});
        g.AddOp(kLiteRtOpCodeTflCast, {pick_bool()}, {f01});
        f01_pool.push_back(f01);
        g.AddOp(
            (rng() % 2 == 0) ? kLiteRtOpCodeTflNotEqual : kLiteRtOpCodeTflEqual,
            {f01, g.Const<float>({}, {(rng() % 2 == 0) ? 0.0f : 1.0f})}, {out});
        bool_pool.push_back(out);
      } else if (kind == 5) {
        Tensor src = (rng() % 2 == 0) ? f_nonfinite : pick_f32();
        Tensor rhs = (rng() % 2 == 0) ? src : g.Const<float>({}, {0.0f});
        Tensor out = g.Bool({2, 3});
        g.AddOp(
            (rng() % 2 == 0) ? kLiteRtOpCodeTflNotEqual : kLiteRtOpCodeTflEqual,
            {src, rhs}, {out});
        bool_pool.push_back(out);
      } else if (kind == 6) {
        Tensor r3d = g.Bool({2, 1, 3}), back = g.Bool({2, 3});
        g.AddReshape(pick_bool(), {2, 1, 3}, r3d);
        g.AddReshape(r3d, {2, 3}, back);
        bool_pool.push_back(back);
      } else if (kind == 7) {
        Tensor b_tiled = g.Bool({2, 6}), sliced = g.Bool({2, 3});
        g.AddTileWrap(pick_bool(), {1, 2}, {2, 3}, {2, 6}, b_tiled);
        g.AddSlice(b_tiled, {0, 2}, {2, 3}, sliced);
        bool_pool.push_back(sliced);
      } else if (kind == 8) {
        Tensor out = g.F32({2, 3});
        g.AddOp(kLiteRtOpCodeTflSelect, {b_batch, pick_f32(), pick_f32()},
                {out});
        farbit_pool.push_back(out);
      } else {
        Tensor t_br = pick_f32(), f_br = pick_f32(), out = g.F32({2, 3});
        const uint32_t c_mode = rng() % 4;
        if (c_mode == 0) f_br = g.Const<float>({}, {0.0f});
        if (c_mode == 1) f_br = g.Const<float>({}, {-0.0f});
        if (c_mode == 2) t_br = g.Const<float>({}, {nan_val});
        g.AddOp(kLiteRtOpCodeTflSelectV2, {pick_bool(), t_br, f_br}, {out});
        farbit_pool.push_back(out);
      }
      if (rng() % 3 == 0) g.MarkOutput(bool_pool.back());
    }

    g.MarkOutput(bool_pool.back());
    g.MarkOutput(farbit_pool.back());
    g.ApplyTo(&subgraph);

    absl::flat_hash_map<const LiteRtTensorT*, EvalVal> feed;
    std::vector<uint8_t> vb0(6), vb1(6),
        vbb = {static_cast<uint8_t>(rng() & 1),
               static_cast<uint8_t>(rng() & 1)};
    for (int i = 0; i < 6; ++i) {
      vb0[i] = static_cast<uint8_t>(rng() & 1u);
      vb1[i] = static_cast<uint8_t>(rng() & 1u);
    }
    feed[b0.Get()] = {kLiteRtElementTypeBool, {2, 3}, {}, {}, vb0};
    feed[b1.Get()] = {kLiteRtElementTypeBool, {2, 3}, {}, {}, vb1};
    feed[b_batch.Get()] = {kLiteRtElementTypeBool, {2}, {}, {}, vbb};
    feed[f_clean.Get()] = {kLiteRtElementTypeFloat32,
                           {2, 3},
                           {-2.5f, 0.0f, 1.25f, 3.5f, -0.75f, 9.0f},
                           {},
                           {}};
    feed[f_tiny.Get()] = {kLiteRtElementTypeFloat32,
                          {2, 3},
                          {1e-12f, -1e-12f, 0.0f, -0.0f, 1e20f, -1e20f},
                          {},
                          {}};
    feed[f_nonfinite.Get()] = {kLiteRtElementTypeFloat32,
                               {2, 3},
                               {1e-12f, -1e-12f, 0.0f, -0.0f, inf, nan_val},
                               {},
                               {}};

    const auto expected = EvalSubgraph(subgraph, feed);
    RunPatterns(&subgraph,
                AllPatterns(/*include_output_not_equal=*/(trial % 2 == 0)));
    ExpectTopologicalOrder(subgraph);
    ExpectSameOutputs(subgraph, expected, EvalSubgraph(subgraph, feed));
  }
}

TEST(BoolMaskTransformationTest,
     DeepNonFiniteSelectBranchLeavesGraphUnmodified) {
  LiteRtModelT model;
  LiteRtSubgraphT& subgraph = model.EmplaceSubgraph();
  GraphBuilder g;
  Tensor cond = g.Bool({1, 2});
  g.MarkInput(cond);
  Tensor branch =
      g.Const<float>({1, 2}, {-std::numeric_limits<float>::infinity(), 3.0f});
  for (int i = 0; i < 26; ++i) {
    Tensor next = g.F32({1, 2});
    g.AddReshape(branch, {1, 2}, next);
    branch = next;
  }
  Tensor out = g.F32({1, 2});
  Op sel = g.AddOp(kLiteRtOpCodeTflSelectV2,
                   {cond, g.Const<float>({1, 2}, {0.0f, 0.0f}), branch}, {out});
  g.MarkOutput(out);
  g.ApplyTo(&subgraph);

  absl::flat_hash_map<const LiteRtTensorT*, EvalVal> feed;
  feed[cond.Get()] = {kLiteRtElementTypeBool, {1, 2}, {}, {}, {1, 0}};
  const auto expected = EvalSubgraph(subgraph, feed);
  ASSERT_EQ(expected.at(out.Get()).f32, (std::vector<float>{0.0f, 3.0f}));

  RunPatterns(&subgraph, AllPatterns(/*include_output_not_equal=*/true));
  ExpectTopologicalOrder(subgraph);
  EXPECT_EQ(out.Get()->DefiningOp(), sel.Get());
  ExpectSameOutputs(subgraph, expected, EvalSubgraph(subgraph, feed));
}

TEST(BoolMaskTransformationTest, SelectBroadcastMismatchLeavesBuilderEmpty) {
  LiteRtModelT model;
  LiteRtSubgraphT& subgraph = model.EmplaceSubgraph();
  GraphBuilder g;
  Tensor cond = g.Bool({2, 3}), x = g.F32({2, 4}), y = g.F32({2, 3}),
         out = g.F32({2, 3});
  for (const Tensor& in : {cond, x, y}) g.MarkInput(in);
  Op bad_sel = g.AddOp(kLiteRtOpCodeTflSelectV2, {cond, x, y}, {out});
  g.MarkOutput(out);
  g.ApplyTo(&subgraph);

  const size_t cond_users_before = cond.Get()->Users().size();
  LiteRtBuilderT builder;
  EXPECT_EQ(BoolMaskOpToFloatTransformation(Ctx(), &builder, bad_sel.Get()),
            kLiteRtStatusPatternNoMatch);
  EXPECT_EQ(builder.Subgraph().Ops().size(), 0u);
  EXPECT_EQ(builder.Subgraph().Tensors().size(), 0u);
  EXPECT_EQ(cond.Get()->Users().size(), cond_users_before);
}

TEST(BoolMaskTransformationTest, BypassCastScalarReshapeLeavesBuilderEmpty) {
  LiteRtModelT model;
  LiteRtSubgraphT& subgraph = model.EmplaceSubgraph();
  GraphBuilder g;
  Tensor b_in = g.Bool({1}), f01 = g.F32({1}), b_mid = g.Bool({1}),
         f_mid = g.F32({1}), scalar_out = g.F32({});
  g.MarkInput(b_in);
  g.AddOp(kLiteRtOpCodeTflCast, {b_in}, {f01});
  g.AddOp(kLiteRtOpCodeTflCast, {f01}, {b_mid});
  g.AddOp(kLiteRtOpCodeTflCast, {b_mid}, {f_mid});
  Op scalar_reshape = g.AddOp(kLiteRtOpCodeTflReshape, {f_mid}, {scalar_out});
  g.SetOptions(scalar_reshape, ReshapeOptions{});
  g.MarkOutput(scalar_out);
  g.ApplyTo(&subgraph);

  LiteRtOp orig_def = scalar_out.Get()->DefiningOp();
  const size_t f01_users_before = f01.Get()->Users().size();
  LiteRtBuilderT builder;
  EXPECT_EQ(
      BoolMaskBypassCastTransformation(Ctx(), &builder, scalar_reshape.Get()),
      kLiteRtStatusPatternNoMatch);
  EXPECT_EQ(builder.Subgraph().Ops().size(), 0u);
  EXPECT_EQ(builder.Subgraph().Tensors().size(), 0u);
  EXPECT_EQ(scalar_out.Get()->DefiningOp(), orig_def);
  EXPECT_EQ(f01.Get()->Users().size(), f01_users_before);
}

TEST(BoolMaskTransformationTest,
     LogicalNotAndComparisonRejectBeforeMutatingBuilder) {
  constexpr int32_t kOversized = (1 << 20) + 1;
  LiteRtModelT model;
  LiteRtSubgraphT& subgraph = model.EmplaceSubgraph();
  GraphBuilder g;
  Tensor i32_in = g.I32({2}), lt_out = g.Bool({2}), lnot_out = g.Bool({2});
  Tensor lhs_neg_zero = g.Const<float>({1}, {-0.0f});
  Tensor rhs_neg_zero =
      g.Const<float>({kOversized}, std::vector<float>(kOversized, -0.0f));
  Tensor ne_out = g.Bool({kOversized});
  g.MarkInput(i32_in);
  g.AddOp(kLiteRtOpCodeTflLess, {i32_in, g.Const<int32_t>({}, {0})}, {lt_out});
  Op lnot = g.AddOp(kLiteRtOpCodeTflLogicalNot, {lt_out}, {lnot_out});
  Op oversized_ne =
      g.AddOp(kLiteRtOpCodeTflNotEqual, {lhs_neg_zero, rhs_neg_zero}, {ne_out});
  g.MarkOutput(lnot_out);
  g.MarkOutput(ne_out);
  g.ApplyTo(&subgraph);

  const size_t lt_users_before = lt_out.Get()->Users().size();
  LiteRtBuilderT b1;
  EXPECT_EQ(BoolMaskOpToFloatTransformation(Ctx(), &b1, lnot.Get()),
            kLiteRtStatusPatternNoMatch);
  EXPECT_EQ(b1.Subgraph().Ops().size(), 0u);
  EXPECT_EQ(b1.Subgraph().Tensors().size(), 0u);
  EXPECT_EQ(lt_out.Get()->Users().size(), lt_users_before);

  LiteRtBuilderT b2;
  EXPECT_EQ(BoolMaskNotEqualTransformation(Ctx(), &b2, oversized_ne.Get()),
            kLiteRtStatusPatternNoMatch);
  EXPECT_EQ(b2.Subgraph().Ops().size(), 0u);
  EXPECT_EQ(b2.Subgraph().Tensors().size(), 0u);
}

}  // namespace
}  // namespace litert::mediatek
