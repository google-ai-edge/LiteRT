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
using ::litert::compiler::SqueezeOptions;
using ::litert::compiler::StridedSliceOptions;
using ::litert::compiler::SubOptions;
using ::litert::compiler::Tensor;

using Dims = std::vector<int32_t>;

struct PatternEntry {
  LiteRtStatus (*fn)(const LiteRtCompilerContext*, LiteRtBuilder, LiteRtOp);
  const char* name;
};

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
    EXPECT_TRUE(t.HasValue());
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
    EXPECT_TRUE(t.HasValue());
    return *t;
  }

  template <typename T>
  Tensor Const(const Dims& dims, const std::vector<T>& data) {
    Tensor t = Typed(GetElementType<T>(), dims);
    auto w = builder_.BuildWeights<T>(absl::MakeConstSpan(data), t);
    EXPECT_TRUE(w.HasValue());
    return t;
  }

  Tensor BoolConst(const Dims& dims, absl::Span<const uint8_t> data) {
    Tensor t = Typed(ElementType::Bool, dims);
    auto buf = std::make_unique<bool[]>(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
      buf[i] = data[i] != 0;
    }
    auto w = builder_.BuildWeights<bool>(
        absl::Span<const bool>(buf.get(), data.size()), t);
    EXPECT_TRUE(w.HasValue());
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

  void ApplyTo(LiteRtSubgraphT* subgraph) {
    impl_.ApplyChanges(subgraph);
    applied_ = true;
    auto is_live = [&](LiteRtTensor candidate) {
      return std::find(subgraph->Tensors().begin(), subgraph->Tensors().end(),
                       candidate) != subgraph->Tensors().end();
    };
    for (LiteRtTensor in : pending_inputs_) {
      if (is_live(in)) {
        subgraph->Inputs().push_back(in);
      }
    }
    for (LiteRtTensor out : pending_outputs_) {
      if (is_live(out)) {
        subgraph->Outputs().push_back(out);
      }
    }
    pending_inputs_.clear();
    pending_outputs_.clear();
  }

  void MarkInput(const Tensor& t, LiteRtSubgraphT* subgraph) {
    if (applied_) {
      subgraph->Inputs().push_back(t.Get());
    } else {
      pending_inputs_.push_back(t.Get());
    }
  }

  void MarkOutput(const Tensor& t, LiteRtSubgraphT* subgraph) {
    if (applied_) {
      subgraph->Outputs().push_back(t.Get());
    } else {
      pending_outputs_.push_back(t.Get());
    }
  }

 private:
  const LiteRtCompilerContext* ctx_;
  LiteRtBuilderT impl_;
  Builder builder_;
  bool applied_ = false;
  std::vector<LiteRtTensor> pending_inputs_;
  std::vector<LiteRtTensor> pending_outputs_;
};

const LiteRtCompilerContext* Ctx() { return LrtGetCompilerContext(); }

void SetConcatAxis(GraphBuilder& g, const Op& op, int32_t axis,
                   uint32_t act = 0) {
  ConcatenationOptions opts;
  opts.axis = axis;
  opts.fused_activation_function = act;
  g.SetOptions(op, std::move(opts));
}

void SetReshapeShape(GraphBuilder& g, const Op& op, const Dims& dims) {
  ReshapeOptions opts;
  opts.new_shape = dims;
  g.SetOptions(op, std::move(opts));
}

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

// Lightweight reference evaluator for CPU parity verification across mask ops.
struct EvalVal {
  LiteRtElementType dtype = kLiteRtElementTypeNone;
  Dims dims;
  std::vector<float> f32;
  std::vector<int64_t> i64;
  std::vector<uint8_t> b;
};

size_t FlatSize(const Dims& dims) {
  size_t n = 1;
  for (int32_t d : dims) {
    n *= static_cast<size_t>(d);
  }
  return n;
}

Dims TensorDims(const LiteRtTensorT* t) {
  const auto& layout = t->Type().second.ranked_tensor_type.layout;
  return Dims(layout.dimensions, layout.dimensions + layout.rank);
}

size_t BroadcastIndex(size_t out_linear, const Dims& out_dims,
                      const Dims& in_dims) {
  if (in_dims.empty()) {
    return 0;
  }
  std::vector<int32_t> coords(out_dims.size(), 0);
  size_t rem = out_linear;
  for (size_t i = out_dims.size(); i > 0; --i) {
    const size_t d = static_cast<size_t>(out_dims[i - 1]);
    coords[i - 1] = static_cast<int32_t>(rem % d);
    rem /= d;
  }
  const size_t offset = out_dims.size() - in_dims.size();
  size_t idx = 0;
  for (size_t i = 0; i < in_dims.size(); ++i) {
    const int32_t c = (in_dims[i] == 1) ? 0 : coords[offset + i];
    idx = idx * static_cast<size_t>(in_dims[i]) + static_cast<size_t>(c);
  }
  return idx;
}

float ApplyActivation(float v, uint32_t act) {
  if (act == 1) {
    return (std::max)(0.0f, v);
  }
  if (act == 3) {
    return (std::min)(6.0f, (std::max)(0.0f, v));
  }
  if (act == 2) {
    return std::tanh(v);
  }
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
    for (size_t i = 0; i < n; ++i) {
      v.b[i] = raw[i] != 0 ? 1 : 0;
    }
  }
  return v;
}

absl::flat_hash_map<const LiteRtTensorT*, EvalVal> EvalSubgraph(
    const LiteRtSubgraphT& subgraph,
    const absl::flat_hash_map<const LiteRtTensorT*, EvalVal>& feed) {
  absl::flat_hash_map<const LiteRtTensorT*, EvalVal> env;
  const EvalVal empty_val{};
  for (const LiteRtTensorT* in : subgraph.Inputs()) {
    if (auto it = feed.find(in); it != feed.end()) {
      env[in] = it->second;
    }
  }
  for (const LiteRtTensorT* t : subgraph.Tensors()) {
    if (t->Weights().Buffer().Size() > 0) {
      env[t] = LoadConstVal(t);
    }
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
               code == kLiteRtOpCodeTflSqueeze ||
               code == kLiteRtOpCodeTflExpandDims) {
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
      const EvalVal& a = in_val(0);
      const EvalVal& b = in_val(1);
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
      const EvalVal& a = in_val(0);
      const EvalVal& b = in_val(1);
      uint32_t act = 0;
      if (code == kLiteRtOpCodeTflAdd) {
        if (auto o = GetOptionsAs<AddOptions>(Ctx(), op); o) {
          act = o->fused_activation_function;
        }
      } else if (code == kLiteRtOpCodeTflSub) {
        if (auto o = GetOptionsAs<SubOptions>(Ctx(), op); o) {
          act = o->fused_activation_function;
        }
      } else if (code == kLiteRtOpCodeTflMul) {
        if (auto o = GetOptionsAs<MulOptions>(Ctx(), op); o) {
          act = o->fused_activation_function;
        }
      }
      out.f32.resize(n);
      for (size_t i = 0; i < n; ++i) {
        const float va = a.f32[BroadcastIndex(i, out.dims, a.dims)];
        const float vb = b.f32[BroadcastIndex(i, out.dims, b.dims)];
        float r = 0.0f;
        if (code == kLiteRtOpCodeTflAdd) r = va + vb;
        if (code == kLiteRtOpCodeTflSub) r = va - vb;
        if (code == kLiteRtOpCodeTflMul) r = va * vb;
        if (code == kLiteRtOpCodeTflMinimum) r = (std::min)(va, vb);
        if (code == kLiteRtOpCodeTflMaximum) r = (std::max)(va, vb);
        out.f32[i] = ApplyActivation(r, act);
      }
    } else if (code == kLiteRtOpCodeTflNotEqual ||
               code == kLiteRtOpCodeTflEqual) {
      const EvalVal& a = in_val(0);
      const EvalVal& b = in_val(1);
      const bool want_eq = (code == kLiteRtOpCodeTflEqual);
      out.b.resize(n);
      for (size_t i = 0; i < n; ++i) {
        const size_t ia = BroadcastIndex(i, out.dims, a.dims);
        const size_t ib = BroadcastIndex(i, out.dims, b.dims);
        if (a.dtype == kLiteRtElementTypeFloat32) {
          const bool cond =
              want_eq ? (a.f32[ia] == b.f32[ib]) : (a.f32[ia] != b.f32[ib]);
          out.b[i] = cond ? 1 : 0;
        } else {
          const bool cond =
              want_eq ? (a.b[ia] == b.b[ib]) : (a.b[ia] != b.b[ib]);
          out.b[i] = cond ? 1 : 0;
        }
      }
    } else if (code == kLiteRtOpCodeTflSelect ||
               code == kLiteRtOpCodeTflSelectV2) {
      const EvalVal& c = in_val(0);
      const EvalVal& x = in_val(1);
      const EvalVal& y = in_val(2);
      const bool select_1d_batch = code == kLiteRtOpCodeTflSelect &&
                                   c.dims.size() == 1 && out.dims.size() > 1 &&
                                   c.dims[0] == out.dims[0];
      Dims eff_c_dims = c.dims;
      if (select_1d_batch) {
        eff_c_dims.assign(out.dims.size(), 1);
        eff_c_dims[0] = c.dims[0];
      }
      if (out.dtype == kLiteRtElementTypeFloat32) {
        out.f32.resize(n);
        for (size_t i = 0; i < n; ++i) {
          const uint8_t cv = c.b[BroadcastIndex(i, out.dims, eff_c_dims)];
          out.f32[i] = cv ? x.f32[BroadcastIndex(i, out.dims, x.dims)]
                          : y.f32[BroadcastIndex(i, out.dims, y.dims)];
        }
      } else {
        out.b.resize(n);
        for (size_t i = 0; i < n; ++i) {
          const uint8_t cv = c.b[BroadcastIndex(i, out.dims, eff_c_dims)];
          out.b[i] = cv ? x.b[BroadcastIndex(i, out.dims, x.dims)]
                        : y.b[BroadcastIndex(i, out.dims, y.dims)];
        }
      }
    } else if (code == kLiteRtOpCodeTflTranspose) {
      const EvalVal& a = in_val(0);
      const EvalVal& perm = in_val(1);
      const size_t rank = out.dims.size();
      auto assign_elem = [&](size_t dst, size_t src) {
        if (out.dtype == kLiteRtElementTypeFloat32) out.f32[dst] = a.f32[src];
        if (out.dtype == kLiteRtElementTypeBool) out.b[dst] = a.b[src];
      };
      if (out.dtype == kLiteRtElementTypeFloat32) out.f32.resize(n);
      if (out.dtype == kLiteRtElementTypeBool) out.b.resize(n);
      for (size_t i = 0; i < n; ++i) {
        std::vector<size_t> out_c(rank), in_c(rank);
        size_t rem = i;
        for (size_t d = rank; d > 0; --d) {
          out_c[d - 1] = rem % static_cast<size_t>(out.dims[d - 1]);
          rem /= static_cast<size_t>(out.dims[d - 1]);
        }
        for (size_t d = 0; d < rank; ++d) {
          in_c[static_cast<size_t>(perm.i64[d])] = out_c[d];
        }
        size_t src = 0;
        for (size_t d = 0; d < rank; ++d) {
          src = src * static_cast<size_t>(a.dims[d]) + in_c[d];
        }
        assign_elem(i, src);
      }
    } else if (code == kLiteRtOpCodeTflSlice) {
      const EvalVal& a = in_val(0);
      const EvalVal& beg = in_val(1);
      const size_t rank = out.dims.size();
      if (out.dtype == kLiteRtElementTypeFloat32) out.f32.resize(n);
      if (out.dtype == kLiteRtElementTypeBool) out.b.resize(n);
      for (size_t i = 0; i < n; ++i) {
        std::vector<size_t> c(rank);
        size_t rem = i;
        for (size_t d = rank; d > 0; --d) {
          c[d - 1] = rem % static_cast<size_t>(out.dims[d - 1]);
          rem /= static_cast<size_t>(out.dims[d - 1]);
        }
        size_t src = 0;
        for (size_t d = 0; d < rank; ++d) {
          src = src * static_cast<size_t>(a.dims[d]) + c[d] +
                static_cast<size_t>(beg.i64[d]);
        }
        if (out.dtype == kLiteRtElementTypeFloat32) out.f32[i] = a.f32[src];
        if (out.dtype == kLiteRtElementTypeBool) out.b[i] = a.b[src];
      }
    } else if (code == kLiteRtOpCodeTflTile) {
      const EvalVal& a = in_val(0);
      const size_t rank = out.dims.size();
      if (out.dtype == kLiteRtElementTypeFloat32) out.f32.resize(n);
      if (out.dtype == kLiteRtElementTypeBool) out.b.resize(n);
      if (out.dtype == kLiteRtElementTypeInt32) out.i64.resize(n);
      for (size_t i = 0; i < n; ++i) {
        std::vector<size_t> c(rank);
        size_t rem = i;
        for (size_t d = rank; d > 0; --d) {
          c[d - 1] = rem % static_cast<size_t>(out.dims[d - 1]);
          rem /= static_cast<size_t>(out.dims[d - 1]);
        }
        size_t src = 0;
        for (size_t d = 0; d < rank; ++d) {
          src = src * static_cast<size_t>(a.dims[d]) +
                (c[d] % static_cast<size_t>(a.dims[d]));
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
      const EvalVal& a = in_val(0);
      const EvalVal& pad = in_val(1);
      const size_t rank = out.dims.size();
      out.f32.assign(n, 0.0f);
      for (size_t i = 0; i < FlatSize(a.dims); ++i) {
        std::vector<size_t> c(rank);
        size_t rem = i;
        for (size_t d = rank; d > 0; --d) {
          c[d - 1] = rem % static_cast<size_t>(a.dims[d - 1]);
          rem /= static_cast<size_t>(a.dims[d - 1]);
        }
        size_t dst = 0;
        for (size_t d = 0; d < rank; ++d) {
          dst = dst * static_cast<size_t>(out.dims[d]) + c[d] +
                static_cast<size_t>(pad.i64[2 * d]);
        }
        out.f32[dst] = a.f32[i];
      }
    } else if (code == kLiteRtOpCodeTflDepthwiseConv2d) {
      const EvalVal& x = in_val(0);
      const EvalVal& w = in_val(1);
      const EvalVal& bias = in_val(2);
      auto opts = GetOptionsAs<DepthwiseConv2dOptions>(Ctx(), op);
      out.f32.resize(n);
      const int32_t b_dim = out.dims[0], oh = out.dims[1], ow = out.dims[2],
                    oc = out.dims[3];
      const int32_t kh = w.dims[1], kw = w.dims[2];
      const int32_t pad_h =
          (opts->padding == 0)
              ? (std::max)(0, ((oh - 1) * opts->stride_h + kh - x.dims[1]) / 2)
              : 0;
      const int32_t pad_w =
          (opts->padding == 0)
              ? (std::max)(0, ((ow - 1) * opts->stride_w + kw - x.dims[2]) / 2)
              : 0;
      for (int32_t bi = 0; bi < b_dim; ++bi) {
        for (int32_t hi = 0; hi < oh; ++hi) {
          for (int32_t wi = 0; wi < ow; ++wi) {
            for (int32_t ci = 0; ci < oc; ++ci) {
              float acc = bias.f32[ci];
              for (int32_t fh = 0; fh < kh; ++fh) {
                for (int32_t fw = 0; fw < kw; ++fw) {
                  const int32_t ih = hi * opts->stride_h + fh - pad_h;
                  const int32_t iw = wi * opts->stride_w + fw - pad_w;
                  if (ih < 0 || ih >= x.dims[1] || iw < 0 || iw >= x.dims[2]) {
                    continue;
                  }
                  const size_t x_idx =
                      (((bi * x.dims[1] + ih) * x.dims[2] + iw) * x.dims[3]) +
                      (ci / opts->depth_multiplier);
                  const size_t w_idx = ((fh * kw + fw) * oc) + ci;
                  acc += x.f32[x_idx] * w.f32[w_idx];
                }
              }
              const size_t o_idx = (((bi * oh + hi) * ow + wi) * oc) + ci;
              out.f32[o_idx] =
                  ApplyActivation(acc, opts->fused_activation_function);
            }
          }
        }
      }
    } else if (code == kLiteRtOpCodeTflQuantize) {
      const EvalVal& a = in_val(0);
      out.f32 = a.f32;
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
    auto it_b = before.find(out);
    auto it_a = after.find(out);
    ASSERT_NE(it_b, before.end());
    ASSERT_NE(it_a, after.end());
    EXPECT_EQ(it_b->second.dtype, it_a->second.dtype);
    EXPECT_EQ(it_b->second.dims, it_a->second.dims);
    if (it_b->second.dtype == kLiteRtElementTypeBool) {
      EXPECT_EQ(it_b->second.b, it_a->second.b);
    } else {
      ASSERT_EQ(it_b->second.f32.size(), it_a->second.f32.size());
      for (size_t i = 0; i < it_b->second.f32.size(); ++i) {
        const float vb = it_b->second.f32[i];
        const float va = it_a->second.f32[i];
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
  bool modified = true;
  while (modified && iterations < max_iters) {
    modified = false;
    ++iterations;
    std::queue<LiteRtOp> worklist;
    for (LiteRtOp op : subgraph->Ops()) {
      worklist.push(op);
    }
    while (!worklist.empty()) {
      LiteRtOp op = worklist.front();
      worklist.pop();
      for (const auto& entry : patterns) {
        LiteRtBuilderT builder;
        builder.SetTargetSubgraph(subgraph);
        if (entry.fn(ctx, &builder, op) == kLiteRtStatusOk) {
          builder.ApplyChanges(subgraph);
          modified = true;
          break;
        }
      }
      if (modified) {
        break;
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
        EXPECT_LT(it->second, i)
            << "Topological violation: op " << it->second << " feeds op " << i;
      }
    }
  }
}

int CountOps(const LiteRtSubgraphT& subgraph, LiteRtOpCode code) {
  int count = 0;
  for (const auto* op : subgraph.Ops()) {
    if (op->OpCode() == code) {
      ++count;
    }
  }
  return count;
}

std::string DumpOps(const LiteRtSubgraphT& subgraph) {
  std::ostringstream os;
  for (size_t i = 0; i < subgraph.Ops().size(); ++i) {
    const auto* op = subgraph.Ops()[i];
    os << "[" << i << "]c=" << op->OpCode() << " i=(";
    for (const auto* in : op->Inputs()) {
      os << in->Type().second.ranked_tensor_type.element_type << ",";
    }
    os << ") o=(";
    for (const auto* out : op->Outputs()) {
      os << out->Type().second.ranked_tensor_type.element_type << "u"
         << out->NumUses() << ",";
    }
    os << ")\n";
  }
  return os.str();
}

int CountBoolOps(const LiteRtSubgraphT& subgraph) {
  int count = 0;
  for (const auto* op : subgraph.Ops()) {
    const LiteRtOpCode code = op->OpCode();
    if (code == kLiteRtOpCodeTflCast || code == kLiteRtOpCodeTflNotEqual) {
      continue;
    }
    bool has_bool = false;
    for (const LiteRtTensor t : op->Inputs()) {
      if (t->Type().first == kLiteRtRankedTensorType &&
          t->Type().second.ranked_tensor_type.element_type ==
              kLiteRtElementTypeBool) {
        has_bool = true;
      }
    }
    for (const LiteRtTensor t : op->Outputs()) {
      if (t->Type().first == kLiteRtRankedTensorType &&
          t->Type().second.ranked_tensor_type.element_type ==
              kLiteRtElementTypeBool) {
        has_bool = true;
      }
    }
    if (has_bool) {
      ++count;
    }
  }
  return count;
}

void BuildAudioMaskSubgraph(LiteRtSubgraphT* subgraph) {
  GraphBuilder g;
  Tensor seg_mask = g.Bool({1, 9});
  Tensor prev_mask = g.Bool({1, 4});
  Tensor logits = g.F32({2, 2, 2, 4});
  Tensor proj_in = g.F32({1, 2, 8});
  g.MarkInput(seg_mask, subgraph);
  g.MarkInput(prev_mask, subgraph);
  g.MarkInput(logits, subgraph);
  g.MarkInput(proj_in, subgraph);

  Tensor dw_filter = g.Const<float>({1, 3, 1, 1}, {0.0f, 1.0f, 0.0f});
  Tensor dw_bias = g.Const<float>({1}, {0.0f});
  Tensor zero_scalar = g.Const<float>({}, {0.0f});
  Tensor neg_inf = g.Const<float>({}, {-58982.398438f});

  Tensor t_cast0 = g.F32({1, 9});
  g.AddOp(kLiteRtOpCodeTflCast, {seg_mask}, {t_cast0});
  Tensor shape_1_9_1_1 = g.Const<int32_t>({4}, {1, 9, 1, 1});
  Tensor t_r0 = g.F32({1, 9, 1, 1});
  Op r0 = g.AddOp(kLiteRtOpCodeTflReshape, {t_cast0, shape_1_9_1_1}, {t_r0});
  SetReshapeShape(g, r0, {1, 9, 1, 1});
  Tensor t_dw0 = g.F32({1, 4, 1, 1});
  Op dw0 = g.AddOp(kLiteRtOpCodeTflDepthwiseConv2d, {t_r0, dw_filter, dw_bias},
                   {t_dw0});
  SetDwConv2dOptions(g, dw0);
  Tensor shape_1_4 = g.Const<int32_t>({2}, {1, 4});
  Tensor t_r1 = g.F32({1, 4});
  Op r1 = g.AddOp(kLiteRtOpCodeTflReshape, {t_dw0, shape_1_4}, {t_r1});
  SetReshapeShape(g, r1, {1, 4});
  Tensor t_mid_bool = g.Bool({1, 4});
  g.AddOp(kLiteRtOpCodeTflNotEqual, {t_r1, zero_scalar}, {t_mid_bool});

  // Stage 2 subsampling consuming t_mid_bool via Cast -> DepthwiseConv2D ->
  // NotEqual -> out_down_mask[SG_OUT]
  Tensor t_cast_mid = g.F32({1, 4});
  g.AddOp(kLiteRtOpCodeTflCast, {t_mid_bool}, {t_cast_mid});
  Tensor shape_1_4_1_1 = g.Const<int32_t>({4}, {1, 4, 1, 1});
  Tensor t_r2 = g.F32({1, 4, 1, 1});
  Op r2 = g.AddOp(kLiteRtOpCodeTflReshape, {t_cast_mid, shape_1_4_1_1}, {t_r2});
  SetReshapeShape(g, r2, {1, 4, 1, 1});
  Tensor t_dw1 = g.F32({1, 2, 1, 1});
  Op dw1 = g.AddOp(kLiteRtOpCodeTflDepthwiseConv2d, {t_r2, dw_filter, dw_bias},
                   {t_dw1});
  SetDwConv2dOptions(g, dw1);
  Tensor shape_1_2 = g.Const<int32_t>({2}, {1, 2});
  Tensor t_r3 = g.F32({1, 2});
  Op r3 = g.AddOp(kLiteRtOpCodeTflReshape, {t_dw1, shape_1_2}, {t_r3});
  SetReshapeShape(g, r3, {1, 2});
  Tensor out_down_mask = g.Bool({1, 2});
  g.AddOp(kLiteRtOpCodeTflNotEqual, {t_r3, zero_scalar}, {out_down_mask});
  g.MarkOutput(out_down_mask, subgraph);

  // Branch A: SelectV2(cond, proj_in, 0.0) -> out_proj[SG_OUT]
  Tensor shape_1_2_1 = g.Const<int32_t>({3}, {1, 2, 1});
  Tensor cond_proj = g.Bool({1, 2, 1});
  Op r_proj = g.AddOp(kLiteRtOpCodeTflReshape, {out_down_mask, shape_1_2_1},
                      {cond_proj});
  SetReshapeShape(g, r_proj, {1, 2, 1});
  Tensor out_proj = g.F32({1, 2, 8});
  g.AddOp(kLiteRtOpCodeTflSelectV2, {cond_proj, proj_in, zero_scalar},
          {out_proj});
  g.MarkOutput(out_proj, subgraph);

  // Branch B: Slice(out_down_mask) -> out_slice[SG_OUT]
  Tensor out_slice = g.Bool({1, 1});
  g.AddOp(kLiteRtOpCodeTflSlice,
          {out_down_mask, g.Const<int32_t>({2}, {0, 1}),
           g.Const<int32_t>({2}, {1, 1})},
          {out_slice});
  g.MarkOutput(out_slice, subgraph);

  // Branch C: Concat(prev_mask, out_down_mask) -> Reshape -> Cast -> Pad ->
  // Cast
  Tensor t_cat_2d = g.Bool({1, 6});
  Op cat2d = g.AddOp(kLiteRtOpCodeTflConcatenation, {prev_mask, out_down_mask},
                     {t_cat_2d});
  SetConcatAxis(g, cat2d, 1);
  Tensor shape_1_1_6 = g.Const<int32_t>({3}, {1, 1, 6});
  Tensor t_cat_3d = g.Bool({1, 1, 6});
  Op r_cat =
      g.AddOp(kLiteRtOpCodeTflReshape, {t_cat_2d, shape_1_1_6}, {t_cat_3d});
  SetReshapeShape(g, r_cat, {1, 1, 6});
  Tensor t_pad_in = g.F32({1, 1, 6});
  g.AddOp(kLiteRtOpCodeTflCast, {t_cat_3d}, {t_pad_in});
  Tensor t_pad_out = g.F32({1, 1, 8});
  g.AddOp(kLiteRtOpCodeTflPad,
          {t_pad_in, g.Const<int32_t>({3, 2}, {0, 0, 0, 0, 0, 2})},
          {t_pad_out});
  Tensor t_pad_bool = g.Bool({1, 1, 8});
  g.AddOp(kLiteRtOpCodeTflCast, {t_pad_out}, {t_pad_bool});

  // Chunking & 4D axis=3 Bool Concat -> I32 Tile wrap -> LogicalAnd ->
  // LogicalNot -> SelectV2
  Tensor t_4_1_2 = g.Bool({4, 1, 2});
  Op r_4 = g.AddOp(kLiteRtOpCodeTflReshape,
                   {t_pad_bool, g.Const<int32_t>({3}, {4, 1, 2})}, {t_4_1_2});
  SetReshapeShape(g, r_4, {4, 1, 2});
  Tensor s0 = g.Bool({2, 1, 2});
  Tensor s1 = g.Bool({2, 1, 2});
  g.AddOp(kLiteRtOpCodeTflSlice,
          {t_4_1_2, g.Const<int32_t>({3}, {0, 0, 0}),
           g.Const<int32_t>({3}, {2, 1, 2})},
          {s0});
  g.AddOp(kLiteRtOpCodeTflSlice,
          {t_4_1_2, g.Const<int32_t>({3}, {1, 0, 0}),
           g.Const<int32_t>({3}, {2, 1, 2})},
          {s1});
  Tensor shape_1_2_1_2 = g.Const<int32_t>({4}, {1, 2, 1, 2});
  Tensor c0 = g.Bool({1, 2, 1, 2});
  Tensor c1 = g.Bool({1, 2, 1, 2});
  SetReshapeShape(g,
                  g.AddOp(kLiteRtOpCodeTflReshape, {s0, shape_1_2_1_2}, {c0}),
                  {1, 2, 1, 2});
  SetReshapeShape(g,
                  g.AddOp(kLiteRtOpCodeTflReshape, {s1, shape_1_2_1_2}, {c1}),
                  {1, 2, 1, 2});
  Tensor cat4d_out = g.Bool({1, 2, 1, 4});
  Op cat4d = g.AddOp(kLiteRtOpCodeTflConcatenation, {c0, c1}, {cat4d_out});
  SetConcatAxis(g, cat4d, 3);

  Tensor t_i32_in = g.I32({1, 2, 1, 4});
  g.AddOp(kLiteRtOpCodeTflCast, {cat4d_out}, {t_i32_in});
  Tensor t_i32_tiled = g.I32({1, 2, 2, 4});
  g.AddOp(kLiteRtOpCodeTflTile, {t_i32_in, g.Const<int64_t>({4}, {1, 1, 2, 1})},
          {t_i32_tiled});
  Tensor t_tiled_bool = g.Bool({1, 2, 2, 4});
  g.AddOp(kLiteRtOpCodeTflCast, {t_i32_tiled}, {t_tiled_bool});

  std::vector<uint8_t> window(16, 1);
  window[0] = 0;
  Tensor t_window = g.BoolConst({1, 2, 2, 4}, window);
  Tensor t_and = g.Bool({1, 2, 2, 4});
  g.AddOp(kLiteRtOpCodeTflLogicalAnd, {t_tiled_bool, t_window}, {t_and});
  Tensor t_not = g.Bool({1, 2, 2, 4});
  g.AddOp(kLiteRtOpCodeTflLogicalNot, {t_and}, {t_not});
  Tensor t_masked = g.F32({2, 2, 2, 4});
  g.AddOp(kLiteRtOpCodeTflSelectV2, {t_not, neg_inf, logits}, {t_masked});
  Tensor out_attn = g.Quant(ElementType::Int16, {2, 2, 2, 4}, 1.8f, 0);
  g.AddOp(kLiteRtOpCodeTflQuantize, {t_masked}, {out_attn});
  g.MarkOutput(out_attn, subgraph);

  // Branch D: Slice(prev_mask) -> Concat(_, out_down_mask) -> out_prev[SG_OUT]
  Tensor prev_tail = g.Bool({1, 2});
  g.AddOp(
      kLiteRtOpCodeTflSlice,
      {prev_mask, g.Const<int32_t>({2}, {0, 2}), g.Const<int32_t>({2}, {1, 2})},
      {prev_tail});
  Tensor out_prev = g.Bool({1, 4});
  Op cat_prev = g.AddOp(kLiteRtOpCodeTflConcatenation,
                        {prev_tail, out_down_mask}, {out_prev});
  SetConcatAxis(g, cat_prev, 1);
  g.MarkOutput(out_prev, subgraph);

  g.ApplyTo(subgraph);
}

TEST(BoolMaskTransformationTest, EndToEndAudioMaskWithOutputNotEqual) {
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

  RunPatterns(&subgraph, AllPatterns(/*include_output_not_equal=*/true));
  ExpectTopologicalOrder(subgraph);
  ExpectSameOutputs(subgraph, expected, EvalSubgraph(subgraph, feed));

  EXPECT_EQ(CountBoolOps(subgraph), 0);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflLogicalAnd), 0);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflLogicalNot), 0);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflSelectV2), 0);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflNotEqual), 3);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflCast), 3) << DumpOps(subgraph);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflTile), 1) << DumpOps(subgraph);

  for (auto* op : subgraph.Ops()) {
    if (op->OpCode() == kLiteRtOpCodeTflTile) {
      EXPECT_EQ(op->Inputs()[0]->Type().second.ranked_tensor_type.element_type,
                kLiteRtElementTypeFloat32);
      EXPECT_EQ(op->Inputs()[1]->Type().second.ranked_tensor_type.element_type,
                kLiteRtElementTypeInt32);
    } else if (op->OpCode() == kLiteRtOpCodeTflCast) {
      EXPECT_EQ(op->Inputs()[0]->Type().second.ranked_tensor_type.element_type,
                kLiteRtElementTypeBool);
      EXPECT_TRUE(op->Inputs()[0]->DefiningOp() == nullptr);
    } else if (op->OpCode() == kLiteRtOpCodeTflNotEqual) {
      const auto& d0 =
          op->Inputs()[0]->Type().second.ranked_tensor_type.layout.dimensions;
      const auto& d1 =
          op->Inputs()[1]->Type().second.ranked_tensor_type.layout.dimensions;
      EXPECT_EQ(op->Inputs()[0]->Type().second.ranked_tensor_type.layout.rank,
                op->Inputs()[1]->Type().second.ranked_tensor_type.layout.rank);
      for (size_t i = 0;
           i < op->Inputs()[0]->Type().second.ranked_tensor_type.layout.rank;
           ++i) {
        EXPECT_EQ(d0[i], d1[i]);
      }
    } else if (op->OpCode() == kLiteRtOpCodeTflConcatenation) {
      auto opts = GetOptionsAs<ConcatenationOptions>(Ctx(), op);
      ASSERT_TRUE(opts.HasValue());
    }
  }
}

TEST(BoolMaskTransformationTest, EndToEndAudioMaskWithoutOutputNotEqual) {
  LiteRtModelT model;
  LiteRtSubgraphT& subgraph = model.EmplaceSubgraph();
  BuildAudioMaskSubgraph(&subgraph);

  RunPatterns(&subgraph, AllPatterns(/*include_output_not_equal=*/false));
  ExpectTopologicalOrder(subgraph);

  EXPECT_EQ(CountBoolOps(subgraph), 0);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflNotEqual), 0);
  // 3 subgraph-input Cast(BOOL->F32) + 3 subgraph-output Cast(F32->BOOL).
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflCast), 6);
}

TEST(BoolMaskTransformationTest,
     ArbitraryFloatsStayExactWithoutScalingMiscompilation) {
  LiteRtModelT model;
  LiteRtSubgraphT& subgraph = model.EmplaceSubgraph();
  GraphBuilder g;
  const float inf = std::numeric_limits<float>::infinity();
  const float nan_val = std::numeric_limits<float>::quiet_NaN();

  // Values that previously miscompiled under 2^30 scaled clamping:
  // 1e-12f (< 2^-30), +/-Inf, NaN, +/-2e38f, -0.0f.
  Tensor x = g.F32({1, 8});
  Tensor y = g.F32({1, 8});
  Tensor zero = g.Const<float>({}, {0.0f});
  g.MarkInput(x, &subgraph);
  g.MarkInput(y, &subgraph);

  // 1. NotEqual(x, 0.0f) -> LogicalNot -> Cast(->f32): for x = 1e-12f,
  //    NotEqual is true and LogicalNot MUST be false (0.0f), never inverted!
  Tensor ne0 = g.Bool({1, 8});
  Tensor inv0 = g.Bool({1, 8});
  Tensor out_inv0 = g.F32({1, 8});
  g.AddOp(kLiteRtOpCodeTflNotEqual, {x, zero}, {ne0});
  g.AddOp(kLiteRtOpCodeTflLogicalNot, {ne0}, {inv0});
  g.AddOp(kLiteRtOpCodeTflCast, {inv0}, {out_inv0});

  // 2. NotEqual(x, y) -> LogicalNot: handles (+Inf, +Inf) -> true,
  //    (2e38f, -2e38f) -> false, (NaN, NaN) -> false.
  Tensor ne_xy = g.Bool({1, 8});
  Tensor eq_xy = g.Bool({1, 8});
  g.AddOp(kLiteRtOpCodeTflNotEqual, {x, y}, {ne_xy});
  g.AddOp(kLiteRtOpCodeTflLogicalNot, {ne_xy}, {eq_xy});

  // 3. Double Cast on arbitrary float x (including -0.0f and 1e-12f) into Add:
  //    must NOT bypass to raw x or scaled clamp.
  Tensor b_cast = g.Bool({1, 8});
  Tensor f_cast = g.F32({1, 8});
  Tensor out_add = g.F32({1, 8});
  g.AddOp(kLiteRtOpCodeTflCast, {x}, {b_cast});
  g.AddOp(kLiteRtOpCodeTflCast, {b_cast}, {f_cast});
  AddOptions add_opts;
  add_opts.fused_activation_function = 0;
  g.SetOptions(g.AddOp(kLiteRtOpCodeTflAdd, {f_cast, zero}, {out_add}),
               std::move(add_opts));

  g.MarkOutput(out_inv0, &subgraph);
  g.MarkOutput(eq_xy, &subgraph);
  g.MarkOutput(out_add, &subgraph);
  g.ApplyTo(&subgraph);

  absl::flat_hash_map<const LiteRtTensorT*, EvalVal> feed;
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
  const auto expected = EvalSubgraph(subgraph, feed);

  RunPatterns(&subgraph, AllPatterns(/*include_output_not_equal=*/true));
  ExpectTopologicalOrder(subgraph);
  const auto actual = EvalSubgraph(subgraph, feed);
  ExpectSameOutputs(subgraph, expected, actual);

  // Verify exact parity on 1e-12f, +Inf, NaN, and -0.0f sign bit.
  EXPECT_FLOAT_EQ(actual.at(out_inv0.Get()).f32[0], 0.0f);
  EXPECT_FLOAT_EQ(actual.at(out_inv0.Get()).f32[2], 1.0f);
  EXPECT_EQ(actual.at(eq_xy.Get()).b[4], 1);                    // +Inf == +Inf
  EXPECT_EQ(actual.at(eq_xy.Get()).b[6], 0);                    // NaN != NaN
  EXPECT_EQ(actual.at(eq_xy.Get()).b[7], 0);                    // 2e38 != -2e38
  EXPECT_FALSE(std::signbit(actual.at(out_add.Get()).f32[3]));  // +0.0f
  EXPECT_FLOAT_EQ(actual.at(out_add.Get()).f32[0], 1.0f);       // 1e-12f -> 1
}

TEST(BoolMaskTransformationTest, DeadCastRespectsSubgraphOutputs) {
  LiteRtModelT model;
  LiteRtSubgraphT& subgraph = model.EmplaceSubgraph();
  GraphBuilder g;
  Tensor x = g.F32({1, 4});
  Tensor dead_b = g.Bool({1, 4});
  Tensor live_b = g.Bool({1, 4});
  g.MarkInput(x, &subgraph);
  g.MarkOutput(live_b, &subgraph);
  g.AddOp(kLiteRtOpCodeTflCast, {x}, {dead_b});
  g.AddOp(kLiteRtOpCodeTflCast, {x}, {live_b});
  g.ApplyTo(&subgraph);

  RunPatterns(&subgraph, AllPatterns(/*include_output_not_equal=*/false));
  ExpectTopologicalOrder(subgraph);

  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflCast), 1);
  EXPECT_EQ(subgraph.Ops()[0]->Outputs()[0], live_b.Get());

  RunPatterns(&subgraph, AllPatterns(/*include_output_not_equal=*/true));
  ExpectTopologicalOrder(subgraph);

  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflCast), 0);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflNotEqual), 1);
  EXPECT_EQ(subgraph.Ops()[0]->Outputs()[0], live_b.Get());
}

TEST(BoolMaskTransformationTest,
     SelectZeroBranchAndLogicalOrPreserveSemantics) {
  LiteRtModelT model;
  LiteRtSubgraphT& subgraph = model.EmplaceSubgraph();
  GraphBuilder g;
  Tensor a = g.Bool({1, 3});
  Tensor b = g.Bool({1, 3});
  Tensor t_val = g.F32({1, 3, 2});
  Tensor f_val = g.F32({1, 3, 2});
  Tensor zero = g.Const<float>({}, {0.0f});
  g.MarkInput(a, &subgraph);
  g.MarkInput(b, &subgraph);
  g.MarkInput(t_val, &subgraph);
  g.MarkInput(f_val, &subgraph);

  Tensor cond_or = g.Bool({1, 3});
  g.AddOp(kLiteRtOpCodeTflLogicalOr, {a, b}, {cond_or});
  Tensor cond_3d = g.Bool({1, 3, 1});
  SetReshapeShape(
      g,
      g.AddOp(kLiteRtOpCodeTflReshape,
              {cond_or, g.Const<int32_t>({3}, {1, 3, 1})}, {cond_3d}),
      {1, 3, 1});
  Tensor out_fzero = g.F32({1, 3, 2});
  g.AddOp(kLiteRtOpCodeTflSelectV2, {cond_3d, t_val, zero}, {out_fzero});
  Tensor out_tzero = g.F32({1, 3, 2});
  g.AddOp(kLiteRtOpCodeTflSelectV2, {cond_3d, zero, f_val}, {out_tzero});
  g.MarkOutput(out_fzero, &subgraph);
  g.MarkOutput(out_tzero, &subgraph);
  g.ApplyTo(&subgraph);

  RunPatterns(&subgraph, AllPatterns(/*include_output_not_equal=*/true));
  ExpectTopologicalOrder(subgraph);

  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflLogicalOr), 0);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflSelectV2), 0);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflMaximum), 1);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflAdd), 0);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflMul), 2);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflSub), 1);
}

TEST(BoolMaskTransformationTest,
     SelectVsSelectV2BroadcastingRulesAndBatchReshape) {
  LiteRtModelT model;
  LiteRtSubgraphT& subgraph = model.EmplaceSubgraph();
  GraphBuilder g;

  Tensor cond_batch = g.Bool({2});     // Leading batch [B] where B=2
  Tensor cond_trailing = g.Bool({4});  // Trailing [W] where W=4 != B=2
  Tensor cond_rank2 = g.Bool({2, 1, 1});
  Tensor cond_scalar = g.Bool({});
  Tensor x_234 = g.F32({2, 3, 4});
  Tensor y_234 = g.F32({2, 3, 4});
  Tensor x_232 = g.F32({2, 3, 2});  // B == W == 2 ambiguity trap!
  Tensor y_232 = g.F32({2, 3, 2});
  Tensor y_134 = g.F32({1, 3, 4});
  g.MarkInput(cond_batch, &subgraph);
  g.MarkInput(cond_trailing, &subgraph);
  g.MarkInput(cond_rank2, &subgraph);
  g.MarkInput(cond_scalar, &subgraph);
  g.MarkInput(x_234, &subgraph);
  g.MarkInput(y_234, &subgraph);
  g.MarkInput(x_232, &subgraph);
  g.MarkInput(y_232, &subgraph);
  g.MarkInput(y_134, &subgraph);

  // 1. Valid Select with 1D [B] batch condition when B != W ([2] on [2,3,4]).
  Tensor out_sel_234 = g.F32({2, 3, 4});
  g.AddOp(kLiteRtOpCodeTflSelect, {cond_batch, x_234, y_234}, {out_sel_234});

  // 2. Valid Select with 1D [B] batch condition when B == W ([2] on [2,3,2]):
  //    MUST reshape cond to [2,1,1] so Mul does NOT broadcast along trailing W!
  Tensor out_sel_232 = g.F32({2, 3, 2});
  g.AddOp(kLiteRtOpCodeTflSelect, {cond_batch, x_232, y_232}, {out_sel_232});

  // 3. Valid Select with rank-0 scalar condition.
  Tensor out_sel_scalar = g.F32({2, 3, 4});
  g.AddOp(kLiteRtOpCodeTflSelect, {cond_scalar, x_234, y_234},
          {out_sel_scalar});

  // 4. Invalid for Select (kept untouched): trailing [4] on [2,3,4], rank
  //    mismatch [2,1,1] on [2,3,4], and mismatched x/y shapes [2,3,4] vs
  //    [1,3,4].
  Tensor out_bad_trail = g.F32({2, 3, 4});
  Op bad_trail = g.AddOp(kLiteRtOpCodeTflSelect, {cond_trailing, x_234, y_234},
                         {out_bad_trail});
  Tensor out_bad_rank = g.F32({2, 3, 4});
  Op bad_rank = g.AddOp(kLiteRtOpCodeTflSelect, {cond_rank2, x_234, y_234},
                        {out_bad_rank});
  Tensor out_bad_xy = g.F32({2, 3, 4});
  Op bad_xy =
      g.AddOp(kLiteRtOpCodeTflSelect, {cond_batch, x_234, y_134}, {out_bad_xy});

  // 5. Valid for SelectV2: trailing [4] and broadcasted y [1,3,4].
  Tensor out_v2 = g.F32({2, 3, 4});
  g.AddOp(kLiteRtOpCodeTflSelectV2, {cond_trailing, x_234, y_134}, {out_v2});

  g.MarkOutput(out_sel_234, &subgraph);
  g.MarkOutput(out_sel_232, &subgraph);
  g.MarkOutput(out_sel_scalar, &subgraph);
  g.MarkOutput(out_v2, &subgraph);
  g.ApplyTo(&subgraph);

  // Check rejected Select variants directly before DCE removes unused ops.
  {
    LiteRtBuilderT b1, b2, b3;
    b1.SetTargetSubgraph(&subgraph);
    b2.SetTargetSubgraph(&subgraph);
    b3.SetTargetSubgraph(&subgraph);
    EXPECT_EQ(BoolMaskOpToFloatTransformation(Ctx(), &b1, bad_trail.Get()),
              kLiteRtStatusPatternNoMatch);
    EXPECT_EQ(BoolMaskOpToFloatTransformation(Ctx(), &b2, bad_rank.Get()),
              kLiteRtStatusPatternNoMatch);
    EXPECT_EQ(BoolMaskOpToFloatTransformation(Ctx(), &b3, bad_xy.Get()),
              kLiteRtStatusPatternNoMatch);
  }
  g.MarkOutput(out_bad_trail, &subgraph);
  g.MarkOutput(out_bad_rank, &subgraph);
  g.MarkOutput(out_bad_xy, &subgraph);

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

  // Evaluate only the valid outputs before & after.
  LiteRtModelT ref_model;
  LiteRtSubgraphT& ref_sg = ref_model.EmplaceSubgraph();
  {
    GraphBuilder rg;
    Tensor cb = rg.Bool({2});
    Tensor ct = rg.Bool({4});
    Tensor cs = rg.Bool({});
    Tensor rx234 = rg.F32({2, 3, 4});
    Tensor ry234 = rg.F32({2, 3, 4});
    Tensor rx232 = rg.F32({2, 3, 2});
    Tensor ry232 = rg.F32({2, 3, 2});
    Tensor ry134 = rg.F32({1, 3, 4});
    rg.MarkInput(cb, &ref_sg);
    rg.MarkInput(ct, &ref_sg);
    rg.MarkInput(cs, &ref_sg);
    rg.MarkInput(rx234, &ref_sg);
    rg.MarkInput(ry234, &ref_sg);
    rg.MarkInput(rx232, &ref_sg);
    rg.MarkInput(ry232, &ref_sg);
    rg.MarkInput(ry134, &ref_sg);
    Tensor o1 = rg.F32({2, 3, 4});
    Tensor o2 = rg.F32({2, 3, 2});
    Tensor o3 = rg.F32({2, 3, 4});
    Tensor o4 = rg.F32({2, 3, 4});
    rg.AddOp(kLiteRtOpCodeTflSelect, {cb, rx234, ry234}, {o1});
    rg.AddOp(kLiteRtOpCodeTflSelect, {cb, rx232, ry232}, {o2});
    rg.AddOp(kLiteRtOpCodeTflSelect, {cs, rx234, ry234}, {o3});
    rg.AddOp(kLiteRtOpCodeTflSelectV2, {ct, rx234, ry134}, {o4});
    rg.MarkOutput(o1, &ref_sg);
    rg.MarkOutput(o2, &ref_sg);
    rg.MarkOutput(o3, &ref_sg);
    rg.MarkOutput(o4, &ref_sg);
    rg.ApplyTo(&ref_sg);
  }
  absl::flat_hash_map<const LiteRtTensorT*, EvalVal> ref_feed;
  ref_feed[ref_sg.Inputs()[0]] = feed[cond_batch.Get()];
  ref_feed[ref_sg.Inputs()[1]] = feed[cond_trailing.Get()];
  ref_feed[ref_sg.Inputs()[2]] = feed[cond_scalar.Get()];
  ref_feed[ref_sg.Inputs()[3]] = feed[x_234.Get()];
  ref_feed[ref_sg.Inputs()[4]] = feed[y_234.Get()];
  ref_feed[ref_sg.Inputs()[5]] = feed[x_232.Get()];
  ref_feed[ref_sg.Inputs()[6]] = feed[y_232.Get()];
  ref_feed[ref_sg.Inputs()[7]] = feed[y_134.Get()];
  const auto ref_before = EvalSubgraph(ref_sg, ref_feed);
  RunPatterns(&ref_sg, AllPatterns(/*include_output_not_equal=*/true));
  ExpectTopologicalOrder(ref_sg);
  EXPECT_EQ(CountOps(ref_sg, kLiteRtOpCodeTflSelect), 0);
  EXPECT_EQ(CountOps(ref_sg, kLiteRtOpCodeTflSelectV2), 0);
  ExpectSameOutputs(ref_sg, ref_before, EvalSubgraph(ref_sg, ref_feed));

  // And on subgraph, the 3 invalid Select ops remain unlowered.
  RunPatterns(&subgraph, AllPatterns(/*include_output_not_equal=*/true));
  ExpectTopologicalOrder(subgraph);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflSelect), 3);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflSelectV2), 0);
}

TEST(BoolMaskTransformationTest,
     NonFiniteAndNegativeZeroConstantsPreserveSemantics) {
  LiteRtModelT model;
  LiteRtSubgraphT& subgraph = model.EmplaceSubgraph();
  GraphBuilder g;
  const float nan_val = std::numeric_limits<float>::quiet_NaN();
  const float neg_inf = -std::numeric_limits<float>::infinity();
  Tensor cond = g.Bool({1, 4});
  Tensor x = g.F32({1, 4});
  Tensor nan_const = g.Const<float>({}, {nan_val});
  Tensor inf_const = g.Const<float>({}, {neg_inf});
  Tensor neg_zero_const = g.Const<float>({}, {-0.0f});
  Tensor tiny_const = g.Const<float>({}, {5e-7f});
  g.MarkInput(cond, &subgraph);
  g.MarkInput(x, &subgraph);

  // SelectV2 with NaN or -Inf constant branches MUST NOT lower to Mul+Add,
  // because `0.0f * NaN` and `0.0f * (-Inf)` produce NaN on the other branch!
  Tensor out_nan_sel = g.F32({1, 4});
  Tensor out_inf_sel = g.F32({1, 4});
  g.AddOp(kLiteRtOpCodeTflSelectV2, {cond, nan_const, x}, {out_nan_sel});
  g.AddOp(kLiteRtOpCodeTflSelectV2, {cond, inf_const, x}, {out_inf_sel});

  // -0.0f and 5e-7f are finite but not canonical +0.0f; both lower via general
  // Sub/Mul/Mul/Add rather than the +0.0f Mul shortcut.
  Tensor out_neg_zero_sel = g.F32({1, 4});
  Tensor out_tiny_sel = g.F32({1, 4});
  g.AddOp(kLiteRtOpCodeTflSelectV2, {cond, x, neg_zero_const},
          {out_neg_zero_sel});
  g.AddOp(kLiteRtOpCodeTflSelectV2, {cond, tiny_const, x}, {out_tiny_sel});

  g.MarkOutput(out_nan_sel, &subgraph);
  g.MarkOutput(out_inf_sel, &subgraph);
  g.MarkOutput(out_neg_zero_sel, &subgraph);
  g.MarkOutput(out_tiny_sel, &subgraph);
  g.ApplyTo(&subgraph);

  absl::flat_hash_map<const LiteRtTensorT*, EvalVal> feed;
  feed[cond.Get()] = {kLiteRtElementTypeBool, {1, 4}, {}, {}, {0, 1, 0, 1}};
  feed[x.Get()] = {
      kLiteRtElementTypeFloat32, {1, 4}, {2.5f, -3.0f, 0.0f, 7.0f}, {}, {}};
  const auto expected = EvalSubgraph(subgraph, feed);

  RunPatterns(&subgraph, AllPatterns(/*include_output_not_equal=*/true));
  ExpectTopologicalOrder(subgraph);
  const auto actual = EvalSubgraph(subgraph, feed);
  ExpectSameOutputs(subgraph, expected, actual);

  // The two non-finite SelectV2 ops remain unlowered; the two finite ones lower
  // via general Sub/Mul/Mul/Add (2 Add ops).
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflSelectV2), 2);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflAdd), 2);
  // Finite false-branch values when cond == false are preserved (not NaN!).
  EXPECT_FLOAT_EQ(actual.at(out_nan_sel.Get()).f32[0], 2.5f);
  EXPECT_FLOAT_EQ(actual.at(out_inf_sel.Get()).f32[0], 2.5f);
  EXPECT_TRUE(std::isnan(actual.at(out_nan_sel.Get()).f32[1]));
  EXPECT_EQ(actual.at(out_inf_sel.Get()).f32[1], neg_inf);
}

TEST(BoolMaskTransformationTest,
     SubgraphOutputWithInternalConsumersSimultaneously) {
  LiteRtModelT model;
  LiteRtSubgraphT& subgraph = model.EmplaceSubgraph();
  GraphBuilder g;

  Tensor b_in = g.Bool({1, 2});
  Tensor x_f = g.F32({1, 4});
  Tensor idx_i32 = g.I32({1, 2});
  g.MarkInput(b_in, &subgraph);
  g.MarkInput(x_f, &subgraph);
  g.MarkInput(idx_i32, &subgraph);

  // 1. BoolMaskOpToFloat + OutputNotEqual: `not_b` is BOTH a subgraph output
  //    AND consumed internally by TileWrap (`Cast->Tile->Cast -> tiled_b`),
  //    where `tiled_b` is ALSO BOTH a subgraph output AND consumed by SelectV2
  //    AND an unrewritable integer Select!
  Tensor not_b = g.Bool({1, 2});
  g.AddOp(kLiteRtOpCodeTflLogicalNot, {b_in}, {not_b});
  g.MarkOutput(not_b, &subgraph);

  Tensor tile_i32_in = g.I32({1, 2});
  Tensor tile_i32_out = g.I32({1, 4});
  Tensor tiled_b = g.Bool({1, 4});
  g.AddOp(kLiteRtOpCodeTflCast, {not_b}, {tile_i32_in});
  g.AddOp(kLiteRtOpCodeTflTile, {tile_i32_in, g.Const<int64_t>({2}, {1, 2})},
          {tile_i32_out});
  g.AddOp(kLiteRtOpCodeTflCast, {tile_i32_out}, {tiled_b});
  g.MarkOutput(tiled_b, &subgraph);

  // Rewritable consumer of tiled_b:
  Tensor sel_f = g.F32({1, 4});
  g.AddOp(kLiteRtOpCodeTflSelectV2, {tiled_b, x_f, g.Const<float>({}, {0.0f})},
          {sel_f});
  g.MarkOutput(sel_f, &subgraph);

  // Unrewritable consumer of tiled_b (Int32 Select): forces `tiled_b` to keep
  // a live Cast(f32 -> tiled_b) feeding BOTH Int32 Select AND subgraph output!
  Tensor tiled_idx = g.I32({1, 4});
  g.AddOp(kLiteRtOpCodeTflTile, {idx_i32, g.Const<int32_t>({2}, {1, 2})},
          {tiled_idx});
  Tensor sel_i32 = g.I32({1, 4});
  g.AddOp(kLiteRtOpCodeTflSelectV2,
          {tiled_b, tiled_idx, g.Const<int32_t>({1, 4}, {0, 0, 0, 0})},
          {sel_i32});
  g.MarkOutput(sel_i32, &subgraph);

  // 2. BypassCast subgraph-output Cast(Bool -> F32) that ALSO has an internal
  //    Mul consumer: out_f32 stays Float32, lowered to Add, and Mul still
  //    works.
  Tensor cast_b_f = g.F32({1, 2});
  g.AddOp(kLiteRtOpCodeTflCast, {not_b}, {cast_b_f});
  g.MarkOutput(cast_b_f, &subgraph);
  Tensor mul_out = g.F32({1, 2});
  MulOptions mul_opts;
  mul_opts.fused_activation_function = 0;
  g.SetOptions(
      g.AddOp(kLiteRtOpCodeTflMul,
              {cast_b_f, g.Const<float>({1, 2}, {3.0f, 5.0f})}, {mul_out}),
      std::move(mul_opts));
  g.MarkOutput(mul_out, &subgraph);

  g.ApplyTo(&subgraph);

  RunPatterns(&subgraph, AllPatterns(/*include_output_not_equal=*/true));
  ExpectTopologicalOrder(subgraph);

  // Every subgraph output tensor retains its original pointer, element type,
  // and a valid defining op!
  EXPECT_EQ(not_b.ElementType(), ElementType::Bool);
  ASSERT_NE(not_b.Get()->DefiningOp(), nullptr);
  // All internal consumers of `not_b` bypassed to Float32, so `not_b` has 0
  // internal uses and was upgraded to NEURON_NOT_EQUAL!
  EXPECT_EQ(not_b.Get()->NumUses(), 0);
  EXPECT_EQ(not_b.Get()->DefiningOp()->OpCode(), kLiteRtOpCodeTflNotEqual);

  // `tiled_b` still has 1 unrewritable internal consumer (Int32 SelectV2) AND
  // is a subgraph output, so its type is still Bool and its defining Cast is
  // preserved (neither erased by CleanupCast nor turned into NotEqual).
  EXPECT_EQ(tiled_b.ElementType(), ElementType::Bool);
  ASSERT_NE(tiled_b.Get()->DefiningOp(), nullptr);
  EXPECT_EQ(tiled_b.Get()->DefiningOp()->OpCode(), kLiteRtOpCodeTflCast);
  EXPECT_EQ(tiled_b.Get()->NumUses(), 1);

  // `cast_b_f` (Float32 subgraph output + internal Mul consumer) is defined by
  // Add (not Bool->Float32 Cast).
  EXPECT_EQ(cast_b_f.ElementType(), ElementType::Float32);
  ASSERT_NE(cast_b_f.Get()->DefiningOp(), nullptr);
  EXPECT_EQ(cast_b_f.Get()->DefiningOp()->OpCode(), kLiteRtOpCodeTflAdd);
}

TEST(BoolMaskTransformationTest,
     ZeroSizedDynamicRankZeroAndStridedSliceGuards) {
  LiteRtModelT model;
  LiteRtSubgraphT& subgraph = model.EmplaceSubgraph();
  GraphBuilder g;

  // 1. Zero-sized ({0, 4}) and dynamic ({-1, 4}) ops are rejected cleanly.
  Tensor zero_b = g.Bool({0, 4});
  Tensor zero_b_out = g.Bool({0, 4});
  Tensor dyn_b = g.Bool({-1, 4});
  Tensor dyn_b_out = g.Bool({-1, 4});
  g.MarkInput(zero_b, &subgraph);
  g.MarkInput(dyn_b, &subgraph);
  g.AddOp(kLiteRtOpCodeTflLogicalNot, {zero_b}, {zero_b_out});
  g.AddOp(kLiteRtOpCodeTflLogicalNot, {dyn_b}, {dyn_b_out});
  g.MarkOutput(zero_b_out, &subgraph);
  g.MarkOutput(dyn_b_out, &subgraph);

  // 2. TileWrap with zero repetition {1, 0} or dynamic reps is rejected.
  Tensor b_in = g.Bool({1, 2});
  g.MarkInput(b_in, &subgraph);
  Tensor ti0 = g.I32({1, 2});
  Tensor to0 = g.I32({1, 0});
  Tensor tb0 = g.Bool({1, 0});
  g.AddOp(kLiteRtOpCodeTflCast, {b_in}, {ti0});
  g.AddOp(kLiteRtOpCodeTflTile, {ti0, g.Const<int32_t>({2}, {1, 0})}, {to0});
  g.AddOp(kLiteRtOpCodeTflCast, {to0}, {tb0});
  g.MarkOutput(tb0, &subgraph);

  // 3. StridedSlice with new_axis_mask != 0 or stride == 0 is rejected.
  Tensor ss_new_axis = g.Bool({1, 1, 2});
  Op ss1 =
      g.AddOp(kLiteRtOpCodeTflStridedSlice,
              {b_in, g.Const<int32_t>({2}, {0, 0}),
               g.Const<int32_t>({2}, {1, 2}), g.Const<int32_t>({2}, {1, 1})},
              {ss_new_axis});
  StridedSliceOptions ss1_opts{};
  ss1_opts.new_axis_mask = 2;
  g.SetOptions(ss1, ss1_opts);
  g.MarkOutput(ss_new_axis, &subgraph);

  Tensor ss_zero_stride = g.Bool({1, 2});
  Op ss2 =
      g.AddOp(kLiteRtOpCodeTflStridedSlice,
              {b_in, g.Const<int32_t>({2}, {0, 0}),
               g.Const<int32_t>({2}, {1, 2}), g.Const<int32_t>({2}, {1, 0})},
              {ss_zero_stride});
  StridedSliceOptions ss2_opts{};
  g.SetOptions(ss2, ss2_opts);
  g.MarkOutput(ss_zero_stride, &subgraph);

  // 4. Rank-0 scalar subgraph outputs: OutputNotEqual and BypassCast output
  //    fallback keep rank-0 boundary Casts intact instead of emitting rank-0
  //    NPU partition outputs.
  Tensor scalar_f = g.F32({});
  Tensor scalar_b_out = g.Bool({});
  g.MarkInput(scalar_f, &subgraph);
  g.AddOp(kLiteRtOpCodeTflCast, {scalar_f}, {scalar_b_out});
  g.MarkOutput(scalar_b_out, &subgraph);

  Tensor scalar_b_const = g.BoolConst({}, {1});
  Tensor scalar_f_out = g.F32({});
  g.AddOp(kLiteRtOpCodeTflCast, {scalar_b_const}, {scalar_f_out});
  g.MarkOutput(scalar_f_out, &subgraph);

  g.ApplyTo(&subgraph);

  const size_t ops_before = subgraph.Ops().size();
  RunPatterns(&subgraph, AllPatterns(/*include_output_not_equal=*/true));
  ExpectTopologicalOrder(subgraph);
  EXPECT_EQ(subgraph.Ops().size(), ops_before) << DumpOps(subgraph);
}

TEST(BoolMaskTransformationTest,
     SelfAliasingDuplicateInputsAndExactBoolXorNotEqual) {
  LiteRtModelT model;
  LiteRtSubgraphT& subgraph = model.EmplaceSubgraph();
  GraphBuilder g;

  Tensor b1 = g.Bool({1, 4});
  Tensor b2 = g.Bool({1, 4});
  Tensor arb_f = g.F32({1, 4});
  g.MarkInput(b1, &subgraph);
  g.MarkInput(b2, &subgraph);
  g.MarkInput(arb_f, &subgraph);

  // Self-aliasing Bool ops: LogicalAnd(b1, b1), LogicalOr(b1, b1),
  // Concat(b1, b1), Select(b1, b1, b2), NotEqual(b1, b1), NotEqual(b1, b2) ->
  // LogicalNot (proving Abs(Sub(b1_f, b2_f)) is recognized as {0.0, 1.0}!).
  Tensor and_self = g.Bool({1, 4});
  g.AddOp(kLiteRtOpCodeTflLogicalAnd, {b1, b1}, {and_self});
  Tensor or_self = g.Bool({1, 4});
  g.AddOp(kLiteRtOpCodeTflLogicalOr, {and_self, and_self}, {or_self});
  Tensor cat_self = g.Bool({1, 8});
  SetConcatAxis(
      g, g.AddOp(kLiteRtOpCodeTflConcatenation, {or_self, or_self}, {cat_self}),
      1);
  g.MarkOutput(cat_self, &subgraph);

  Tensor sel_self = g.Bool({1, 4});
  g.AddOp(kLiteRtOpCodeTflSelect, {b1, b1, b2}, {sel_self});
  Tensor ne_self = g.Bool({1, 4});
  g.AddOp(kLiteRtOpCodeTflNotEqual, {b1, b1}, {ne_self});
  Tensor ne_xor = g.Bool({1, 4});
  g.AddOp(kLiteRtOpCodeTflNotEqual, {sel_self, b2}, {ne_xor});
  Tensor xnor = g.Bool({1, 4});
  g.AddOp(kLiteRtOpCodeTflLogicalNot, {ne_xor}, {xnor});
  g.MarkOutput(ne_self, &subgraph);
  g.MarkOutput(xnor, &subgraph);

  // Self-aliasing NotEqual(arb_f, arb_f) (`isnan(x)` idiom) + LogicalNot:
  // MUST NOT lower to Sub(arb_f, arb_f), preserving NaN != NaN == true!
  Tensor isnan_b = g.Bool({1, 4});
  Tensor not_nan_b = g.Bool({1, 4});
  g.AddOp(kLiteRtOpCodeTflNotEqual, {arb_f, arb_f}, {isnan_b});
  g.AddOp(kLiteRtOpCodeTflLogicalNot, {isnan_b}, {not_nan_b});
  g.MarkOutput(not_nan_b, &subgraph);

  g.ApplyTo(&subgraph);

  absl::flat_hash_map<const LiteRtTensorT*, EvalVal> feed;
  feed[b1.Get()] = {kLiteRtElementTypeBool, {1, 4}, {}, {}, {0, 1, 0, 1}};
  feed[b2.Get()] = {kLiteRtElementTypeBool, {1, 4}, {}, {}, {0, 0, 1, 1}};
  feed[arb_f.Get()] = {kLiteRtElementTypeFloat32,
                       {1, 4},
                       {0.0f, 1.5f, std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::infinity()},
                       {},
                       {}};
  const auto expected = EvalSubgraph(subgraph, feed);

  RunPatterns(&subgraph, AllPatterns(/*include_output_not_equal=*/true));
  ExpectTopologicalOrder(subgraph);
  const auto actual = EvalSubgraph(subgraph, feed);
  ExpectSameOutputs(subgraph, expected, actual);

  // NaN element (index 2) has isnan == true -> not_nan == false (0).
  EXPECT_EQ(actual.at(not_nan_b.Get()).b[0], 1);
  EXPECT_EQ(actual.at(not_nan_b.Get()).b[2], 0);
  EXPECT_EQ(actual.at(not_nan_b.Get()).b[3], 1);
}

TEST(BoolMaskTransformationTest,
     FusedActivationsAndQuantizedTensorsAreHandledSafely) {
  LiteRtModelT model;
  LiteRtSubgraphT& subgraph = model.EmplaceSubgraph();
  GraphBuilder g;

  // 1. BypassCast preserves non-zero fused_activation_function (RELU6 = 3,
  //    TANH = 2) on Add/Mul/Conv2d consumers, while IsKnownZeroOneFloat
  //    rejects TANH (2) outputs because tanh(1.0) != 1.0!
  Tensor b_in = g.Bool({1, 4});
  g.MarkInput(b_in, &subgraph);
  Tensor f01 = g.F32({1, 4});
  g.AddOp(kLiteRtOpCodeTflCast, {b_in}, {f01});
  Tensor b_mid = g.Bool({1, 4});
  g.AddOp(kLiteRtOpCodeTflCast, {f01}, {b_mid});
  Tensor f_mid = g.F32({1, 4});
  g.AddOp(kLiteRtOpCodeTflCast, {b_mid}, {f_mid});

  Tensor tanh_mul_out = g.F32({1, 4});
  MulOptions tanh_mul_opts;
  tanh_mul_opts.fused_activation_function = 2;  // TANH
  g.SetOptions(g.AddOp(kLiteRtOpCodeTflMul, {f_mid, f_mid}, {tanh_mul_out}),
               tanh_mul_opts);

  // Because tanh(1.0) = 0.76159 != 1.0, NotEqual(tanh_mul_out, 1.0f) MUST NOT
  // treat tanh_mul_out as {0.0, 1.0}!
  Tensor ne_tanh = g.Bool({1, 4});
  Tensor not_ne_tanh = g.Bool({1, 4});
  g.AddOp(kLiteRtOpCodeTflNotEqual, {tanh_mul_out, g.Const<float>({}, {1.0f})},
          {ne_tanh});
  g.AddOp(kLiteRtOpCodeTflLogicalNot, {ne_tanh}, {not_ne_tanh});
  g.MarkOutput(not_ne_tanh, &subgraph);

  // 2. Quantized Float32 / Bool / Int32 tensors are never treated as
  //    unquantized mask tensors.
  Tensor q_bool = g.Quant(ElementType::Bool, {1, 4}, 1.0f, 0);
  Tensor q_bool_out = g.Quant(ElementType::Bool, {1, 4}, 1.0f, 0);
  g.MarkInput(q_bool, &subgraph);
  g.AddOp(kLiteRtOpCodeTflLogicalNot, {q_bool}, {q_bool_out});
  g.MarkOutput(q_bool_out, &subgraph);

  Tensor q_f32 = g.Quant(ElementType::Float32, {1, 4}, 0.5f, 0);
  Tensor q_ne_out = g.Bool({1, 4});
  Tensor q_ne_inv = g.Bool({1, 4});
  g.MarkInput(q_f32, &subgraph);
  g.AddOp(kLiteRtOpCodeTflNotEqual, {q_f32, g.Const<float>({}, {0.0f})},
          {q_ne_out});
  g.AddOp(kLiteRtOpCodeTflLogicalNot, {q_ne_out}, {q_ne_inv});
  g.MarkOutput(q_ne_inv, &subgraph);

  g.ApplyTo(&subgraph);

  absl::flat_hash_map<const LiteRtTensorT*, EvalVal> feed;
  feed[b_in.Get()] = {kLiteRtElementTypeBool, {1, 4}, {}, {}, {0, 1, 1, 0}};
  feed[q_bool.Get()] = {kLiteRtElementTypeBool, {1, 4}, {}, {}, {1, 0, 1, 0}};
  feed[q_f32.Get()] = {
      kLiteRtElementTypeFloat32, {1, 4}, {0.0f, 1.0f, 0.5f, 0.0f}, {}, {}};
  const auto expected = EvalSubgraph(subgraph, feed);

  RunPatterns(&subgraph, AllPatterns(/*include_output_not_equal=*/true));
  ExpectTopologicalOrder(subgraph);
  const auto actual = EvalSubgraph(subgraph, feed);
  ExpectSameOutputs(subgraph, expected, actual);

  // tanh(1.0) != 1.0 -> ne_tanh is true -> not_ne_tanh is false (0)!
  EXPECT_EQ(actual.at(not_ne_tanh.Get()).b[1], 0);
  // Mul kept TANH fused activation after BypassCast cloned it.
  bool found_tanh_mul = false;
  for (LiteRtOp op : subgraph.Ops()) {
    if (op->OpCode() == kLiteRtOpCodeTflMul) {
      auto opts = GetOptionsAs<MulOptions>(Ctx(), op);
      ASSERT_TRUE(opts.HasValue());
      EXPECT_EQ(opts->fused_activation_function, 2u);
      found_tanh_mul = true;
    }
  }
  EXPECT_TRUE(found_tanh_mul);
}

TEST(BoolMaskTransformationTest,
     SqueezeExpandDimsAndTransposeMasksLowerCleanly) {
  LiteRtModelT model;
  LiteRtSubgraphT& subgraph = model.EmplaceSubgraph();
  GraphBuilder g;

  Tensor b_in = g.Bool({1, 3, 1});
  Tensor x = g.F32({3, 1, 2});
  g.MarkInput(b_in, &subgraph);
  g.MarkInput(x, &subgraph);

  // Bool Squeeze([1, 3, 1], squeeze_dims={-1}) -> [1, 3]
  Tensor sq = g.Bool({1, 3});
  Op sq_op = g.AddOp(kLiteRtOpCodeTflSqueeze, {b_in}, {sq});
  SqueezeOptions sq_opts;
  sq_opts.squeeze_dims = {-1};
  g.SetOptions(sq_op, std::move(sq_opts));

  // Bool Transpose([1, 3], perm=[1, 0] Int64) -> [3, 1]
  Tensor tr = g.Bool({3, 1});
  g.AddOp(kLiteRtOpCodeTflTranspose, {sq, g.Const<int64_t>({2}, {1, 0})}, {tr});

  // Bool ExpandDims([3, 1], axis=-1 Int64) -> [3, 1, 1]
  Tensor ex = g.Bool({3, 1, 1});
  g.AddOp(kLiteRtOpCodeTflExpandDims, {tr, g.Const<int64_t>({}, {-1})}, {ex});

  // SelectV2([3, 1, 1], x [3, 1, 2], 0.0f) -> [3, 1, 2]
  Tensor out_sel = g.F32({3, 1, 2});
  g.AddOp(kLiteRtOpCodeTflSelectV2, {ex, x, g.Const<float>({}, {0.0f})},
          {out_sel});
  g.MarkOutput(out_sel, &subgraph);

  // Also verify Float32 Squeeze + ExpandDims propagate IsKnownZeroOneFloat
  // into NotEqual(f_ex, 0.0f).
  Tensor f0 = g.F32({1, 3, 1});
  g.AddOp(kLiteRtOpCodeTflCast, {b_in}, {f0});
  Tensor f_sq = g.F32({3});
  SqueezeOptions sq_all;
  g.SetOptions(g.AddOp(kLiteRtOpCodeTflSqueeze, {f0}, {f_sq}),
               std::move(sq_all));
  Tensor f_ex = g.F32({1, 3});
  g.AddOp(kLiteRtOpCodeTflExpandDims, {f_sq, g.Const<int32_t>({}, {0})},
          {f_ex});
  Tensor ne_f = g.Bool({1, 3});
  g.AddOp(kLiteRtOpCodeTflNotEqual, {f_ex, g.Const<float>({}, {0.0f})}, {ne_f});
  Tensor not_ne_f = g.Bool({1, 3});
  g.AddOp(kLiteRtOpCodeTflLogicalNot, {ne_f}, {not_ne_f});
  g.MarkOutput(not_ne_f, &subgraph);

  g.ApplyTo(&subgraph);

  absl::flat_hash_map<const LiteRtTensorT*, EvalVal> feed;
  feed[b_in.Get()] = {kLiteRtElementTypeBool, {1, 3, 1}, {}, {}, {1, 0, 1}};
  feed[x.Get()] = {kLiteRtElementTypeFloat32,
                   {3, 1, 2},
                   {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                   {},
                   {}};
  const auto expected = EvalSubgraph(subgraph, feed);

  RunPatterns(&subgraph, AllPatterns(/*include_output_not_equal=*/true));
  ExpectTopologicalOrder(subgraph);
  ExpectSameOutputs(subgraph, expected, EvalSubgraph(subgraph, feed));

  EXPECT_EQ(CountBoolOps(subgraph), 0) << DumpOps(subgraph);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflSelectV2), 0);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflLogicalNot), 0);
}

TEST(BoolMaskTransformationTest, ZeroOneAnalysisMemoizesSharedProducers) {
  LiteRtModelT model;
  LiteRtSubgraphT& subgraph = model.EmplaceSubgraph();
  GraphBuilder g;
  Tensor b = g.Bool({1, 4});
  g.MarkInput(b, &subgraph);
  Tensor t = g.F32({1, 4});
  g.AddOp(kLiteRtOpCodeTflCast, {b}, {t});
  for (int i = 0; i < 23; ++i) {
    Tensor next = g.F32({1, 4});
    MulOptions opts;
    opts.fused_activation_function = 0;
    g.SetOptions(g.AddOp(kLiteRtOpCodeTflMul, {t, t}, {next}), std::move(opts));
    t = next;
  }
  Tensor zero = g.Const<float>({}, {0.0f});
  Tensor ne_out = g.Bool({1, 4});
  Op ne = g.AddOp(kLiteRtOpCodeTflNotEqual, {t, zero}, {ne_out});
  Tensor not_out = g.Bool({1, 4});
  g.AddOp(kLiteRtOpCodeTflLogicalNot, {ne_out}, {not_out});
  g.MarkOutput(not_out, &subgraph);
  g.ApplyTo(&subgraph);

  LiteRtBuilderT builder;
  builder.SetTargetSubgraph(&subgraph);
  ASSERT_EQ(BoolMaskNotEqualTransformation(Ctx(), &builder, ne.Get()),
            kLiteRtStatusOk);
  builder.ApplyChanges(&subgraph);
  ExpectTopologicalOrder(subgraph);

  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflNotEqual), 0);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflAbs), 0) << DumpOps(subgraph);
}

TEST(BoolMaskTransformationTest, NativeSameShapeOutputNotEqualIsPreserved) {
  LiteRtModelT model;
  LiteRtSubgraphT& subgraph = model.EmplaceSubgraph();
  GraphBuilder g;
  Tensor x = g.F32({1, 4});
  Tensor y = g.F32({1, 4});
  Tensor ne_out = g.Bool({1, 4});
  g.MarkInput(x, &subgraph);
  g.MarkInput(y, &subgraph);
  g.AddOp(kLiteRtOpCodeTflNotEqual, {x, y}, {ne_out});
  g.MarkOutput(ne_out, &subgraph);
  g.ApplyTo(&subgraph);

  RunPatterns(&subgraph, AllPatterns(/*include_output_not_equal=*/true));
  ExpectTopologicalOrder(subgraph);

  ASSERT_EQ(subgraph.Ops().size(), 1) << DumpOps(subgraph);
  const LiteRtOpT* op = subgraph.Ops()[0];
  EXPECT_EQ(op->OpCode(), kLiteRtOpCodeTflNotEqual);
  EXPECT_EQ(op->Inputs()[0], x.Get());
  EXPECT_EQ(op->Inputs()[1], y.Get());
  EXPECT_EQ(op->Outputs()[0], ne_out.Get());
}

TEST(BoolMaskTransformationTest,
     BroadcastZeroOneOutputNotEqualIsLoweredToSameShapeNotEqual) {
  LiteRtModelT model;
  LiteRtSubgraphT& subgraph = model.EmplaceSubgraph();
  GraphBuilder g;
  Tensor b = g.Bool({2, 1});
  Tensor x = g.F32({2, 1});
  Tensor y_mixed = g.Const<float>({1, 4}, {1.0f, 0.0f, 1.0f, 0.0f});
  Tensor ne_bcast = g.Bool({2, 4});
  Tensor b_same = g.Bool({1, 4});
  Tensor x_same = g.F32({1, 4});
  Tensor y_one = g.Const<float>({1, 1}, {1.0f});
  Tensor ne_one = g.Bool({1, 4});
  g.MarkInput(b, &subgraph);
  g.MarkInput(b_same, &subgraph);
  g.AddOp(kLiteRtOpCodeTflCast, {b}, {x});
  g.AddOp(kLiteRtOpCodeTflNotEqual, {x, y_mixed}, {ne_bcast});
  g.AddOp(kLiteRtOpCodeTflCast, {b_same}, {x_same});
  g.AddOp(kLiteRtOpCodeTflNotEqual, {x_same, y_one}, {ne_one});
  g.MarkOutput(ne_bcast, &subgraph);
  g.MarkOutput(ne_one, &subgraph);
  g.ApplyTo(&subgraph);

  absl::flat_hash_map<const LiteRtTensorT*, EvalVal> feed;
  feed[b.Get()] = {kLiteRtElementTypeBool, {2, 1}, {}, {}, {0, 1}};
  feed[b_same.Get()] = {kLiteRtElementTypeBool, {1, 4}, {}, {}, {0, 1, 1, 0}};
  const auto expected = EvalSubgraph(subgraph, feed);

  RunPatterns(&subgraph, AllPatterns(/*include_output_not_equal=*/true));
  ExpectTopologicalOrder(subgraph);
  ExpectSameOutputs(subgraph, expected, EvalSubgraph(subgraph, feed));

  // 2-way broadcast NotEqual({2,1}, {1,4}) lowers via Sub + Abs -> NotEqual,
  // while NotEqual({1,4}, {1,1}=1.0f) takes the single-Sub fast path ->
  // NotEqual (no Abs needed).
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflSub), 2) << DumpOps(subgraph);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflAbs), 1);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflNotEqual), 2);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflCast), 2);
}

TEST(BoolMaskTransformationTest,
     EqualOnBoolAndZeroOneFloatLowersWhileArbitraryFloatStaysExact) {
  LiteRtModelT model;
  LiteRtSubgraphT& subgraph = model.EmplaceSubgraph();
  GraphBuilder g;

  Tensor b1 = g.Bool({1, 4});
  Tensor b2 = g.Bool({1, 4});
  Tensor arb = g.F32({1, 4});
  g.MarkInput(b1, &subgraph);
  g.MarkInput(b2, &subgraph);
  g.MarkInput(arb, &subgraph);

  // 1. Equal(b1, b2) -> XNOR on Bool -> LogicalNot -> XOR
  Tensor eq_bb = g.Bool({1, 4});
  Tensor neq_bb = g.Bool({1, 4});
  g.AddOp(kLiteRtOpCodeTflEqual, {b1, b2}, {eq_bb});
  g.AddOp(kLiteRtOpCodeTflLogicalNot, {eq_bb}, {neq_bb});
  g.MarkOutput(neq_bb, &subgraph);

  // 2. Equal(f01, 0.0f) and Equal(f01, 1.0f) fast paths -> LogicalAnd
  Tensor f01 = g.F32({1, 4});
  g.AddOp(kLiteRtOpCodeTflCast, {b1}, {f01});
  Tensor eq_z = g.Bool({1, 4});
  Tensor eq_o = g.Bool({1, 4});
  Tensor both = g.Bool({1, 4});
  g.AddOp(kLiteRtOpCodeTflEqual, {f01, g.Const<float>({}, {0.0f})}, {eq_z});
  g.AddOp(kLiteRtOpCodeTflEqual, {g.Const<float>({}, {1.0f}), f01}, {eq_o});
  g.AddOp(kLiteRtOpCodeTflLogicalOr, {eq_z, eq_o}, {both});
  g.MarkOutput(both, &subgraph);

  // 3. Arbitrary float Equal(arb, arb) (`!isnan(x)`) stays untouched on NaN!
  Tensor eq_arb = g.Bool({1, 4});
  g.AddOp(kLiteRtOpCodeTflEqual, {arb, arb}, {eq_arb});
  g.MarkOutput(eq_arb, &subgraph);

  g.ApplyTo(&subgraph);

  absl::flat_hash_map<const LiteRtTensorT*, EvalVal> feed;
  feed[b1.Get()] = {kLiteRtElementTypeBool, {1, 4}, {}, {}, {0, 1, 0, 1}};
  feed[b2.Get()] = {kLiteRtElementTypeBool, {1, 4}, {}, {}, {0, 0, 1, 1}};
  feed[arb.Get()] = {kLiteRtElementTypeFloat32,
                     {1, 4},
                     {0.0f, 1e-12f, std::numeric_limits<float>::quiet_NaN(),
                      std::numeric_limits<float>::infinity()},
                     {},
                     {}};
  const auto expected = EvalSubgraph(subgraph, feed);

  RunPatterns(&subgraph, AllPatterns(/*include_output_not_equal=*/true));
  ExpectTopologicalOrder(subgraph);
  const auto actual = EvalSubgraph(subgraph, feed);
  ExpectSameOutputs(subgraph, expected, actual);

  // Only the single arbitrary-float Equal(arb, arb) remains!
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflEqual), 1) << DumpOps(subgraph);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflLogicalNot), 0);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflLogicalOr), 0);
  EXPECT_EQ(actual.at(eq_arb.Get()).b[2], 0);  // NaN == NaN is false
  EXPECT_EQ(actual.at(eq_arb.Get()).b[3], 1);  // +Inf == +Inf is true
}

TEST(BoolMaskTransformationTest, RandomSeededMaskDagParityStress) {
  const float inf = std::numeric_limits<float>::infinity();
  const float nan_val = std::numeric_limits<float>::quiet_NaN();
  for (uint32_t trial = 0; trial < 200; ++trial) {
    std::mt19937 rng(20261120u + trial);
    LiteRtModelT model;
    LiteRtSubgraphT& subgraph = model.EmplaceSubgraph();
    GraphBuilder g;

    Tensor b0 = g.Bool({2, 3});
    Tensor b1 = g.Bool({2, 3});
    Tensor b_batch = g.Bool({2});
    Tensor f_clean = g.F32({2, 3});
    Tensor f_tiny = g.F32({2, 3});
    Tensor f_nonfinite = g.F32({2, 3});
    g.MarkInput(b0, &subgraph);
    g.MarkInput(b1, &subgraph);
    g.MarkInput(b_batch, &subgraph);
    g.MarkInput(f_clean, &subgraph);
    g.MarkInput(f_tiny, &subgraph);
    g.MarkInput(f_nonfinite, &subgraph);

    std::vector<Tensor> bool_pool = {b0, b1};
    std::vector<Tensor> f01_pool;
    // Finite extreme floats (including sub-2^-30 1e-12f and -0.0f) for
    // dynamic Select/SelectV2 branches; runtime NaN/Inf is tested via
    // NotEqual/Equal (`f_nonfinite`) and constant NaN/-Inf SelectV2 branches.
    std::vector<Tensor> farbit_pool = {f_clean, f_tiny};

    const int num_steps = 8 + static_cast<int>(rng() % 7);
    for (int step = 0; step < num_steps; ++step) {
      const uint32_t kind = rng() % 11;
      auto pick_bool = [&]() -> Tensor {
        return bool_pool[rng() % bool_pool.size()];
      };
      auto pick_f32 = [&]() -> Tensor {
        if (!f01_pool.empty() && (rng() % 3 == 0)) {
          return f01_pool[rng() % f01_pool.size()];
        }
        return farbit_pool[rng() % farbit_pool.size()];
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
        Tensor f01 = g.F32({2, 3});
        g.AddOp(kLiteRtOpCodeTflCast, {pick_bool()}, {f01});
        f01_pool.push_back(f01);
        Tensor out = g.Bool({2, 3});
        const float target = (rng() % 2 == 0) ? 0.0f : 1.0f;
        g.AddOp(
            (rng() % 2 == 0) ? kLiteRtOpCodeTflNotEqual : kLiteRtOpCodeTflEqual,
            {f01, g.Const<float>({}, {target})}, {out});
        bool_pool.push_back(out);
      } else if (kind == 5) {
        // Interleave arbitrary/non-finite float comparison into Bool pool!
        Tensor src = (rng() % 2 == 0) ? f_nonfinite : pick_f32();
        Tensor rhs = (rng() % 2 == 0) ? src : g.Const<float>({}, {0.0f});
        Tensor out = g.Bool({2, 3});
        g.AddOp(
            (rng() % 2 == 0) ? kLiteRtOpCodeTflNotEqual : kLiteRtOpCodeTflEqual,
            {src, rhs}, {out});
        bool_pool.push_back(out);
      } else if (kind == 6) {
        // ExpandDims -> Squeeze round-trip on Bool mask
        Tensor ex = g.Bool({2, 1, 3});
        g.AddOp(kLiteRtOpCodeTflExpandDims,
                {pick_bool(), g.Const<int64_t>({}, {1})}, {ex});
        Tensor sq = g.Bool({2, 3});
        SqueezeOptions sq_opts;
        sq_opts.squeeze_dims = {1};
        g.SetOptions(g.AddOp(kLiteRtOpCodeTflSqueeze, {ex}, {sq}),
                     std::move(sq_opts));
        bool_pool.push_back(sq);
      } else if (kind == 7) {
        // Double Transpose round-trip on Bool mask
        Tensor tr = g.Bool({3, 2});
        Tensor perm = g.Const<int64_t>({2}, {1, 0});
        g.AddOp(kLiteRtOpCodeTflTranspose, {pick_bool(), perm}, {tr});
        Tensor back = g.Bool({2, 3});
        g.AddOp(kLiteRtOpCodeTflTranspose, {tr, perm}, {back});
        bool_pool.push_back(back);
      } else if (kind == 8) {
        // TileWrap (Cast->Int32 Tile->Cast) + Slice back to {2, 3}
        Tensor i_in = g.I32({2, 3});
        Tensor i_out = g.I32({2, 6});
        Tensor b_tiled = g.Bool({2, 6});
        g.AddOp(kLiteRtOpCodeTflCast, {pick_bool()}, {i_in});
        g.AddOp(kLiteRtOpCodeTflTile, {i_in, g.Const<int64_t>({2}, {1, 2})},
                {i_out});
        g.AddOp(kLiteRtOpCodeTflCast, {i_out}, {b_tiled});
        Tensor sliced = g.Bool({2, 3});
        g.AddOp(kLiteRtOpCodeTflSlice,
                {b_tiled, g.Const<int32_t>({2}, {0, 2}),
                 g.Const<int32_t>({2}, {2, 3})},
                {sliced});
        bool_pool.push_back(sliced);
      } else if (kind == 9) {
        // 1D leading-batch Select on {2, 3}
        Tensor out = g.F32({2, 3});
        g.AddOp(kLiteRtOpCodeTflSelect, {b_batch, pick_f32(), pick_f32()},
                {out});
        farbit_pool.push_back(out);
      } else {
        // SelectV2 with occasional +0.0f / -0.0f / NaN constant branch
        Tensor t_br = pick_f32();
        Tensor f_br = pick_f32();
        const uint32_t c_mode = rng() % 4;
        if (c_mode == 0) {
          f_br = g.Const<float>({}, {0.0f});
        } else if (c_mode == 1) {
          f_br = g.Const<float>({}, {-0.0f});
        } else if (c_mode == 2) {
          t_br = g.Const<float>({}, {nan_val});
        }
        Tensor out = g.F32({2, 3});
        g.AddOp(kLiteRtOpCodeTflSelectV2, {pick_bool(), t_br, f_br}, {out});
        farbit_pool.push_back(out);
      }

      // Periodically expose intermediate tensors as subgraph outputs while
      // later steps still consume them internally!
      if (rng() % 3 == 0) {
        g.MarkOutput(bool_pool.back(), &subgraph);
      }
    }

    g.MarkOutput(bool_pool.back(), &subgraph);
    g.MarkOutput(farbit_pool.back(), &subgraph);
    g.ApplyTo(&subgraph);

    absl::flat_hash_map<const LiteRtTensorT*, EvalVal> feed;
    std::vector<uint8_t> vb0(6), vb1(6), vbb(2);
    for (int i = 0; i < 6; ++i) {
      vb0[i] = static_cast<uint8_t>(rng() & 1u);
      vb1[i] = static_cast<uint8_t>(rng() & 1u);
    }
    vbb[0] = static_cast<uint8_t>(rng() & 1u);
    vbb[1] = static_cast<uint8_t>(rng() & 1u);
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
    const auto actual = EvalSubgraph(subgraph, feed);
    ExpectSameOutputs(subgraph, expected, actual);
  }
}

TEST(BoolMaskTransformationTest,
     UnrewrittenProducerBoolChainsStayIntactWithoutPingPong) {
  LiteRtModelT model;
  LiteRtSubgraphT& subgraph = model.EmplaceSubgraph();
  GraphBuilder g;

  // 1. EntryEmbedding / FloorDiv style: integer comparisons (Less,
  //    GreaterEqual, NotEqual on Int32) -> LogicalOr -> LogicalAnd -> Reshape
  //    -> SelectV2. BoolMask must NOT synthesize Cast(Bool->F32) on outputs of
  //    unrewritten ops, preserving multi-op patterns and avoiding NPU-CPU
  //    ping-pong islands.
  Tensor coord = g.I32({1, 4});
  Tensor table = g.F32({1, 4, 8});
  Tensor arb_mask_f = g.F32({1, 4, 1, 4});
  g.MarkInput(coord, &subgraph);
  g.MarkInput(table, &subgraph);
  g.MarkInput(arb_mask_f, &subgraph);

  Tensor lt = g.Bool({1, 4});
  Tensor ge = g.Bool({1, 4});
  Tensor lor = g.Bool({1, 4});
  Tensor ne_i32 = g.Bool({1, 4});
  Tensor land = g.Bool({1, 4});
  Tensor r_cond = g.Bool({1, 4, 1});
  Tensor sel_out = g.F32({1, 4, 8});
  g.AddOp(kLiteRtOpCodeTflLess, {coord, g.Const<int32_t>({}, {0})}, {lt});
  g.AddOp(kLiteRtOpCodeTflGreaterEqual, {coord, g.Const<int32_t>({}, {16})},
          {ge});
  g.AddOp(kLiteRtOpCodeTflLogicalOr, {lt, ge}, {lor});
  g.AddOp(kLiteRtOpCodeTflNotEqual, {coord, g.Const<int32_t>({}, {-1})},
          {ne_i32});
  g.AddOp(kLiteRtOpCodeTflLogicalAnd, {lor, ne_i32}, {land});
  SetReshapeShape(g,
                  g.AddOp(kLiteRtOpCodeTflReshape,
                          {land, g.Const<int32_t>({3}, {1, 4, 1})}, {r_cond}),
                  {1, 4, 1});
  g.AddOp(kLiteRtOpCodeTflSelectV2, {r_cond, g.Const<float>({}, {0.0f}), table},
          {sel_out});
  g.MarkOutput(sel_out, &subgraph);

  // 2. AttentionMask Pattern 2 style: NotEqual(arbitrary_f32, 0.0f) ->
  //    LogicalNot -> Cast(->F32). LogicalNot must NOT be rewritten before
  //    AttentionMaskTransformation visits the root Cast!
  Tensor ne_f = g.Bool({1, 4, 1, 4});
  Tensor lnot_f = g.Bool({1, 4, 1, 4});
  Tensor cast_f = g.F32({1, 4, 1, 4});
  g.AddOp(kLiteRtOpCodeTflNotEqual, {arb_mask_f, g.Const<float>({}, {0.0f})},
          {ne_f});
  g.AddOp(kLiteRtOpCodeTflLogicalNot, {ne_f}, {lnot_f});
  g.AddOp(kLiteRtOpCodeTflCast, {lnot_f}, {cast_f});
  g.MarkOutput(cast_f, &subgraph);

  g.ApplyTo(&subgraph);

  const size_t ops_before = subgraph.Ops().size();
  RunPatterns(&subgraph, AllPatterns(/*include_output_not_equal=*/true));
  ExpectTopologicalOrder(subgraph);
  EXPECT_EQ(subgraph.Ops().size(), ops_before) << DumpOps(subgraph);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflLogicalOr), 1);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflLogicalAnd), 1);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflLogicalNot), 1);
  EXPECT_EQ(CountOps(subgraph, kLiteRtOpCodeTflSelectV2), 1);
}

}  // namespace
}  // namespace litert::mediatek
