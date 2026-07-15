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

#ifndef LITERT_VENDORS_INTEL_OPENVINO_COMPILER_NPU_OPTIMIZER_H_
#define LITERT_VENDORS_INTEL_OPENVINO_COMPILER_NPU_OPTIMIZER_H_

#include <memory>

#include "openvino/core/model.hpp"
#include "openvino/pass/graph_rewrite.hpp"

namespace litert {
namespace openvino {

// Eliminates FakeQuantize nodes that immediately follow MatMul operations.
//
// The FakeQuantize directly after a MatMul re-quantize its output to a
// calibrated range. This pass detects the MatMul -> FakeQuantize pattern
// and removes the FakeQuantize, rewiring downstream consumers to read
// directly from the MatMul output to reduce NPU overhead.
class EliminateMatMulFakeQuantize : public ov::pass::MatcherPass {
 public:
  OPENVINO_MATCHER_PASS_RTTI("EliminateMatMulFakeQuantize");
  EliminateMatMulFakeQuantize();
};

// The Intel NPU compiler's IE.Sign op only accepts float operands
// (f16/f32/f64). When the TFLite frontend produces a Sign node with integer
// input, NPU compilation fails. This pass wraps such Sign nodes with Convert
// ops:
//   Convert(int->f32) -> Sign -> Convert(f32->original_int_type)
class CastIntegerSignToFloat : public ov::pass::MatcherPass {
 public:
  OPENVINO_MATCHER_PASS_RTTI("CastIntegerSignToFloat");
  CastIntegerSignToFloat();
};

// Fuses the "split-attention" sub-graph produced by some Gemma-style models
// (attention computed separately against the persistent KV cache and the
// current step's KV, merged via Concat) into a single
// `ov::op::v13::ScaledDotProductAttention`.
//
// Pattern matched (Q is shared between both qk MatMuls):
//
//   qk_cache = MatMul(Q, K_cache)
//   qk_new   = MatMul(Q, K_new)
//   scores   = Concat(qk_cache, qk_new, axis=-1)
//   masked   = Add(scores, mask)
//   probs    = Softmax(masked, axis=-1)
//   pc       = Slice/StridedSlice(probs)        // cache half
//   pn       = Slice/StridedSlice(probs)        // new half
//   attn_c   = MatMul(pc, V_cache)
//   attn_n   = MatMul(pn, V_new)
//   output   = Add(attn_c, attn_n)              // <-- match root
//
// Restricted to static 4-D ranks; other shapes are left untouched. Scale is
// set to 1.0 because the matched pattern has no explicit scale node (any
// scaling is assumed pre-applied to Q by the model, e.g. Gemma's
// `query_pre_attn_scalar`). When the MatMul transpose_b flags imply K or V
// is stored as [B,H,D,S], an explicit Transpose(0,1,3,2) is inserted to
// restore the [B,H,S,D] layout required by v13::SDPA.
//
// Ported from openvinotoolkit/openvino PR #36153.
class FuseSplitAttentionToSDPA : public ov::pass::MatcherPass {
 public:
  OPENVINO_MATCHER_PASS_RTTI("FuseSplitAttentionToSDPA");
  // |pad_kv_to_alignment| controls behavior when the merged KV sequence length
  // is not a multiple of the NPU SDPA kernel's required alignment (16). When
  // false, such blocks are left unfused. When true(default), the merged K/V
  // is padded up to the next multiple of 16 and the mask is padded with a
  // large negative bias so the padded key positions are excluded from softmax.
  explicit FuseSplitAttentionToSDPA(bool pad_kv_to_alignment);
};

// Fuses the "standard" single-branch attention sub-graph into a single
// `ov::op::v13::ScaledDotProductAttention`. This is the shape produced after an
// offline transform merges a Gemma-style split KV cache into one tensor with a
// DynamicUpdateSlice (see inplace_kv_conversion/convert_inplace_kv.py), so there
// is no Concat/Slice pair and only one QK / one PV MatMul remain:
//
//   scores = MatMul(Q, K)                    // K may be [B,H,S,D] or [B,H,D,S]
//   masked = Add(scores, mask)               // optional
//   probs  = Softmax(masked, axis=-1)
//   output = MatMul(probs, V)                // V may be [B,H,S,D] or [B,H,D,S]
//                                            //   <-- match root
//
// The K/V operands typically arrive via CAST(DynamicUpdateSlice(...)); that is
// transparent to this matcher, which anchors on the PV MatMul and reads the
// MatMul transpose flags as a layout hint (inserting a Transpose(0,1,3,2) when
// K or V is stored [B,H,D,S], as Gemma stores V). Restricted to static 4-D
// ranks; scale is set to 1.0 (any scaling assumed pre-applied to Q). The merged
// KV length is expected to already be a multiple of the NPU SDPA kernel's
// alignment, so no KV/mask padding is performed here.
//
// The stock ov::pass::SDPAFusion does not cover this case because it requires
// the PV MatMul to consume V in [S_kv, Ev] layout (transpose_b=false), whereas
// Gemma stores V transposed as [B,H,D,S] and consumes it with transpose_b=true.
class FuseStandardAttentionToSDPA : public ov::pass::MatcherPass {
 public:
  OPENVINO_MATCHER_PASS_RTTI("FuseStandardAttentionToSDPA");
  FuseStandardAttentionToSDPA();
};

// Configurable runner for NPU-specific optimization passes.
// Use the setter APIs to toggle individual optimizations, then call Run() to
// apply the enabled passes to a model.
class NpuOptimizer {
 public:
  // Toggles the EliminateMatMulFakeQuantize pass. Disabled by default.
  NpuOptimizer& SetEliminateMatMulFakeQuantize(bool enable) {
    eliminate_matmul_fq_ = enable;
    return *this;
  }

  // Toggles the CastIntegerSignToFloat pass. Enabled by default because the
  // NPU plugin cannot lower integer Sign operations.
  NpuOptimizer& SetCastIntegerSignToFloat(bool enable) {
    cast_integer_sign_to_float_ = enable;
    return *this;
  }

  // Toggles the FuseSplitAttentionToSDPA pass. Disabled by default; enable
  // via config key "fuse_split_attention_to_sdpa" = "true" to collapse the
  // Gemma-style split-attention pattern into a single SDPA op.
  NpuOptimizer& SetFuseSplitAttentionToSDPA(bool enable) {
    fuse_split_attention_to_sdpa_ = enable;
    return *this;
  }

  // Controls whether the SDPA fusion pads an unaligned merged KV sequence
  // length up to the NPU kernel's required alignment.
  NpuOptimizer& SetSdpaPadKvToAlignment(bool enable) {
    sdpa_pad_kv_to_alignment_ = enable;
    return *this;
  }

  // Toggles the FuseStandardAttentionToSDPA pass. Disabled by default; enable
  // via config key "fuse_standard_attention_to_sdpa" = "true" to collapse the
  // single-branch (DUS-merged) attention pattern into a single SDPA op.
  NpuOptimizer& SetFuseStandardAttentionToSDPA(bool enable) {
    fuse_standard_attention_to_sdpa_ = enable;
    return *this;
  }

  // Runs all currently-enabled passes on |model|.
  void Run(const std::shared_ptr<ov::Model>& model) const;

 private:
  bool eliminate_matmul_fq_ = false;
  bool cast_integer_sign_to_float_ = true;
  bool fuse_split_attention_to_sdpa_ = true;
  bool sdpa_pad_kv_to_alignment_ = true;
  bool fuse_standard_attention_to_sdpa_ = true;
};

}  // namespace openvino
}  // namespace litert

#endif  // LITERT_VENDORS_INTEL_OPENVINO_COMPILER_NPU_OPTIMIZER_H_
