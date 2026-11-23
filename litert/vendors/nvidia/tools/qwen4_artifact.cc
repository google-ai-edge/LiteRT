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

// qwen4_artifact: a whole Qwen 4 (qwen4_exp) text model at reduced sizes with
// random weights, for checking the NVIDIA backend end to end without a Qwen 4
// export. Every component of the architecture, in the reference's order
// (sglang srt/models/qwen4_exp.py) and in the op spelling of the one export
// of this family there is (Qwen 3.8 27B, litert-torch's Qwen3_5Static
// recipes: FULLY_CONNECTED, odml.rms_norm, odml.swiglu, odml.short_conv_step,
// the gated_delta_update custom op, plain elementwise ops):
//  - token embedding, the hyper-connection streams (hc = 4 copies);
//  - per-layer embeddings (PLE) on layers 1 and 4: the n-gram ids of each
//    token and its two predecessors (the n-gram context is model state,
//    kv_cache_ngram), their table rows, the key/value projections, the
//    grouped norms, the sign-sqrt-sigmoid gate and the causal convolution at
//    dilation 3 (odml.short_conv_step with 10 taps, the dilated ones nonzero:
//    its 9-column state is exactly the reference's);
//  - layers 1-3 Gated DeltaNet (Qwen 3.8's ops; output gate sigmoid as
//    Qwen 4's config says, or silu with --gdn_gate=silu), layer 4 QSA
//    attention (odml.qsa_indexer_update, odml.cache_update,
//    odml.sparse_sdpa_transposed) with the attention output gate;
//  - around every mixer and feed-forward the hyper-connection mix (grouped
//    norm, low-rank sigmoid gates, mean over the streams) and combine
//    (injection weights);
//  - dense SwiGLU feed-forward (the vLLM reference's num_experts = 0);
//  - the final mixer and the head.
// The n-gram hash is the reference's (ids = ((t0 m0) ^ (t-1 m1) [^ (t-2 m2)])
// mod prime + offset, with the end-of-sequence rule) in int32: at these
// sizes every product fits, the reference's int64 does not change the ops.
// The hash and the table lookups of every PLE layer come first (sglang's
// forward prepares its PLE batch first; LiteRT's default partitioning follows
// the op order, and lookups between the layers would cut the GPU engine at
// every PLE layer), and stay on the CPU with the tables.
//
// USAGE
//   qwen4_artifact --write_model=FILE.tflite [--gdn_gate=sigmoid|silu]
//   qwen4_artifact --model=FILE --accelerator=cpu|npu [--runtime_libs=DIR]
//                  --out_dir=DIR
//   qwen4_artifact --compare --cpu_dir=DIR --npu_dir=DIR [--tolerance=T]
//                  [--strict_selection=false]
// qwen4_artifact_check.sh runs all of it for both output gates and both
// activation types.
// Signatures prefill_16 and decode share the states (kv_cache_*), bound in
// place on the NPU as LiteRT-LM binds them.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "litert/c/litert_common.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/experimental/custom_ops/gated_delta_net/gated_delta_update_litert_custom_op.h"
#include "litert/vendors/nvidia/tools/qsa_decompositions.h"
#include "litert/vendors/nvidia/tools/tflite_builder.h"
#include "tflite/schema/schema_generated.h"

ABSL_FLAG(std::string, write_model, "", "Write the model to this file.");
ABSL_FLAG(std::string, gdn_gate, "sigmoid",
          "The Gated DeltaNet output gate: sigmoid (Qwen 4) or silu.");
ABSL_FLAG(std::string, model, "", "The model to run.");
ABSL_FLAG(std::string, accelerator, "cpu", "cpu, or npu (the NVIDIA plugin).");
ABSL_FLAG(std::string, runtime_libs, "",
          "The directory of the NVIDIA compiler plugin and dispatch library.");
ABSL_FLAG(std::string, out_dir, "", "Where a run writes its outputs.");
ABSL_FLAG(int, decode_steps, 12, "Decode steps after the prompt.");
ABSL_FLAG(std::string, weights, "fp32",
          "The FULLY_CONNECTED weights of --write_model: fp32, or int8 (per "
          "output channel, as an export's weight-only quantization).");
ABSL_FLAG(
    bool, ple, true,
    "Whether the PLE layers have their per-layer embeddings (false: "
    "none, to time them by difference; for --write_model and --compare).");
ABSL_FLAG(bool, compare, false, "Compare the outputs of two runs.");
ABSL_FLAG(std::string, cpu_dir, "", "The CPU run's outputs.");
ABSL_FLAG(std::string, npu_dir, "", "The NPU run's outputs.");
ABSL_FLAG(bool, strict_selection, true,
          "Whether a QSA selection that differs fails the comparison (with "
          "BF16 activations, rounding can move a selection between two blocks "
          "of nearly equal scores: reported only).");
ABSL_FLAG(double, tolerance, 2e-2,
          "Largest error allowed, relative to the largest magnitude of an "
          "output.");

namespace {

using litert::Error;
using litert::Expected;
using namespace litert::nvidia::artifact;  // NOLINT: shapes and builders

#ifndef QWEN4_ARTIFACT_WIDE
// The reduced Qwen 4: hidden kWidth (256, kHidden of qsa_decompositions.h),
// hc 4 streams, low rank 32; Gated DeltaNet with 2 key heads and 4 value heads
// of 128 (the fused layer's head size); PLE of 16 heads of 16 values; the QSA
// layer of qsa_decompositions.h; SwiGLU of 512.
constexpr int kWidth = kHidden;
constexpr int kLowRank = 32;
constexpr int kNumLayers = 4;
constexpr bool kIsGdn[kNumLayers] = {true, true, true, false};
constexpr bool kHasPle[kNumLayers] = {true, false, false, true};
constexpr int kGdnKeyHeads = 2;
constexpr int kGdnValueHeads = 4;
constexpr int kInter = 512;
#else
// qwen4_artifact_wide, for timing: the real model's widths (hidden 2560, hc 4,
// low rank 320, Gated DeltaNet 16 key and 32 value heads of 128, PLE 16 heads
// of 160), a SwiGLU of 5120 (about the bytes of the experts a token reads,
// 10 of 512 plus the shared one), two Gated DeltaNet layers, PLE on the first.
constexpr int kWidth = 2560;
constexpr int kLowRank = 320;
constexpr int kNumLayers = 2;
constexpr bool kIsGdn[kNumLayers] = {true, true};
constexpr bool kHasPle[kNumLayers] = {true, false};
constexpr int kGdnKeyHeads = 16;
constexpr int kGdnValueHeads = 32;
constexpr int kInter = 5120;
#endif
constexpr int kVocab = 512;
constexpr int kHc = 4;
constexpr int kStreams = kHc * kWidth;
constexpr int kGdnDim = 128;
constexpr int kKeyDim = kGdnKeyHeads * kGdnDim;
constexpr int kValueDim = kGdnValueHeads * kGdnDim;
constexpr int kConvDim = 2 * kKeyDim + kValueDim;
constexpr int kConvTaps = 4;
constexpr int kNgram = 3;
constexpr int kPleHeads = 16;  // (kNgram - 1) * 8
constexpr int kPleDim = kWidth / kPleHeads;
constexpr int kPleTaps = (kConvTaps - 1) * kNgram + 1;  // 10, dilation 3
constexpr int kEos = 1;
constexpr int kRows = 16;  // prefill chunk
constexpr int kPrompt = 40;
// The attention projection: q, its gate, k, v, the indexer's q and k.
constexpr int kAttnProjection = 2 * kQueryHeads * kDepth +
                                2 * kKvHeads * kDepth +
                                kIndexHeads * kIndexDim + kIndexDim;

std::string StateName(const std::string& kind, int layer) {
  return absl::StrCat("kv_cache_", kind, "_", layer);
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

std::vector<float> Ones(size_t n, float jitter, Random& random) {
  std::vector<float> values(n);
  for (auto& v : values) v = 1.0f + jitter * random.Next();
  return values;
}

std::vector<float> Matrix(Random& random, int rows, int columns,
                          float gain = 1.0f) {
  return random.Values(static_cast<size_t>(rows) * columns,
                       gain / std::sqrt(static_cast<float>(columns)));
}

struct HcWeights {
  std::vector<float> norm, down, up, inject;  // norm [hc, H] (1 + w)
};

HcWeights MakeHc(Random& random) {
  HcWeights w;
  w.norm = Ones(kStreams, 0.2f, random);
  w.down = Matrix(random, kLowRank, kStreams, 2.0f);
  w.up = Matrix(random, kStreams, kLowRank, 2.0f);
  w.inject = Matrix(random, kHc, kStreams, 2.0f);
  return w;
}

struct PleWeights {
  std::vector<float> key, value, norm_key, norm_query, norm_conv, conv, table;
  std::vector<int32_t> sizes, offsets;  // per head
  int32_t multipliers[kNgram];
  float scale;
};

bool IsPrime(int n) {
  if (n < 2) return false;
  for (int d = 2; d * d <= n; ++d) {
    if (n % d == 0) return false;
  }
  return true;
}

PleWeights MakePle(Random& random, int index) {
  PleWeights w;
  w.key = Matrix(random, kStreams, kWidth);
  w.value = Matrix(random, kWidth, kWidth);
  w.norm_key = Ones(kStreams, 0.2f, random);
  w.norm_query = Ones(kStreams, 0.2f, random);
  w.norm_conv = Ones(kStreams, 0.2f, random);
  // The 4 taps at dilation 3 in a 10-tap kernel: oldest first.
  w.conv.assign(static_cast<size_t>(kStreams) * kPleTaps, 0.0f);
  for (int c = 0; c < kStreams; ++c) {
    for (int k = 0; k < kConvTaps; ++k) {
      w.conv[static_cast<size_t>(c) * kPleTaps + k * kNgram] =
          0.5f * random.Next();
    }
  }
  // Each head's rows: successive primes (the reference's sizes are the
  // primes from 20M on, numbered across the layers' heads).
  int candidate = 101 + 400 * index;
  int32_t offset = 0;
  for (int h = 0; h < kPleHeads; ++h) {
    while (!IsPrime(candidate)) ++candidate;
    w.sizes.push_back(candidate);
    w.offsets.push_back(offset);
    offset += candidate;
    ++candidate;
  }
  for (int i = 0; i < kNgram; ++i) {
    w.multipliers[i] =
        2 * ((1234 + 10007 * index + 40503 * (i + 1)) % 4000) + 1;
  }
  w.table = random.Values(static_cast<size_t>(offset) * kPleDim, 1.0f);
  w.scale = 0.75f;
  return w;
}

struct LayerWeights {
  HcWeights attn_hc, mlp_hc;
  bool ple = false;
  PleWeights ple_weights;
  // Gated DeltaNet.
  std::vector<float> qkv, z, b, a, a_bias, decay, conv, q_norm, k_norm,
      out_norm, out;
  // QSA attention.
  std::vector<float> projection, o, k_norm_idx, cos, sin;
  // Feed-forward.
  std::vector<float> gate_up, down;
};

struct Weights {
  std::vector<float> embedding, head;
  std::vector<LayerWeights> layers;
  HcWeights final_mixer;
};

Weights MakeWeights() {
  Weights w;
  Random random(4242);
  w.embedding = random.Values(static_cast<size_t>(kVocab) * kWidth, 1.0f);
  w.head = Matrix(random, kVocab, kWidth);
  int ple_index = 0;
  for (int l = 0; l < kNumLayers; ++l) {
    LayerWeights lw;
    lw.attn_hc = MakeHc(random);
    lw.mlp_hc = MakeHc(random);
    lw.ple = kHasPle[l] && absl::GetFlag(FLAGS_ple);
    if (lw.ple) lw.ple_weights = MakePle(random, ple_index++);
    if (kIsGdn[l]) {
      lw.qkv = Matrix(random, kConvDim, kWidth);
      lw.z = Matrix(random, kValueDim, kWidth);
      lw.b = Matrix(random, kGdnValueHeads, kWidth);
      lw.a = Matrix(random, kGdnValueHeads, kWidth);
      lw.a_bias = random.Values(kGdnValueHeads, 0.5f);
      lw.decay.resize(kGdnValueHeads);
      for (auto& d : lw.decay) d = -std::exp(0.5f * random.Next());
      lw.conv = random.Values(static_cast<size_t>(kConvDim) * kConvTaps, 0.5f);
      // The l2 norms of q and k as RMS norms (Qwen 3.8's export): k / |k| is
      // rms_norm(k) / sqrt(d), q also scaled by d^-0.5.
      lw.q_norm.assign(kGdnDim, 1.0f / kGdnDim);
      lw.k_norm.assign(kGdnDim, 1.0f / std::sqrt(static_cast<float>(kGdnDim)));
      lw.out_norm = Ones(kGdnDim, 0.2f, random);
      lw.out = Matrix(random, kWidth, kValueDim, 0.5f);
    } else {
      lw.projection = Matrix(random, kAttnProjection, kWidth);
      lw.o = Matrix(random, kWidth, kQueryHeads * kDepth, 0.5f);
      lw.k_norm_idx = Ones(kIndexDim, 0.2f, random);
      lw.cos.resize(kBlocks * kRotary);
      lw.sin.resize(kBlocks * kRotary);
      for (int b = 0; b < kBlocks; ++b) {
        for (int i = 0; i < kRotary; ++i) {
          const double angle =
              4.0 * b * std::pow(10000.0, -2.0 * (i % (kRotary / 2)) / kRotary);
          lw.cos[b * kRotary + i] = static_cast<float>(std::cos(angle));
          lw.sin[b * kRotary + i] = static_cast<float>(std::sin(angle));
        }
      }
    }
    lw.gate_up = Matrix(random, 2 * kInter, kWidth);
    lw.down = Matrix(random, kWidth, kInter, 0.5f);
    w.layers.push_back(std::move(lw));
  }
  w.final_mixer = MakeHc(random);
  return w;
}

// ---------------------------------------------------------------------------
// Decompositions of the composites (shared per shape)
// ---------------------------------------------------------------------------

class Decompositions {
 public:
  explicit Decompositions(Builder& b) : b_(b) {}

  // odml.rms_norm over the last dim of `shape`.
  int32_t RmsNorm(const Shape& shape) {
    const std::string key = absl::StrCat("rms", ShapeKey(shape));
    if (auto it = cache_.find(key); it != cache_.end()) return it->second;
    const int32_t current = b_.Current();
    const int32_t graph = b_.NewSubgraph("odml.rms_norm.impl_" + key);
    const int32_t x = b_.Input(b_.F32("x", shape));
    const int32_t w = b_.Input(b_.F32("w", {shape.back()}));
    Shape reduced = shape;
    reduced.back() = 1;
    const int32_t mean =
        b_.Reduce(tflite::BuiltinOperator_MEAN, b_.MulF(x, x, shape),
                  {static_cast<int32_t>(shape.size()) - 1}, true, reduced);
    const int32_t inv = b_.Unary(
        tflite::BuiltinOperator_RSQRT,
        b_.AddF(mean, b_.ConstF32("eps", Shape(shape.size(), 1), {kEps}),
                reduced),
        TensorType_FLOAT32, reduced);
    b_.Output(b_.MulF(b_.MulF(x, inv, shape), w, shape));
    b_.Select(current);
    cache_[key] = graph;
    return graph;
  }

  // odml.swiglu over [1, n, 2N].
  int32_t Swiglu(int n, int gate) {
    const std::string key = absl::StrCat("swiglu", n, "_", gate);
    if (auto it = cache_.find(key); it != cache_.end()) return it->second;
    const int32_t current = b_.Current();
    const int32_t graph = b_.NewSubgraph("odml.swiglu.impl_" + key);
    const int32_t x = b_.Input(b_.F32("x", {1, n, 2 * gate}));
    const int32_t g = b_.Slice(x, {0, 0, 0}, {1, n, gate});
    const int32_t u = b_.Slice(x, {0, 0, gate}, {1, n, gate});
    const int32_t act = b_.MulF(g,
                                b_.Unary(tflite::BuiltinOperator_LOGISTIC, g,
                                         TensorType_FLOAT32, {1, n, gate}),
                                {1, n, gate});
    b_.Output(b_.MulF(act, u, {1, n, gate}));
    b_.Select(current);
    cache_[key] = graph;
    return graph;
  }

  // odml.short_conv_step(x [1, n, C], state [1, C, L - 1], w [C, 1, L]
  // [, valid [1]]) -> (silu(conv) [1, n, C], state'), Qwen 3.8's
  // decomposition: the taps over concat(state, x), oldest first; the new
  // state the L - 1 columns ending at the last valid row.
  int32_t ShortConv(int n, int channels, int taps, bool valid_input) {
    const std::string key =
        absl::StrCat("conv", n, "_", channels, "_", taps, valid_input);
    if (auto it = cache_.find(key); it != cache_.end()) return it->second;
    const int32_t current = b_.Current();
    const int32_t graph = b_.NewSubgraph("odml.short_conv_step.impl_" + key);
    const int C = channels, L = taps;
    const int32_t x = b_.Input(b_.F32("x", {1, n, C}));
    const int32_t state = b_.Input(b_.F32("state", {1, C, L - 1}));
    const int32_t w = b_.Input(b_.F32("w", {C, 1, L}));
    int32_t valid = -1;
    if (valid_input) valid = b_.Input(b_.I32("valid", {1}));
    const int32_t columns = b_.Transpose(x, {0, 2, 1}, {1, C, n});
    const int32_t full = b_.Concat({state, columns}, 2, {1, C, L - 1 + n});
    const int32_t wr = b_.Reshape(w, {C, L});
    int32_t sum = -1;
    for (int j = 0; j < L; ++j) {
      const int32_t tap = b_.MulF(b_.Slice(full, {0, 0, j}, {1, C, n}),
                                  b_.Slice(wr, {0, j}, {C, 1}), {1, C, n});
      sum = sum < 0 ? tap : b_.AddF(sum, tap, {1, C, n});
    }
    const int32_t act = b_.MulF(sum,
                                b_.Unary(tflite::BuiltinOperator_LOGISTIC, sum,
                                         TensorType_FLOAT32, {1, C, n}),
                                {1, C, n});
    b_.Output(b_.Transpose(act, {0, 2, 1}, {1, n, C}));
    std::vector<int32_t> iota(L - 1);
    for (int i = 0; i < L - 1; ++i) iota[i] = i;
    const int32_t start = valid_input ? valid : b_.OneI32("valid_n", n);
    const int32_t indices =
        b_.AddI(start, b_.ConstI32("iota", {L - 1}, iota), {L - 1});
    b_.Output(b_.Gather(full, indices, 2, {1, C, L - 1}));
    b_.Select(current);
    cache_[key] = graph;
    return graph;
  }

 private:
  static std::string ShapeKey(const Shape& shape) {
    std::string key;
    for (int32_t d : shape) absl::StrAppend(&key, "_", d);
    return key;
  }

  Builder& b_;
  std::map<std::string, int32_t> cache_;
};

// ---------------------------------------------------------------------------
// The model
// ---------------------------------------------------------------------------

class Model {
 public:
  Model(Builder& b, Decompositions& d, const Weights& w, int rows, bool prefill,
        bool sigmoid_gate)
      : b_(b),
        d_(d),
        w_(w),
        n_(rows),
        prefill_(prefill),
        sigmoid_gate_(sigmoid_gate),
        int8_weights_(absl::GetFlag(FLAGS_weights) == "int8") {}

  int32_t Const(const std::string& name, const Shape& shape,
                const std::vector<float>& values) {
    return b_.ConstF32(absl::StrCat(name, "_", counter_++), shape, values);
  }

  int32_t RmsNorm(int32_t x, const Shape& shape, const std::vector<float>& w) {
    const int32_t out = b_.F32(absl::StrCat("rms_", counter_++), shape);
    b_.Composite("odml.rms_norm", d_.RmsNorm(shape),
                 [](flexbuffers::Builder* a) { a->Float("epsilon", kEps); },
                 {x, Const("rms_w", {shape.back()}, w)}, {out});
    return out;
  }

  int32_t Fc(int32_t x, const std::vector<float>& weights, int out_dim,
             int in_dim, const std::vector<float>* bias = nullptr) {
    const int32_t weight =
        int8_weights_ ? b_.ConstInt8PerRow(absl::StrCat("fc_w_", counter_++),
                                           out_dim, in_dim, weights)
                      : Const("fc", {out_dim, in_dim}, weights);
    const int32_t out =
        b_.F32(absl::StrCat("fc_", counter_++), {1, n_, out_dim});
    auto& op =
        b_.Op(tflite::BuiltinOperator_FULLY_CONNECTED,
              {x, weight, bias ? Const("bias", {out_dim}, *bias) : -1}, {out});
    tflite::FullyConnectedOptionsT options;
    options.keep_num_dims = true;
    op.builtin_options.Set(options);
    return out;
  }

  int32_t Silu(int32_t x, const Shape& s) {
    return b_.MulF(
        x, b_.Unary(tflite::BuiltinOperator_LOGISTIC, x, TensorType_FLOAT32, s),
        s);
  }
  int32_t Sigmoid(int32_t x, const Shape& s) {
    return b_.Unary(tflite::BuiltinOperator_LOGISTIC, x, TensorType_FLOAT32, s);
  }
  int32_t Scale(int32_t x, float value, const Shape& s) {
    return b_.MulF(x, Const("scale", {1}, {value}), s);
  }

  // The grouped Gemma RMS norm of the hc streams [1, n, hc*H] with per-stream
  // weights: odml.rms_norm per stream (unit weight), times (1 + w) [hc, H].
  int32_t GroupedNorm(int32_t x, const std::vector<float>& weight) {
    const Shape s4 = {1, n_, kHc, kWidth};
    const int32_t normed =
        RmsNorm(b_.Reshape(x, s4), s4, std::vector<float>(kWidth, 1.0f));
    return b_.MulF(normed, Const("hc_norm", {kHc, kWidth}, weight), s4);
  }

  // The hyper-connection mix: the layer input [1, n, H] and the normed
  // streams [1, n, hc*H].
  std::pair<int32_t, int32_t> Mix(int32_t streams, const HcWeights& w) {
    const int32_t normed =
        b_.Reshape(GroupedNorm(streams, w.norm), {1, n_, kStreams});
    const Shape low = {1, n_, kLowRank};
    const int32_t t = Silu(
        Scale(Fc(normed, w.down, kLowRank, kStreams), 1.0f / kHc, low), low);
    const int32_t u =
        Sigmoid(Fc(t, w.up, kStreams, kLowRank), {1, n_, kStreams});
    const int32_t gated =
        b_.Reshape(b_.MulF(u, normed, {1, n_, kStreams}), {1, n_, kHc, kWidth});
    const int32_t mixed = b_.Reduce(tflite::BuiltinOperator_MEAN, gated, {2},
                                    false, {1, n_, kWidth});
    return {mixed, normed};
  }

  // The hyper-connection combine: streams + o * 2 sigmoid(inject / hc).
  int32_t Combine(int32_t streams, int32_t normed, int32_t o,
                  const HcWeights& w) {
    const Shape s = {1, n_, kHc};
    const int32_t inject = Scale(
        Sigmoid(Scale(Fc(normed, w.inject, kHc, kStreams), 1.0f / kHc, s), s),
        2.0f, s);
    const int32_t product =
        b_.MulF(b_.Reshape(o, {1, n_, 1, kWidth}),
                b_.Reshape(inject, {1, n_, kHc, 1}), {1, n_, kHc, kWidth});
    return b_.AddF(streams, b_.Reshape(product, {1, n_, kStreams}),
                   {1, n_, kStreams});
  }

  // The n-gram ids of every token: ctx [1, n + 2] (the context state, then
  // the tokens) -> ids [1, n, 16] of the PLE layer's tables.
  int32_t NgramIds(int32_t ctx, const PleWeights& p) {
    const Shape s = {1, n_};
    const int32_t t0 = b_.Slice(ctx, {0, 2}, s);
    const int32_t t1 = b_.Slice(ctx, {0, 1}, s);
    const int32_t t2 = b_.Slice(ctx, {0, 0}, s);
    const int32_t eos =
        b_.ConstI32(absl::StrCat("eos_", counter_++), {1}, {kEos});
    const int32_t either = b_.Binary(
        tflite::BuiltinOperator_LOGICAL_OR,
        b_.Compare(BuiltinOperator_EQUAL, t2, eos, s),
        b_.Compare(BuiltinOperator_EQUAL, t1, eos, s), TensorType_BOOL, s);
    const int32_t p2 = b_.Select(either, eos, t2, TensorType_INT32, s);
    const auto times = [&](int32_t t, int i) {
      return b_.MulI(
          t, b_.OneI32(absl::StrCat("m_", counter_++), p.multipliers[i]), s);
    };
    const auto xor_of = [&](int32_t a, int32_t c) {
      return b_.Binary(tflite::BuiltinOperator_BITWISE_XOR, a, c,
                       TensorType_INT32, s);
    };
    const int32_t mix2 = xor_of(times(t0, 0), times(t1, 1));
    const int32_t mix3 = xor_of(mix2, times(p2, 2));
    const int half = kPleHeads / 2;
    const auto heads = [&](int32_t mix, int first) {
      const Shape h = {1, n_, half};
      std::vector<int32_t> sizes(p.sizes.begin() + first,
                                 p.sizes.begin() + first + half);
      std::vector<int32_t> offsets(p.offsets.begin() + first,
                                   p.offsets.begin() + first + half);
      const int32_t rem = b_.Binary(
          tflite::BuiltinOperator_FLOOR_MOD, b_.Reshape(mix, {1, n_, 1}),
          b_.ConstI32(absl::StrCat("sizes_", counter_++), {half}, sizes),
          TensorType_INT32, h);
      return b_.AddI(
          rem,
          b_.ConstI32(absl::StrCat("offsets_", counter_++), {half}, offsets),
          h);
    };
    return b_.Concat({heads(mix2, 0), heads(mix3, half)}, 2,
                     {1, n_, kPleHeads});
  }

  // The rows of a PLE layer's tables for every token [1, n, 16, 160]: the
  // n-gram hash and the lookups, which stay on LiteRT's CPU with the tables.
  int32_t PleRows(int32_t ctx, const PleWeights& p) {
    const int rows = static_cast<int>(p.table.size() / kPleDim);
    const int32_t ids = NgramIds(ctx, p);
    const int32_t table = Const("ple_table", {rows, kPleDim}, p.table);
    return b_.Gather(table, ids, 0, {1, n_, kPleHeads, kPleDim});
  }

  // The PLE layer's output added to the streams, from its rows (PleRows).
  int32_t Ple(int32_t streams, int32_t looked, const PleWeights& p,
              int32_t conv_state, int32_t* conv_state_out, int layer,
              int32_t row_mask, int32_t valid) {
    const int32_t e =
        Scale(b_.Reshape(looked, {1, n_, kWidth}), p.scale, {1, n_, kWidth});
    const Shape s4 = {1, n_, kHc, kWidth};
    const auto group_norm = [&](int32_t x, const std::vector<float>& w) {
      return b_.MulF(RmsNorm(x, s4, std::vector<float>(kWidth, 1.0f)),
                     Const("ple_norm", {kHc, kWidth}, w), s4);
    };
    const int32_t key =
        group_norm(b_.Reshape(Fc(e, p.key, kStreams, kWidth), s4), p.norm_key);
    const int32_t query = group_norm(b_.Reshape(streams, s4), p.norm_query);
    const Shape g3 = {1, n_, kHc};
    const int32_t dot =
        Scale(b_.Reduce(tflite::BuiltinOperator_SUM, b_.MulF(key, query, s4),
                        {3}, false, g3),
              1.0f / std::sqrt(static_cast<float>(kWidth)), g3);
    const int32_t sign =
        b_.Unary(tflite::BuiltinOperator_SIGN, dot, TensorType_FLOAT32, g3);
    const int32_t magnitude = b_.Binary(
        tflite::BuiltinOperator_MAXIMUM,
        b_.Unary(tflite::BuiltinOperator_ABS, dot, TensorType_FLOAT32, g3),
        Const("ple_floor", {1}, {1e-6f}), TensorType_FLOAT32, g3);
    const int32_t gate =
        Sigmoid(b_.MulF(sign,
                        b_.Unary(tflite::BuiltinOperator_SQRT, magnitude,
                                 TensorType_FLOAT32, g3),
                        g3),
                g3);
    const int32_t value = Fc(e, p.value, kWidth, kWidth);
    const int32_t gv = b_.MulF(b_.Reshape(gate, {1, n_, kHc, 1}),
                               b_.Reshape(value, {1, n_, 1, kWidth}), s4);
    int32_t conv_in =
        b_.Reshape(group_norm(gv, p.norm_conv), {1, n_, kStreams});
    if (row_mask >= 0) {
      conv_in = b_.MulF(conv_in, row_mask, {1, n_, kStreams});
    }
    const int32_t conv_out =
        b_.F32(absl::StrCat("ple_conv_", counter_++), {1, n_, kStreams});
    *conv_state_out =
        b_.F32(StateName("ple", layer) + "_out", {1, kStreams, kPleTaps - 1});
    std::vector<int32_t> inputs = {
        conv_in, conv_state,
        Const("ple_conv_w", {kStreams, 1, kPleTaps}, p.conv)};
    if (valid >= 0) inputs.push_back(valid);
    b_.Composite("odml.short_conv_step",
                 d_.ShortConv(n_, kStreams, kPleTaps, valid >= 0),
                 [](flexbuffers::Builder* a) {
                   a->Int("conv_L_cache", kPleTaps);
                   a->Bool("is_gated", false);
                   a->Bool("use_silu", true);
                 },
                 inputs, {conv_out, *conv_state_out});
    const int32_t out =
        b_.AddF(b_.Reshape(gv, {1, n_, kStreams}), conv_out, {1, n_, kStreams});
    return b_.AddF(streams, out, {1, n_, kStreams});
  }

  // A Gated DeltaNet layer in Qwen 3.8's ops (decode: reshapes; prefill:
  // row masks, transposes and the valid length).
  int32_t GatedDeltaNet(int32_t x, const LayerWeights& w, int32_t conv_state,
                        int32_t state, int32_t* conv_out, int32_t* state_out,
                        int layer, int32_t row_mask, int32_t valid,
                        int32_t valid_scalar) {
    const int n = n_;
    int32_t qkv = Fc(x, w.qkv, kConvDim, kWidth);
    const int32_t b_proj = Fc(x, w.b, kGdnValueHeads, kWidth);
    int32_t mask3 = -1, mask4 = -1;
    if (prefill_) {
      // The layer's own casts of the row mask, as Qwen 3.8's export has them
      // (the fused layer's matcher takes the masks from inside its partition).
      mask3 =
          b_.Cast(b_.Reshape(row_mask_bool_, {1, n, 1}), TensorType_FLOAT32);
      qkv = b_.MulF(qkv, mask3, {1, n, kConvDim});
    }
    const int32_t conv =
        b_.F32(absl::StrCat("gdn_conv_", counter_++), {1, n, kConvDim});
    *conv_out =
        b_.F32(StateName("c", layer) + "_out", {1, kConvDim, kConvTaps - 1});
    std::vector<int32_t> conv_inputs = {
        qkv, conv_state, Const("gdn_conv_w", {kConvDim, 1, kConvTaps}, w.conv)};
    if (prefill_) conv_inputs.push_back(valid);
    b_.Composite("odml.short_conv_step",
                 d_.ShortConv(n, kConvDim, kConvTaps, prefill_),
                 [](flexbuffers::Builder* a) {
                   a->Int("conv_L_cache", kConvTaps);
                   a->Bool("is_gated", false);
                   a->Bool("use_silu", true);
                 },
                 conv_inputs, {conv, *conv_out});
    int32_t q = b_.Reshape(b_.Slice(conv, {0, 0, 0}, {1, n, kKeyDim}),
                           {1, n, kGdnKeyHeads, kGdnDim});
    int32_t k = b_.Reshape(b_.Slice(conv, {0, 0, kKeyDim}, {1, n, kKeyDim}),
                           {1, n, kGdnKeyHeads, kGdnDim});
    int32_t v =
        b_.Reshape(b_.Slice(conv, {0, 0, 2 * kKeyDim}, {1, n, kValueDim}),
                   {1, n, kGdnValueHeads, kGdnDim});
    const Shape hv = {1, n, kGdnValueHeads};
    int32_t beta = Sigmoid(b_proj, hv);
    const int32_t a = Fc(x, w.a, kGdnValueHeads, kWidth, &w.a_bias);
    int32_t g = b_.MulF(
        b_.Unary(tflite::BuiltinOperator_LOG,
                 b_.AddF(b_.Unary(tflite::BuiltinOperator_EXP,
                                  b_.Binary(tflite::BuiltinOperator_MINIMUM, a,
                                            Const("g_cap", {}, {50.0f}),
                                            TensorType_FLOAT32, hv),
                                  TensorType_FLOAT32, hv),
                         Const("g_one", {}, {1.0f}), hv),
                 TensorType_FLOAT32, hv),
        Const("g_decay", {1, 1, kGdnValueHeads}, w.decay), hv);
    const Shape qs = {1, n, kGdnKeyHeads, kGdnDim};
    const Shape vs = {1, n, kGdnValueHeads, kGdnDim};
    if (prefill_) {
      mask4 =
          b_.Cast(b_.Reshape(row_mask_bool_, {1, n, 1, 1}), TensorType_FLOAT32);
      q = b_.MulF(q, mask4, qs);
      k = b_.MulF(k, mask4, qs);
      v = b_.MulF(v, mask4, vs);
      beta = b_.MulF(beta, mask3, hv);
      g = b_.MulF(g, mask3, hv);
    }
    q = RmsNorm(q, qs, w.q_norm);
    k = RmsNorm(k, qs, w.k_norm);
    const Shape qh = {1, kGdnKeyHeads, n, kGdnDim};
    const Shape vh = {1, kGdnValueHeads, n, kGdnDim};
    const Shape bh = {1, kGdnValueHeads, n};
    if (prefill_) {
      q = b_.Transpose(q, {0, 2, 1, 3}, qh);
      k = b_.Transpose(k, {0, 2, 1, 3}, qh);
      v = b_.Transpose(v, {0, 2, 1, 3}, vh);
      beta = b_.Transpose(beta, {0, 2, 1}, bh);
      g = b_.Transpose(g, {0, 2, 1}, bh);
    } else {
      q = b_.Reshape(q, qh);
      k = b_.Reshape(k, qh);
      v = b_.Reshape(v, vh);
      beta = b_.Reshape(beta, bh);
      g = b_.Reshape(g, bh);
    }
    int32_t o = b_.F32(absl::StrCat("gdn_out_", counter_++), vh);
    *state_out = b_.F32(StateName("r", layer) + "_out",
                        {1, kGdnValueHeads, kGdnDim, kGdnDim});
    std::vector<int32_t> gdn_inputs = {q, k, v, beta, g, state};
    if (prefill_) gdn_inputs.push_back(valid_scalar);
    b_.Custom("gated_delta_update", gdn_inputs, {o, *state_out});
    o = prefill_ ? b_.Transpose(o, {0, 2, 1, 3}, vs) : b_.Reshape(o, vs);
    const int32_t z = b_.Reshape(Fc(x, w.z, kValueDim, kWidth), vs);
    const int32_t normed = RmsNorm(o, vs, w.out_norm);
    const int32_t zs = Sigmoid(z, vs);
    const int32_t gated = sigmoid_gate_
                              ? b_.MulF(normed, zs, vs)
                              : b_.MulF(normed, b_.MulF(z, zs, vs), vs);
    return Fc(b_.Reshape(gated, {1, n, kValueDim}), w.out, kWidth, kValueDim);
  }

  // The QSA attention layer with its output gate.
  int32_t Attention(int32_t x, const LayerWeights& w, int32_t start,
                    int32_t mask, int32_t param, int32_t k_start,
                    int32_t v_start, const std::vector<int32_t>& state,
                    std::vector<int32_t>* state_out, int32_t* topk_out,
                    int layer, int32_t indexer, int32_t cache_update,
                    int32_t sparse) {
    const int n = n_;
    const int32_t projected = Fc(x, w.projection, kAttnProjection, kWidth);
    int offset = 0;
    const auto part = [&](int width) {
      const int32_t slice = b_.Slice(projected, {0, 0, offset}, {1, n, width});
      offset += width;
      return slice;
    };
    const auto heads = [&](int32_t t, int count) {
      return b_.Transpose(b_.Reshape(t, {1, n, count, kDepth}), {0, 2, 1, 3},
                          {1, count, n, kDepth});
    };
    const int32_t q = heads(part(kQueryHeads * kDepth), kQueryHeads);
    const int32_t gate = part(kQueryHeads * kDepth);
    const int32_t k = heads(part(kKvHeads * kDepth), kKvHeads);
    const int32_t v = heads(part(kKvHeads * kDepth), kKvHeads);
    const int32_t q_idx = b_.Reshape(part(kIndexHeads * kIndexDim),
                                     {1, n, kIndexHeads, kIndexDim});
    const int32_t k_idx = b_.Reshape(part(kIndexDim), {1, n, kIndexDim});
    const int32_t topk = b_.I32(absl::StrCat("topk_", layer), {1, n, kTopk});
    const int32_t ring_out =
        b_.F32(StateName("qsa_ring", layer) + "_out", {1, 4, kIndexDim});
    const int32_t pool_out =
        b_.F32(StateName("qsa_pool", layer) + "_out", {1, kBlocks, kIndexDim});
    const float index_scale = 1.0f / std::sqrt(static_cast<float>(kIndexDim));
    b_.Composite(
        "odml.qsa_indexer_update", indexer,
        [&](flexbuffers::Builder* a) {
          a->Int("block_topk", kTopk);
          a->Int("compress_ratio", 4);
          a->Float("rms_norm_eps", kEps);
          a->Float("scale", index_scale);
        },
        {q_idx, k_idx, state[0], state[1],
         Const("k_norm", {kIndexDim}, w.k_norm_idx),
         Const("block_cos", {1, kBlocks, kRotary}, w.cos),
         Const("block_sin", {1, kBlocks, kRotary}, w.sin), start, mask},
        {topk, ring_out, pool_out});
    const int32_t k_out = b_.F32(StateName("k", layer) + "_out",
                                 {1, kKvHeads, kCacheLen, kDepth});
    const int32_t v_out = b_.F32(StateName("v", layer) + "_out",
                                 {1, kKvHeads, kDepth, kCacheLen});
    b_.Composite("odml.cache_update", cache_update,
                 [&](flexbuffers::Builder* a) {
                   a->Int("cache_size", kCacheLen);
                   a->Int("head_size", kDepth);
                   a->Int("kv_cache_batch_size", kKvHeads);
                 },
                 {k, v, param, state[2], state[3], k_start, v_start},
                 {k_out, v_out});
    const int32_t attention =
        b_.F32(absl::StrCat("attention_", layer), {1, n, kQueryHeads, kDepth});
    b_.Composite("odml.sparse_sdpa_transposed", sparse,
                 [&](flexbuffers::Builder* a) {
                   a->Int("block_topk", kTopk);
                   a->Int("compress_ratio", 4);
                   a->Int("k_ts_idx", 2);
                   a->Float("scale",
                            1.0f / std::sqrt(static_cast<float>(kDepth)));
                   a->Int("v_ts_idx", 3);
                 },
                 {q, k_out, v_out, topk, mask}, {attention});
    *state_out = {ring_out, pool_out, k_out, v_out};
    *topk_out = topk;
    const Shape s = {1, n, kQueryHeads * kDepth};
    const int32_t gated =
        b_.MulF(b_.Reshape(attention, s), Sigmoid(gate, s), s);
    return Fc(gated, w.o, kWidth, kQueryHeads * kDepth);
  }

  int32_t FeedForward(int32_t x, const LayerWeights& w) {
    const int32_t gate_up = Fc(x, w.gate_up, 2 * kInter, kWidth);
    const int32_t act =
        b_.F32(absl::StrCat("swiglu_", counter_++), {1, n_, kInter});
    b_.Composite("odml.swiglu", d_.Swiglu(n_, kInter),
                 [](flexbuffers::Builder* a) { a->Int("gate_size", kInter); },
                 {gate_up}, {act});
    return Fc(act, w.down, kWidth, kInter);
  }

  void SetRowMask(int32_t bools) { row_mask_bool_ = bools; }

 private:
  Builder& b_;
  Decompositions& d_;
  const Weights& w_;
  const int n_;
  const bool prefill_;
  const bool sigmoid_gate_;
  const bool int8_weights_;
  int32_t row_mask_bool_ = -1;
  int counter_ = 0;
};

void AddSignature(Builder& b, Decompositions& d, const std::string& key,
                  int rows, bool prefill, const Weights& weights,
                  bool sigmoid_gate) {
  const int32_t indexer = AddIndexerDecomposition(
      b, rows, 1.0f / std::sqrt(static_cast<float>(kIndexDim)));
  const int32_t cache_update = AddCacheUpdateDecomposition(b, rows);
  const int32_t sparse = AddSparseDecomposition(
      b, rows, 1.0f / std::sqrt(static_cast<float>(kDepth)));
  const int32_t graph = b.NewSubgraph(key);
  Model m(b, d, weights, rows, prefill, sigmoid_gate);
  const int n = rows;
  std::vector<std::pair<std::string, int32_t>> inputs, outputs;
  const auto input = [&](const std::string& name, int32_t tensor) {
    b.Input(tensor);
    inputs.push_back({name, tensor});
    return tensor;
  };
  const auto output = [&](const std::string& name, int32_t tensor) {
    b.Output(tensor);
    outputs.push_back({name, tensor});
  };
  const int32_t tokens = input("input_tokens", b.I32("tokens", {1, n}));
  const int32_t positions = input("input_pos", b.I32("input_pos", {n}));
  const int32_t start = input("start_pos", b.I32("start_pos", {1}));
  const int32_t mask = input("mask", b.Bool("mask", {1, 1, n, kCacheLen}));
  const int32_t param = input("param_tensor", b.I32("param", {1, 1, 1, 7}));
  const int32_t k_start = input("k_start", b.I32("k_start", {4}));
  const int32_t v_start = input("v_start", b.I32("v_start", {4}));
  const int32_t ngram =
      input("kv_cache_ngram", b.I32("kv_cache_ngram", {1, kNgram - 1}));
  // The token embedding and the hc streams.
  const int32_t e =
      b.Gather(b.ConstF32("embedding", {kVocab, kWidth}, weights.embedding),
               tokens, 0, {1, n, kWidth});
  int32_t x = b.Concat({e, e, e, e}, 2, {1, n, kStreams});
  // The n-gram context: the context state, then the tokens.
  const int32_t ctx = b.Concat({ngram, tokens}, 1, {1, n + kNgram - 1});
  // Every PLE layer's rows before the first layer (as sglang's forward
  // prepares the PLE batch first): the CPU work in one place, ahead of one GPU
  // engine. LiteRT's default partitioning follows the op order, and lookups
  // between the layers would split the layers into an engine per PLE layer.
  std::vector<int32_t> ple_rows(kNumLayers, -1);
  for (int l = 0; l < kNumLayers; ++l) {
    if (weights.layers[l].ple) {
      ple_rows[l] = m.PleRows(ctx, weights.layers[l].ple_weights);
    }
  }
  // The valid rows of a prefill chunk (Qwen 3.8's: a position above the one
  // before it), their count: after the CPU work above, where the GPU engine
  // starts.
  int32_t row_mask = -1, valid = -1, valid_scalar = -1;
  if (prefill) {
    const int32_t later = b.Slice(positions, {1}, {n - 1});
    const int32_t earlier = b.Slice(positions, {0}, {n - 1});
    const int32_t rising =
        b.Reshape(b.Compare(BuiltinOperator_GREATER, later, earlier, {n - 1}),
                  {1, n - 1});
    const int32_t first = b.Constant("first_row", TensorType_BOOL, {1, 1},
                                     std::vector<uint8_t>{1});
    const int32_t bools = b.Concat({first, rising}, 1, {1, n});
    m.SetRowMask(bools);
    row_mask = b.Cast(b.Reshape(bools, {1, n, 1}), TensorType_FLOAT32);
    const int32_t count = b.Reduce(
        tflite::BuiltinOperator_SUM,
        b.Cast(b.Reshape(bools, {n}), TensorType_FLOAT32), {0}, false, {});
    valid_scalar = b.Cast(count, TensorType_INT32);
    valid = b.Reshape(valid_scalar, {1});
  }
  std::vector<std::pair<std::string, int32_t>> states;
  std::vector<int32_t> topks;
  for (int l = 0; l < kNumLayers; ++l) {
    const auto& lw = weights.layers[l];
    if (lw.ple) {
      const int32_t ple_state =
          input(StateName("ple", l),
                b.F32(StateName("ple", l), {1, kStreams, kPleTaps - 1}));
      int32_t ple_out = -1;
      x = m.Ple(x, ple_rows[l], lw.ple_weights, ple_state, &ple_out, l,
                row_mask, valid);
      states.push_back({StateName("ple", l), ple_out});
    }
    auto [a, normed] = m.Mix(x, lw.attn_hc);
    int32_t o = -1;
    if (kIsGdn[l]) {
      const int32_t conv_state =
          input(StateName("c", l),
                b.F32(StateName("c", l), {1, kConvDim, kConvTaps - 1}));
      const int32_t state = input(
          StateName("r", l),
          b.F32(StateName("r", l), {1, kGdnValueHeads, kGdnDim, kGdnDim}));
      int32_t conv_out = -1, state_out = -1;
      o = m.GatedDeltaNet(a, lw, conv_state, state, &conv_out, &state_out, l,
                          row_mask, valid, valid_scalar);
      states.push_back({StateName("c", l), conv_out});
      states.push_back({StateName("r", l), state_out});
    } else {
      std::vector<int32_t> state;
      for (const auto& [kind, shape] :
           std::vector<std::pair<const char*, Shape>>{
               {"qsa_ring", {1, 4, kIndexDim}},
               {"qsa_pool", {1, kBlocks, kIndexDim}},
               {"k", {1, kKvHeads, kCacheLen, kDepth}},
               {"v", {1, kKvHeads, kDepth, kCacheLen}}}) {
        state.push_back(
            input(StateName(kind, l), b.F32(StateName(kind, l), shape)));
      }
      std::vector<int32_t> state_out;
      int32_t topk = -1;
      o = m.Attention(a, lw, start, mask, param, k_start, v_start, state,
                      &state_out, &topk, l, indexer, cache_update, sparse);
      const char* kinds[] = {"qsa_ring", "qsa_pool", "k", "v"};
      for (int i = 0; i < 4; ++i)
        states.push_back({StateName(kinds[i], l), state_out[i]});
      topks.push_back(topk);
    }
    x = m.Combine(x, normed, o, lw.attn_hc);
    auto [f, normed2] = m.Mix(x, lw.mlp_hc);
    x = m.Combine(x, normed2, m.FeedForward(f, lw), lw.mlp_hc);
  }
  auto [h, unused] = m.Mix(x, weights.final_mixer);
  output("logits", m.Fc(h, weights.head, kVocab, kWidth));
  // The n-gram context's next value, the last two real tokens (on the CPU,
  // after the engine).
  const int32_t last = prefill ? valid : b.OneI32("one_row", 1);
  output("kv_cache_ngram",
         b.Gather(ctx,
                  b.AddI(last, b.ConstI32("ngram_iota", {kNgram - 1}, {0, 1}),
                         {kNgram - 1}),
                  1, {1, kNgram - 1}));
  for (size_t i = 0; i < topks.size(); ++i)
    output(absl::StrCat("topk_", i), topks[i]);
  for (const auto& [name, tensor] : states) output(name, tensor);
  b.Signature(key, graph, inputs, outputs);
}

std::string BuildModel(bool sigmoid_gate) {
  const Weights weights = MakeWeights();
  Builder b;
  Decompositions d(b);
  AddSignature(b, d, "prefill_16", kRows, true, weights, sigmoid_gate);
  AddSignature(b, d, "decode", 1, false, weights, sigmoid_gate);
  return b.Pack();
}

// ---------------------------------------------------------------------------
// Running
// ---------------------------------------------------------------------------

Expected<void> WriteFile(const std::string& path, const void* data,
                         size_t size) {
  std::ofstream out(path, std::ios::binary);
  out.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
  if (!out) return Error(kLiteRtStatusErrorFileIO, "Cannot write " + path);
  return {};
}

Expected<std::string> ReadFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) return Error(kLiteRtStatusErrorFileIO, "Cannot open " + path);
  std::string bytes(static_cast<size_t>(in.tellg()), '\0');
  in.seekg(0);
  in.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  return bytes;
}

template <typename T>
Expected<void> Fill(litert::TensorBuffer& buffer,
                    const std::vector<T>& values) {
  LITERT_ASSIGN_OR_RETURN(auto lock,
                          litert::TensorBufferScopedLock::Create<T>(
                              buffer, litert::TensorBuffer::LockMode::kWrite));
  LITERT_ASSIGN_OR_RETURN(const size_t size, buffer.PackedSize());
  if (size != values.size() * sizeof(T)) {
    return Error(kLiteRtStatusErrorInvalidArgument, "Input size mismatch");
  }
  std::memcpy(lock.second, values.data(), size);
  return {};
}

Expected<std::string> Bytes(litert::TensorBuffer& buffer) {
  LITERT_ASSIGN_OR_RETURN(const size_t size, buffer.PackedSize());
  LITERT_ASSIGN_OR_RETURN(auto lock,
                          litert::TensorBufferScopedLock::Create<const uint8_t>(
                              buffer, litert::TensorBuffer::LockMode::kRead));
  return std::string(reinterpret_cast<const char*>(lock.second), size);
}

struct Step {
  bool prefill;
  int start;
  int valid;
};

std::vector<Step> Script(int decode_steps) {
  std::vector<Step> steps;
  for (int start = 0; start < kPrompt; start += kRows) {
    steps.push_back({true, start, std::min(kRows, kPrompt - start)});
  }
  for (int i = 0; i < decode_steps; ++i) {
    steps.push_back({false, kPrompt + i, 1});
  }
  return steps;
}

// The token sequence both backends are fed (teacher forcing): pseudo-random
// tokens with an end-of-sequence token now and then.
std::vector<int32_t> Tokens(int count) {
  std::vector<int32_t> tokens(count);
  uint32_t state = 99;
  for (int i = 0; i < count; ++i) {
    state = state * 1664525u + 1013904223u;
    tokens[i] = (state >> 9) % 13 == 0
                    ? kEos
                    : 2 + static_cast<int32_t>((state >> 8) % (kVocab - 2));
  }
  return tokens;
}

Expected<void> Run() {
  const bool npu = absl::GetFlag(FLAGS_accelerator) == "npu";
  std::vector<litert::EnvironmentOptions::Option> env_options;
  const std::string libs = absl::GetFlag(FLAGS_runtime_libs);
  if (npu) {
    env_options.push_back(
        {litert::EnvironmentOptions::Tag::kDispatchLibraryDir, libs.c_str()});
    env_options.push_back(
        {litert::EnvironmentOptions::Tag::kCompilerPluginLibraryDir,
         libs.c_str()});
  }
  LITERT_ASSIGN_OR_RETURN(
      auto env,
      litert::Environment::Create(litert::EnvironmentOptions(env_options)));
  LITERT_ASSIGN_OR_RETURN(auto options, litert::Options::Create());
  LITERT_RETURN_IF_ERROR(options.SetHardwareAccelerators(
      npu ? litert::HwAccelerators::kNpu | litert::HwAccelerators::kCpu
          : litert::HwAcceleratorSet(litert::HwAccelerators::kCpu)));
  LITERT_RETURN_IF_ERROR(
      litert::gated_delta_net::RegisterGatedDeltaNetCustomOps(options));
  LITERT_ASSIGN_OR_RETURN(
      auto model,
      litert::CompiledModel::Create(env, absl::GetFlag(FLAGS_model), options));
  LITERT_ASSIGN_OR_RETURN(const size_t prefill,
                          model.GetSignatureIndex("prefill_16"));
  LITERT_ASSIGN_OR_RETURN(const size_t decode,
                          model.GetSignatureIndex("decode"));
  std::vector<std::string> state_names;
  LITERT_ASSIGN_OR_RETURN(auto prefill_inputs,
                          model.GetSignatureInputNames(prefill));
  for (const auto& name : prefill_inputs) {
    if (name.rfind("kv_cache_", 0) == 0)
      state_names.push_back(std::string(name));
  }
  std::vector<litert::TensorBuffer> states;
  for (const auto& name : state_names) {
    LITERT_ASSIGN_OR_RETURN(auto buffer,
                            model.CreateInputBuffer("prefill_16", name));
    LITERT_ASSIGN_OR_RETURN(const size_t size, buffer.PackedSize());
    if (name == "kv_cache_ngram") {
      LITERT_RETURN_IF_ERROR(
          Fill(buffer, std::vector<int32_t>(size / 4, kEos)));
    } else {
      LITERT_RETURN_IF_ERROR(Fill(buffer, std::vector<uint8_t>(size, 0)));
    }
    states.push_back(std::move(buffer));
  }
  const auto steps = Script(absl::GetFlag(FLAGS_decode_steps));
  const std::vector<int32_t> sequence = Tokens(kPrompt + 64);
  const std::string out_dir = absl::GetFlag(FLAGS_out_dir);
  int index = 0;
  for (const Step& step : steps) {
    const size_t signature = step.prefill ? prefill : decode;
    const int n = step.prefill ? kRows : 1;
    LITERT_ASSIGN_OR_RETURN(auto input_names,
                            model.GetSignatureInputNames(signature));
    LITERT_ASSIGN_OR_RETURN(auto output_names,
                            model.GetSignatureOutputNames(signature));
    LITERT_ASSIGN_OR_RETURN(auto inputs, model.CreateInputBuffers(signature));
    LITERT_ASSIGN_OR_RETURN(auto outputs, model.CreateOutputBuffers(signature));
    const auto input = [&](const std::string& name) -> litert::TensorBuffer& {
      for (size_t i = 0; i < input_names.size(); ++i) {
        if (input_names[i] == name) return inputs[i];
      }
      ABSL_LOG(FATAL) << "No input " << name;
    };
    const auto output_index = [&](const std::string& name) -> size_t {
      for (size_t i = 0; i < output_names.size(); ++i) {
        if (output_names[i] == name) return i;
      }
      ABSL_LOG(FATAL) << "No output " << name;
    };
    std::vector<int32_t> tokens(n, kEos), positions(n, 0);
    for (int t = 0; t < step.valid; ++t) {
      tokens[t] = sequence[step.start + t];
      positions[t] = step.start + t;
    }
    LITERT_RETURN_IF_ERROR(Fill(input("input_tokens"), tokens));
    LITERT_RETURN_IF_ERROR(Fill(input("input_pos"), positions));
    LITERT_RETURN_IF_ERROR(
        Fill(input("start_pos"), std::vector<int32_t>{step.start}));
    std::vector<uint8_t> mask(static_cast<size_t>(n) * kCacheLen, 0);
    for (int t = 0; t < step.valid; ++t) {
      for (int c = 0; c <= step.start + t; ++c) mask[t * kCacheLen + c] = 1;
    }
    LITERT_RETURN_IF_ERROR(Fill(input("mask"), mask));
    LITERT_RETURN_IF_ERROR(
        Fill(input("param_tensor"),
             std::vector<int32_t>{step.start, step.start + step.valid, 0, 0, 0,
                                  0, 0}));
    LITERT_RETURN_IF_ERROR(
        Fill(input("k_start"), std::vector<int32_t>{0, 0, step.start, 0}));
    LITERT_RETURN_IF_ERROR(
        Fill(input("v_start"), std::vector<int32_t>{0, 0, 0, step.start}));
    for (size_t s = 0; s < states.size(); ++s) {
      LITERT_ASSIGN_OR_RETURN(input(state_names[s]), states[s].Duplicate());
      if (npu) {
        LITERT_ASSIGN_OR_RETURN(outputs[output_index(state_names[s])],
                                states[s].Duplicate());
      }
    }
    LITERT_RETURN_IF_ERROR(model.Run(signature, inputs, outputs));
    const std::string prefix =
        absl::StrCat(out_dir, "/step", index < 10 ? "0" : "", index);
    for (const auto& name : output_names) {
      if (name.rfind("kv_cache_", 0) == 0) continue;
      LITERT_ASSIGN_OR_RETURN(const std::string bytes,
                              Bytes(outputs[output_index(std::string(name))]));
      LITERT_RETURN_IF_ERROR(
          WriteFile(absl::StrCat(prefix, ".", std::string(name), ".raw"),
                    bytes.data(), bytes.size()));
    }
    for (size_t s = 0; s < states.size(); ++s) {
      if (!npu) {
        LITERT_ASSIGN_OR_RETURN(const std::string bytes,
                                Bytes(outputs[output_index(state_names[s])]));
        LITERT_RETURN_IF_ERROR(
            Fill(states[s], std::vector<uint8_t>(bytes.begin(), bytes.end())));
      }
      LITERT_ASSIGN_OR_RETURN(const std::string bytes, Bytes(states[s]));
      LITERT_RETURN_IF_ERROR(
          WriteFile(absl::StrCat(prefix, ".", state_names[s], ".raw"),
                    bytes.data(), bytes.size()));
    }
    ABSL_LOG(INFO) << prefix << (step.prefill ? " prefill" : " decode")
                   << " start " << step.start << " rows " << step.valid;
    ++index;
  }
  return {};
}

// ---------------------------------------------------------------------------
// Comparing
// ---------------------------------------------------------------------------

std::vector<float> Floats(const std::string& bytes) {
  std::vector<float> values(bytes.size() / 4);
  std::memcpy(values.data(), bytes.data(), values.size() * 4);
  return values;
}

// The logits of the real rows and every float state to the tolerance
// (relative to the largest magnitude), the n-gram context exactly, the QSA
// selections as sets. The QSA ring's slots past the incomplete block are not
// compared (LiteRT's CPU keeps a completed block's keys there).
Expected<bool> Compare() {
  const std::string cpu = absl::GetFlag(FLAGS_cpu_dir);
  const std::string npu = absl::GetFlag(FLAGS_npu_dir);
  const double tolerance = absl::GetFlag(FLAGS_tolerance);
  bool ok = true;
  const auto steps = Script(absl::GetFlag(FLAGS_decode_steps));
  for (size_t index = 0; index < steps.size(); ++index) {
    const Step& step = steps[index];
    const std::string prefix =
        absl::StrCat("/step", index < 10 ? "0" : "", index, ".");
    std::string line =
        absl::StrCat("step ", index, step.prefill ? " prefill" : " decode",
                     " start ", step.start, ":");
    const auto compare = [&](const std::string& name, size_t first,
                             size_t count) -> Expected<void> {
      LITERT_ASSIGN_OR_RETURN(const std::string a,
                              ReadFile(cpu + prefix + name + ".raw"));
      LITERT_ASSIGN_OR_RETURN(const std::string b,
                              ReadFile(npu + prefix + name + ".raw"));
      const auto x = Floats(a);
      const auto y = Floats(b);
      count = std::min({count, x.size() - first, y.size() - first});
      double largest = 0.0, error = 0.0;
      bool finite = true;
      for (size_t i = first; i < first + count; ++i) {
        finite &= std::isfinite(y[i]) || !std::isfinite(x[i]);
        largest = std::max(largest, std::fabs(static_cast<double>(x[i])));
        error = std::max(error, std::fabs(static_cast<double>(x[i]) - y[i]));
      }
      const double relative = largest > 0.0 ? error / largest : error;
      const bool fine = finite && x.size() == y.size() && relative <= tolerance;
      ok &= fine;
      absl::StrAppend(&line, " ", name, " ", relative, fine ? "" : " (!)");
      return {};
    };
    LITERT_RETURN_IF_ERROR(
        compare("logits", 0, static_cast<size_t>(step.valid) * kVocab));
    for (int l = 0; l < kNumLayers; ++l) {
      if (kHasPle[l] && absl::GetFlag(FLAGS_ple)) {
        LITERT_RETURN_IF_ERROR(compare(StateName("ple", l), 0, SIZE_MAX));
      }
      if (kIsGdn[l]) {
        LITERT_RETURN_IF_ERROR(compare(StateName("c", l), 0, SIZE_MAX));
        LITERT_RETURN_IF_ERROR(compare(StateName("r", l), 0, SIZE_MAX));
      } else {
        const int end = step.start + step.valid;
        LITERT_RETURN_IF_ERROR(
            compare(StateName("qsa_ring", l), 0,
                    static_cast<size_t>(end % 4) * kIndexDim));
        LITERT_RETURN_IF_ERROR(compare(StateName("qsa_pool", l), 0, SIZE_MAX));
      }
    }
    LITERT_ASSIGN_OR_RETURN(const std::string ca,
                            ReadFile(cpu + prefix + "kv_cache_ngram.raw"));
    LITERT_ASSIGN_OR_RETURN(const std::string na,
                            ReadFile(npu + prefix + "kv_cache_ngram.raw"));
    ok &= ca == na;
    absl::StrAppend(&line, " ngram ", ca == na ? "equal" : "differs (!)");
    LITERT_ASSIGN_OR_RETURN(const std::string ta,
                            ReadFile(cpu + prefix + "topk_0.raw"));
    LITERT_ASSIGN_OR_RETURN(const std::string tb,
                            ReadFile(npu + prefix + "topk_0.raw"));
    int differing = 0;
    for (int r = 0; r < step.valid; ++r) {
      std::vector<int32_t> x(kTopk), y(kTopk);
      std::memcpy(x.data(), ta.data() + r * kTopk * 4, kTopk * 4);
      std::memcpy(y.data(), tb.data() + r * kTopk * 4, kTopk * 4);
      std::sort(x.begin(), x.end());
      std::sort(y.begin(), y.end());
      differing += x != y;
    }
    if (absl::GetFlag(FLAGS_strict_selection)) ok &= differing == 0;
    absl::StrAppend(&line, " topk rows differing ", differing);
    std::printf("%s\n", line.c_str());
  }
  std::printf("%s\n", ok ? "MATCH" : "MISMATCH");
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  if (!absl::GetFlag(FLAGS_write_model).empty()) {
    const std::string gate = absl::GetFlag(FLAGS_gdn_gate);
    if (gate != "sigmoid" && gate != "silu") {
      ABSL_LOG(ERROR) << "--gdn_gate must be sigmoid or silu";
      return EXIT_FAILURE;
    }
    const std::string bytes = BuildModel(gate == "sigmoid");
    if (!WriteFile(absl::GetFlag(FLAGS_write_model), bytes.data(),
                   bytes.size())) {
      return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
  }
  if (absl::GetFlag(FLAGS_compare)) {
    auto ok = Compare();
    if (!ok) {
      ABSL_LOG(ERROR) << ok.Error().Message();
      return EXIT_FAILURE;
    }
    return *ok ? EXIT_SUCCESS : EXIT_FAILURE;
  }
  if (auto result = Run(); !result) {
    ABSL_LOG(ERROR) << result.Error().Message();
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
