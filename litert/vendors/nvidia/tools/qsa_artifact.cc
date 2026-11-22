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

// qsa_artifact: a model with the attention layers of Qwen 4 at reduced sizes
// and random weights, for checking the NVIDIA backend's QSA lowering end to
// end without a Qwen 4 export. Each of its two layers is an RMS norm, a fused
// projection to the queries, keys and values of the attention and of its
// sparse-attention indexer, odml.qsa_indexer_update, odml.cache_update,
// odml.sparse_sdpa_transposed reading the updated caches with the indexer's
// selection, an output projection and a residual add. Every composite has a
// decomposition in TFLite builtin ops (written here from the composites'
// contracts, ml_drift_delegate/delegate/composite/*_parser.h); LiteRT's CPU
// backend runs the XNNPACK delegate's own kernels for the QSA composites
// (tflite/delegates/xnnpack/qsa_delegate_kernel.cc) and the NVIDIA plugin
// its kernels, so the comparison is of two independent implementations.
// Signatures "prefill_16" and "decode" share the states, named as LiteRT-LM
// names caches (kv_cache_*), which LiteRT-LM binds in place.
//
// USAGE
//
//   qsa_artifact --write_model=FILE.tflite
//   qsa_artifact --model=FILE.tflite --accelerator=cpu --out_dir=DIR
//   qsa_artifact --model=FILE.tflite --accelerator=npu \
//                --runtime_libs=DIR --out_dir=DIR
//   qsa_artifact --compare --cpu_dir=DIR --npu_dir=DIR
//
// A run feeds a 40-token prompt in prefill chunks (the last one
// right-padded) and --decode_steps decode steps, with pseudo-random
// embeddings from a fixed seed, carrying every state between steps (in place
// on the NPU, as LiteRT-LM does; copied back on the CPU), and writes each
// step's hidden states, selections, indexer rings and pools. --compare
// requires the selections to agree and the states and hidden states to
// agree to --tolerance; the KV caches stay in each backend's own layout.
// qsa_artifact_check.sh runs the whole check.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
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
#include "litert/vendors/nvidia/tools/qsa_decompositions.h"
#include "litert/vendors/nvidia/tools/tflite_builder.h"
#include "tflite/schema/schema_generated.h"

ABSL_FLAG(std::string, write_model, "", "Write the model to this file.");
ABSL_FLAG(std::string, model, "", "The model to run.");
ABSL_FLAG(std::string, accelerator, "cpu", "cpu, or npu (the NVIDIA plugin).");
ABSL_FLAG(std::string, runtime_libs, "",
          "The directory of the NVIDIA compiler plugin and dispatch library.");
ABSL_FLAG(std::string, out_dir, "", "Where a run writes its outputs.");
ABSL_FLAG(int, decode_steps, 12, "Decode steps after the prompt.");
ABSL_FLAG(bool, compare, false, "Compare the outputs of two runs.");
ABSL_FLAG(std::string, cpu_dir, "", "The CPU run's outputs.");
ABSL_FLAG(std::string, npu_dir, "", "The NPU run's outputs.");
ABSL_FLAG(std::string, quant_layer, "",
          "Adds a third layer of attention over quantized KV caches (int4, "
          "uint8, int8 or fused_int4; see QuantLayerSpec) when writing the "
          "model; when comparing, the form of the model's layer.");
ABSL_FLAG(double, tolerance, 2e-2,
          "Largest error allowed, relative to the largest magnitude of an "
          "output.");

namespace {

using litert::Error;
using litert::Expected;

using namespace litert::nvidia::artifact;  // NOLINT: the shared shapes and
                                           // builders
constexpr int kPrefill = 16;
constexpr int kLayers = 2;
constexpr int kProjection = kQueryHeads * kDepth + 2 * kKvHeads * kDepth +
                            kIndexHeads * kIndexDim + kIndexDim;
constexpr int kPrompt = 40;

// ---------------------------------------------------------------------------
// The quantized attention layer (--quant_layer)
// ---------------------------------------------------------------------------

// A third layer of full attention over quantized KV caches, in one of the
// forms the general attention plugin takes
// (compiler/general_attention_plugin.h,
// LITERT_NVIDIA_TENSORRT_GENERAL_ATTENTION=1):
//   int4: float keys and values into packed INT4 caches (odml.cache_update),
//         odml.sdpa_transposed with a float query, an additive float mask, a
//         scale and a soft cap;
//   uint8: keys and values quantized to INT8 of their own scale into UINT8
//         caches, an INT8 query and a bool mask;
//   int8: the form of Gemma 4's local layers (INT8 query, keys and values on
//         the caches' INT8 grid, a bool mask) at depth 64 instead of 256;
//   fused_int4: odml.fused_sdpa_cache_update over packed INT4 ring caches of
//         32 slots with a float chunk, a float mask and a soft cap.
// The decompositions are TFLite builtin ops (QUANTIZE, DEQUANTIZE,
// DYNAMIC_UPDATE_SLICE, BATCH_MATMUL, SOFTMAX), which LiteRT's CPU runs.
constexpr int kQuantLayer = kLayers;  // its index
constexpr int kRing = 32;
constexpr float kQueryPrescale = 0.125f;  // 1 / sqrt(kDepth)
constexpr float kQueryScale = 0.0016f;
constexpr float kRowScale = 0.017f;
constexpr float kQuantSoftcap = 20.0f;

struct QuantSpec {
  std::string kind;
  tflite::TensorType cache_type = tflite::TensorType_INT8;
  int64_t zero_point = 0;
  float k_scale = 0.0f;
  float v_scale = 0.0f;
  int cache_len = kCacheLen;
  bool fused() const { return kind == "fused_int4"; }
};

std::optional<QuantSpec> QuantLayerSpec(const std::string& kind) {
  QuantSpec spec;
  spec.kind = kind;
  if (kind == "int4" || kind == "fused_int4") {
    spec.cache_type = tflite::TensorType_INT4;
    spec.k_scale = 0.25f;
    spec.v_scale = 0.25f;
    if (kind == "fused_int4") spec.cache_len = kRing;
  } else if (kind == "uint8") {
    spec.cache_type = tflite::TensorType_UINT8;
    spec.zero_point = 128;
    spec.k_scale = 0.02f;
    spec.v_scale = 0.025f;
  } else if (kind == "int8") {
    spec.k_scale = 0.02f;
    spec.v_scale = 0.025f;
  } else {
    return std::nullopt;
  }
  return spec;
}

int32_t QuantCache(Builder& b, const std::string& name, const QuantSpec& spec,
                   bool value) {
  const Shape shape = value ? Shape{1, kKvHeads, kDepth, spec.cache_len}
                            : Shape{1, kKvHeads, spec.cache_len, kDepth};
  return b.Quantized(name, spec.cache_type, shape,
                     value ? spec.v_scale : spec.k_scale, spec.zero_point);
}

// odml.cache_update into the quantized caches. The keys and values are float
// (int4) or INT8 (uint8: kRowScale; int8: the caches' scales).
int32_t AddQuantCacheUpdateDecomposition(Builder& b, int rows,
                                         const QuantSpec& spec) {
  const int32_t graph =
      b.NewSubgraph(absl::StrCat("odml.cache_update.quant_impl_", rows));
  const Shape row_shape = {1, kKvHeads, rows, kDepth};
  const auto update = [&](const char* name, float cache_scale) {
    if (spec.kind == "int4") return b.Input(b.F32(name, row_shape));
    return b.Input(b.Quantized(name, tflite::TensorType_INT8, row_shape,
                               spec.kind == "uint8" ? kRowScale : cache_scale,
                               0));
  };
  const int32_t k = update("k", spec.k_scale);
  const int32_t v = update("v", spec.v_scale);
  b.Input(b.I32("param_tensor", {1, 1, 1, 7}));
  const int32_t k_cache = b.Input(QuantCache(b, "k_cache", spec, false));
  const int32_t v_cache = b.Input(QuantCache(b, "v_cache", spec, true));
  const int32_t k_start = b.Input(b.I32("k_start", {4}));
  const int32_t v_start = b.Input(b.I32("v_start", {4}));
  // The rows on the caches' grid: as they are when they already are on it.
  int32_t k_rows = k;
  int32_t v_columns = 0;
  if (spec.kind == "int8") {
    v_columns = b.Quantized("v_columns", tflite::TensorType_INT8,
                            {1, kKvHeads, kDepth, rows}, spec.v_scale, 0);
    b.Op(tflite::BuiltinOperator_TRANSPOSE,
         {v, b.ConstI32("perm", {4}, {0, 1, 3, 2})}, {v_columns});
  } else {
    const int32_t k_float = spec.kind == "int4" ? k : b.Dequantize(k);
    const int32_t v_float = spec.kind == "int4" ? v : b.Dequantize(v);
    k_rows =
        b.Quantize(k_float, spec.cache_type, spec.k_scale, spec.zero_point);
    v_columns = b.Quantize(
        b.Transpose(v_float, {0, 1, 3, 2}, {1, kKvHeads, kDepth, rows}),
        spec.cache_type, spec.v_scale, spec.zero_point);
  }
  const int32_t k_out = QuantCache(b, "k_cache_out", spec, false);
  b.Op(tflite::BuiltinOperator_DYNAMIC_UPDATE_SLICE, {k_cache, k_rows, k_start},
       {k_out});
  const int32_t v_out = QuantCache(b, "v_cache_out", spec, true);
  b.Op(tflite::BuiltinOperator_DYNAMIC_UPDATE_SLICE,
       {v_cache, v_columns, v_start}, {v_out});
  b.Output(k_out);
  b.Output(v_out);
  return graph;
}

// odml.sdpa_transposed over the quantized caches: q [1, Hq, n, D] float
// (int4) or INT8 (kQueryScale), the mask float (int4, additive) or bool
// (fill -10000), the scale and soft cap of int4.
int32_t AddQuantSdpaDecomposition(Builder& b, int rows, const QuantSpec& spec) {
  const int32_t graph =
      b.NewSubgraph(absl::StrCat("odml.sdpa_transposed.quant_impl_", rows));
  const int n = rows, L = spec.cache_len, R = kGroup * rows;
  const bool float_forms = spec.kind == "int4";
  const Shape q_shape = {1, kQueryHeads, n, kDepth};
  const int32_t q =
      b.Input(float_forms ? b.F32("query", q_shape)
                          : b.Quantized("query", tflite::TensorType_INT8,
                                        q_shape, kQueryScale, 0));
  const int32_t k = b.Input(QuantCache(b, "key_cache", spec, false));
  const int32_t v = b.Input(QuantCache(b, "value_cache", spec, true));
  const int32_t mask = b.Input(float_forms ? b.F32("mask", {1, 1, n, L})
                                           : b.Bool("mask", {1, 1, n, L}));
  b.Input(b.I32("param_tensor", {1, 1, 1, 7}));
  const int32_t q_float = float_forms ? q : b.Dequantize(q);
  int32_t logits = b.BatchMatMul(b.Reshape(q_float, {1, kKvHeads, R, kDepth}),
                                 b.Dequantize(k), true, {1, kKvHeads, R, L});
  if (float_forms) {
    logits =
        b.Softcap(b.MulF(logits, b.ConstF32("scale", {1}, {kQueryPrescale}),
                         {1, kKvHeads, R, L}),
                  kQuantSoftcap, {1, kKvHeads, R, L});
  }
  const int32_t per_head = b.Reshape(logits, {1, kQueryHeads, n, L});
  const int32_t masked =
      float_forms
          ? b.AddF(per_head, mask, {1, kQueryHeads, n, L})
          : b.Select(mask, per_head, b.ConstF32("fill", {1}, {-10000.0f}),
                     TensorType_FLOAT32, {1, kQueryHeads, n, L});
  const int32_t probs = b.Reshape(b.Softmax(masked), {1, kKvHeads, R, L});
  const int32_t out =
      b.BatchMatMul(probs, b.Dequantize(v), true, {1, kKvHeads, R, kDepth});
  b.Output(b.Reshape(out, q_shape));
  return graph;
}

// odml.fused_sdpa_cache_update over the INT4 ring: attention over the ring
// followed by the chunk (on the ring's grid), soft-capped, with the additive
// mask [1, 1, n, W + n] (row r of a KV head reads mask row r % n); then the
// valid tokens t >= max(0, valid - W) go to slots (p0 + t) % W.
int32_t AddQuantFusedDecomposition(Builder& b, int rows,
                                   const QuantSpec& spec) {
  const int32_t graph = b.NewSubgraph(
      absl::StrCat("odml.fused_sdpa_cache_update.quant_impl_", rows));
  const int n = rows, W = spec.cache_len, R = kGroup * rows, H = kKvHeads;
  const int32_t q = b.Input(b.F32("query", {1, H, R, kDepth}));
  const int32_t k_cache = b.Input(QuantCache(b, "k_cache", spec, false));
  const int32_t v_cache = b.Input(QuantCache(b, "v_cache", spec, true));
  const int32_t k_new = b.Input(b.F32("k", {1, H, n, kDepth}));
  const int32_t v_new = b.Input(b.F32("v", {1, H, kDepth, n}));
  const int32_t mask = b.Input(b.F32("mask", {1, 1, n, W + n}));
  const int32_t param = b.Input(b.I32("param_tensor", {1, 1, 1, 7}));
  const int32_t k_grid = b.Dequantize(
      b.Quantize(k_new, spec.cache_type, spec.k_scale, spec.zero_point));
  const int32_t v_grid = b.Dequantize(
      b.Quantize(v_new, spec.cache_type, spec.v_scale, spec.zero_point));
  const int32_t k_old = b.Dequantize(k_cache);
  const int32_t v_old = b.Dequantize(v_cache);
  const int32_t keys = b.Concat({k_old, k_grid}, 2, {1, H, W + n, kDepth});
  const int32_t values = b.Concat({v_old, v_grid}, 3, {1, H, kDepth, W + n});
  const int32_t logits =
      b.Softcap(b.BatchMatMul(q, keys, true, {1, H, R, W + n}), kQuantSoftcap,
                {1, H, R, W + n});
  const int32_t masked =
      b.AddF(b.Reshape(logits, {H * kGroup, n, W + n}),
             b.Reshape(mask, {1, n, W + n}), {H * kGroup, n, W + n});
  const int32_t probs = b.Reshape(b.Softmax(masked), {1, H, R, W + n});
  b.Output(b.BatchMatMul(probs, values, true, {1, H, R, kDepth}));
  // The ring write as a product with a one-hot scatter matrix [n, W].
  const int32_t params = b.Reshape(param, {7});
  const int32_t p0 = b.Slice(params, {0}, {1});
  const int32_t p1 = b.Slice(params, {1}, {1});
  const int32_t zero = b.OneI32("zero", 0);
  const int32_t valid =
      b.MinI(b.MaxI(b.SubI(p1, p0, {1}), zero, {1}), b.OneI32("rows", n), {1});
  const int32_t first =
      b.MaxI(b.SubI(valid, b.OneI32("ring", W), {1}), zero, {1});
  const int32_t t = b.Iota("t", n);
  const int32_t written_rows =
      b.And(b.And(b.Compare(BuiltinOperator_GREATER_EQUAL, t, first, {n}),
                  b.Compare(BuiltinOperator_LESS, t, valid, {n}), {n}),
            b.Compare(BuiltinOperator_GREATER_EQUAL, p0, zero, {1}), {n});
  const int32_t slot =
      b.Binary(tflite::BuiltinOperator_FLOOR_MOD, b.AddI(p0, t, {n}),
               b.OneI32("ring2", W), TensorType_INT32, {n});
  const int32_t hot = b.F32("slot_hot", {n, W});
  {
    auto& op = b.Op(tflite::BuiltinOperator_ONE_HOT,
                    {slot, b.ScalarI32("depth", W), b.ScalarF32("on", 1.0f),
                     b.ScalarF32("off", 0.0f)},
                    {hot});
    tflite::OneHotOptionsT options;
    options.axis = -1;
    op.builtin_options.Set(options);
  }
  const int32_t scatter = b.MulF(
      hot, b.Reshape(b.Cast(written_rows, TensorType_FLOAT32), {n, 1}), {n, W});
  const int32_t written =
      b.Reduce(tflite::BuiltinOperator_REDUCE_MAX, scatter, {0}, false, {W});
  const int32_t keep =
      b.AddF(b.MulF(written, b.ConstF32("minus_one", {1}, {-1.0f}), {W}),
             b.ConstF32("one", {1}, {1.0f}), {W});
  const int32_t k_ring =
      b.AddF(b.MulF(k_old, b.Reshape(keep, {1, 1, W, 1}), {1, H, W, kDepth}),
             b.BatchMatMul(
                 b.Reshape(b.Transpose(scatter, {1, 0}, {W, n}), {1, 1, W, n}),
                 k_grid, false, {1, H, W, kDepth}),
             {1, H, W, kDepth});
  const int32_t v_ring =
      b.AddF(b.MulF(v_old, b.Reshape(keep, {1, 1, 1, W}), {1, H, kDepth, W}),
             b.BatchMatMul(v_grid, b.Reshape(scatter, {1, 1, n, W}), false,
                           {1, H, kDepth, W}),
             {1, H, kDepth, W});
  b.Output(b.Quantize(k_ring, spec.cache_type, spec.k_scale, spec.zero_point));
  b.Output(b.Quantize(v_ring, spec.cache_type, spec.v_scale, spec.zero_point));
  return graph;
}

struct LayerWeights {
  std::vector<float> norm, projection, output, k_norm, cos, sin;
};

std::vector<LayerWeights> MakeWeights() {
  std::vector<LayerWeights> layers(kLayers + 1);
  for (int l = 0; l < kLayers + 1; ++l) {
    Random random(100 + l);
    auto& w = layers[l];
    w.norm = random.Values(kHidden, 0.2f);
    for (auto& x : w.norm) x += 1.0f;
    w.projection = random.Values(static_cast<size_t>(kProjection) * kHidden,
                                 1.0f / std::sqrt(static_cast<float>(kHidden)));
    w.output = random.Values(
        static_cast<size_t>(kHidden) * kQueryHeads * kDepth,
        0.5f / std::sqrt(static_cast<float>(kQueryHeads * kDepth)));
    w.k_norm = random.Values(kIndexDim, 0.2f);
    for (auto& x : w.k_norm) x += 1.0f;
    w.cos.resize(kBlocks * kRotary);
    w.sin.resize(kBlocks * kRotary);
    for (int b = 0; b < kBlocks; ++b) {
      for (int i = 0; i < kRotary; ++i) {
        const double angle =
            4.0 * b * std::pow(10000.0, -2.0 * (i % (kRotary / 2)) / kRotary);
        w.cos[b * kRotary + i] = static_cast<float>(std::cos(angle));
        w.sin[b * kRotary + i] = static_cast<float>(std::sin(angle));
      }
    }
  }
  return layers;
}

std::string StateName(const char* kind, int layer) {
  return absl::StrCat("kv_cache_", kind, "_", layer);
}

// The quantized attention layer (QuantLayerSpec) on the hidden states x
// [1, n, kHidden]: RMS norm, the projection to queries, keys and values,
// the attention composites over the layer's caches (state_in's last two,
// whose updates go to state_out), the output projection and the residual.
int32_t AddQuantLayer(Builder& b, int32_t x, int n, const LayerWeights& w,
                      const QuantSpec& spec,
                      const std::vector<int32_t>& state_in,
                      std::vector<int32_t>& state_out, int32_t mask,
                      int32_t param, int32_t k_start, int32_t v_start,
                      int32_t fused_mask, int32_t update_graph,
                      int32_t sdpa_graph, int32_t fused_graph) {
  const int32_t k_cache = state_in[state_in.size() - 2];
  const int32_t v_cache = state_in[state_in.size() - 1];
  const int32_t square =
      b.Reduce(tflite::BuiltinOperator_MEAN, b.MulF(x, x, {1, n, kHidden}), {2},
               true, {1, n, 1});
  const int32_t inv =
      b.Unary(tflite::BuiltinOperator_RSQRT,
              b.AddF(square, b.ConstF32("quant_eps", {1}, {kEps}), {1, n, 1}),
              TensorType_FLOAT32, {1, n, 1});
  const int32_t normed =
      b.MulF(b.MulF(x, inv, {1, n, kHidden}),
             b.ConstF32("quant_norm", {kHidden}, w.norm), {1, n, kHidden});
  const int32_t projected = b.FullyConnected(
      b.Reshape(normed, {n, kHidden}),
      b.ConstF32("quant_projection", {kProjection, kHidden}, w.projection),
      {n, kProjection});
  int offset = 0;
  const auto heads = [&](int count) {
    const int32_t slice = b.Slice(projected, {0, offset}, {n, count * kDepth});
    offset += count * kDepth;
    return b.Reshape(b.Transpose(b.Reshape(slice, {n, count, kDepth}),
                                 {1, 0, 2}, {count, n, kDepth}),
                     {1, count, n, kDepth});
  };
  const int32_t q = heads(kQueryHeads);
  const int32_t k = heads(kKvHeads);
  const int32_t v = heads(kKvHeads);
  const std::string k_name = StateName("k", kQuantLayer);
  const std::string v_name = StateName("v", kQuantLayer);
  const int32_t k_out = QuantCache(b, k_name + "_out", spec, false);
  const int32_t v_out = QuantCache(b, v_name + "_out", spec, true);
  int32_t attention = 0;  // [1, Hq, n, D]
  if (spec.fused()) {
    const int R = kGroup * n;
    const int32_t grouped =
        b.Reshape(b.MulF(q, b.ConstF32("quant_prescale", {1}, {kQueryPrescale}),
                         {1, kQueryHeads, n, kDepth}),
                  {1, kKvHeads, R, kDepth});
    const int32_t fused_out =
        b.F32("quant_attention", {1, kKvHeads, R, kDepth});
    b.Composite(
        "odml.fused_sdpa_cache_update", fused_graph,
        [&](flexbuffers::Builder* a) { a->Float("softcap", kQuantSoftcap); },
        {grouped, k_cache, v_cache, k,
         b.Transpose(v, {0, 1, 3, 2}, {1, kKvHeads, kDepth, n}), fused_mask,
         param},
        {fused_out, k_out, v_out});
    attention = b.Reshape(fused_out, {1, kQueryHeads, n, kDepth});
  } else {
    int32_t q_in = q, k_in = k, v_in = v, mask_in = mask;
    if (spec.kind == "int4") {
      // The additive form of the bool mask: 0 seen, -30000 hidden (a fill
      // FP16 holds: with FP16 activations the engine computes this mask in
      // FP16, where -1e9 would be infinite and its products undefined).
      mask_in =
          b.MulF(b.AddF(b.Cast(mask, TensorType_FLOAT32),
                        b.ConstF32("quant_minus_one", {1}, {-1.0f}),
                        {1, 1, n, kCacheLen}),
                 b.ConstF32("quant_hidden", {1}, {3e4f}), {1, 1, n, kCacheLen});
    } else {
      q_in = b.Quantize(
          b.MulF(q, b.ConstF32("quant_prescale", {1}, {kQueryPrescale}),
                 {1, kQueryHeads, n, kDepth}),
          tflite::TensorType_INT8, kQueryScale, 0);
      const bool own = spec.kind == "uint8";
      k_in = b.Quantize(k, tflite::TensorType_INT8,
                        own ? kRowScale : spec.k_scale, 0);
      v_in = b.Quantize(v, tflite::TensorType_INT8,
                        own ? kRowScale : spec.v_scale, 0);
    }
    b.Composite("odml.cache_update", update_graph,
                [&](flexbuffers::Builder* a) {
                  a->Int("cache_size", kCacheLen);
                  a->Int("head_size", kDepth);
                  a->Int("k_cache_ts_idx", 2);
                  a->Int("k_update_ts_idx", 2);
                  a->Int("kv_cache_batch_size", kKvHeads);
                  a->Float("scale_k", spec.k_scale);
                  a->Float("scale_v", spec.v_scale);
                  a->Int("v_cache_ts_idx", 3);
                  a->Int("v_update_ts_idx", 2);
                },
                {k_in, v_in, param, k_cache, v_cache, k_start, v_start},
                {k_out, v_out});
    attention = b.F32("quant_attention", {1, kQueryHeads, n, kDepth});
    b.Composite("odml.sdpa_transposed", sdpa_graph,
                [&](flexbuffers::Builder* a) {
                  a->Bool("from_cache_update", true);
                  a->Int("k_ts_idx", 2);
                  if (spec.kind == "int4") {
                    a->Float("scale", kQueryPrescale);
                    a->Float("softcap", kQuantSoftcap);
                  }
                  a->Int("v_ts_idx", 3);
                },
                {q_in, k_out, v_out, mask_in, param}, {attention});
  }
  state_out.push_back(k_out);
  state_out.push_back(v_out);
  const int32_t rows = b.Reshape(
      b.Transpose(attention, {0, 2, 1, 3}, {1, n, kQueryHeads, kDepth}),
      {n, kQueryHeads * kDepth});
  const int32_t projected_out = b.FullyConnected(
      rows,
      b.ConstF32("quant_output", {kHidden, kQueryHeads * kDepth}, w.output),
      {n, kHidden});
  return b.AddF(x, b.Reshape(projected_out, {1, n, kHidden}), {1, n, kHidden});
}

// One signature over `rows` tokens: inputs embeddings, start_pos, mask,
// param_tensor, k_start, v_start, then each layer's states (ring, pool, key
// cache, value cache); outputs hidden, each layer's topk, then the states.
void AddSignature(Builder& b, const std::string& key, int rows,
                  const std::vector<LayerWeights>& weights,
                  const std::optional<QuantSpec>& quant) {
  const float index_scale = 1.0f / std::sqrt(static_cast<float>(kIndexDim));
  const float attention_scale = 1.0f / std::sqrt(static_cast<float>(kDepth));
  const int32_t indexer = AddIndexerDecomposition(b, rows, index_scale);
  const int32_t cache_update = AddCacheUpdateDecomposition(b, rows);
  const int32_t sparse = AddSparseDecomposition(b, rows, attention_scale);
  int32_t quant_update = -1, quant_sdpa = -1, quant_fused = -1;
  if (quant.has_value() && quant->fused()) {
    quant_fused = AddQuantFusedDecomposition(b, rows, *quant);
  } else if (quant.has_value()) {
    quant_update = AddQuantCacheUpdateDecomposition(b, rows, *quant);
    quant_sdpa = AddQuantSdpaDecomposition(b, rows, *quant);
  }
  const int32_t graph = b.NewSubgraph(key);
  const int n = rows;
  std::vector<std::pair<std::string, int32_t>> inputs, outputs;
  const auto input = [&](const std::string& name, int32_t tensor) {
    b.Input(tensor);
    inputs.push_back({name, tensor});
    return tensor;
  };
  int32_t x = input("embeddings", b.F32("embeddings", {1, n, kHidden}));
  const int32_t start = input("start_pos", b.I32("start_pos", {1}));
  const int32_t mask = input("mask", b.Bool("mask", {1, 1, n, kCacheLen}));
  const int32_t param = input("param_tensor", b.I32("param", {1, 1, 1, 7}));
  const int32_t k_start = input("k_start", b.I32("k_start", {4}));
  const int32_t v_start = input("v_start", b.I32("v_start", {4}));
  std::vector<int32_t> state_in, state_out;
  for (int l = 0; l < kLayers; ++l) {
    for (const auto& [kind, shape] : std::vector<std::pair<const char*, Shape>>{
             {"qsa_ring", {1, 4, kIndexDim}},
             {"qsa_pool", {1, kBlocks, kIndexDim}},
             {"k", {1, kKvHeads, kCacheLen, kDepth}},
             {"v", {1, kKvHeads, kDepth, kCacheLen}}}) {
      const std::string name = StateName(kind, l);
      state_in.push_back(input(name, b.F32(name, shape)));
    }
  }
  int32_t fused_mask = -1;
  if (quant.has_value()) {
    for (bool value : {false, true}) {
      const std::string name = StateName(value ? "v" : "k", kQuantLayer);
      state_in.push_back(input(name, QuantCache(b, name, *quant, value)));
    }
    if (quant->fused()) {
      fused_mask =
          input("fused_mask", b.F32("fused_mask", {1, 1, n, kRing + n}));
    }
  }
  std::vector<int32_t> topks;
  for (int l = 0; l < kLayers; ++l) {
    const auto& w = weights[l];
    const int32_t ring = state_in[4 * l + 0];
    const int32_t pool = state_in[4 * l + 1];
    const int32_t k_cache = state_in[4 * l + 2];
    const int32_t v_cache = state_in[4 * l + 3];
    // RMS norm.
    const int32_t square =
        b.Reduce(tflite::BuiltinOperator_MEAN, b.MulF(x, x, {1, n, kHidden}),
                 {2}, true, {1, n, 1});
    const int32_t inv =
        b.Unary(tflite::BuiltinOperator_RSQRT,
                b.AddF(square, b.ConstF32(absl::StrCat("eps", l), {1}, {kEps}),
                       {1, n, 1}),
                TensorType_FLOAT32, {1, n, 1});
    const int32_t normed =
        b.MulF(b.MulF(x, inv, {1, n, kHidden}),
               b.ConstF32(absl::StrCat("norm", l), {kHidden}, w.norm),
               {1, n, kHidden});
    // The fused projection and its parts.
    const int32_t projected =
        b.FullyConnected(b.Reshape(normed, {n, kHidden}),
                         b.ConstF32(absl::StrCat("projection", l),
                                    {kProjection, kHidden}, w.projection),
                         {n, kProjection});
    int offset = 0;
    const auto part = [&](int width) {
      const int32_t slice = b.Slice(projected, {0, offset}, {n, width});
      offset += width;
      return slice;
    };
    const int32_t q = b.Reshape(
        b.Transpose(
            b.Reshape(part(kQueryHeads * kDepth), {n, kQueryHeads, kDepth}),
            {1, 0, 2}, {kQueryHeads, n, kDepth}),
        {1, kQueryHeads, n, kDepth});
    const int32_t k = b.Reshape(
        b.Transpose(b.Reshape(part(kKvHeads * kDepth), {n, kKvHeads, kDepth}),
                    {1, 0, 2}, {kKvHeads, n, kDepth}),
        {1, kKvHeads, n, kDepth});
    const int32_t v = b.Reshape(
        b.Transpose(b.Reshape(part(kKvHeads * kDepth), {n, kKvHeads, kDepth}),
                    {1, 0, 2}, {kKvHeads, n, kDepth}),
        {1, kKvHeads, n, kDepth});
    const int32_t q_idx = b.Reshape(part(kIndexHeads * kIndexDim),
                                    {1, n, kIndexHeads, kIndexDim});
    const int32_t k_idx = b.Reshape(part(kIndexDim), {1, n, kIndexDim});

    const int32_t topk = b.I32(absl::StrCat("topk_", l), {1, n, kTopk});
    const int32_t ring_out =
        b.F32(StateName("qsa_ring", l) + "_out", {1, 4, kIndexDim});
    const int32_t pool_out =
        b.F32(StateName("qsa_pool", l) + "_out", {1, kBlocks, kIndexDim});
    b.Composite(
        "odml.qsa_indexer_update", indexer,
        [&](flexbuffers::Builder* a) {
          a->Int("block_topk", kTopk);
          a->Int("compress_ratio", 4);
          a->Float("rms_norm_eps", kEps);
          a->Float("scale", index_scale);
        },
        {q_idx, k_idx, ring, pool,
         b.ConstF32(absl::StrCat("k_norm", l), {kIndexDim}, w.k_norm),
         b.ConstF32(absl::StrCat("block_cos", l), {1, kBlocks, kRotary}, w.cos),
         b.ConstF32(absl::StrCat("block_sin", l), {1, kBlocks, kRotary}, w.sin),
         start, mask},
        {topk, ring_out, pool_out});
    const int32_t k_out =
        b.F32(StateName("k", l) + "_out", {1, kKvHeads, kCacheLen, kDepth});
    const int32_t v_out =
        b.F32(StateName("v", l) + "_out", {1, kKvHeads, kDepth, kCacheLen});
    b.Composite("odml.cache_update", cache_update,
                [&](flexbuffers::Builder* a) {
                  a->Int("cache_size", kCacheLen);
                  a->Int("head_size", kDepth);
                  a->Int("k_cache_ts_idx", 2);
                  a->Int("k_update_ts_idx", 2);
                  a->Int("kv_cache_batch_size", kKvHeads);
                  a->Int("v_cache_ts_idx", 3);
                  a->Int("v_update_ts_idx", 2);
                },
                {k, v, param, k_cache, v_cache, k_start, v_start},
                {k_out, v_out});
    const int32_t attention =
        b.F32(absl::StrCat("attention_", l), {1, n, kQueryHeads, kDepth});
    b.Composite("odml.sparse_sdpa_transposed", sparse,
                [&](flexbuffers::Builder* a) {
                  a->Int("block_topk", kTopk);
                  a->Int("compress_ratio", 4);
                  a->Int("k_ts_idx", 2);
                  a->Float("scale", attention_scale);
                  a->Int("v_ts_idx", 3);
                },
                {q, k_out, v_out, topk, mask}, {attention});
    const int32_t projected_out =
        b.FullyConnected(b.Reshape(attention, {n, kQueryHeads * kDepth}),
                         b.ConstF32(absl::StrCat("output", l),
                                    {kHidden, kQueryHeads * kDepth}, w.output),
                         {n, kHidden});
    x = b.AddF(x, b.Reshape(projected_out, {1, n, kHidden}), {1, n, kHidden});
    topks.push_back(topk);
    state_out.insert(state_out.end(), {ring_out, pool_out, k_out, v_out});
  }
  if (quant.has_value()) {
    x = AddQuantLayer(b, x, n, weights[kQuantLayer], *quant, state_in,
                      state_out, mask, param, k_start, v_start, fused_mask,
                      quant_update, quant_sdpa, quant_fused);
  }
  const auto output = [&](const std::string& name, int32_t tensor) {
    b.Output(tensor);
    outputs.push_back({name, tensor});
  };
  output("hidden", x);
  for (int l = 0; l < kLayers; ++l) output(absl::StrCat("topk_", l), topks[l]);
  for (int l = 0; l < kLayers; ++l) {
    output(StateName("qsa_ring", l), state_out[4 * l + 0]);
    output(StateName("qsa_pool", l), state_out[4 * l + 1]);
    output(StateName("k", l), state_out[4 * l + 2]);
    output(StateName("v", l), state_out[4 * l + 3]);
  }
  if (quant.has_value()) {
    output(StateName("k", kQuantLayer), state_out[4 * kLayers]);
    output(StateName("v", kQuantLayer), state_out[4 * kLayers + 1]);
  }
  b.Signature(key, graph, inputs, outputs);
}

Expected<std::string> BuildModel() {
  const auto weights = MakeWeights();
  std::optional<QuantSpec> quant;
  if (const std::string kind = absl::GetFlag(FLAGS_quant_layer);
      !kind.empty()) {
    quant = QuantLayerSpec(kind);
    if (!quant.has_value()) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Unknown --quant_layer " + kind);
    }
  }
  Builder b;
  AddSignature(b, "prefill_16", kPrefill, weights, quant);
  AddSignature(b, "decode", 1, weights, quant);
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

// One step of the script: `rows` positions from `start`, of which `valid`
// are real.
struct Step {
  bool prefill;
  int start;
  int valid;
};

std::vector<Step> Script(int decode_steps) {
  std::vector<Step> steps;
  for (int start = 0; start < kPrompt; start += kPrefill) {
    steps.push_back({true, start, std::min(kPrefill, kPrompt - start)});
  }
  for (int i = 0; i < decode_steps; ++i) {
    steps.push_back({false, kPrompt + i, 1});
  }
  return steps;
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
  LITERT_ASSIGN_OR_RETURN(
      auto model,
      litert::CompiledModel::Create(env, absl::GetFlag(FLAGS_model), options));
  LITERT_ASSIGN_OR_RETURN(const size_t prefill,
                          model.GetSignatureIndex("prefill_16"));
  LITERT_ASSIGN_OR_RETURN(const size_t decode,
                          model.GetSignatureIndex("decode"));
  // The states, one buffer each, shared by both signatures; on the NPU also
  // their outputs (in place, as LiteRT-LM binds caches).
  std::vector<std::string> state_names;
  LITERT_ASSIGN_OR_RETURN(auto prefill_inputs,
                          model.GetSignatureInputNames(prefill));
  for (const auto& name : prefill_inputs) {
    if (name.rfind("kv_cache_", 0) == 0)
      state_names.push_back(std::string(name));
  }
  const std::string quant_k = StateName("k", kQuantLayer);
  const std::string quant_v = StateName("v", kQuantLayer);
  std::vector<litert::TensorBuffer> states;
  for (const auto& name : state_names) {
    LITERT_ASSIGN_OR_RETURN(auto buffer,
                            model.CreateInputBuffer("prefill_16", name));
    LITERT_ASSIGN_OR_RETURN(const size_t size, buffer.PackedSize());
    LITERT_RETURN_IF_ERROR(Fill(buffer, std::vector<uint8_t>(size, 0)));
    states.push_back(std::move(buffer));
  }
  const std::string out_dir = absl::GetFlag(FLAGS_out_dir);
  int index = 0;
  for (const Step& step : Script(absl::GetFlag(FLAGS_decode_steps))) {
    const size_t signature = step.prefill ? prefill : decode;
    const int n = step.prefill ? kPrefill : 1;
    LITERT_ASSIGN_OR_RETURN(auto input_names,
                            model.GetSignatureInputNames(signature));
    LITERT_ASSIGN_OR_RETURN(auto output_names,
                            model.GetSignatureOutputNames(signature));
    LITERT_ASSIGN_OR_RETURN(auto inputs, model.CreateInputBuffers(signature));
    LITERT_ASSIGN_OR_RETURN(auto outputs, model.CreateOutputBuffers(signature));
    // The buffers by name (the signature's order is the runtime's).
    const auto has_input = [&](const std::string& name) {
      for (const auto& have : input_names) {
        if (have == name) return true;
      }
      return false;
    };
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
    Random random(1000 + index);
    LITERT_RETURN_IF_ERROR(
        Fill(input("embeddings"),
             random.Values(static_cast<size_t>(n) * kHidden, 1.0f)));
    LITERT_RETURN_IF_ERROR(
        Fill(input("start_pos"), std::vector<int32_t>{step.start}));
    std::vector<uint8_t> mask(static_cast<size_t>(n) * kCacheLen, 0);
    for (int t = 0; t < step.valid; ++t) {
      for (int c = 0; c <= step.start + t; ++c) mask[t * kCacheLen + c] = 1;
    }
    LITERT_RETURN_IF_ERROR(Fill(input("mask"), mask));
    if (has_input("fused_mask")) {
      // The ring holds the kRing positions before the step; a row sees
      // those within kRing of its own and the step's tokens up to its own.
      const int width = kRing + n;
      std::vector<float> ring_mask(static_cast<size_t>(n) * width, -1e9f);
      for (int t = 0; t < step.valid; ++t) {
        const int position = step.start + t;
        for (int j = 0; j < kRing; ++j) {
          const int held =
              step.start - 1 - ((step.start - 1 - j) % kRing + kRing) % kRing;
          if (step.start > 0 && held >= 0 && held > position - kRing) {
            ring_mask[t * width + j] = 0.0f;
          }
        }
        for (int u = 0; u <= t; ++u) ring_mask[t * width + kRing + u] = 0.0f;
      }
      LITERT_RETURN_IF_ERROR(Fill(input("fused_mask"), ring_mask));
    }
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
    LITERT_ASSIGN_OR_RETURN(const std::string hidden,
                            Bytes(outputs[output_index("hidden")]));
    LITERT_RETURN_IF_ERROR(
        WriteFile(prefix + ".hidden.raw", hidden.data(), hidden.size()));
    for (int l = 0; l < kLayers; ++l) {
      const std::string name = absl::StrCat("topk_", l);
      LITERT_ASSIGN_OR_RETURN(const std::string topk,
                              Bytes(outputs[output_index(name)]));
      LITERT_RETURN_IF_ERROR(WriteFile(absl::StrCat(prefix, ".", name, ".raw"),
                                       topk.data(), topk.size()));
    }
    for (size_t s = 0; s < states.size(); ++s) {
      // The CPU's states come back through its outputs.
      if (!npu) {
        LITERT_ASSIGN_OR_RETURN(const std::string bytes,
                                Bytes(outputs[output_index(state_names[s])]));
        LITERT_RETURN_IF_ERROR(
            Fill(states[s], std::vector<uint8_t>(bytes.begin(), bytes.end())));
      }
      if (state_names[s].find("qsa") != std::string::npos ||
          state_names[s] == quant_k || state_names[s] == quant_v) {
        LITERT_ASSIGN_OR_RETURN(const std::string bytes, Bytes(states[s]));
        LITERT_RETURN_IF_ERROR(
            WriteFile(absl::StrCat(prefix, ".", state_names[s], ".raw"),
                      bytes.data(), bytes.size()));
      }
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

// Compares two runs: the hidden states and the pools to the tolerance; each
// row's selection as a set (LiteRT's CPU kernels, in the XNNPACK delegate,
// list a selection by score, the GPU delegate and this backend by block);
// the rings in the slots that hold the incomplete block's keys (the CPU
// keeps a completed block's keys in the others, the GPU delegate and this
// backend clear them; no step reads them).
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
    const auto compare = [&](const std::string& name,
                             size_t count) -> Expected<void> {
      LITERT_ASSIGN_OR_RETURN(const std::string a,
                              ReadFile(cpu + prefix + name + ".raw"));
      LITERT_ASSIGN_OR_RETURN(const std::string b,
                              ReadFile(npu + prefix + name + ".raw"));
      const auto x = Floats(a);
      const auto y = Floats(b);
      count = std::min({count, x.size(), y.size()});
      double largest = 0.0, error = 0.0;
      for (size_t i = 0; i < count; ++i) {
        largest = std::max(largest, std::fabs(static_cast<double>(x[i])));
        error = std::max(error, std::fabs(static_cast<double>(x[i]) - y[i]));
      }
      const double relative = largest > 0.0 ? error / largest : error;
      const bool fine = x.size() == y.size() && relative <= tolerance;
      ok &= fine;
      absl::StrAppend(&line, " ", name, " ", relative, fine ? "" : " (!)");
      return {};
    };
    LITERT_RETURN_IF_ERROR(compare("hidden", SIZE_MAX));
    // The incomplete block's keys: slots [0, end % 4).
    const int end = step.start + step.valid;
    for (int l = 0; l < kLayers; ++l) {
      LITERT_RETURN_IF_ERROR(compare(absl::StrCat("kv_cache_qsa_ring_", l),
                                     static_cast<size_t>(end % 4) * kIndexDim));
      LITERT_RETURN_IF_ERROR(
          compare(absl::StrCat("kv_cache_qsa_pool_", l), SIZE_MAX));
    }
    for (int l = 0; l < kLayers; ++l) {
      const std::string name = absl::StrCat("topk_", l);
      LITERT_ASSIGN_OR_RETURN(const std::string a,
                              ReadFile(cpu + prefix + name + ".raw"));
      LITERT_ASSIGN_OR_RETURN(const std::string b,
                              ReadFile(npu + prefix + name + ".raw"));
      const int rows = step.prefill ? kPrefill : 1;
      int differing = 0;
      for (int r = 0; r < rows; ++r) {
        std::vector<int32_t> x(kTopk), y(kTopk);
        std::memcpy(x.data(), a.data() + r * kTopk * 4, kTopk * 4);
        std::memcpy(y.data(), b.data() + r * kTopk * 4, kTopk * 4);
        std::sort(x.begin(), x.end());
        std::sort(y.begin(), y.end());
        differing += x != y;
      }
      ok &= differing == 0;
      absl::StrAppend(&line, " ", name, " rows differing ", differing);
    }
    if (const std::string kind = absl::GetFlag(FLAGS_quant_layer);
        !kind.empty()) {
      const auto spec = QuantLayerSpec(kind);
      if (!spec.has_value()) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "Unknown --quant_layer");
      }
      // The caches' integers: equal, or one apart where the two backends'
      // values of a key straddle a rounding boundary.
      for (const char* kind_name : {"k", "v"}) {
        const std::string name = StateName(kind_name, kQuantLayer);
        LITERT_ASSIGN_OR_RETURN(const std::string a,
                                ReadFile(cpu + prefix + name + ".raw"));
        LITERT_ASSIGN_OR_RETURN(const std::string c,
                                ReadFile(npu + prefix + name + ".raw"));
        const auto values = [&](const std::string& bytes) {
          std::vector<int> out;
          for (unsigned char byte : bytes) {
            if (spec->cache_type == tflite::TensorType_INT4) {
              for (int nibble : {byte & 0xF, byte >> 4}) {
                out.push_back(nibble >= 8 ? nibble - 16 : nibble);
              }
            } else if (spec->cache_type == tflite::TensorType_UINT8) {
              out.push_back(static_cast<int>(byte) - 128);
            } else {
              out.push_back(static_cast<int>(static_cast<signed char>(byte)));
            }
          }
          return out;
        };
        const auto x = values(a);
        const auto y = values(c);
        int one_apart = 0, further = 0;
        for (size_t i = 0; i < std::min(x.size(), y.size()); ++i) {
          const int difference = std::abs(x[i] - y[i]);
          one_apart += difference == 1;
          further += difference > 1;
        }
        const bool fine = x.size() == y.size() && further == 0;
        ok &= fine;
        absl::StrAppend(&line, " ", name, " one apart ", one_apart, "/",
                        x.size(), " further ", further, fine ? "" : " (!)");
      }
    }
    std::printf("%s\n", line.c_str());
  }
  std::printf("%s\n", ok ? "MATCH" : "MISMATCH");
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  if (!absl::GetFlag(FLAGS_write_model).empty()) {
    const auto bytes = BuildModel();
    if (!bytes || !WriteFile(absl::GetFlag(FLAGS_write_model), bytes->data(),
                             bytes->size())) {
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
