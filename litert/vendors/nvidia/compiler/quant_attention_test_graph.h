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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_QUANT_ATTENTION_TEST_GRAPH_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_QUANT_ATTENTION_TEST_GRAPH_H_

// Test graphs with the composites over quantized KV caches, as the model
// that has them builds them: 8 query heads on 2 KV heads of depth 256, INT8
// ring caches of 512 entries (local layers) and INT4 caches of the context
// length (global layers).

#include <cstdint>
#include <deque>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/core/model/model.h"
#include "tflite/schema/schema_generated.h"

namespace litert::nvidia::quant_test {

inline constexpr int kHeads = 2;  // KV heads
inline constexpr int kQueryHeads = 8;
inline constexpr int kGroup = kQueryHeads / kHeads;
inline constexpr int kDepth = 256;
inline constexpr int kRing = 512;

// The quantization of a block: those of the model's blocks 0 (local) and 4
// (global).
struct Scales {
  int bits;
  float q;
  float k;
  float v;
};
inline constexpr Scales kLocal = {8, 0.006032f, 0.0389767661690712f,
                                  0.05182601884007454f};
inline constexpr Scales kGlobal = {4, 0.007865f, 0.382475346326828f,
                                   0.7135795950889587f};

// A graph under construction, with the storage its constants point to.
class Graph {
 public:
  Graph() : graph_(model_.EmplaceSubgraph()) {}

  LiteRtTensorT& Tensor(const char* name, LiteRtElementType type,
                        std::vector<int32_t> dims, float scale = 0.0f,
                        int64_t zero_point = 0) {
    auto& tensor = graph_.EmplaceTensor();
    tensor.SetName(name);
    tensor.SetType(MakeRankedTensorType(type, std::move(dims)));
    if (scale != 0.0f) {
      tensor.SetQarams(MakePerTensorQuantization(scale, zero_point));
    }
    return tensor;
  }

  LiteRtTensorT& Input(const char* name, LiteRtElementType type,
                       std::vector<int32_t> dims, float scale = 0.0f,
                       int64_t zero_point = 0) {
    auto& tensor = Tensor(name, type, std::move(dims), scale, zero_point);
    graph_.Inputs().push_back(&tensor);
    return tensor;
  }

  void Output(LiteRtTensorT& tensor) { graph_.Outputs().push_back(&tensor); }

  LiteRtTensorT& Constant(const char* name, std::vector<int32_t> values) {
    constants_.push_back(std::move(values));
    const auto& stored = constants_.back();
    auto& tensor = Tensor(name, kLiteRtElementTypeInt32,
                          {static_cast<int32_t>(stored.size())});
    SetWeightsFromUnownedBuffer(
        tensor.Weights(), litert::BufferRef<uint8_t>(
                              stored.data(), stored.size() * sizeof(int32_t)));
    return tensor;
  }

  // A constant of zeros.
  LiteRtTensorT& Zeros(const char* name, LiteRtElementType type,
                       std::vector<int32_t> dims, size_t bytes) {
    zeros_.emplace_back(bytes, 0);
    auto& tensor = Tensor(name, type, std::move(dims));
    SetWeightsFromUnownedBuffer(
        tensor.Weights(),
        litert::BufferRef<uint8_t>(zeros_.back().data(), bytes));
    return tensor;
  }

  LiteRtOpT& Op(LiteRtOpCode code, std::vector<LiteRtTensorT*> inputs,
                std::vector<LiteRtTensorT*> outputs) {
    auto& op = graph_.EmplaceOp();
    op.SetOpCode(code);
    for (auto* tensor : inputs) {
      litert::internal::AttachInput(tensor, op);
    }
    for (auto* tensor : outputs) {
      litert::internal::AttachOutput(tensor, op);
    }
    return op;
  }

  // QUANTIZE of `input` into a tensor of `bits` bits.
  LiteRtTensorT& Quantize(LiteRtTensorT& input, const char* name, int bits,
                          std::vector<int32_t> dims, float scale) {
    auto& output = Tensor(
        name, bits == 8 ? kLiteRtElementTypeInt8 : kLiteRtElementTypeInt4,
        std::move(dims), scale);
    Op(kLiteRtOpCodeTflQuantize, {&input}, {&output});
    return output;
  }

  LiteRtTensorT& Reshape(LiteRtTensorT& input, const char* name,
                         LiteRtElementType type, std::vector<int32_t> dims,
                         float scale) {
    auto& output = Tensor(name, type, std::move(dims), scale);
    Op(kLiteRtOpCodeTflReshape, {&input}, {&output});
    return output;
  }

  LiteRtTensorT& Transpose(LiteRtTensorT& input, const char* name,
                           LiteRtElementType type, std::vector<int32_t> dims,
                           float scale, std::vector<int32_t> permutation) {
    auto& output = Tensor(name, type, std::move(dims), scale);
    auto& perm =
        Constant((std::string(name) + "_perm").c_str(), std::move(permutation));
    Op(kLiteRtOpCodeTflTranspose, {&input, &perm}, {&output});
    return output;
  }

  LiteRtOpT& Composite(const char* name,
                       const std::function<void(flexbuffers::Builder*)>& attrs,
                       std::vector<LiteRtTensorT*> inputs,
                       std::vector<LiteRtTensorT*> outputs) {
    auto& op =
        Op(kLiteRtOpCodeShloComposite, std::move(inputs), std::move(outputs));
    flexbuffers::Builder attributes;
    const auto map = attributes.StartMap();
    attrs(&attributes);
    attributes.EndMap(map);
    attributes.Finish();
    tflite::StableHLOCompositeOptionsT composite_options;
    composite_options.name = name;
    composite_options.composite_attributes = attributes.GetBuffer();
    tflite::BuiltinOptions2Union options;
    options.Set(std::move(composite_options));
    litert::internal::SetTflOptions2(op, std::move(options));
    return op;
  }

  // What the graphs put after odml.sdpa_transposed: the attention
  // [1, Hq, n, D] (FLOAT16) as [1, n, Hq, D], with zero for the tokens whose
  // row of the mask [1, 1, n, S] sees no key:
  //   SELECT_V2(NOT_EQUAL(REDUCE_MAX(CAST(mask as [1, n, 1, S])), 0), ., 0).
  // `suffix` tells the tensors of several of them apart.
  LiteRtTensorT& MaskSelect(LiteRtTensorT& attention, LiteRtTensorT& mask,
                            int tokens, int query_heads, int depth,
                            int cache_len, const std::string& suffix = "") {
    const auto name = [&](const char* base) { return base + suffix; };
    const std::vector<int32_t> rows_dims = {1, tokens, query_heads, depth};
    auto& rows = tokens == 1
                     ? Reshape(attention, name("attention_rows").c_str(),
                               kLiteRtElementTypeFloat16, rows_dims, 0.0f)
                     : Transpose(attention, name("attention_rows").c_str(),
                                 kLiteRtElementTypeFloat16, rows_dims, 0.0f,
                                 {0, 2, 1, 3});
    LiteRtTensorT* mask_rows = &mask;
    if (tokens > 1) {
      mask_rows =
          &Reshape(mask, name("mask_rows").c_str(), kLiteRtElementTypeBool,
                   {1, tokens, 1, cache_len}, 0.0f);
    }
    auto& mask_float =
        Tensor(name("mask_float").c_str(), kLiteRtElementTypeFloat32,
               {1, tokens, 1, cache_len});
    Op(kLiteRtOpCodeTflCast, {mask_rows}, {&mask_float});
    auto& any = Tensor(name("mask_any").c_str(), kLiteRtElementTypeFloat32,
                       {1, tokens, 1, 1});
    auto& reduce =
        Op(kLiteRtOpCodeTflReduceMax,
           {&mask_float, &Constant(name("mask_axis").c_str(), {3})}, {&any});
    tflite::ReducerOptionsT reducer;
    reducer.keep_dims = true;
    tflite::BuiltinOptionsUnion reducer_options;
    reducer_options.Set(std::move(reducer));
    litert::internal::SetTflOptions(reduce, std::move(reducer_options));
    auto& seen = Tensor(name("mask_seen").c_str(), kLiteRtElementTypeBool,
                        {1, tokens, 1, 1});
    Op(kLiteRtOpCodeTflNotEqual,
       {&any, &Zeros(name("no_key").c_str(), kLiteRtElementTypeFloat32, {}, 4)},
       {&seen});
    auto& selected = Tensor(name("attention_selected").c_str(),
                            kLiteRtElementTypeFloat16, rows_dims);
    Op(kLiteRtOpCodeTflSelectV2,
       {&seen, &rows,
        &Zeros(name("zero").c_str(), kLiteRtElementTypeFloat16, {}, 2)},
       {&selected});
    return selected;
  }

  LiteRtSubgraphT& Get() { return graph_; }

 private:
  LiteRtModelT model_;
  LiteRtSubgraphT& graph_;
  std::deque<std::vector<int32_t>> constants_;
  std::deque<std::vector<uint8_t>> zeros_;
};

inline void CacheUpdateAttributes(flexbuffers::Builder* attributes,
                                  int cache_len, bool ring,
                                  const Scales& scales) {
  attributes->Int("cache_size", cache_len);
  attributes->Int("head_size", kDepth);
  if (ring) {
    attributes->Bool("is_ring_buffer", true);
  }
  attributes->Int("kv_cache_batch_size", kHeads);
  attributes->Float("scale_k", scales.k);
  attributes->Float("scale_v", scales.v);
}

inline void SdpaAttributes(flexbuffers::Builder* attributes, bool global) {
  attributes->Bool("from_cache_update", true);
  attributes->Bool("is_causal", true);
  attributes->Bool("is_global", global);
  attributes->Int("k_ts_idx", 2);
  attributes->Float("scale", 1.0f);
  attributes->Int("v_ts_idx", 3);
}

// A decode block and a later block that attends to the same updated caches
// (the model's blocks 20 to 49 read the caches of blocks 18 and 19):
//   k, v, q, q2 float -> QUANTIZE (-> RESHAPE for v) -> odml.cache_update ->
//   odml.sdpa_transposed twice.
// Inputs: k [1, H, 1, D], v [1, 1, H, D], q and q2 [1, 8, 1, D] (FLOAT16, or
// FLOAT32 with `float32`), the key cache [1, H, S, D] and the value cache
// [1, H, D, S] (INT8 for the local scales, INT4 for the global ones), the
// mask [1, 1, 1, S], the cache parameters and the two start index vectors.
// Outputs: the updated caches and the two attentions [1, 8, 1, D], or with
// `mask_select` (FLOAT16 only) what Graph::MaskSelect makes of them,
// [1, 1, 8, D].
inline void AddDecodeBlock(Graph* graph, const Scales& scales, int cache_len,
                           bool float32, bool mask_select = false) {
  const bool ring = scales.bits == 8;
  const LiteRtElementType float_type =
      float32 ? kLiteRtElementTypeFloat32 : kLiteRtElementTypeFloat16;
  const LiteRtElementType cache_type =
      scales.bits == 8 ? kLiteRtElementTypeInt8 : kLiteRtElementTypeInt4;
  auto& k = graph->Input("k", float_type, {1, kHeads, 1, kDepth});
  auto& v = graph->Input("v", float_type, {1, 1, kHeads, kDepth});
  auto& q = graph->Input("q", float_type, {1, kQueryHeads, 1, kDepth});
  auto& q2 = graph->Input("q2", float_type, {1, kQueryHeads, 1, kDepth});
  auto& k_cache = graph->Input("kv_cache_k_0", cache_type,
                               {1, kHeads, cache_len, kDepth}, scales.k);
  auto& v_cache = graph->Input("kv_cache_v_0", cache_type,
                               {1, kHeads, kDepth, cache_len}, scales.v);
  auto& mask =
      graph->Input("mask", kLiteRtElementTypeBool, {1, 1, 1, cache_len});
  auto& params =
      graph->Input("param_tensor", kLiteRtElementTypeInt32, {1, 1, 1, 7});
  auto& k_start = graph->Input("k_start", kLiteRtElementTypeInt32, {4});
  auto& v_start = graph->Input("v_start", kLiteRtElementTypeInt32, {4});
  auto& k_quantized = graph->Quantize(k, "k_quantized", scales.bits,
                                      {1, kHeads, 1, kDepth}, scales.k);
  auto& v_quantized = graph->Quantize(v, "v_quantized", scales.bits,
                                      {1, 1, kHeads, kDepth}, scales.v);
  auto& v_rows = graph->Reshape(v_quantized, "v_rows", cache_type,
                                {1, kHeads, 1, kDepth}, scales.v);
  auto& k_out = graph->Tensor("kv_cache_k_0_out", cache_type,
                              {1, kHeads, cache_len, kDepth}, scales.k);
  auto& v_out = graph->Tensor("kv_cache_v_0_out", cache_type,
                              {1, kHeads, kDepth, cache_len}, scales.v);
  graph->Composite(
      "odml.cache_update",
      [&](flexbuffers::Builder* attributes) {
        CacheUpdateAttributes(attributes, cache_len, ring, scales);
      },
      {&k_quantized, &v_rows, &params, &k_cache, &v_cache, &k_start, &v_start},
      {&k_out, &v_out});
  graph->Output(k_out);
  graph->Output(v_out);
  int reader = 0;
  for (LiteRtTensorT* query : {&q, &q2}) {
    const std::string suffix = std::to_string(reader++);
    auto& quantized = graph->Quantize(*query, ("q_quantized" + suffix).c_str(),
                                      8, {1, kQueryHeads, 1, kDepth}, scales.q);
    auto& attention = graph->Tensor(("attention" + suffix).c_str(), float_type,
                                    {1, kQueryHeads, 1, kDepth});
    graph->Composite("odml.sdpa_transposed",
                     [&](flexbuffers::Builder* attributes) {
                       SdpaAttributes(attributes, !ring);
                     },
                     {&quantized, &k_out, &v_out, &mask, &params},
                     {&attention});
    graph->Output(mask_select ? graph->MaskSelect(attention, mask, /*tokens=*/1,
                                                  kQueryHeads, kDepth,
                                                  cache_len, suffix)
                              : attention);
  }
}

}  // namespace litert::nvidia::quant_test

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_QUANT_ATTENTION_TEST_GRAPH_H_
