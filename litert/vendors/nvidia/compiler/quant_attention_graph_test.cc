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

// The composites over quantized KV caches, built as the graphs of the model
// that has them build them (their attributes, tensor types and shapes, and
// the QUANTIZE, RESHAPE and TRANSPOSE ops in front of them), lowered by the
// graph builder and compared with the host references of
// trtllm/quant_attention_reference.h: a decode block over an INT8 ring cache
// and over an INT4 cache, each with a second reader of the updated caches,
// and the global and the local block of a prefill chunk.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "cuda_runtime_api.h"
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/core/model/model.h"
#include "litert/vendors/nvidia/compiler/quant_attention_test_graph.h"
#include "litert/vendors/nvidia/compiler/tensorrt_graph_builder.h"
#include "litert/vendors/nvidia/tensorrt_logger.h"
#include "NvInfer.h"
#include "litert/vendors/nvidia/trtllm/quant_attention_reference.h"
#include "tflite/schema/schema_generated.h"

namespace {

using litert::nvidia::quant_reference::AttentionInputs;
using litert::nvidia::quant_reference::Bf16BitsToFloat;
using litert::nvidia::quant_reference::Caches;
using litert::nvidia::quant_reference::CompareRows;
using litert::nvidia::quant_reference::FloatToBf16Bits;
using litert::nvidia::quant_reference::FloatToHalfBits;
using litert::nvidia::quant_reference::HalfBitsToFloat;
using litert::nvidia::quant_reference::PackInt4;
using litert::nvidia::quant_reference::Quantize;
using litert::nvidia::quant_reference::RowsMatch;
using litert::nvidia::quant_reference::UnpackInt4;
using litert::nvidia::quant_test::AddDecodeBlock;
using litert::nvidia::quant_test::CacheUpdateAttributes;
using litert::nvidia::quant_test::Graph;
using litert::nvidia::quant_test::kDepth;
using litert::nvidia::quant_test::kGlobal;
using litert::nvidia::quant_test::kGroup;
using litert::nvidia::quant_test::kHeads;
using litert::nvidia::quant_test::kLocal;
using litert::nvidia::quant_test::kQueryHeads;
using litert::nvidia::quant_test::kRing;
using litert::nvidia::quant_test::Scales;
using litert::nvidia::quant_test::SdpaAttributes;

// The fills of the decompositions.
constexpr float kSdpaFill = -45856.0f;
constexpr float kFusedFill = -10000.0f;

class Random {
 public:
  explicit Random(uint32_t seed) : state_(seed) {}
  // Uniform in [-1, 1].
  float Next() {
    state_ = state_ * 1664525u + 1013904223u;
    return static_cast<float>((state_ >> 8) & 0xFFFF) / 65535.0f * 2.0f - 1.0f;
  }
  int Int(int lo, int hi) {
    state_ = state_ * 1664525u + 1013904223u;
    return lo +
           static_cast<int>((state_ >> 8) % static_cast<uint32_t>(hi - lo + 1));
  }

 private:
  uint32_t state_;
};

// A float tensor of the graph: FLOAT16, which the engine takes as it is, or
// FLOAT32, which the engine casts to its BF16 activations. `held` is what
// the composites then see.
struct FloatTensor {
  std::vector<uint8_t> bytes;
  std::vector<float> held;
};

FloatTensor MakeFloats(const std::vector<float>& values, bool float32) {
  FloatTensor tensor;
  tensor.held.resize(values.size());
  tensor.bytes.resize(values.size() * (float32 ? 4 : 2));
  for (size_t i = 0; i < values.size(); ++i) {
    if (float32) {
      std::memcpy(tensor.bytes.data() + 4 * i, &values[i], 4);
      tensor.held[i] = Bf16BitsToFloat(FloatToBf16Bits(values[i]));
    } else {
      const uint16_t bits = FloatToHalfBits(values[i]);
      std::memcpy(tensor.bytes.data() + 2 * i, &bits, 2);
      tensor.held[i] = HalfBitsToFloat(bits);
    }
  }
  return tensor;
}

std::vector<float> ReadFloats(const std::vector<uint8_t>& bytes, bool float32) {
  std::vector<float> values(bytes.size() / (float32 ? 4 : 2));
  for (size_t i = 0; i < values.size(); ++i) {
    if (float32) {
      std::memcpy(&values[i], bytes.data() + 4 * i, 4);
    } else {
      uint16_t bits = 0;
      std::memcpy(&bits, bytes.data() + 2 * i, 2);
      values[i] = HalfBitsToFloat(bits);
    }
  }
  return values;
}

std::vector<float> RandomValues(size_t count, float range, Random* random) {
  std::vector<float> values(count);
  for (auto& value : values) {
    value = random->Next() * range;
  }
  return values;
}

Caches RandomCaches(int cache_len, int bits, Random* random) {
  Caches caches(kHeads, cache_len, kDepth);
  for (auto& key : caches.k) {
    key = static_cast<int8_t>(
        random->Int(bits == 8 ? -40 : -8, bits == 8 ? 42 : 7));
  }
  for (auto& value : caches.v) {
    value = static_cast<int8_t>(
        random->Int(bits == 8 ? -128 : -8, bits == 8 ? 127 : 7));
  }
  return caches;
}

std::vector<uint8_t> StoredCache(const std::vector<int8_t>& values, int bits) {
  if (bits == 4) {
    return PackInt4(values);
  }
  return std::vector<uint8_t>(values.begin(), values.end());
}

std::vector<int8_t> LoadedCache(const std::vector<uint8_t>& stored, int bits,
                                size_t count) {
  if (bits == 4) {
    return UnpackInt4(stored, count);
  }
  return std::vector<int8_t>(stored.begin(), stored.end());
}

// The quantization of float rows [heads][tokens][depth], read token-major
// when the graph has them as [tokens][heads][depth].
std::vector<int8_t> QuantizeRows(const std::vector<float>& held, int tokens,
                                 bool token_major, float scale, int bits) {
  std::vector<int8_t> rows(held.size());
  for (int h = 0; h < kHeads; ++h) {
    for (int t = 0; t < tokens; ++t) {
      for (int d = 0; d < kDepth; ++d) {
        const size_t to = (static_cast<size_t>(h) * tokens + t) * kDepth + d;
        const size_t from =
            token_major ? (static_cast<size_t>(t) * kHeads + h) * kDepth + d
                        : to;
        rows[to] = Quantize(held[from], scale, bits);
      }
    }
  }
  return rows;
}

std::vector<int8_t> QuantizeQuery(const std::vector<float>& held, float scale) {
  std::vector<int8_t> query(held.size());
  for (size_t i = 0; i < held.size(); ++i) {
    query[i] = Quantize(held[i], scale, 8);
  }
  return query;
}

// LiteRT-LM's masks (runtime/executor/litert_compiled_model_executor_utils.cc):
// a decode step over a ring cache, and a causal prefill chunk over one.
std::vector<uint8_t> RingDecodeMask(int ring, int step, int window) {
  std::vector<uint8_t> mask(ring, 0);
  for (int col = 0; col < ring; ++col) {
    const int distance = (((step - col) % ring) + ring) % ring;
    mask[col] =
        distance <= window - 1 && distance <= step && distance <= ring - 1;
  }
  return mask;
}

std::vector<uint8_t> RingPrefillMask(int ring, int tokens, int start, int steps,
                                     int window) {
  const int keys = ring + tokens;
  std::vector<uint8_t> mask(static_cast<size_t>(tokens) * keys, 0);
  for (int step = 0; step < steps; ++step) {
    const int query = start + step;
    uint8_t* row = mask.data() + static_cast<size_t>(step) * keys;
    if (start > 0) {
      const int last_cached = start - 1;
      const int current = ((last_cached % ring) + ring) % ring;
      for (int col = 0; col < ring; ++col) {
        const int raw = current - col;
        const int position = last_cached - (raw >= 0 ? raw : raw + ring);
        row[col] = position >= 0 && position < start && position <= query &&
                   position >= query - window + 1;
      }
    }
    for (int key = 0; key < steps; ++key) {
      row[ring + key] = key <= step && key >= step - window + 1;
    }
  }
  return mask;
}

// An engine built from a graph, with a device buffer per input and output.
// The outputs named in `aliases` share the buffer of an input: the caches,
// which the runtime binds in place.
class Session {
 public:
  void Build(Graph* graph, const std::vector<size_t>& input_bytes,
             const std::vector<size_t>& output_bytes,
             const std::vector<std::pair<int, int>>& aliases) {
    const litert::compiler::Subgraph subgraph(LrtGetCompilerContext(),
                                              &graph->Get());
    for (const auto& op : subgraph.Ops()) {
      EXPECT_TRUE(litert::nvidia::IsTensorRtOpSupported(op))
          << "op code " << op.Code();
    }
    auto built = litert::nvidia::BuildTensorRtEngine(subgraph);
    ASSERT_TRUE(built.HasValue()) << built.Error().Message();
    input_names_ = built->input_names;
    output_names_ = built->output_names;
    ASSERT_EQ(input_names_.size(), input_bytes.size());
    ASSERT_EQ(output_names_.size(), output_bytes.size());
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    ASSERT_NE(runtime_, nullptr);
    engine_.reset(runtime_->deserializeCudaEngine(built->engine.data(),
                                                  built->engine.size()));
    ASSERT_NE(engine_, nullptr);
    std::unique_ptr<nvinfer1::IEngineInspector> inspector(
        engine_->createEngineInspector());
    ASSERT_NE(inspector, nullptr);
    layers_ = inspector->getEngineInformation(
        nvinfer1::LayerInformationFormat::kJSON);
    context_.reset(engine_->createExecutionContext());
    ASSERT_NE(context_, nullptr);
    inputs_.resize(input_bytes.size(), nullptr);
    outputs_.resize(output_bytes.size(), nullptr);
    for (size_t i = 0; i < inputs_.size(); ++i) {
      ASSERT_EQ(cudaMalloc(&inputs_[i], input_bytes[i]), cudaSuccess);
      ASSERT_TRUE(
          context_->setTensorAddress(input_names_[i].c_str(), inputs_[i]))
          << input_names_[i];
    }
    for (const auto& [output, input] : aliases) {
      outputs_[output] = inputs_[input];
    }
    for (size_t i = 0; i < outputs_.size(); ++i) {
      if (outputs_[i] == nullptr) {
        ASSERT_EQ(cudaMalloc(&outputs_[i], output_bytes[i]), cudaSuccess);
        owned_outputs_.push_back(outputs_[i]);
      }
      ASSERT_TRUE(
          context_->setTensorAddress(output_names_[i].c_str(), outputs_[i]))
          << output_names_[i];
    }
    input_bytes_ = input_bytes;
    output_bytes_ = output_bytes;
    ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess);
  }

  ~Session() {
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
    for (void* pointer : inputs_) cudaFree(pointer);
    for (void* pointer : owned_outputs_) cudaFree(pointer);
  }

  template <typename T>
  void SetInput(int index, const std::vector<T>& host) {
    ASSERT_EQ(host.size() * sizeof(T), input_bytes_[index])
        << input_names_[index];
    ASSERT_EQ(cudaMemcpy(inputs_[index], host.data(), input_bytes_[index],
                         cudaMemcpyHostToDevice),
              cudaSuccess);
  }

  void Run() {
    ASSERT_TRUE(context_->enqueueV3(stream_));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
  }

  std::vector<uint8_t> Output(int index) {
    std::vector<uint8_t> host(output_bytes_[index]);
    EXPECT_EQ(cudaMemcpy(host.data(), outputs_[index], host.size(),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    return host;
  }

  // Whether the engine's layer list has `text`.
  bool LayersMention(const std::string& text) const {
    return layers_.find(text) != std::string::npos;
  }

  // How often the engine's layers name `layer`.
  int LayerNames(const std::string& layer) const {
    int count = 0;
    for (size_t at = layers_.find("\"" + layer); at != std::string::npos;
         at = layers_.find("\"" + layer, at + 1)) {
      ++count;
    }
    return count;
  }

  const nvinfer1::ICudaEngine& engine() const { return *engine_; }
  const std::vector<std::string>& input_names() const { return input_names_; }

 private:
  litert::nvidia::TensorRtLogger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<void*> inputs_;
  std::vector<void*> outputs_;
  std::vector<void*> owned_outputs_;
  std::vector<size_t> input_bytes_;
  std::vector<size_t> output_bytes_;
  std::string layers_;
  cudaStream_t stream_ = nullptr;
};

// `actual` against the reference, to `tolerance` but for the rare row an
// FP16 step of a score moves (RowsMatch).
void ExpectRowsMatch(const std::vector<float>& actual,
                     const std::vector<float>& expected, float tolerance) {
  const auto errors = CompareRows(actual, expected, kDepth, tolerance);
  EXPECT_TRUE(RowsMatch(errors))
      << errors.rows_off << " of " << errors.rows
      << " rows are off, the largest error is " << errors.max_error;
}

// A decode block and a later block that reads the same updated caches
// (AddDecodeBlock), run step by step with the caches in place. `float32`
// graph tensors reach the composites as BF16 activations. With `mask_select`
// the graph has the SELECT that gives a token whose mask sees no key zero,
// and the step `blind_step` runs under such a mask.
void RunDecodeBlock(const Scales& scales, int cache_len, bool float32,
                    const std::vector<int>& steps, uint32_t seed,
                    bool mask_select = false, int blind_step = -1) {
  SCOPED_TRACE(::testing::Message()
               << "bits=" << scales.bits << " cache_len=" << cache_len
               << " float32=" << float32 << " mask_select=" << mask_select);
  const bool ring = scales.bits == 8;
  const size_t float_bytes = float32 ? 4 : 2;
  const size_t cache_bytes =
      static_cast<size_t>(kHeads) * cache_len * kDepth * scales.bits / 8;
  Graph graph;
  AddDecodeBlock(&graph, scales, cache_len, float32, mask_select);

  const size_t row_bytes = static_cast<size_t>(kHeads) * kDepth * float_bytes;
  const size_t query_bytes =
      static_cast<size_t>(kQueryHeads) * kDepth * float_bytes;
  Session session;
  ASSERT_NO_FATAL_FAILURE(session.Build(
      &graph,
      {row_bytes, row_bytes, query_bytes, query_bytes, cache_bytes, cache_bytes,
       static_cast<size_t>(cache_len), 7 * sizeof(int32_t), 4 * sizeof(int32_t),
       4 * sizeof(int32_t)},
      {cache_bytes, cache_bytes, query_bytes, query_bytes}, {{0, 4}, {1, 5}}));
  // One launch for the update and one layer per attention.
  EXPECT_GT(session.LayerNames("cuda_quant_cache_update"), 0);
  EXPECT_GT(session.LayerNames("cuda_quant_attention"), 0);
  // The plugin's zero for a row that sees no key makes the SELECT after it
  // the identity: neither it nor the mask's reduction is in the engine.
  EXPECT_FALSE(session.LayersMention("Select"));
  EXPECT_FALSE(session.LayersMention("Reduce"));
  // An INT4 cache crosses the engine's boundary as its bytes.
  for (int i : {4, 5}) {
    const char* name = session.input_names()[i].c_str();
    EXPECT_EQ(session.engine().getTensorDataType(name),
              nvinfer1::DataType::kINT8);
    const nvinfer1::Dims dims = session.engine().getTensorShape(name);
    size_t volume = 1;
    for (int d = 0; d < dims.nbDims; ++d) volume *= dims.d[d];
    EXPECT_EQ(volume, cache_bytes);
  }

  Random random(seed);
  Caches caches = RandomCaches(cache_len, scales.bits, &random);
  session.SetInput(4, StoredCache(caches.k, scales.bits));
  session.SetInput(5, StoredCache(caches.v, scales.bits));
  for (const int step : steps) {
    SCOPED_TRACE(::testing::Message() << "step=" << step);
    const float range = scales.bits == 8 ? 140.0f : 9.5f;
    const size_t row_count = static_cast<size_t>(kHeads) * kDepth;
    const FloatTensor k_step =
        MakeFloats(RandomValues(row_count, range * scales.k, &random), float32);
    const FloatTensor v_step =
        MakeFloats(RandomValues(row_count, range * scales.v, &random), float32);
    const size_t query_count = static_cast<size_t>(kQueryHeads) * kDepth;
    const FloatTensor q_step = MakeFloats(
        RandomValues(query_count, 110.0f * scales.q, &random), float32);
    const FloatTensor q2_step = MakeFloats(
        RandomValues(query_count, 110.0f * scales.q, &random), float32);
    // The ring's write position is reduced in the graph; the global caches
    // are written at the token's position.
    const int position = ring ? step % cache_len : step;
    const bool blind = step == blind_step;
    std::vector<uint8_t> mask_step(cache_len, 0);
    if (blind) {
      // No key: the graph's SELECT gives the token zero.
    } else if (ring) {
      mask_step = RingDecodeMask(cache_len, step, cache_len);
    } else {
      std::fill_n(mask_step.begin(), step + 1, 1);
    }
    session.SetInput(0, k_step.bytes);
    session.SetInput(1, v_step.bytes);
    session.SetInput(2, q_step.bytes);
    session.SetInput(3, q2_step.bytes);
    session.SetInput(6, mask_step);
    session.SetInput(
        7, std::vector<int32_t>{step, step + 1, step + 1, 0, 0, 0, 0});
    session.SetInput(8, std::vector<int32_t>{0, 0, position, 0});
    session.SetInput(9, std::vector<int32_t>{0, 0, 0, position});
    ASSERT_NO_FATAL_FAILURE(session.Run());

    litert::nvidia::quant_reference::CacheUpdate(
        &caches, QuantizeRows(k_step.held, 1, false, scales.k, scales.bits),
        QuantizeRows(v_step.held, 1, true, scales.v, scales.bits), 1, position,
        position);
    EXPECT_EQ(LoadedCache(session.Output(0), scales.bits, caches.k.size()),
              caches.k);
    EXPECT_EQ(LoadedCache(session.Output(1), scales.bits, caches.v.size()),
              caches.v);
    int output = 2;
    for (const FloatTensor* query : {&q_step, &q2_step}) {
      const std::vector<int8_t> quantized =
          QuantizeQuery(query->held, scales.q);
      AttentionInputs reference;
      reference.q = &quantized;
      reference.rows = kGroup;
      reference.q_scale = scales.q;
      reference.k_scale = scales.k;
      reference.v_scale = scales.v;
      reference.caches = &caches;
      reference.mask = &mask_step;
      reference.mask_rows = 1;
      reference.fill = kSdpaFill;
      auto expected = litert::nvidia::quant_reference::Attention(reference);
      if (blind) {
        std::fill(expected.begin(), expected.end(), 0.0f);
      }
      SCOPED_TRACE(::testing::Message() << "reader " << output - 2);
      // FP16 results are good to 0.004 at these magnitudes, BF16 ones to
      // 0.03.
      ExpectRowsMatch(ReadFloats(session.Output(output), float32), expected,
                      float32 ? 0.05f : 0.02f);
      ++output;
    }
  }
}

TEST(QuantAttentionGraphTest, DecodeBlockOverAnInt8RingCache) {
  // The ring fills up, then positions past its 512 entries overwrite its
  // first slots while the mask shows every slot.
  RunDecodeBlock(kLocal, kRing, /*float32=*/false, {0, 1, 2, 300}, 11);
  RunDecodeBlock(kLocal, kRing, /*float32=*/false, {510, 511, 512, 513, 1030},
                 12);
  RunDecodeBlock(kLocal, kRing, /*float32=*/true, {40, 511, 512}, 13);
}

TEST(QuantAttentionGraphTest, DecodeBlockWithTheSelectOfTokensThatSeeAKey) {
  // The graphs' SELECT after odml.sdpa_transposed, here for one token: a
  // reshape of the attention and the mask itself.
  RunDecodeBlock(kLocal, kRing, /*float32=*/false, {7, 8, 9}, 14,
                 /*mask_select=*/true, /*blind_step=*/8);
  RunDecodeBlock(kGlobal, 2048, /*float32=*/false, {100, 101}, 24,
                 /*mask_select=*/true, /*blind_step=*/101);
}

TEST(QuantAttentionGraphTest, DecodeBlockOverAnInt4Cache) {
  RunDecodeBlock(kGlobal, 2048, /*float32=*/false, {0, 1, 777, 2047}, 21);
  RunDecodeBlock(kGlobal, 2048, /*float32=*/true, {5, 1500}, 22);
  // A context length that leaves the INT4 value rows off the byte boundaries.
  RunDecodeBlock(kGlobal, 1027, /*float32=*/false, {0, 600, 1026}, 23);
}

TEST(QuantAttentionGraphTest, GlobalBlockOfAPrefillChunk) {
  // k float -> QUANTIZE (INT4); v float [1, M, H, D] -> QUANTIZE ->
  // TRANSPOSE [1, H, M, D]; odml.cache_update writes the chunk at its first
  // position, padding included; q float -> QUANTIZE (INT8) ->
  // odml.sdpa_transposed over the updated caches; then the SELECT that gives
  // the tokens of padding zero, which the plugin's result makes the identity.
  constexpr int kTokens = 128;
  constexpr int kCacheLen = 1024;
  const Scales& scales = kGlobal;
  const size_t cache_bytes =
      static_cast<size_t>(kHeads) * kCacheLen * kDepth / 2;
  Graph graph;
  auto& k =
      graph.Input("k", kLiteRtElementTypeFloat16, {1, kHeads, kTokens, kDepth});
  auto& v =
      graph.Input("v", kLiteRtElementTypeFloat16, {1, kTokens, kHeads, kDepth});
  auto& q = graph.Input("q", kLiteRtElementTypeFloat16,
                        {1, kQueryHeads, kTokens, kDepth});
  auto& k_cache = graph.Input("kv_cache_k_4", kLiteRtElementTypeInt4,
                              {1, kHeads, kCacheLen, kDepth}, scales.k);
  auto& v_cache = graph.Input("kv_cache_v_4", kLiteRtElementTypeInt4,
                              {1, kHeads, kDepth, kCacheLen}, scales.v);
  auto& mask =
      graph.Input("mask", kLiteRtElementTypeBool, {1, 1, kTokens, kCacheLen});
  auto& params =
      graph.Input("param_tensor", kLiteRtElementTypeInt32, {1, 1, 1, 7});
  auto& k_start = graph.Input("k_start", kLiteRtElementTypeInt32, {4});
  auto& v_start = graph.Input("v_start", kLiteRtElementTypeInt32, {4});
  auto& k_quantized = graph.Quantize(k, "k_quantized", 4,
                                     {1, kHeads, kTokens, kDepth}, scales.k);
  auto& v_quantized = graph.Quantize(v, "v_quantized", 4,
                                     {1, kTokens, kHeads, kDepth}, scales.v);
  auto& v_rows =
      graph.Transpose(v_quantized, "v_rows", kLiteRtElementTypeInt4,
                      {1, kHeads, kTokens, kDepth}, scales.v, {0, 2, 1, 3});
  auto& k_out = graph.Tensor("kv_cache_k_4_out", kLiteRtElementTypeInt4,
                             {1, kHeads, kCacheLen, kDepth}, scales.k);
  auto& v_out = graph.Tensor("kv_cache_v_4_out", kLiteRtElementTypeInt4,
                             {1, kHeads, kDepth, kCacheLen}, scales.v);
  graph.Composite(
      "odml.cache_update",
      [&](flexbuffers::Builder* attributes) {
        CacheUpdateAttributes(attributes, kCacheLen, /*ring=*/false, scales);
      },
      {&k_quantized, &v_rows, &params, &k_cache, &v_cache, &k_start, &v_start},
      {&k_out, &v_out});
  auto& q_quantized = graph.Quantize(
      q, "q_quantized", 8, {1, kQueryHeads, kTokens, kDepth}, scales.q);
  auto& attention = graph.Tensor("attention", kLiteRtElementTypeFloat16,
                                 {1, kQueryHeads, kTokens, kDepth});
  graph.Composite("odml.sdpa_transposed",
                  [&](flexbuffers::Builder* attributes) {
                    SdpaAttributes(attributes, /*global=*/true);
                  },
                  {&q_quantized, &k_out, &v_out, &mask, &params}, {&attention});
  graph.Output(k_out);
  graph.Output(v_out);
  graph.Output(graph.MaskSelect(attention, mask, kTokens, kQueryHeads, kDepth,
                                kCacheLen));

  const size_t row_count = static_cast<size_t>(kHeads) * kTokens * kDepth;
  const size_t query_count =
      static_cast<size_t>(kQueryHeads) * kTokens * kDepth;
  Session session;
  ASSERT_NO_FATAL_FAILURE(session.Build(
      &graph,
      {row_count * 2, row_count * 2, query_count * 2, cache_bytes, cache_bytes,
       static_cast<size_t>(kTokens) * kCacheLen, 7 * sizeof(int32_t),
       4 * sizeof(int32_t), 4 * sizeof(int32_t)},
      {cache_bytes, cache_bytes, query_count * 2}, {{0, 3}, {1, 4}}));
  EXPECT_GT(session.LayerNames("cuda_quant_cache_update"), 0);
  EXPECT_GT(session.LayerNames("cuda_quant_attention"), 0);
  // Neither the SELECT nor the reduction of the [tokens, context] mask is in
  // the engine.
  EXPECT_FALSE(session.LayersMention("Select"));
  EXPECT_FALSE(session.LayersMention("Reduce"));

  Random random(31);
  Caches caches = RandomCaches(kCacheLen, 4, &random);
  session.SetInput(3, StoredCache(caches.k, 4));
  session.SetInput(4, StoredCache(caches.v, 4));
  struct Chunk {
    int start;
    int steps;  // the tokens that are not padding
  };
  for (const Chunk chunk : {Chunk{0, 128}, Chunk{128, 128}, Chunk{256, 100}}) {
    SCOPED_TRACE(::testing::Message() << "start=" << chunk.start);
    const FloatTensor k_chunk =
        MakeFloats(RandomValues(row_count, 9.5f * scales.k, &random), false);
    const FloatTensor v_chunk =
        MakeFloats(RandomValues(row_count, 9.5f * scales.v, &random), false);
    const FloatTensor q_chunk = MakeFloats(
        RandomValues(query_count, 110.0f * scales.q, &random), false);
    std::vector<uint8_t> mask_chunk(static_cast<size_t>(kTokens) * kCacheLen,
                                    0);
    for (int t = 0; t < chunk.steps; ++t) {
      std::fill_n(mask_chunk.begin() + static_cast<size_t>(t) * kCacheLen,
                  chunk.start + t + 1, 1);
    }
    session.SetInput(0, k_chunk.bytes);
    session.SetInput(1, v_chunk.bytes);
    session.SetInput(2, q_chunk.bytes);
    session.SetInput(5, mask_chunk);
    session.SetInput(
        6, std::vector<int32_t>{chunk.start, chunk.start + chunk.steps,
                                chunk.start + chunk.steps, 0, 0, 0, 0});
    session.SetInput(7, std::vector<int32_t>{0, 0, chunk.start, 0});
    session.SetInput(8, std::vector<int32_t>{0, 0, 0, chunk.start});
    ASSERT_NO_FATAL_FAILURE(session.Run());

    litert::nvidia::quant_reference::CacheUpdate(
        &caches, QuantizeRows(k_chunk.held, kTokens, false, scales.k, 4),
        QuantizeRows(v_chunk.held, kTokens, true, scales.v, 4), kTokens,
        chunk.start, chunk.start);
    EXPECT_EQ(LoadedCache(session.Output(0), 4, caches.k.size()), caches.k);
    EXPECT_EQ(LoadedCache(session.Output(1), 4, caches.v.size()), caches.v);
    const std::vector<int8_t> quantized = QuantizeQuery(q_chunk.held, scales.q);
    AttentionInputs reference;
    reference.q = &quantized;
    reference.rows = kGroup * kTokens;
    reference.q_scale = scales.q;
    reference.k_scale = scales.k;
    reference.v_scale = scales.v;
    reference.caches = &caches;
    reference.mask = &mask_chunk;
    reference.mask_rows = kTokens;
    reference.fill = kSdpaFill;
    const auto attended = litert::nvidia::quant_reference::Attention(reference);
    // The graph's result is [tokens][query heads][depth], with zero for the
    // tokens of padding, whose mask rows see no key.
    std::vector<float> expected(attended.size(), 0.0f);
    for (int t = 0; t < chunk.steps; ++t) {
      for (int head = 0; head < kQueryHeads; ++head) {
        const size_t from =
            (static_cast<size_t>(head / kGroup) * kGroup * kTokens +
             (head % kGroup) * kTokens + t) *
            kDepth;
        std::copy_n(attended.begin() + from, kDepth,
                    expected.begin() +
                        (static_cast<size_t>(t) * kQueryHeads + head) * kDepth);
      }
    }
    ExpectRowsMatch(ReadFloats(session.Output(2), false), expected, 0.02f);
  }
}

TEST(QuantAttentionGraphTest, LocalBlockOfAPrefillChunk) {
  // q float [1, 8, M, D] -> QUANTIZE -> RESHAPE [1, 2, 4 M, D]; v float
  // [1, M, H, D] -> QUANTIZE -> TRANSPOSE [1, H, D, M]; k float -> QUANTIZE;
  // odml.fused_sdpa_cache_update attends to the ring cache and the chunk,
  // then writes the chunk into the ring.
  constexpr int kTokens = 128;
  const Scales& scales = kLocal;
  const size_t cache_bytes = static_cast<size_t>(kHeads) * kRing * kDepth;
  Graph graph;
  auto& q = graph.Input("q", kLiteRtElementTypeFloat16,
                        {1, kQueryHeads, kTokens, kDepth});
  auto& k =
      graph.Input("k", kLiteRtElementTypeFloat16, {1, kHeads, kTokens, kDepth});
  auto& v =
      graph.Input("v", kLiteRtElementTypeFloat16, {1, kTokens, kHeads, kDepth});
  auto& k_cache = graph.Input("kv_cache_k_5", kLiteRtElementTypeInt8,
                              {1, kHeads, kRing, kDepth}, scales.k);
  auto& v_cache = graph.Input("kv_cache_v_5", kLiteRtElementTypeInt8,
                              {1, kHeads, kDepth, kRing}, scales.v);
  auto& mask = graph.Input("local_mask", kLiteRtElementTypeBool,
                           {1, 1, kTokens, kRing + kTokens});
  auto& params =
      graph.Input("param_tensor", kLiteRtElementTypeInt32, {1, 1, 1, 7});
  auto& q_quantized = graph.Quantize(
      q, "q_quantized", 8, {1, kQueryHeads, kTokens, kDepth}, scales.q);
  auto& q_rows = graph.Reshape(q_quantized, "q_rows", kLiteRtElementTypeInt8,
                               {1, kHeads, kGroup * kTokens, kDepth}, scales.q);
  auto& v_quantized = graph.Quantize(v, "v_quantized", 8,
                                     {1, kTokens, kHeads, kDepth}, scales.v);
  auto& v_columns =
      graph.Transpose(v_quantized, "v_columns", kLiteRtElementTypeInt8,
                      {1, kHeads, kDepth, kTokens}, scales.v, {0, 2, 3, 1});
  auto& k_quantized = graph.Quantize(k, "k_quantized", 8,
                                     {1, kHeads, kTokens, kDepth}, scales.k);
  auto& attention = graph.Tensor("attention", kLiteRtElementTypeFloat16,
                                 {1, kHeads, kGroup * kTokens, kDepth});
  auto& k_out = graph.Tensor("kv_cache_k_5_out", kLiteRtElementTypeInt8,
                             {1, kHeads, kRing, kDepth}, scales.k);
  auto& v_out = graph.Tensor("kv_cache_v_5_out", kLiteRtElementTypeInt8,
                             {1, kHeads, kDepth, kRing}, scales.v);
  graph.Composite(
      "odml.fused_sdpa_cache_update",
      [&](flexbuffers::Builder* attributes) {
        attributes->Int("cache_size", kRing);
        attributes->Float("scale_k", scales.k);
        attributes->Float("scale_q", scales.q);
        attributes->Float("scale_v", scales.v);
        attributes->Bool("update_cache", true);
      },
      {&q_rows, &k_cache, &v_cache, &k_quantized, &v_columns, &mask, &params},
      {&attention, &k_out, &v_out});
  graph.Output(attention);
  graph.Output(k_out);
  graph.Output(v_out);

  const size_t row_count = static_cast<size_t>(kHeads) * kTokens * kDepth;
  const size_t query_count =
      static_cast<size_t>(kQueryHeads) * kTokens * kDepth;
  Session session;
  ASSERT_NO_FATAL_FAILURE(session.Build(
      &graph,
      {query_count * 2, row_count * 2, row_count * 2, cache_bytes, cache_bytes,
       static_cast<size_t>(kTokens) * (kRing + kTokens), 7 * sizeof(int32_t)},
      {query_count * 2, cache_bytes, cache_bytes}, {{1, 3}, {2, 4}}));
  EXPECT_GT(session.LayerNames("cuda_quant_fused_attention"), 0);

  Random random(41);
  Caches caches = RandomCaches(kRing, 8, &random);
  session.SetInput(3, caches.k);
  session.SetInput(4, caches.v);
  struct Chunk {
    int start;
    int steps;   // the tokens that are not padding
    int length;  // the update length LiteRT-LM puts into the parameters
  };
  // A first chunk (LiteRT-LM counts the pending token into its length), a
  // chunk that wraps around the ring, and a padded one.
  for (const Chunk chunk :
       {Chunk{0, 127, 128}, Chunk{127, 128, 128}, Chunk{255, 128, 128},
        Chunk{383, 128, 128}, Chunk{511, 128, 128}, Chunk{639, 70, 70}}) {
    SCOPED_TRACE(::testing::Message() << "start=" << chunk.start);
    const FloatTensor q_chunk = MakeFloats(
        RandomValues(query_count, 110.0f * scales.q, &random), false);
    const FloatTensor k_chunk =
        MakeFloats(RandomValues(row_count, 140.0f * scales.k, &random), false);
    const FloatTensor v_chunk =
        MakeFloats(RandomValues(row_count, 140.0f * scales.v, &random), false);
    const std::vector<uint8_t> mask_chunk =
        RingPrefillMask(kRing, kTokens, chunk.start, chunk.steps, kRing);
    session.SetInput(0, q_chunk.bytes);
    session.SetInput(1, k_chunk.bytes);
    session.SetInput(2, v_chunk.bytes);
    session.SetInput(5, mask_chunk);
    session.SetInput(
        6, std::vector<int32_t>{chunk.start, chunk.start + chunk.length,
                                chunk.start + chunk.length, 0, 0, 0, 0});
    ASSERT_NO_FATAL_FAILURE(session.Run());

    const std::vector<int8_t> quantized = QuantizeQuery(q_chunk.held, scales.q);
    const std::vector<int8_t> k_rows =
        QuantizeRows(k_chunk.held, kTokens, false, scales.k, 8);
    const std::vector<int8_t> v_rows =
        QuantizeRows(v_chunk.held, kTokens, true, scales.v, 8);
    AttentionInputs reference;
    reference.q = &quantized;
    reference.rows = kGroup * kTokens;
    reference.q_scale = scales.q;
    reference.k_scale = scales.k;
    reference.v_scale = scales.v;
    reference.caches = &caches;
    reference.k_new = &k_rows;
    reference.v_new = &v_rows;
    reference.new_len = kTokens;
    reference.mask = &mask_chunk;
    reference.mask_rows = kTokens;
    reference.fill = kFusedFill;
    auto expected = litert::nvidia::quant_reference::Attention(reference);
    // The rows of padding see no key: the plugin gives them zero where the
    // decomposition averages every value. Nothing reads them.
    for (size_t row = 0; row < expected.size() / kDepth; ++row) {
      if (static_cast<int>(row % kTokens) >= chunk.steps) {
        std::fill_n(expected.begin() + row * kDepth, kDepth, 0.0f);
      }
    }
    ExpectRowsMatch(ReadFloats(session.Output(0), false), expected, 0.02f);
    litert::nvidia::quant_reference::RingUpdate(&caches, k_rows, v_rows,
                                                kTokens, chunk.start,
                                                chunk.start + chunk.length);
    EXPECT_EQ(LoadedCache(session.Output(1), 8, caches.k.size()), caches.k);
    EXPECT_EQ(LoadedCache(session.Output(2), 8, caches.v.size()), caches.v);
  }
}

TEST(QuantAttentionGraphTest, RejectsWhatThePluginDoesNotRun) {
  // Shapes outside the kernels' contracts stay off the engine: a prefill
  // chunk whose rows are not whole blocks of the tensor-core kernel, and a
  // head depth the kernels are not built for.
  const auto sdpa_supported = [](int tokens, int depth, int cache_len) {
    Graph graph;
    auto& q = graph.Input("q", kLiteRtElementTypeInt8,
                          {1, kQueryHeads, tokens, depth}, kLocal.q);
    auto& k_cache = graph.Input("k_cache", kLiteRtElementTypeInt8,
                                {1, kHeads, cache_len, depth}, kLocal.k);
    auto& v_cache = graph.Input("v_cache", kLiteRtElementTypeInt8,
                                {1, kHeads, depth, cache_len}, kLocal.v);
    auto& mask =
        graph.Input("mask", kLiteRtElementTypeBool, {1, 1, tokens, cache_len});
    auto& params =
        graph.Input("param_tensor", kLiteRtElementTypeInt32, {1, 1, 1, 7});
    auto& attention = graph.Tensor("attention", kLiteRtElementTypeFloat16,
                                   {1, kQueryHeads, tokens, depth});
    graph.Composite("odml.sdpa_transposed",
                    [&](flexbuffers::Builder* attributes) {
                      SdpaAttributes(attributes, /*global=*/false);
                    },
                    {&q, &k_cache, &v_cache, &mask, &params}, {&attention});
    graph.Output(attention);
    const litert::compiler::Subgraph subgraph(LrtGetCompilerContext(),
                                              &graph.Get());
    return litert::nvidia::IsTensorRtOpSupported(subgraph.Ops()[0]);
  };
  EXPECT_TRUE(sdpa_supported(1, kDepth, kRing));
  EXPECT_TRUE(sdpa_supported(4, kDepth, kRing));  // 16 rows
  EXPECT_TRUE(sdpa_supported(128, kDepth, 1024));
  EXPECT_FALSE(sdpa_supported(5, kDepth, kRing));  // 20 rows
  EXPECT_FALSE(sdpa_supported(100, kDepth, 1024));
  EXPECT_FALSE(sdpa_supported(128, kDepth, 1000));  // not whole tiles
  EXPECT_FALSE(sdpa_supported(1, 128, kRing));
}

}  // namespace
