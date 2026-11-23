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

// Qwen 4's sparse-attention block as a graph holds it --
// odml.qsa_indexer_update over the indexer's ring and pool, odml.cache_update
// over the KV caches and odml.sparse_sdpa_transposed reading the updated caches
// with the indexer's selection -- lowered by the graph builder and run step by
// step with every state in place (as LiteRT-LM binds it), against the host
// references of trtllm/qsa_reference.h: decode steps across block boundaries, a
// verify step, right-padded prefill chunks, and state outputs in buffers of
// their own.

#include <algorithm>
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
#include "litert/vendors/nvidia/trtllm/qsa_reference.h"

namespace {

namespace ref = litert::nvidia::qsa_reference;
using litert::nvidia::quant_test::Graph;

// A reduced Qwen 4 attention layer: 8 query heads on 2 KV heads of 128, an
// indexer of 4 heads of 64 with 32 rotary dims, a 256-key cache (64 blocks)
// and 8 blocks per row.
constexpr int kQueryHeads = 8;
constexpr int kKvHeads = 2;
constexpr int kDepth = 128;
constexpr int kIndexHeads = 4;
constexpr int kIndexDim = 64;
constexpr int kRotary = 32;
constexpr int kCacheLen = 256;
constexpr int kBlocks = kCacheLen / 4;
constexpr int kTopk = 8;
constexpr float kIndexScale = 0.125f;
constexpr float kAttentionScale = 0.08838834764f;

class Random {
 public:
  explicit Random(uint32_t seed) : state_(seed * 2654435761u + 7u) {}
  float Next() {  // [-1, 1)
    state_ = state_ * 1664525u + 1013904223u;
    return static_cast<float>(state_ >> 8) / 8388608.0f - 1.0f;
  }
  std::vector<float> Values(size_t count, float range = 1.0f) {
    std::vector<float> values(count);
    for (auto& value : values) value = Next() * range;
    return values;
  }

 private:
  uint32_t state_;
};

// The values a FLOAT32 graph input has inside the engine (BF16 activations),
// and what an FP16 cache then holds of them.
std::vector<float> AsBf16(std::vector<float> values) {
  for (auto& value : values) {
    value = ref::Bf16BitsToFloat(ref::FloatToBf16Bits(value));
  }
  return values;
}
std::vector<float> AsHalf(std::vector<float> values) {
  for (auto& value : values) value = ref::RoundToHalf(value);
  return values;
}
std::vector<uint16_t> HalfBits(const std::vector<float>& values) {
  std::vector<uint16_t> bits(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    bits[i] = ref::FloatToHalfBits(values[i]);
  }
  return bits;
}
std::vector<float> FromHalfBytes(const std::vector<uint8_t>& bytes) {
  std::vector<float> values(bytes.size() / 2);
  for (size_t i = 0; i < values.size(); ++i) {
    uint16_t bits;
    std::memcpy(&bits, bytes.data() + 2 * i, 2);
    values[i] = ref::HalfBitsToFloat(bits);
  }
  return values;
}
template <typename T>
std::vector<T> FromBytes(const std::vector<uint8_t>& bytes) {
  std::vector<T> values(bytes.size() / sizeof(T));
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

// The block's graph for steps of `tokens` rows. Inputs, in order: q_idx,
// k_idx_new, ring, pool, start_pos, mask, q, k, v, params, k cache, v cache,
// k start, v start. Outputs: attention, topk, ring', pool', k cache',
// v cache'.
struct BlockGraph {
  Graph graph;
  std::vector<float> norm, cos, sin;

  explicit BlockGraph(int tokens) {
    Random random(1234);
    norm = random.Values(kIndexDim, 0.4f);
    for (auto& w : norm) w += 1.0f;
    cos.resize(kBlocks * kRotary);
    sin.resize(kBlocks * kRotary);
    for (int b = 0; b < kBlocks; ++b) {
      for (int i = 0; i < kRotary; ++i) {
        const double angle =
            4.0 * b * std::pow(10000.0, -(2.0 * (i % (kRotary / 2))) / kRotary);
        cos[b * kRotary + i] = static_cast<float>(std::cos(angle));
        sin[b * kRotary + i] = static_cast<float>(std::sin(angle));
      }
    }
    const auto f32 = kLiteRtElementTypeFloat32;
    const auto i32 = kLiteRtElementTypeInt32;
    auto& q_idx =
        graph.Input("q_idx", f32, {1, tokens, kIndexHeads, kIndexDim});
    auto& k_idx = graph.Input("k_idx_new", f32, {1, tokens, kIndexDim});
    auto& ring = graph.Input("kv_cache_qsa_ring_0", f32, {1, 4, kIndexDim});
    auto& pool =
        graph.Input("kv_cache_qsa_pool_0", f32, {1, kBlocks, kIndexDim});
    auto& start = graph.Input("start_pos", i32, {1});
    auto& mask =
        graph.Input("mask", kLiteRtElementTypeBool, {1, 1, tokens, kCacheLen});
    auto& q = graph.Input("q", f32, {1, kQueryHeads, tokens, kDepth});
    auto& k = graph.Input("k", f32, {1, kKvHeads, tokens, kDepth});
    auto& v = graph.Input("v", f32, {1, kKvHeads, tokens, kDepth});
    auto& params = graph.Input("param_tensor", i32, {1, 1, 1, 7});
    auto& k_cache =
        graph.Input("kv_cache_k_0", f32, {1, kKvHeads, kCacheLen, kDepth});
    auto& v_cache =
        graph.Input("kv_cache_v_0", f32, {1, kKvHeads, kDepth, kCacheLen});
    auto& k_start = graph.Input("k_start", i32, {4});
    auto& v_start = graph.Input("v_start", i32, {4});

    auto& norm_t = FloatConstant("k_norm_weight", {kIndexDim}, norm);
    auto& cos_t = FloatConstant("block_cos", {1, kBlocks, kRotary}, cos);
    auto& sin_t = FloatConstant("block_sin", {1, kBlocks, kRotary}, sin);
    auto& topk = graph.Tensor("topk", i32, {1, tokens, kTopk});
    auto& ring_out =
        graph.Tensor("kv_cache_qsa_ring_0_out", f32, {1, 4, kIndexDim});
    auto& pool_out =
        graph.Tensor("kv_cache_qsa_pool_0_out", f32, {1, kBlocks, kIndexDim});
    graph.Composite(
        "odml.qsa_indexer_update",
        [](flexbuffers::Builder* attributes) {
          attributes->Int("block_topk", kTopk);
          attributes->Int("compress_ratio", 4);
          attributes->Float("rms_norm_eps", 1e-6f);
          attributes->Float("scale", kIndexScale);
        },
        {&q_idx, &k_idx, &ring, &pool, &norm_t, &cos_t, &sin_t, &start, &mask},
        {&topk, &ring_out, &pool_out});

    auto& k_out =
        graph.Tensor("kv_cache_k_0_out", f32, {1, kKvHeads, kCacheLen, kDepth});
    auto& v_out =
        graph.Tensor("kv_cache_v_0_out", f32, {1, kKvHeads, kDepth, kCacheLen});
    graph.Composite("odml.cache_update",
                    [](flexbuffers::Builder* attributes) {
                      attributes->Int("cache_size", kCacheLen);
                      attributes->Int("head_size", kDepth);
                      attributes->Int("kv_cache_batch_size", kKvHeads);
                    },
                    {&k, &v, &params, &k_cache, &v_cache, &k_start, &v_start},
                    {&k_out, &v_out});

    auto& attention =
        graph.Tensor("attention", f32, {1, tokens, kQueryHeads, kDepth});
    graph.Composite("odml.sparse_sdpa_transposed",
                    [](flexbuffers::Builder* attributes) {
                      attributes->Int("block_topk", kTopk);
                      attributes->Int("compress_ratio", 4);
                      attributes->Int("k_ts_idx", 2);
                      attributes->Float("scale", kAttentionScale);
                      attributes->Int("v_ts_idx", 3);
                    },
                    {&q, &k_out, &v_out, &topk, &mask}, {&attention});
    for (auto* output :
         {&attention, &topk, &ring_out, &pool_out, &k_out, &v_out}) {
      graph.Output(*output);
    }
  }

  LiteRtTensorT& FloatConstant(const char* name, std::vector<int32_t> dims,
                               const std::vector<float>& values) {
    auto& tensor = graph.Tensor(name, kLiteRtElementTypeFloat32, dims);
    SetWeightsFromUnownedBuffer(
        tensor.Weights(),
        litert::BufferRef<uint8_t>(values.data(), values.size() * 4));
    return tensor;
  }
};

// The engine of a block graph with a device buffer per input and output; the
// state outputs share their inputs' buffers when `in_place`.
class Session {
 public:
  void Build(BlockGraph* block, int tokens, bool in_place) {
    tokens_ = tokens;
    const litert::compiler::Subgraph subgraph(LrtGetCompilerContext(),
                                              &block->graph.Get());
    for (const auto& op : subgraph.Ops()) {
      EXPECT_TRUE(litert::nvidia::IsTensorRtOpSupported(op))
          << "op code " << op.Code();
    }
    auto built = litert::nvidia::BuildTensorRtEngine(subgraph);
    ASSERT_TRUE(built.HasValue()) << built.Error().Message();
    input_names_ = built->input_names;
    output_names_ = built->output_names;
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    engine_.reset(runtime_->deserializeCudaEngine(built->engine.data(),
                                                  built->engine.size()));
    ASSERT_NE(engine_, nullptr);
    std::unique_ptr<nvinfer1::IEngineInspector> inspector(
        engine_->createEngineInspector());
    layers_ = inspector->getEngineInformation(
        nvinfer1::LayerInformationFormat::kJSON);
    context_.reset(engine_->createExecutionContext());
    ASSERT_NE(context_, nullptr);
    const size_t f = 4;  // FLOAT32 boundary
    const size_t cache = static_cast<size_t>(kKvHeads) * kCacheLen * kDepth * 2;
    input_bytes_ = {static_cast<size_t>(tokens) * kIndexHeads * kIndexDim * f,
                    static_cast<size_t>(tokens) * kIndexDim * f,
                    4 * kIndexDim * f,
                    static_cast<size_t>(kBlocks) * kIndexDim * f,
                    4,
                    static_cast<size_t>(tokens) * kCacheLen,
                    static_cast<size_t>(kQueryHeads) * tokens * kDepth * f,
                    static_cast<size_t>(kKvHeads) * tokens * kDepth * f,
                    static_cast<size_t>(kKvHeads) * tokens * kDepth * f,
                    7 * 4,
                    cache,
                    cache,
                    16,
                    16};
    output_bytes_ = {static_cast<size_t>(tokens) * kQueryHeads * kDepth * f,
                     static_cast<size_t>(tokens) * kTopk * 4,
                     input_bytes_[2],
                     input_bytes_[3],
                     cache,
                     cache};
    ASSERT_EQ(input_names_.size(), input_bytes_.size());
    ASSERT_EQ(output_names_.size(), output_bytes_.size());
    inputs_.assign(input_bytes_.size(), nullptr);
    outputs_.assign(output_bytes_.size(), nullptr);
    for (size_t i = 0; i < inputs_.size(); ++i) {
      ASSERT_EQ(cudaMalloc(&inputs_[i], input_bytes_[i]), cudaSuccess);
      ASSERT_TRUE(
          context_->setTensorAddress(input_names_[i].c_str(), inputs_[i]));
    }
    // The KV caches always share their buffers (the engine aliases them);
    // the indexer's state when `in_place`.
    outputs_[4] = inputs_[10];
    outputs_[5] = inputs_[11];
    if (in_place) {
      outputs_[2] = inputs_[2];
      outputs_[3] = inputs_[3];
    }
    for (size_t i = 0; i < outputs_.size(); ++i) {
      if (outputs_[i] == nullptr) {
        ASSERT_EQ(cudaMalloc(&outputs_[i], output_bytes_[i]), cudaSuccess);
        owned_.push_back(outputs_[i]);
      }
      ASSERT_TRUE(
          context_->setTensorAddress(output_names_[i].c_str(), outputs_[i]));
    }
    ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess);
  }

  ~Session() {
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
    for (void* pointer : inputs_) cudaFree(pointer);
    for (void* pointer : owned_) cudaFree(pointer);
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
  bool LayersMention(const std::string& text) const {
    return layers_.find(text) != std::string::npos;
  }
  const nvinfer1::ICudaEngine& engine() const { return *engine_; }
  const std::vector<std::string>& input_names() const { return input_names_; }

 private:
  int tokens_ = 0;
  litert::nvidia::TensorRtLogger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  std::vector<std::string> input_names_, output_names_;
  std::vector<void*> inputs_, outputs_, owned_;
  std::vector<size_t> input_bytes_, output_bytes_;
  std::string layers_;
  cudaStream_t stream_ = nullptr;
};

// The block's state on the host, as the engine holds it: FP32 ring and pool,
// FP16 caches with the values as rows [H][L][D].
struct State {
  std::vector<float> ring, pool, k, v;
};

// One step of `tokens` rows at `start` of which the first `valid` are real,
// run on the engine and the references, compared, the state carried on.
void Step(Session& session, BlockGraph& block, State& state, int tokens,
          int start, int valid, uint32_t seed) {
  SCOPED_TRACE(::testing::Message() << "tokens=" << tokens << " start=" << start
                                    << " valid=" << valid);
  Random random(seed);
  const auto q_idx =
      random.Values(static_cast<size_t>(tokens) * kIndexHeads * kIndexDim);
  const auto k_idx = random.Values(static_cast<size_t>(tokens) * kIndexDim);
  const auto q =
      random.Values(static_cast<size_t>(kQueryHeads) * tokens * kDepth);
  const auto k = random.Values(static_cast<size_t>(kKvHeads) * tokens * kDepth);
  const auto v = random.Values(static_cast<size_t>(kKvHeads) * tokens * kDepth);
  // LiteRT-LM's causal mask; padded rows see nothing.
  std::vector<uint8_t> mask(static_cast<size_t>(tokens) * kCacheLen, 0);
  for (int t = 0; t < valid; ++t) {
    for (int c = 0; c <= start + t; ++c) mask[t * kCacheLen + c] = 1;
  }
  session.SetInput(0, q_idx);
  session.SetInput(1, k_idx);
  session.SetInput(2, state.ring);
  session.SetInput(3, state.pool);
  session.SetInput(4, std::vector<int32_t>{start});
  session.SetInput(5, mask);
  session.SetInput(6, q);
  session.SetInput(7, k);
  session.SetInput(8, v);
  session.SetInput(9,
                   std::vector<int32_t>{start, start + tokens, 0, 0, 0, 0, 0});
  session.SetInput(10, HalfBits(state.k));
  session.SetInput(11, HalfBits(state.v));
  session.SetInput(12, std::vector<int32_t>{0, 0, start, 0});
  session.SetInput(13, std::vector<int32_t>{0, 0, 0, start});
  ASSERT_NO_FATAL_FAILURE(session.Run());

  // The indexer on the values the engine holds (BF16 activations).
  ref::IndexerShape shape;
  shape.rows = tokens;
  shape.index_heads = kIndexHeads;
  shape.index_dim = kIndexDim;
  shape.rotary_dim = kRotary;
  shape.max_blocks = kBlocks;
  shape.block_topk = kTopk;
  shape.mask_len = kCacheLen;
  shape.scale = kIndexScale;
  const auto q_idx_held = AsBf16(q_idx);
  const auto k_idx_held = AsBf16(k_idx);
  ref::Mask view;
  view.bools = mask.data();
  view.len = kCacheLen;
  ref::IndexerUpdate(shape, q_idx_held, k_idx_held, state.ring, state.pool,
                     block.norm, block.cos, block.sin, start, view);
  const auto ring = FromBytes<float>(session.Output(2));
  const auto pool = FromBytes<float>(session.Output(3));
  for (size_t i = 0; i < ring.size(); ++i) {
    ASSERT_NEAR(ring[i], state.ring[i], 1e-6f) << "ring " << i;
  }
  for (size_t i = 0; i < pool.size(); ++i) {
    ASSERT_NEAR(pool[i], state.pool[i], 2e-5f * (1.0f + std::fabs(pool[i])))
        << "pool " << i;
  }
  // The engine's selection against the scores of its own pool.
  const auto topk = FromBytes<int32_t>(session.Output(1));
  const auto scored = ref::IndexerScores(shape, q_idx_held, pool, start, view);
  for (int t = 0; t < tokens; ++t) {
    std::string why;
    EXPECT_TRUE(ref::IsTopk(scored.scores.data() + t * scored.bound,
                            scored.bound, topk.data() + t * kTopk, kTopk, 2e-5,
                            &why))
        << "row " << t << ": " << why;
  }
  state.ring = ring;
  state.pool = pool;

  // The caches: the step's rows written at start (FP16 of the BF16 values).
  const auto k_held = AsHalf(AsBf16(k));
  const auto v_held = AsHalf(AsBf16(v));
  for (int h = 0; h < kKvHeads; ++h) {
    for (int t = 0; t < tokens; ++t) {
      for (int d = 0; d < kDepth; ++d) {
        const size_t row =
            (static_cast<size_t>(h) * kCacheLen + start + t) * kDepth + d;
        const size_t from = (static_cast<size_t>(h) * tokens + t) * kDepth + d;
        state.k[row] = k_held[from];
        state.v[row] = v_held[from];
      }
    }
  }
  const auto k_cache = FromHalfBytes(session.Output(4));
  const auto v_cache = FromHalfBytes(session.Output(5));
  ASSERT_EQ(k_cache, state.k);
  ASSERT_EQ(v_cache, state.v);

  // The attention over the engine's selection.
  ref::AttentionShape attention;
  attention.rows = tokens;
  attention.q_heads = kQueryHeads;
  attention.kv_heads = kKvHeads;
  attention.head_dim = kDepth;
  attention.cache_len = kCacheLen;
  attention.block_topk = kTopk;
  attention.scale = kAttentionScale;
  const auto want = ref::SparseAttention(
      attention, AsBf16(q), state.k, state.v, topk,
      [&](int row, int token) { return mask[row * kCacheLen + token] != 0; });
  const auto got = FromBytes<float>(session.Output(0));
  ASSERT_EQ(got.size(), want.size());
  for (size_t i = 0; i < got.size(); ++i) {
    ASSERT_NEAR(got[i], want[i], 1e-2f * (1.0f + std::fabs(want[i])))
        << "attention " << i;
  }
}

State RandomState(uint32_t seed) {
  Random random(seed);
  State state;
  state.ring = random.Values(4 * kIndexDim);
  state.pool = random.Values(static_cast<size_t>(kBlocks) * kIndexDim);
  state.k =
      AsHalf(random.Values(static_cast<size_t>(kKvHeads) * kCacheLen * kDepth));
  state.v =
      AsHalf(random.Values(static_cast<size_t>(kKvHeads) * kCacheLen * kDepth));
  return state;
}

void CheckLowered(const Session& session) {
  EXPECT_TRUE(session.LayersMention("cuda_qsa_indexer"));
  EXPECT_TRUE(session.LayersMention("cuda_qsa_attention"));
  // The value cache is held as rows [1, H, L, D], and the composites leave
  // no top-k of their decompositions behind.
  const nvinfer1::Dims value =
      session.engine().getTensorShape(session.input_names()[11].c_str());
  ASSERT_EQ(value.nbDims, 4);
  EXPECT_EQ(value.d[2], kCacheLen);
  EXPECT_EQ(value.d[3], kDepth);
  EXPECT_FALSE(session.LayersMention("TopK"));
}

TEST(QsaGraphTest, DecodeStepsAcrossBlockBoundaries) {
  BlockGraph block(/*tokens=*/1);
  Session session;
  ASSERT_NO_FATAL_FAILURE(session.Build(&block, 1, /*in_place=*/true));
  CheckLowered(session);
  State state = RandomState(1);
  // Positions 101..110 complete blocks 25 and 26; 26 blocks and more compete
  // for 8 places.
  for (int start = 101; start <= 110; ++start) {
    ASSERT_NO_FATAL_FAILURE(
        Step(session, block, state, 1, start, 1, 100 + start));
  }
}

TEST(QsaGraphTest, VerifyStepAndStateInBuffersOfItsOwn) {
  BlockGraph block(/*tokens=*/4);
  for (bool in_place : {true, false}) {
    Session session;
    ASSERT_NO_FATAL_FAILURE(session.Build(&block, 4, in_place));
    State state = RandomState(2);
    ASSERT_NO_FATAL_FAILURE(Step(session, block, state, 4, 61, 4, 3));
    ASSERT_NO_FATAL_FAILURE(Step(session, block, state, 4, 65, 4, 4));
  }
}

TEST(QsaGraphTest, RightPaddedPrefillChunks) {
  BlockGraph block(/*tokens=*/16);
  Session session;
  ASSERT_NO_FATAL_FAILURE(session.Build(&block, 16, /*in_place=*/true));
  CheckLowered(session);
  State state = RandomState(5);
  std::fill(state.ring.begin(), state.ring.end(), 0.0f);
  // Chunks of a 45-token prompt: two full ones (no selection yet, then 8
  // blocks for 8 places) and a padded third (13 real rows) with selection.
  ASSERT_NO_FATAL_FAILURE(Step(session, block, state, 16, 0, 16, 6));
  ASSERT_NO_FATAL_FAILURE(Step(session, block, state, 16, 16, 16, 7));
  ASSERT_NO_FATAL_FAILURE(Step(session, block, state, 16, 32, 13, 8));
}

}  // namespace
