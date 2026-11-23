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

#include "litert/vendors/nvidia/experimental/megakernel/megakernel.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "cuda_runtime_api.h"
#include "litert/vendors/nvidia/experimental/megakernel/megakernel_lab_data.h"
#include "litert/vendors/nvidia/experimental/megakernel/program.h"

// The megakernel against the reference step (today's per-op launches) on a
// small synthetic step: three layers of Gemma 4 26B-A4B's shapes (sliding,
// global, sliding), short caches and a small vocabulary. The megakernel
// sums in its own order (README.md, Numerics), so its tensors are held to
// the reference's within bounds, and to its own bits from step to step.

namespace {

constexpr int kLayers = 3;
constexpr int kDim = 2816;
constexpr int kVocab = 8192;
constexpr int kExperts = 128;
constexpr int kTopK = 8;
constexpr int kExpertHidden = 704;
constexpr int kDenseHidden = 2112;
constexpr int kLocalCache = 1024;
constexpr int kGlobalCache = 2048;
// The grid: as many CTAs as the device keeps resident, the engine's choice.
constexpr int kCtas = 0;
// The megakernel's tensors against the reference's, as the root mean square
// of the differences over that of the values. A BF16 tensor's unit in the
// last place is up to 0.8% of a value and each layer rounds its stream a few
// times: on the RTX 5080 the three layers' inputs differ by 0, 0.01% and
// 0.2%, the output by 0.3%, the logits by 0.3%. The bound leaves room for
// another grid's order of sums, not for a wrong term.
constexpr double kStreamBound = 0.02;

bool IsGlobal(int l) { return l == 1; }

// Device memory, all freed with the model.
class Memory {
 public:
  ~Memory() {
    for (void* p : blocks_) cudaFree(p);
  }
  template <typename T = uint8_t>
  T* Get(size_t bytes) {
    void* p = nullptr;
    if (cudaMalloc(&p, std::max<size_t>(bytes, 256)) != cudaSuccess) {
      ok_ = false;
      return nullptr;
    }
    blocks_.push_back(p);
    return static_cast<T*>(p);
  }
  void Check(cudaError_t status) { ok_ = ok_ && status == cudaSuccess; }
  bool ok() const { return ok_; }

 private:
  std::vector<void*> blocks_;
  bool ok_ = true;
};

// The synthetic model: weights and norms from a hash of a seed (the lab's
// fills), in the ranges of the 26B's.
class Model {
 public:
  explicit Model(Memory& memory) : memory_(memory) {}

  LiteRtNvidiaMegakernelProjection Projection(int rows, int cols, float lo,
                                              float hi) {
    const size_t weights = static_cast<size_t>(rows) * cols / 2;
    const size_t scales = static_cast<size_t>(rows) * (cols / 32);
    auto* w = memory_.Get(weights);
    auto* s = memory_.Get(scales * 2);
    if (w == nullptr || s == nullptr) return {};
    memory_.Check(LiteRtNvidiaMegakernelLabFillBytes(w, weights, seed_++, 0,
                                                     255, nullptr));
    memory_.Check(
        LiteRtNvidiaMegakernelLabFillHalf(s, scales, seed_++, lo, hi, nullptr));
    return {w, s, rows, cols};
  }

  float* Vector(int count, float lo, float hi) {
    float* v = memory_.Get<float>(static_cast<size_t>(count) * 4);
    if (v == nullptr) return nullptr;
    memory_.Check(
        LiteRtNvidiaMegakernelLabFillFloat(v, count, seed_++, lo, hi, nullptr));
    return v;
  }

  // Compact expert scales: a byte per block of each row, and the rows'
  // maxima as FP16.
  void CompactScales(size_t rows, int cols, float max_lo, float max_hi,
                     const void** fractions, const void** maxima) {
    const size_t blocks = rows * (cols / 32);
    auto* f = memory_.Get(blocks);
    auto* m = memory_.Get(rows * 2);
    if (f == nullptr || m == nullptr) return;
    memory_.Check(LiteRtNvidiaMegakernelLabFillBytes(f, blocks, seed_++, 64,
                                                     255, nullptr));
    memory_.Check(LiteRtNvidiaMegakernelLabFillHalf(m, rows, seed_++, max_lo,
                                                    max_hi, nullptr));
    *fractions = f;
    *maxima = m;
  }

  // The experts as the MoE plugin's arena holds them (expert major, compact
  // scales), shared by the layers.
  LiteRtNvidiaMoeArena Experts() {
    LiteRtNvidiaMoeArena a{};
    const size_t gate_rows = static_cast<size_t>(kExperts) * kExpertHidden;
    const size_t down_rows = static_cast<size_t>(kExperts) * kDim;
    const size_t gate_bytes = gate_rows * kDim / 2;
    const size_t down_bytes = down_rows * kExpertHidden / 2;
    auto* gate = memory_.Get(gate_bytes);
    auto* up = memory_.Get(gate_bytes);
    auto* down = memory_.Get(down_bytes);
    if (gate == nullptr || up == nullptr || down == nullptr) return a;
    memory_.Check(LiteRtNvidiaMegakernelLabFillBytes(gate, gate_bytes, seed_++,
                                                     0, 255, nullptr));
    memory_.Check(LiteRtNvidiaMegakernelLabFillBytes(up, gate_bytes, seed_++, 0,
                                                     255, nullptr));
    memory_.Check(LiteRtNvidiaMegakernelLabFillBytes(down, down_bytes, seed_++,
                                                     0, 255, nullptr));
    a.gate = gate;
    a.up = up;
    a.down = down;
    CompactScales(gate_rows, kDim, 0.005f, 0.008f, &a.gate_scales,
                  &a.gate_scale_max);
    CompactScales(gate_rows, kDim, 0.005f, 0.008f, &a.up_scales,
                  &a.up_scale_max);
    CompactScales(down_rows, kExpertHidden, 0.02f, 0.03f, &a.down_scales,
                  &a.down_scale_max);
    a.compact_scales = 1;
    a.expert_major = 1;
    return a;
  }

  // With `pointers` off the model is its shapes alone.
  void Build(bool pointers) {
    LiteRtNvidiaMoeArena experts{};
    experts.compact_scales = 1;
    experts.expert_major = 1;
    if (pointers) experts = Experts();
    layers_.resize(kLayers);
    for (int l = 0; l < kLayers; ++l) {
      LiteRtNvidiaMegakernelLayer& layer = layers_[l];
      std::memset(&layer, 0, sizeof(layer));
      const bool global = IsGlobal(l);
      layer.shared_kv = global ? 1 : 0;
      layer.q_heads = 16;
      layer.kv_heads = global ? 2 : 8;
      layer.head_dim = global ? 512 : 256;
      layer.rotary_dim = global ? 128 : 256;
      layer.rope_min_timescale = 1.0f;
      layer.rope_max_timescale = global ? 1e6f : 1e4f;
      const int qkv_rows =
          (layer.q_heads + (global ? 1 : 2) * layer.kv_heads) * layer.head_dim;
      layer.qkv = {nullptr, nullptr, qkv_rows, kDim};
      layer.cache_len = global ? kGlobalCache : kLocalCache;
      layer.ring = global ? 0 : 1;
      layer.mask = global ? 1 : 0;
      layer.o = {nullptr, nullptr, kDim, layer.q_heads * layer.head_dim};
      layer.arena = experts;
      layer.experts = kExperts;
      layer.top_k = kTopK;
      layer.expert_hidden = kExpertHidden;
      layer.expert_gate = 1;
      layer.dense_hidden = kDenseHidden;
      layer.dense_gate = 1;
      layer.router_multiplier = 1.0f;
      layer.residual_multiplier = 0.6f + 0.1f * static_cast<float>(l);
      if (!pointers) continue;
      layer.input_norm = Vector(kDim, 0.5f, 1.5f);
      layer.qkv = Projection(qkv_rows, kDim, 0.002f, 0.006f);
      layer.q_norm = Vector(layer.head_dim, 0.5f, 1.5f);
      layer.k_norm = Vector(layer.head_dim, 0.05f, 0.15f);
      const size_t cache = static_cast<size_t>(layer.kv_heads) *
                           layer.cache_len * layer.head_dim;
      layer.k_cache = memory_.Get(cache * 2);
      layer.v_cache = memory_.Get(cache * 2);
      if (layer.k_cache == nullptr || layer.v_cache == nullptr) return;
      memory_.Check(LiteRtNvidiaMegakernelLabFillHalf(
          layer.k_cache, cache, seed_++, -0.17f, 0.17f, nullptr));
      memory_.Check(LiteRtNvidiaMegakernelLabFillHalf(
          layer.v_cache, cache, seed_++, -1.7f, 1.7f, nullptr));
      layer.o =
          Projection(kDim, layer.q_heads * layer.head_dim, 0.0015f, 0.0045f);
      layer.post_attention_norm = Vector(kDim, 0.5f, 1.5f);
      const LiteRtNvidiaMegakernelProjection router =
          Projection(kExperts, kDim, 0.002f, 0.006f);
      layer.arena.router = router.weights;
      layer.arena.router_scales = router.scales;
      const LiteRtNvidiaMegakernelProjection gate_up =
          Projection(2 * kDenseHidden, kDim, 0.002f, 0.006f);
      layer.arena.dense_gate_up = gate_up.weights;
      layer.arena.dense_gate_up_scales = gate_up.scales;
      const LiteRtNvidiaMegakernelProjection down =
          Projection(kDim, kDenseHidden, 0.0025f, 0.0075f);
      layer.arena.dense_down = down.weights;
      layer.arena.dense_down_scales = down.scales;
      layer.per_expert_scale = Vector(kExperts, 0.5f, 1.5f);
      layer.router_norm = Vector(kDim, 0.5f, 1.5f);
      layer.expert_norm = Vector(kDim, 0.5f, 1.5f);
      layer.dense_norm = Vector(kDim, 0.5f, 1.5f);
      layer.dense_post = Vector(kDim, 0.5f, 1.5f);
      layer.moe_post = Vector(kDim, 0.5f, 1.5f);
      layer.sum_post = Vector(kDim, 0.5f, 1.5f);
    }
    model_.layers = kLayers;
    model_.model_dim = kDim;
    model_.vocab = kVocab;
    model_.norm_eps = 1e-6f;
    model_.layer = layers_.data();
    model_.head = {nullptr, nullptr, kVocab, kDim};
    model_.soft_cap = 30.0f;
    model_.soft_cap_in = 1.0f / 30.0f;
    if (pointers) {
      model_.final_norm = Vector(kDim, 0.5f, 1.5f);
      model_.head = Projection(kVocab, kDim, 0.002f, 0.006f);
      memory_.Check(cudaDeviceSynchronize());
    }
  }

  const LiteRtNvidiaMegakernelModel& model() const { return model_; }
  const std::vector<LiteRtNvidiaMegakernelLayer>& layers() const {
    return layers_;
  }

 private:
  Memory& memory_;
  uint64_t seed_ = 0x51ed270b;
  std::vector<LiteRtNvidiaMegakernelLayer> layers_;
  LiteRtNvidiaMegakernelModel model_{};
};

template <typename T>
std::vector<T> Fetch(const void* device, size_t count) {
  std::vector<T> host(count);
  if (cudaMemcpy(host.data(), device, count * sizeof(T),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    ADD_FAILURE() << "cudaMemcpy";
  }
  return host;
}

float Bf16ToFloat(uint16_t bits) {
  const uint32_t word = static_cast<uint32_t>(bits) << 16;
  float value;
  std::memcpy(&value, &word, 4);
  return value;
}

float HalfToFloat(uint16_t bits) {
  const int exponent = (bits >> 10) & 0x1f;
  const int mantissa = bits & 0x3ff;
  const float sign = (bits & 0x8000) != 0 ? -1.0f : 1.0f;
  if (exponent == 0x1f) return mantissa != 0 ? NAN : sign * INFINITY;
  if (exponent == 0)
    return sign * std::ldexp(static_cast<float>(mantissa), -24);
  return sign * std::ldexp(static_cast<float>(1024 + mantissa), exponent - 25);
}

std::vector<float> Bf16Values(const void* device, size_t count) {
  const auto bits = Fetch<uint16_t>(device, count);
  std::vector<float> values(count);
  for (size_t i = 0; i < count; ++i) values[i] = Bf16ToFloat(bits[i]);
  return values;
}

// How far `got` is from `ref`: the root mean square of the differences over
// that of the reference's values. Every value finite.
double RelativeError(const std::vector<float>& ref,
                     const std::vector<float>& got) {
  double difference = 0.0, magnitude = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    if (!std::isfinite(ref[i]) || !std::isfinite(got[i])) return INFINITY;
    difference += (static_cast<double>(ref[i]) - got[i]) * (ref[i] - got[i]);
    magnitude += static_cast<double>(ref[i]) * ref[i];
  }
  return magnitude > 0.0 ? std::sqrt(difference / magnitude) : INFINITY;
}

// The rows a step at `position` writes into a layer's caches (FP16), the
// keys' then the values'.
std::vector<float> WrittenRows(const LiteRtNvidiaMegakernelLayer& layer,
                               int position) {
  const int row = layer.ring != 0 ? position % layer.cache_len
                                  : std::min(position, layer.cache_len - 1);
  std::vector<float> rows;
  for (const void* cache : {layer.k_cache, layer.v_cache}) {
    for (int h = 0; h < layer.kv_heads; ++h) {
      const auto bits = Fetch<uint16_t>(
          static_cast<const uint16_t*>(cache) +
              (static_cast<size_t>(h) * layer.cache_len + row) * layer.head_dim,
          layer.head_dim);
      for (const uint16_t value : bits) rows.push_back(HalfToFloat(value));
    }
  }
  return rows;
}

int Argmax(const std::vector<float>& values) {
  return static_cast<int>(std::max_element(values.begin(), values.end()) -
                          values.begin());
}

TEST(MegakernelTest, BuildsTheShapesOfAModelWithoutPointers) {
  Memory memory;
  Model shapes(memory);
  shapes.Build(/*pointers=*/false);
  auto program = std::make_unique<LiteRtNvidiaMegakernelProgram>();
  const cudaError_t built =
      LiteRtNvidiaMegakernelBuildProgram(&shapes.model(), kCtas, program.get());
  if (built == cudaErrorNotSupported) {
    GTEST_SKIP() << "The device does not run the megakernel.";
  }
  ASSERT_EQ(built, cudaSuccess);
  EXPECT_EQ(program->version, kLiteRtNvidiaMegakernelVersion);
  EXPECT_GT(program->ctas, 0);
  EXPECT_LE(program->ctas, kLiteRtNvidiaMegakernelMaxCtas);
  EXPECT_EQ(program->layers, kLayers);
  EXPECT_GT(program->instruction_count, kLayers);
  EXPECT_LE(program->instruction_count, kLiteRtNvidiaMegakernelMaxInstructions);
  EXPECT_GT(program->workspace_bytes, 0u);
  EXPECT_GT(program->shared_bytes, 0);
  // Every instruction waits for earlier ones only.
  for (int i = 0; i < program->instruction_count; ++i) {
    for (const int32_t wait : program->instruction[i].waits) {
      EXPECT_LT(wait, i) << "instruction " << i;
    }
  }
  // A layer the kernels do not take: heads of another depth.
  std::vector<LiteRtNvidiaMegakernelLayer> layers = shapes.layers();
  layers[0].head_dim = 128;
  layers[0].qkv.rows = (layers[0].q_heads + 2 * layers[0].kv_heads) * 128;
  layers[0].o.cols = layers[0].q_heads * 128;
  LiteRtNvidiaMegakernelModel other = shapes.model();
  other.layer = layers.data();
  EXPECT_EQ(LiteRtNvidiaMegakernelBuildProgram(&other, kCtas, program.get()),
            cudaErrorNotSupported);
  EXPECT_EQ(LiteRtNvidiaMegakernelBuildProgram(nullptr, kCtas, program.get()),
            cudaErrorInvalidValue);
}

TEST(MegakernelTest, AStepMatchesTheReferenceStep) {
  Memory memory;
  Model model(memory);
  model.Build(/*pointers=*/true);
  ASSERT_TRUE(memory.ok());
  auto program = std::make_unique<LiteRtNvidiaMegakernelProgram>();
  const cudaError_t built =
      LiteRtNvidiaMegakernelBuildProgram(&model.model(), kCtas, program.get());
  if (built == cudaErrorNotSupported) {
    GTEST_SKIP() << "The device does not run the megakernel.";
  }
  ASSERT_EQ(built, cudaSuccess);
  auto* device_program =
      memory.Get<LiteRtNvidiaMegakernelProgram>(sizeof(*program));
  ASSERT_NE(device_program, nullptr);
  ASSERT_EQ(cudaMemcpy(device_program, program.get(), sizeof(*program),
                       cudaMemcpyHostToDevice),
            cudaSuccess);

  // A step in the middle of the caches: every row of the sliding layers'
  // ring is visible, and the global layer's keys up to the token.
  const int position = 700;
  auto* position_d = memory.Get<int32_t>(4);
  auto* local_mask = memory.Get<bool>(kLocalCache);
  auto* global_mask = memory.Get<bool>(kGlobalCache);
  float* embeddings = memory.Get<float>(kDim * 4);
  void* state = memory.Get(LiteRtNvidiaMegakernelStateBytes());
  ASSERT_TRUE(memory.ok());
  ASSERT_EQ(cudaMemcpy(position_d, &position, 4, cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemset(local_mask, 1, kLocalCache), cudaSuccess);
  ASSERT_EQ(cudaMemset(global_mask, 0, kGlobalCache), cudaSuccess);
  ASSERT_EQ(cudaMemset(global_mask, 1, position + 1), cudaSuccess);
  ASSERT_EQ(cudaMemset(state, 0, LiteRtNvidiaMegakernelStateBytes()),
            cudaSuccess);
  ASSERT_EQ(LiteRtNvidiaMegakernelLabFillFloat(embeddings, kDim, 99, -2.0f,
                                               2.0f, nullptr),
            cudaSuccess);
  // The reference's outputs and workspace, then the megakernel's.
  LiteRtNvidiaMegakernelStep steps[2];
  char* workspaces[2];
  for (int r = 0; r < 2; ++r) {
    steps[r].embeddings = embeddings;
    steps[r].position = position_d;
    steps[r].mask[0] = local_mask;
    steps[r].mask[1] = global_mask;
    steps[r].activations = memory.Get<float>(kDim * 4);
    steps[r].logits = memory.Get<float>(static_cast<size_t>(kVocab) * 4);
    workspaces[r] = memory.Get<char>(program->workspace_bytes);
    ASSERT_TRUE(memory.ok());
    ASSERT_EQ(cudaMemset(workspaces[r], 0, program->workspace_bytes),
              cudaSuccess);
  }
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  cudaStream_t stream = nullptr, side = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
            cudaSuccess);
  ASSERT_EQ(cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking),
            cudaSuccess);
  LiteRtNvidiaMegakernelReferenceStreams streams{side, nullptr, nullptr};
  ASSERT_EQ(cudaEventCreateWithFlags(&streams.fork, cudaEventDisableTiming),
            cudaSuccess);
  ASSERT_EQ(cudaEventCreateWithFlags(&streams.join, cudaEventDisableTiming),
            cudaSuccess);

  ASSERT_EQ(LiteRtNvidiaMegakernelLaunchReference(
                program.get(), &steps[0], workspaces[0], &streams, stream),
            cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  const auto& layers = model.layers();
  std::vector<std::vector<float>> reference_rows;
  for (const auto& layer : layers) {
    reference_rows.push_back(WrittenRows(layer, position));
  }
  const auto launch = [&] {
    return LiteRtNvidiaMegakernelLaunch(program.get(), device_program,
                                        &steps[1], workspaces[1], state,
                                        nullptr, stream);
  };
  ASSERT_EQ(launch(), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  // The residual stream into every layer and out of the last, and the rows
  // the step wrote into the caches.
  for (int l = 0; l <= kLayers; ++l) {
    const uint64_t at = l < kLayers ? program->scratch[l].x : program->final_x;
    const double x = RelativeError(Bf16Values(workspaces[0] + at, kDim),
                                   Bf16Values(workspaces[1] + at, kDim));
    std::printf("layer %d input: relative error %.3g\n", l, x);
    EXPECT_LT(x, kStreamBound) << "the input of layer " << l;
    if (l == kLayers) break;
    const double rows =
        RelativeError(reference_rows[l], WrittenRows(layers[l], position));
    std::printf("layer %d cache rows: relative error %.3g\n", l, rows);
    EXPECT_LT(rows, kStreamBound) << "the cache rows of layer " << l;
  }
  const std::vector<float> reference_logits =
      Fetch<float>(steps[0].logits, kVocab);
  const std::vector<float> logits = Fetch<float>(steps[1].logits, kVocab);
  const double logits_error = RelativeError(reference_logits, logits);
  const double activations_error =
      RelativeError(Fetch<float>(steps[0].activations, kDim),
                    Fetch<float>(steps[1].activations, kDim));
  std::printf("logits: relative error %.3g; activations: %.3g\n", logits_error,
              activations_error);
  EXPECT_LT(logits_error, kStreamBound);
  EXPECT_LT(activations_error, kStreamBound);
  EXPECT_EQ(Argmax(logits), Argmax(reference_logits));

  // The same step again, in the next epoch of the kernel's state, gives the
  // same bits: the kernel's order does not depend on the run.
  const std::vector<float> activations =
      Fetch<float>(steps[1].activations, kDim);
  for (int repeat = 0; repeat < 3; ++repeat) {
    ASSERT_EQ(cudaMemset(steps[1].logits, 0, static_cast<size_t>(kVocab) * 4),
              cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(launch(), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    const std::vector<float> again = Fetch<float>(steps[1].logits, kVocab);
    EXPECT_EQ(std::memcmp(again.data(), logits.data(), kVocab * 4), 0)
        << "repeat " << repeat;
    const std::vector<float> activations_again =
        Fetch<float>(steps[1].activations, kDim);
    EXPECT_EQ(
        std::memcmp(activations_again.data(), activations.data(), kDim * 4), 0)
        << "repeat " << repeat;
  }
  cudaEventDestroy(streams.join);
  cudaEventDestroy(streams.fork);
  cudaStreamDestroy(side);
  cudaStreamDestroy(stream);
}

}  // namespace
