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

// The decode megakernel's lab: a Gemma 4 26B-A4B decode step with the real
// shapes and the runtime's formats on synthetic data (INT4 weights with FP16
// scales, the MoE arenas as the MoE plugin lays them out, FP16 caches filled
// to a context), run (a) as today's per-op launches
// (LiteRtNvidiaMegakernelLaunchReference) and (b) as the megakernel; every
// layer's residual stream, the caches' new rows, the logits and the
// activations are compared bit for bit, and the two steps are timed
// interleaved, each a CUDA graph replayed in turn.
//
//   megakernel_lab [--context N] [--ctas N] [--replays N] [--arenas N]
//                  [--trace FILE.json] [--detail 1] [--seed N]
//
// --context: the token's position (1024: a step at 1k tokens). --ctas: the
// megakernel's grid (0: as many as the device keeps resident). --arenas:
// distinct expert arenas the layers share (each 405 MB; the experts a layer
// reads are as far from the last layer's in memory as in the model). --trace
// writes one traced launch as Chrome trace JSON and prints its summary.
// --detail 1: the intermediate buffers of the first layer that differs.
//
// Every layer's input x and h, and the logits, are reported against the
// reference: the values that differ, the largest absolute and relative
// differences, the top-1 token and the overlap of the top 40.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iterator>
#include <string>
#include <vector>

#include "cuda_runtime_api.h"
#include "litert/vendors/nvidia/experimental/megakernel/megakernel.h"
#include "litert/vendors/nvidia/experimental/megakernel/megakernel_lab_data.h"
#include "litert/vendors/nvidia/experimental/megakernel/mktrace.h"
#include "litert/vendors/nvidia/experimental/megakernel/program.h"
#include "litert/vendors/nvidia/trtllm/int4_block_gemv.h"

namespace {

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    const cudaError_t status_ = (call);                                  \
    if (status_ != cudaSuccess) {                                        \
      std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call, \
                   cudaGetErrorString(status_));                         \
      std::exit(1);                                                      \
    }                                                                    \
  } while (0)

struct Options {
  int context = 1024;
  int ctas = 0;
  int replays = 20;
  int arenas = 4;
  bool detail = false;
  uint64_t seed = 1;
  std::string trace;
};

Options Parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string flag = argv[i];
    const char* value = argv[i + 1];
    if (flag == "--context") {
      o.context = std::atoi(value);
    } else if (flag == "--ctas") {
      o.ctas = std::atoi(value);
    } else if (flag == "--replays") {
      o.replays = std::atoi(value);
    } else if (flag == "--arenas") {
      o.arenas = std::max(1, std::atoi(value));
    } else if (flag == "--detail") {
      o.detail = std::atoi(value) != 0;
    } else if (flag == "--trace") {
      o.trace = value;
    } else if (flag == "--seed") {
      o.seed = std::strtoull(value, nullptr, 10);
    } else {
      std::fprintf(stderr, "unknown flag %s\n", flag.c_str());
      std::exit(2);
    }
  }
  return o;
}

// Device memory, all freed at exit.
class Memory {
 public:
  ~Memory() {
    for (void* p : blocks_) cudaFree(p);
  }
  template <typename T = uint8_t>
  T* Get(size_t bytes) {
    void* p = nullptr;
    CHECK_CUDA(cudaMalloc(&p, std::max<size_t>(bytes, 256)));
    blocks_.push_back(p);
    total_ += bytes;
    return static_cast<T*>(p);
  }
  size_t total() const { return total_; }

 private:
  std::vector<void*> blocks_;
  size_t total_ = 0;
};

// The 26B's shapes.
constexpr int kLayers = 30;
constexpr int kDim = 2816;
constexpr int kVocab = 262144;
constexpr int kExperts = 128;
constexpr int kTopK = 8;
constexpr int kExpertHidden = 704;
constexpr int kDenseHidden = 2112;
constexpr int kLocalCache = 1024;
constexpr int kGlobalCache = 65537;

bool IsGlobal(int l) { return l % 6 == 5; }

class Model {
 public:
  Model(const Options& options, Memory& memory)
      : options_(options), memory_(memory) {}

  uint64_t NextSeed() { return options_.seed * 0x100000001b3ull + seed_++; }

  // A blockwise INT4 projection with FP16 scales uniform in [lo, hi].
  LiteRtNvidiaMegakernelProjection Projection(int rows, int cols, float lo,
                                              float hi) {
    const size_t weights = static_cast<size_t>(rows) * cols / 2;
    const size_t scales = static_cast<size_t>(rows) * (cols / 32);
    auto* w = memory_.Get(weights);
    auto* s = memory_.Get(scales * 2);
    CHECK_CUDA(LiteRtNvidiaMegakernelLabFillBytes(w, weights, NextSeed(), 0,
                                                  255, nullptr));
    CHECK_CUDA(LiteRtNvidiaMegakernelLabFillHalf(s, scales, NextSeed(), lo, hi,
                                                 nullptr));
    return {w, s, rows, cols};
  }

  float* Vector(int count, float lo, float hi) {
    float* v = memory_.Get<float>(static_cast<size_t>(count) * 4);
    CHECK_CUDA(LiteRtNvidiaMegakernelLabFillFloat(v, count, NextSeed(), lo, hi,
                                                  nullptr));
    return v;
  }

  // Compact expert scales: a byte per block of each row, then the rows'
  // maxima as FP16.
  void CompactScales(int rows, int cols, float max_lo, float max_hi,
                     const void** fractions, const void** maxima) {
    const size_t blocks = static_cast<size_t>(rows) * (cols / 32);
    auto* f = memory_.Get(blocks);
    auto* m = memory_.Get(static_cast<size_t>(rows) * 2);
    CHECK_CUDA(LiteRtNvidiaMegakernelLabFillBytes(f, blocks, NextSeed(), 64,
                                                  255, nullptr));
    CHECK_CUDA(LiteRtNvidiaMegakernelLabFillHalf(m, rows, NextSeed(), max_lo,
                                                 max_hi, nullptr));
    *fractions = f;
    *maxima = m;
  }

  // An expert arena of the MoE plugin's layout (expert major, compact
  // scales) without its router and dense pieces.
  LiteRtNvidiaMoeArena Experts() {
    LiteRtNvidiaMoeArena a{};
    const size_t gate_rows = static_cast<size_t>(kExperts) * kExpertHidden;
    const size_t down_rows = static_cast<size_t>(kExperts) * kDim;
    auto* gate = memory_.Get(gate_rows * kDim / 2);
    auto* up = memory_.Get(gate_rows * kDim / 2);
    auto* down = memory_.Get(down_rows * kExpertHidden / 2);
    CHECK_CUDA(LiteRtNvidiaMegakernelLabFillBytes(gate, gate_rows * kDim / 2,
                                                  NextSeed(), 0, 255, nullptr));
    CHECK_CUDA(LiteRtNvidiaMegakernelLabFillBytes(up, gate_rows * kDim / 2,
                                                  NextSeed(), 0, 255, nullptr));
    CHECK_CUDA(LiteRtNvidiaMegakernelLabFillBytes(
        down, down_rows * kExpertHidden / 2, NextSeed(), 0, 255, nullptr));
    a.gate = gate;
    a.up = up;
    a.down = down;
    CompactScales(static_cast<int>(gate_rows), kDim, 0.005f, 0.008f,
                  &a.gate_scales, &a.gate_scale_max);
    CompactScales(static_cast<int>(gate_rows), kDim, 0.005f, 0.008f,
                  &a.up_scales, &a.up_scale_max);
    CompactScales(static_cast<int>(down_rows), kExpertHidden, 0.02f, 0.03f,
                  &a.down_scales, &a.down_scale_max);
    a.compact_scales = 1;
    a.expert_major = 1;
    return a;
  }

  void Build() {
    std::vector<LiteRtNvidiaMoeArena> arenas;
    for (int a = 0; a < options_.arenas; ++a) arenas.push_back(Experts());
    layers_.resize(kLayers);
    std::srand(static_cast<unsigned>(options_.seed));
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
      layer.input_norm = Vector(kDim, 0.5f, 1.5f);
      const int qkv_rows =
          (layer.q_heads + (global ? 1 : 2) * layer.kv_heads) * layer.head_dim;
      layer.qkv = Projection(qkv_rows, kDim, 0.002f, 0.006f);
      layer.q_norm = Vector(layer.head_dim, 0.5f, 1.5f);
      layer.k_norm = Vector(layer.head_dim, 0.05f, 0.15f);
      layer.cache_len = global ? kGlobalCache : kLocalCache;
      layer.ring = global ? 0 : 1;
      layer.mask = global ? 1 : 0;
      const size_t cache = static_cast<size_t>(layer.kv_heads) *
                           layer.cache_len * layer.head_dim;
      layer.k_cache = memory_.Get(cache * 2);
      layer.v_cache = memory_.Get(cache * 2);
      CHECK_CUDA(LiteRtNvidiaMegakernelLabFillHalf(
          layer.k_cache, cache, NextSeed(), -0.17f, 0.17f, nullptr));
      CHECK_CUDA(LiteRtNvidiaMegakernelLabFillHalf(
          layer.v_cache, cache, NextSeed(), -1.7f, 1.7f, nullptr));
      layer.o =
          Projection(kDim, layer.q_heads * layer.head_dim, 0.0015f, 0.0045f);
      layer.post_attention_norm = Vector(kDim, 0.5f, 1.5f);
      layer.arena = arenas[l % arenas.size()];
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
      layer.experts = kExperts;
      layer.top_k = kTopK;
      layer.expert_hidden = kExpertHidden;
      layer.expert_gate = 1;
      layer.dense_hidden = kDenseHidden;
      layer.dense_gate = 1;
      layer.per_expert_scale = Vector(kExperts, 0.5f, 1.5f);
      layer.router_norm = Vector(kDim, 0.5f, 1.5f);
      layer.router_multiplier = 1.0f;
      layer.expert_norm = Vector(kDim, 0.5f, 1.5f);
      layer.dense_norm = Vector(kDim, 0.5f, 1.5f);
      layer.dense_post = Vector(kDim, 0.5f, 1.5f);
      layer.moe_post = Vector(kDim, 0.5f, 1.5f);
      layer.sum_post = Vector(kDim, 0.5f, 1.5f);
      layer.residual_multiplier =
          0.5f + 0.5f * static_cast<float>(std::rand()) / RAND_MAX;
    }
    model_.layers = kLayers;
    model_.model_dim = kDim;
    model_.vocab = kVocab;
    model_.norm_eps = 1e-6f;
    model_.layer = layers_.data();
    model_.final_norm = Vector(kDim, 0.5f, 1.5f);
    model_.head = Projection(kVocab, kDim, 0.002f, 0.006f);
    model_.soft_cap = 30.0f;
    model_.soft_cap_in = 1.0f / 30.0f;
    CHECK_CUDA(cudaDeviceSynchronize());
  }

  const LiteRtNvidiaMegakernelModel& model() const { return model_; }
  const std::vector<LiteRtNvidiaMegakernelLayer>& layers() const {
    return layers_;
  }

 private:
  const Options& options_;
  Memory& memory_;
  uint64_t seed_ = 1;
  std::vector<LiteRtNvidiaMegakernelLayer> layers_;
  LiteRtNvidiaMegakernelModel model_{};
};

template <typename T>
std::vector<T> Fetch(const void* device, size_t count) {
  std::vector<T> host(count);
  CHECK_CUDA(cudaMemcpy(host.data(), device, count * sizeof(T),
                        cudaMemcpyDeviceToHost));
  return host;
}

float Bf16ToFloat(uint16_t bits) {
  const uint32_t word = static_cast<uint32_t>(bits) << 16;
  float value;
  std::memcpy(&value, &word, 4);
  return value;
}

// How a megakernel tensor differs from the reference's: the values that
// differ, the largest absolute difference, and the largest relative one,
// |diff| / |ref| with |ref| floored at 1% of the tensor's largest magnitude
// (values near zero would otherwise dominate).
struct Diff {
  size_t count = 0;
  size_t mismatches = 0;
  double max_abs = 0;
  double max_rel = 0;
  double max_ref = 0;
  std::string Text() const {
    char line[160];
    std::snprintf(
        line, sizeof(line),
        "%zu/%zu differ, max |diff| %.3g, max rel %.3g (max |ref| %.3g)",
        mismatches, count, max_abs, max_rel, max_ref);
    return line;
  }
};

Diff CompareValues(const std::vector<double>& ref,
                   const std::vector<double>& got, size_t mismatches) {
  Diff d;
  d.count = ref.size();
  d.mismatches = mismatches;
  for (const double a : ref) d.max_ref = std::max(d.max_ref, std::fabs(a));
  const double floor = 0.01 * d.max_ref;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double diff = std::fabs(ref[i] - got[i]);
    d.max_abs = std::max(d.max_abs, diff);
    d.max_rel = std::max(d.max_rel, diff / std::max(std::fabs(ref[i]), floor));
  }
  return d;
}

Diff CompareBf16(const std::vector<uint16_t>& ref,
                 const std::vector<uint16_t>& got) {
  std::vector<double> a(ref.size()), b(ref.size());
  size_t mismatches = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    a[i] = Bf16ToFloat(ref[i]);
    b[i] = Bf16ToFloat(got[i]);
    mismatches += ref[i] != got[i] ? 1 : 0;
  }
  return CompareValues(a, b, mismatches);
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

Diff CompareHalf(const std::vector<uint16_t>& ref,
                 const std::vector<uint16_t>& got) {
  std::vector<double> a(ref.size()), b(ref.size());
  size_t mismatches = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    a[i] = HalfToFloat(ref[i]);
    b[i] = HalfToFloat(got[i]);
    mismatches += ref[i] != got[i] ? 1 : 0;
  }
  return CompareValues(a, b, mismatches);
}

Diff CompareFloat(const std::vector<float>& ref,
                  const std::vector<float>& got) {
  std::vector<double> a(ref.size()), b(ref.size());
  size_t mismatches = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    a[i] = ref[i];
    b[i] = got[i];
    mismatches += std::memcmp(&ref[i], &got[i], 4) != 0 ? 1 : 0;
  }
  return CompareValues(a, b, mismatches);
}

// The indices of the k largest values, largest first (ties to the lower
// index).
std::vector<int> TopK(const std::vector<float>& values, int k) {
  std::vector<int> order(values.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int>(i);
  std::partial_sort(
      order.begin(), order.begin() + k, order.end(), [&](int a, int b) {
        return values[a] > values[b] || (values[a] == values[b] && a < b);
      });
  order.resize(k);
  return order;
}

double Median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values.empty() ? 0.0 : values[values.size() / 2];
}

// int4_block_gemv.cu's decode workspace of the MoE layer block (one row):
// the routing, the router's logits, the experts' hidden rows and partials,
// the dense feed-forward's hidden row and output, each 256-byte aligned.
struct MoeBlockBuffers {
  uint64_t indices, weights, logits, hidden, partial, dense_hidden, dense_out;
};

MoeBlockBuffers MoeBuffers(const LiteRtNvidiaMegakernelLayer& l, int dim) {
  auto up = [](uint64_t bytes) { return (bytes + 255) / 256 * 256; };
  MoeBlockBuffers b{};
  uint64_t at = 0;
  b.indices = at;
  at += up(l.top_k * 4);
  b.weights = at;
  at += up(l.top_k * 4);
  b.logits = at;
  at += up(l.experts * 2);
  b.hidden = at;
  at += up(static_cast<uint64_t>(l.top_k) * l.expert_hidden * 4);
  b.partial = at;
  at += up(static_cast<uint64_t>(l.top_k) * dim * 4);
  b.dense_hidden = at;
  at += up(l.dense_hidden * 2);
  b.dense_out = at;
  return b;
}

}  // namespace

int main(int argc, char** argv) {
  const Options options = Parse(argc, argv);
  Memory memory;
  Model model(options, memory);
  model.Build();

  LiteRtNvidiaMegakernelProgram* program = new LiteRtNvidiaMegakernelProgram;
  CHECK_CUDA(LiteRtNvidiaMegakernelBuildProgram(&model.model(), options.ctas,
                                                program));
  LiteRtNvidiaMegakernelProgram* device_program =
      memory.Get<LiteRtNvidiaMegakernelProgram>(sizeof(*program));
  CHECK_CUDA(cudaMemcpy(device_program, program, sizeof(*program),
                        cudaMemcpyHostToDevice));
  std::printf(
      "program: %d instructions, %d CTAs x %d threads, %d B shared memory, "
      "workspace %.1f MB; model and caches %.2f GB\n",
      program->instruction_count, program->ctas, program->threads,
      program->shared_bytes, program->workspace_bytes / 1e6,
      memory.total() / 1e9);

  // The step's inputs and the two runs' outputs and workspaces.
  const int position = options.context;
  int32_t* position_d = memory.Get<int32_t>(4);
  CHECK_CUDA(cudaMemcpy(position_d, &position, 4, cudaMemcpyHostToDevice));
  auto* local_mask = memory.Get<bool>(kLocalCache);
  CHECK_CUDA(cudaMemset(local_mask, 1, kLocalCache));
  auto* global_mask = memory.Get<bool>(kGlobalCache);
  CHECK_CUDA(cudaMemset(global_mask, 0, kGlobalCache));
  CHECK_CUDA(cudaMemset(global_mask, 1, std::min(position + 1, kGlobalCache)));
  float* embeddings = memory.Get<float>(kDim * 4);
  CHECK_CUDA(LiteRtNvidiaMegakernelLabFillFloat(
      embeddings, kDim, options.seed + 99, -2.0f, 2.0f, nullptr));
  LiteRtNvidiaMegakernelStep steps[2];
  void* workspaces[2];
  for (int r = 0; r < 2; ++r) {
    steps[r].embeddings = embeddings;
    steps[r].position = position_d;
    steps[r].mask[0] = local_mask;
    steps[r].mask[1] = global_mask;
    steps[r].activations = memory.Get<float>(kDim * 4);
    steps[r].logits = memory.Get<float>(static_cast<size_t>(kVocab) * 4);
    workspaces[r] = memory.Get(program->workspace_bytes);
    CHECK_CUDA(cudaMemset(workspaces[r], 0, program->workspace_bytes));
  }
  void* state = memory.Get(LiteRtNvidiaMegakernelStateBytes());
  CHECK_CUDA(cudaMemset(state, 0, LiteRtNvidiaMegakernelStateBytes()));
  cudaStream_t stream, side;
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CHECK_CUDA(cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking));
  LiteRtNvidiaMegakernelReferenceStreams streams{side, nullptr, nullptr};
  CHECK_CUDA(cudaEventCreateWithFlags(&streams.fork, cudaEventDisableTiming));
  CHECK_CUDA(cudaEventCreateWithFlags(&streams.join, cudaEventDisableTiming));
  auto reference = [&] {
    return LiteRtNvidiaMegakernelLaunchReference(
        program, &steps[0], workspaces[0], &streams, stream);
  };
  auto megakernel = [&](uint64_t* trace) {
    return LiteRtNvidiaMegakernelLaunch(program, device_program, &steps[1],
                                        workspaces[1], state, trace, stream);
  };

  // --- The numbers: every layer's input and h, the caches' new rows, the
  // logits and the activations. The device memory each way's first step
  // takes beyond the lab's own buffers is the driver's for its kernels.
  auto device_used_mb = [] {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    return cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess
               ? (total_bytes - free_bytes) / 1048576.0
               : 0.0;
  };
  const double used_before = device_used_mb();
  CHECK_CUDA(reference());
  CHECK_CUDA(cudaStreamSynchronize(stream));
  const double used_reference = device_used_mb();
  const auto& layers = model.layers();
  std::vector<std::vector<uint16_t>> rows_ref(2 * kLayers);
  for (int l = 0; l < kLayers; ++l) {
    const auto& layer = layers[l];
    const int row = layer.ring != 0 ? position % layer.cache_len
                                    : std::min(position, layer.cache_len - 1);
    for (int kv = 0; kv < 2; ++kv) {
      auto& out = rows_ref[2 * l + kv];
      for (int h = 0; h < layer.kv_heads; ++h) {
        const auto* cache =
            static_cast<const uint16_t*>(kv == 0 ? layer.k_cache
                                                 : layer.v_cache) +
            (static_cast<size_t>(h) * layer.cache_len + row) * layer.head_dim;
        const auto part = Fetch<uint16_t>(cache, layer.head_dim);
        out.insert(out.end(), part.begin(), part.end());
      }
    }
  }
  CHECK_CUDA(megakernel(nullptr));
  {
    const cudaError_t status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess) {
      std::fprintf(stderr, "megakernel: %s\n", cudaGetErrorString(status));
      return 1;
    }
  }
  std::printf(
      "device memory in use: %.0f MiB with the model, +%.0f MiB at the "
      "reference's first step, +%.0f MiB at the megakernel's\n",
      used_before, used_reference - used_before,
      device_used_mb() - used_reference);
  auto* ws_ref = static_cast<char*>(workspaces[0]);
  auto* ws_mk = static_cast<char*>(workspaces[1]);
  bool exact = true;
  int first_bad = -1;
  int routing_differs = 0;
  for (int l = 0; l <= kLayers; ++l) {
    const uint64_t x_at =
        l < kLayers ? program->scratch[l].x : program->final_x;
    const Diff x = CompareBf16(Fetch<uint16_t>(ws_ref + x_at, kDim),
                               Fetch<uint16_t>(ws_mk + x_at, kDim));
    char head[64];
    std::snprintf(head, sizeof(head), "layer %2d%s: x ", l,
                  l == kLayers ? " (out)" : "");
    std::string line = head + x.Text();
    bool bad = x.mismatches != 0;
    if (l < kLayers) {
      const Diff h =
          CompareBf16(Fetch<uint16_t>(ws_ref + program->scratch[l].h, kDim),
                      Fetch<uint16_t>(ws_mk + program->scratch[l].h, kDim));
      line += "; h " + h.Text();
      bad = bad || h.mismatches != 0;
      const auto& layer = layers[l];
      const int row = layer.ring != 0 ? position % layer.cache_len
                                      : std::min(position, layer.cache_len - 1);
      for (int kv = 0; kv < 2; ++kv) {
        std::vector<uint16_t> got;
        for (int h2 = 0; h2 < layer.kv_heads; ++h2) {
          const auto* cache =
              static_cast<const uint16_t*>(kv == 0 ? layer.k_cache
                                                   : layer.v_cache) +
              (static_cast<size_t>(h2) * layer.cache_len + row) *
                  layer.head_dim;
          const auto part = Fetch<uint16_t>(cache, layer.head_dim);
          got.insert(got.end(), part.begin(), part.end());
        }
        if (got != rows_ref[2 * l + kv]) {
          // The caches hold FP16.
          const Diff row_diff = CompareHalf(rows_ref[2 * l + kv], got);
          char text[96];
          std::snprintf(
              text, sizeof(text), "; %s row max |diff| %.3g, max rel %.3g",
              kv == 0 ? "k" : "v", row_diff.max_abs, row_diff.max_rel);
          line += text;
          bad = true;
        }
      }
      // The experts the layer routes to (as sets), and when they differ the
      // reference's margin: its top_k-th largest router logit minus the
      // next one.
      const MoeBlockBuffers moe = MoeBuffers(layer, kDim);
      auto ri = Fetch<int32_t>(ws_ref + program->scratch[l].moe + moe.indices,
                               layer.top_k);
      auto mi =
          Fetch<int32_t>(ws_mk + program->scratch[l].routing, layer.top_k);
      std::sort(ri.begin(), ri.end());
      std::sort(mi.begin(), mi.end());
      if (ri != mi) {
        std::vector<int32_t> common;
        std::set_intersection(ri.begin(), ri.end(), mi.begin(), mi.end(),
                              std::back_inserter(common));
        const auto bits = Fetch<uint16_t>(
            ws_ref + program->scratch[l].moe + moe.logits, layer.experts);
        std::vector<double> logits(bits.size());
        for (size_t e = 0; e < bits.size(); ++e)
          logits[e] = Bf16ToFloat(bits[e]);
        std::sort(logits.begin(), logits.end(), std::greater<double>());
        char text[128];
        std::snprintf(
            text, sizeof(text),
            "; ROUTING: %d of %d experts differ (reference margin %.3g)",
            layer.top_k - static_cast<int>(common.size()), layer.top_k,
            logits[layer.top_k - 1] - logits[layer.top_k]);
        line += text;
        ++routing_differs;
      }
    }
    if (bad && first_bad < 0) first_bad = l;
    exact = exact && !bad;
    std::printf("%s\n", line.c_str());
  }
  std::printf("routing: %d of %d layers route to other experts\n",
              routing_differs, kLayers);
  const Diff logits = CompareFloat(Fetch<float>(steps[0].logits, kVocab),
                                   Fetch<float>(steps[1].logits, kVocab));
  const Diff activations =
      CompareFloat(Fetch<float>(steps[0].activations, kDim),
                   Fetch<float>(steps[1].activations, kDim));
  std::printf("logits: %s\nactivations: %s\n", logits.Text().c_str(),
              activations.Text().c_str());
  {
    // The token's ranking: the top-1 and the overlap of the top 40.
    const auto ref_logits = Fetch<float>(steps[0].logits, kVocab);
    const auto mk_logits = Fetch<float>(steps[1].logits, kVocab);
    const std::vector<int> ref_top = TopK(ref_logits, 40);
    const std::vector<int> mk_top = TopK(mk_logits, 40);
    int overlap = 0;
    for (const int i : ref_top) {
      overlap += std::count(mk_top.begin(), mk_top.end(), i) > 0 ? 1 : 0;
    }
    std::printf(
        "logits top-1: reference %d, megakernel %d (%s); top-40 overlap "
        "%d/40\n",
        ref_top[0], mk_top[0], ref_top[0] == mk_top[0] ? "agree" : "DIFFER",
        overlap);
  }
  exact = exact && logits.mismatches == 0 && activations.mismatches == 0;
  if (options.detail && first_bad >= 0 && first_bad < kLayers) {
    // Where the first layer that differs starts to.
    const auto& s = program->scratch[first_bad];
    const auto& layer = layers[first_bad];
    const MoeBlockBuffers moe = MoeBuffers(layer, kDim);
    const int depth = layer.head_dim;
    auto bf16 = [&](const char* name, uint64_t ref_at, uint64_t mk_at,
                    size_t n) {
      std::printf("  %s: %s\n", name,
                  CompareBf16(Fetch<uint16_t>(ws_ref + ref_at, n),
                              Fetch<uint16_t>(ws_mk + mk_at, n))
                      .Text()
                      .c_str());
    };
    auto f32 = [&](const char* name, uint64_t ref_at, uint64_t mk_at,
                   size_t n) {
      std::printf("  %s: %s\n", name,
                  CompareFloat(Fetch<float>(ws_ref + ref_at, n),
                               Fetch<float>(ws_mk + mk_at, n))
                      .Text()
                      .c_str());
    };
    bf16("qkv", s.qkv, s.qkv, layer.qkv.rows);
    bf16("q", s.q, s.q, layer.q_heads * depth);
    bf16("k", s.k, s.k, layer.kv_heads * depth);
    bf16("v", s.v, s.v, layer.kv_heads * depth);
    bf16("attention", s.attention_out, s.attention_out, layer.q_heads * depth);
    bf16("o", s.o, s.o, kDim);
    bf16("router", s.moe + moe.logits, s.router, layer.experts);
    std::printf("  routing ref:");
    const auto ri = Fetch<int32_t>(ws_ref + s.moe + moe.indices, layer.top_k);
    const auto mi = Fetch<int32_t>(ws_mk + s.routing, layer.top_k);
    for (int j = 0; j < layer.top_k; ++j) std::printf(" %d", ri[j]);
    std::printf(" mk:");
    for (int j = 0; j < layer.top_k; ++j) std::printf(" %d", mi[j]);
    std::printf("\n");
    f32("routing weights", s.moe + moe.weights, s.routing + layer.top_k * 4,
        layer.top_k);
    f32("expert hidden", s.moe + moe.hidden, s.expert_hidden,
        static_cast<size_t>(layer.top_k) * layer.expert_hidden);
    bf16("dense hidden", s.moe + moe.dense_hidden, s.dense_hidden,
         layer.dense_hidden);
    bf16("dense out", s.moe + moe.dense_out, s.dense_out, kDim);
    {
      // The experts' sum: the reference's partials summed in slot order.
      const auto partial =
          Fetch<float>(ws_ref + s.moe + moe.partial,
                       static_cast<size_t>(layer.top_k) * kDim);
      std::vector<float> sum(kDim, 0.0f);
      for (int o = 0; o < kDim; ++o) {
        float m = 0.0f;
        for (int j = 0; j < layer.top_k; ++j) m += partial[j * kDim + o];
        sum[o] = m;
      }
      std::printf("  experts out: %s\n",
                  CompareFloat(sum, Fetch<float>(ws_mk + s.experts_out, kDim))
                      .Text()
                      .c_str());
    }
  }
  std::printf("numerics: %s\n",
              exact ? "bit for bit" : "not bit for bit (see the stats)");

  // --- The times: each step a CUDA graph, replayed in turn.
  cudaGraph_t graphs[2];
  cudaGraphExec_t execs[2];
  bool graph_ok[2] = {false, false};
  for (int r = 0; r < 2; ++r) {
    CHECK_CUDA(
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    const cudaError_t launched = r == 0 ? reference() : megakernel(nullptr);
    const cudaError_t captured = cudaStreamEndCapture(stream, &graphs[r]);
    if (launched == cudaSuccess && captured == cudaSuccess &&
        cudaGraphInstantiate(&execs[r], graphs[r], 0) == cudaSuccess) {
      graph_ok[r] = true;
    } else {
      std::printf("%s: no graph (%s / %s), launched directly\n",
                  r == 0 ? "reference" : "megakernel",
                  cudaGetErrorString(launched), cudaGetErrorString(captured));
      cudaGetLastError();
    }
  }
  auto run = [&](int r) {
    if (graph_ok[r]) return cudaGraphLaunch(execs[r], stream);
    return r == 0 ? reference() : megakernel(nullptr);
  };
  cudaEvent_t begin, end;
  CHECK_CUDA(cudaEventCreate(&begin));
  CHECK_CUDA(cudaEventCreate(&end));
  for (int warm = 0; warm < 3; ++warm) {
    CHECK_CUDA(run(0));
    CHECK_CUDA(run(1));
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));
  std::vector<double> times[2];
  for (int i = 0; i < options.replays; ++i) {
    for (int k = 0; k < 2; ++k) {
      const int r = (i + k) % 2;  // alternate which goes first
      CHECK_CUDA(cudaEventRecord(begin, stream));
      CHECK_CUDA(run(r));
      CHECK_CUDA(cudaEventRecord(end, stream));
      CHECK_CUDA(cudaEventSynchronize(end));
      float ms = 0;
      CHECK_CUDA(cudaEventElapsedTime(&ms, begin, end));
      times[r].push_back(ms);
    }
  }
  const double ref_ms = Median(times[0]);
  const double mk_ms = Median(times[1]);
  std::printf(
      "step at %d tokens, median of %d interleaved replays: reference %.3f ms, "
      "megakernel %.3f ms (%.2fx); min %.3f / %.3f ms\n",
      options.context, options.replays, ref_ms, mk_ms, ref_ms / mk_ms,
      *std::min_element(times[0].begin(), times[0].end()),
      *std::min_element(times[1].begin(), times[1].end()));

  if (!options.trace.empty()) {
    const size_t trace_bytes = LiteRtNvidiaMegakernelTraceBytes(program);
    auto* trace = memory.Get<uint64_t>(trace_bytes);
    CHECK_CUDA(cudaMemset(trace, 0, trace_bytes));
    CHECK_CUDA(megakernel(trace));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    const auto host = Fetch<uint64_t>(trace, trace_bytes / 8);
    if (!litert::nvidia::megakernel::WriteChromeTrace(*program, host.data(),
                                                      options.trace)) {
      std::fprintf(stderr, "could not write %s\n", options.trace.c_str());
    }
    std::printf(
        "%s", litert::nvidia::megakernel::SummarizeTrace(*program, host.data())
                  .c_str());
  }
  delete program;
  return 0;
}
