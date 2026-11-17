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

// The kernels of the composites over quantized KV caches against the host
// references of quant_attention_reference.h, with the shapes of the model
// that has them: 2 KV heads of depth 256 under 8 query heads, INT8 ring
// caches of 512 entries and INT4 caches of the context length.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "litert/vendors/nvidia/trtllm/decode_attention.h"
#include "litert/vendors/nvidia/trtllm/quant_attention_reference.h"
#include "litert/vendors/nvidia/trtllm/quant_cache.h"
#include "litert/vendors/nvidia/trtllm/tiled_attention.h"

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

constexpr int kHeads = 2;
constexpr int kDepth = 256;
constexpr int kGroup = 4;  // query heads per KV head
constexpr int kRing = 512;
// The fills of the decompositions: odml.sdpa_transposed (FP16 0xF999) and
// odml.fused_sdpa_cache_update.
constexpr float kSdpaFill = -45856.0f;
constexpr float kFusedFill = -10000.0f;

struct CudaDeleter {
  void operator()(void* pointer) const { cudaFree(pointer); }
};
using CudaAllocation = std::unique_ptr<void, CudaDeleter>;

CudaAllocation AllocateCuda(size_t bytes) {
  void* pointer = nullptr;
  const cudaError_t status = cudaMalloc(&pointer, std::max<size_t>(bytes, 1));
  if (status != cudaSuccess) {
    ADD_FAILURE() << "cudaMalloc(" << bytes
                  << ") failed: " << cudaGetErrorString(status);
  }
  return CudaAllocation(pointer);
}

template <typename T>
CudaAllocation Upload(const std::vector<T>& host) {
  auto device = AllocateCuda(host.size() * sizeof(T));
  if (device != nullptr && !host.empty()) {
    EXPECT_EQ(cudaMemcpy(device.get(), host.data(), host.size() * sizeof(T),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
  }
  return device;
}

template <typename T>
std::vector<T> Download(const CudaAllocation& device, size_t count) {
  std::vector<T> host(count);
  EXPECT_EQ(cudaMemcpy(host.data(), device.get(), count * sizeof(T),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  return host;
}

class Random {
 public:
  explicit Random(uint32_t seed) : state_(seed) {}
  // Uniform in [-1, 1].
  float Next() {
    state_ = state_ * 1664525u + 1013904223u;
    return static_cast<float>((state_ >> 8) & 0xFFFF) / 65535.0f * 2.0f - 1.0f;
  }
  // Uniform in [lo, hi].
  int Int(int lo, int hi) {
    state_ = state_ * 1664525u + 1013904223u;
    return lo +
           static_cast<int>((state_ >> 8) % static_cast<uint32_t>(hi - lo + 1));
  }

 private:
  uint32_t state_;
};

// Float values as the bits of an FP16 or a BF16 tensor, and what those hold.
std::vector<uint16_t> FloatBits(const std::vector<float>& values, bool bf16) {
  std::vector<uint16_t> bits(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    bits[i] = bf16 ? FloatToBf16Bits(values[i]) : FloatToHalfBits(values[i]);
  }
  return bits;
}

std::vector<float> BitsToFloats(const std::vector<uint16_t>& bits, bool bf16) {
  std::vector<float> values(bits.size());
  for (size_t i = 0; i < bits.size(); ++i) {
    values[i] = bf16 ? Bf16BitsToFloat(bits[i]) : HalfBitsToFloat(bits[i]);
  }
  return values;
}

// Caches of random values: keys of a moderate range, so that the scores do
// not single out one key, values of the whole range.
Caches RandomCaches(int cache_len, int bits, Random* random) {
  Caches caches(kHeads, cache_len, kDepth);
  const int limit = bits == 8 ? 127 : 7;
  for (auto& key : caches.k) {
    key = static_cast<int8_t>(
        random->Int(bits == 8 ? -40 : -8, bits == 8 ? 42 : 7));
  }
  for (auto& value : caches.v) {
    value = static_cast<int8_t>(random->Int(-limit - 1, limit));
  }
  return caches;
}

// The caches as the model stores them.
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

// LiteRT-LM's mask of a decode step over a ring cache
// (FillRingBufferDecodeAttentionMask): the slots written within `window`
// steps up to `step`.
std::vector<uint8_t> RingDecodeMask(int ring, int step, int window) {
  std::vector<uint8_t> mask(ring, 0);
  for (int col = 0; col < ring; ++col) {
    const int distance = (((step - col) % ring) + ring) % ring;
    mask[col] =
        distance <= window - 1 && distance <= step && distance <= ring - 1;
  }
  return mask;
}

// LiteRT-LM's causal mask of a prefill chunk over a ring cache
// (FillRingBufferPrefillAttentionMask): [tokens][ring + tokens], the rows
// from `steps` on are padding and see nothing.
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

TEST(QuantCacheTest, QueryIsRoundedHalfAwayFromZeroAndSaturated) {
  // With a scale of 0.5 the quotients are exact: ties go away from zero
  // (TFLite's QUANTIZE; TensorRT's rounds them to even), and what leaves the
  // INT8 range saturates.
  const float scale = 0.5f;
  const std::vector<float> values = {
      0.25f, 0.75f,  -0.25f, -0.75f, 1.25f,  -1.25f, 0.2f, -0.3f,
      63.5f, 63.75f, 100.0f, -64.0f, -64.3f, -90.0f, 0.0f, 3.1f};
  const std::vector<float> expected = {
      0.5f,  1.0f,  -0.5f, -1.0f,  1.5f,   -1.5f,  0.0f, -0.5f,
      63.5f, 63.5f, 63.5f, -64.0f, -64.0f, -64.0f, 0.0f, 3.0f};
  for (const bool bf16 : {false, true}) {
    SCOPED_TRACE(bf16 ? "bf16" : "fp16");
    const auto q = Upload(FloatBits(values, bf16));
    auto out = AllocateCuda(values.size() * 2);
    ASSERT_EQ(LiteRtNvidiaLaunchQuantizeQuery(q.get(), bf16, values.size(),
                                              scale, out.get(), nullptr),
              cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto actual =
        BitsToFloats(Download<uint16_t>(out, values.size()), false);
    for (size_t i = 0; i < values.size(); ++i) {
      EXPECT_EQ(actual[i], expected[i]) << "value " << values[i];
    }
  }
}

// The keys and values of `tokens` tokens before their QUANTIZE, and their
// quantization. v is [heads][tokens][depth], or token-major as the graphs
// have it before the transpose that follows its QUANTIZE.
struct NewRows {
  std::vector<uint16_t> k_bits;
  std::vector<uint16_t> v_bits;
  std::vector<int8_t> k;  // [heads][tokens][depth]
  std::vector<int8_t> v;
};

NewRows RandomRows(int tokens, int bits, float k_scale, float v_scale,
                   bool bf16, bool v_token_major, Random* random) {
  const size_t count = static_cast<size_t>(kHeads) * tokens * kDepth;
  const float range = bits == 8 ? 140.0f : 9.5f;  // some values saturate
  std::vector<float> k(count), v(count);
  for (size_t i = 0; i < count; ++i) {
    k[i] = random->Next() * range * k_scale;
    v[i] = random->Next() * range * v_scale;
  }
  NewRows rows;
  rows.k_bits = FloatBits(k, bf16);
  rows.v_bits = FloatBits(v, bf16);
  const auto k_held = BitsToFloats(rows.k_bits, bf16);
  const auto v_held = BitsToFloats(rows.v_bits, bf16);
  rows.k.resize(count);
  rows.v.resize(count);
  for (int h = 0; h < kHeads; ++h) {
    for (int t = 0; t < tokens; ++t) {
      for (int d = 0; d < kDepth; ++d) {
        const size_t to = (static_cast<size_t>(h) * tokens + t) * kDepth + d;
        const size_t v_from =
            v_token_major ? (static_cast<size_t>(t) * kHeads + h) * kDepth + d
                          : to;
        rows.k[to] = Quantize(k_held[to], k_scale, bits);
        rows.v[to] = Quantize(v_held[v_from], v_scale, bits);
      }
    }
  }
  return rows;
}

TEST(QuantCacheTest, CacheUpdateWritesQuantizedRowsAtTheClampedPositions) {
  // odml.cache_update in its forms: one token into an INT8 ring cache and
  // into an INT4 cache (decode), a chunk into an INT4 cache (prefill), and a
  // cache whose length leaves its INT4 value rows off the byte boundaries.
  struct Case {
    int bits;
    int cache_len;
    int tokens;
    int k_position;
    int v_position;
    bool bf16;
    bool v_token_major;
  };
  uint32_t seed = 100;
  for (const Case c : {Case{8, kRing, 1, 37, 37, false, false},
                       Case{8, kRing, 1, 511, 511, true, false},
                       Case{4, 2048, 1, 1001, 1001, false, false},
                       Case{4, 2048, 1, 1000, 1000, true, true},
                       Case{4, 2048, 128, 300, 300, false, true},
                       Case{4, 2048, 128, 1999, 1999, false, true},  // clamped
                       Case{4, 2048, 128, -5, 3, false, false},
                       Case{4, 1027, 1, 1026, 1026, false, false},
                       Case{4, 1027, 129, 77, 78, false, true},
                       Case{8, 1027, 32, 995, 995, false, true}}) {
    SCOPED_TRACE(::testing::Message()
                 << "bits=" << c.bits << " cache_len=" << c.cache_len
                 << " tokens=" << c.tokens << " k_position=" << c.k_position
                 << " bf16=" << c.bf16 << " token_major=" << c.v_token_major);
    Random random(seed++);
    const float k_scale = c.bits == 8 ? 0.039f : 0.3825f;
    const float v_scale = c.bits == 8 ? 0.0518f : 0.7136f;
    Caches caches = RandomCaches(c.cache_len, c.bits, &random);
    const NewRows rows = RandomRows(c.tokens, c.bits, k_scale, v_scale, c.bf16,
                                    c.v_token_major, &random);
    const size_t count = caches.k.size();
    auto k_cache = Upload(StoredCache(caches.k, c.bits));
    auto v_cache = Upload(StoredCache(caches.v, c.bits));
    const auto k = Upload(rows.k_bits);
    const auto v = Upload(rows.v_bits);
    const auto positions =
        Upload(std::vector<int32_t>{c.k_position, c.v_position});
    ASSERT_EQ(
        LiteRtNvidiaLaunchQuantCacheWrite(
            k_cache.get(), v_cache.get(), k.get(), v.get(), c.bf16,
            c.v_token_major, static_cast<const int32_t*>(positions.get()),
            static_cast<const int32_t*>(positions.get()) + 1, c.bits, k_scale,
            v_scale, kHeads, c.cache_len, kDepth, c.tokens, nullptr),
        cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    litert::nvidia::quant_reference::CacheUpdate(
        &caches, rows.k, rows.v, c.tokens, c.k_position, c.v_position);
    const size_t bytes = count * c.bits / 8;
    EXPECT_EQ(LoadedCache(Download<uint8_t>(k_cache, bytes), c.bits, count),
              caches.k);
    EXPECT_EQ(LoadedCache(Download<uint8_t>(v_cache, bytes), c.bits, count),
              caches.v);
  }
}

TEST(QuantCacheTest, RingUpdateKeepsTheNewestEntries) {
  // The cache update of odml.fused_sdpa_cache_update: the chunk's tokens go
  // to their positions modulo the ring, past its end around to its start,
  // and of more tokens than slots the newest stay.
  struct Case {
    int tokens;
    int start;
    int end;
  };
  uint32_t seed = 200;
  for (const Case c :
       {Case{128, 0, 40}, Case{128, 0, 128}, Case{128, 300, 428},
        Case{128, 450, 578},    // wraps around
        Case{128, 1000, 1001},  // one token, far into the ring's reuse
        Case{128, 77, 77},      // no token
        Case{128, 77, 300},     // more than the chunk holds: clipped
        Case{1024, 0, 1024},    // twice the ring
        Case{1024, 700, 1500}, Case{1024, 5, 517}}) {
    SCOPED_TRACE(::testing::Message() << "tokens=" << c.tokens << " start="
                                      << c.start << " end=" << c.end);
    Random random(seed++);
    Caches caches = RandomCaches(kRing, 8, &random);
    const size_t count = static_cast<size_t>(kHeads) * c.tokens * kDepth;
    std::vector<int8_t> k_new(count), v_new(count);
    for (size_t i = 0; i < count; ++i) {
      k_new[i] = static_cast<int8_t>(random.Int(-128, 127));
      v_new[i] = static_cast<int8_t>(random.Int(-128, 127));
    }
    auto k_cache = Upload(caches.k);
    auto v_cache = Upload(caches.v);
    const auto k = Upload(k_new);
    const auto v = Upload(v_new);
    const auto params =
        Upload(std::vector<int32_t>{c.start, c.end, c.end, 0, 0, 0, 0});
    ASSERT_EQ(LiteRtNvidiaLaunchQuantRingUpdate(
                  k_cache.get(), v_cache.get(), k.get(), v.get(),
                  static_cast<const int32_t*>(params.get()), kHeads, kRing,
                  kDepth, c.tokens, nullptr),
              cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    // What the decomposition computes is what writing the valid tokens in
    // order at their positions modulo the ring leaves.
    Caches in_order = caches;
    const int valid = std::min(std::max(c.end - c.start, 0), c.tokens);
    for (int t = 0; t < valid; ++t) {
      for (int h = 0; h < kHeads; ++h) {
        for (int d = 0; d < kDepth; ++d) {
          const size_t from =
              (static_cast<size_t>(h) * c.tokens + t) * kDepth + d;
          in_order.Key(h, (c.start + t) % kRing, d) = k_new[from];
          in_order.Value(h, (c.start + t) % kRing, d) = v_new[from];
        }
      }
    }
    litert::nvidia::quant_reference::RingUpdate(&caches, k_new, v_new, c.tokens,
                                                c.start, c.end);
    ASSERT_EQ(caches.k, in_order.k);
    ASSERT_EQ(caches.v, in_order.v);
    EXPECT_EQ(Download<int8_t>(k_cache, caches.k.size()), caches.k);
    EXPECT_EQ(Download<int8_t>(v_cache, caches.v.size()), caches.v);
  }
}

// A query before its QUANTIZE and its quantization: [heads][rows][depth].
struct Query {
  std::vector<uint16_t> bits;
  std::vector<int8_t> quantized;
};

Query RandomQuery(int rows, float scale, bool bf16, Random* random) {
  std::vector<float> values(static_cast<size_t>(kHeads) * rows * kDepth);
  for (auto& value : values) {
    value = random->Next() * 100.0f * scale;
  }
  // A few values saturate.
  values[3] = 200.0f * scale;
  values[values.size() - 5] = -300.0f * scale;
  Query query;
  query.bits = FloatBits(values, bf16);
  const auto held = BitsToFloats(query.bits, bf16);
  query.quantized.resize(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    query.quantized[i] = Quantize(held[i], scale, 8);
  }
  return query;
}

TEST(QuantDecodeAttentionTest, MatchesTheDecomposition) {
  // odml.sdpa_transposed of a decode step: the query heads of a KV head as
  // rows over an INT8 ring cache with LiteRT-LM's ring mask, or over an INT4
  // cache with the causal mask. More rows, a mask row per row and a cache
  // whose value rows are not whole vectors exercise the other paths.
  struct Case {
    int bits;
    int cache_len;
    int rows;
    int mask_rows;
    int step;  // the position of the token being decoded
    bool bf16;
  };
  uint32_t seed = 300;
  for (const Case c :
       {Case{8, kRing, kGroup, 1, 0, false},
        Case{8, kRing, kGroup, 1, 5, false},
        Case{8, kRing, kGroup, 1, 511, true},
        Case{8, kRing, kGroup, 1, 700, false},
        Case{4, 2048, kGroup, 1, 0, false}, Case{4, 2048, kGroup, 1, 93, false},
        Case{4, 2048, kGroup, 1, 2047, true},
        Case{4, 32768, kGroup, 1, 12345, false},
        Case{4, 1027, kGroup, 1, 700, false},
        Case{8, 1000, kGroup, 1, 999, false},
        Case{4, 448, kGroup, 1, 300, false}, Case{8, 200, kGroup, 1, 150, true},
        Case{4, 37, kGroup, 1, 20, false}, Case{8, kRing, 16, 16, 400, false},
        Case{4, 2048, 12, 12, 1500, false}}) {
    SCOPED_TRACE(::testing::Message()
                 << "bits=" << c.bits << " cache_len=" << c.cache_len
                 << " rows=" << c.rows << " mask_rows=" << c.mask_rows
                 << " step=" << c.step << " bf16=" << c.bf16);
    Random random(seed++);
    const LiteRtNvidiaQuantAttentionParams params = {
        0.006f, c.bits == 8 ? 0.039f : 0.3825f, c.bits == 8 ? 0.0518f : 0.7136f,
        c.bits};
    const Caches caches = RandomCaches(c.cache_len, c.bits, &random);
    const Query query = RandomQuery(c.rows, params.q_scale, c.bf16, &random);
    // Mask row r: the ring's (or the causal) mask of position step - r, so
    // every row sees a key.
    std::vector<uint8_t> mask;
    for (int r = 0; r < c.mask_rows; ++r) {
      std::vector<uint8_t> row(c.cache_len, 0);
      if (c.bits == 8 && c.cache_len == kRing) {
        row = RingDecodeMask(kRing, c.step - r, kRing);
      } else {
        std::fill(row.begin(), row.begin() + c.step - r + 1, 1);
      }
      mask.insert(mask.end(), row.begin(), row.end());
    }
    const auto q = Upload(query.bits);
    const auto k = Upload(StoredCache(caches.k, c.bits));
    const auto v = Upload(StoredCache(caches.v, c.bits));
    const auto mask_device = Upload(mask);
    auto out = AllocateCuda(query.bits.size() * 2);
    auto workspace = AllocateCuda(LiteRtNvidiaDecodeAttentionWorkspaceBytes(
        kHeads, c.rows, c.cache_len, kDepth));
    ASSERT_TRUE(
        LiteRtNvidiaQuantDecodeAttentionSupports(c.rows, kDepth, c.bits));
    ASSERT_EQ(LiteRtNvidiaLaunchQuantDecodeAttention(
                  q.get(), c.bf16, k.get(), v.get(), &params,
                  static_cast<const bool*>(mask_device.get()), c.mask_rows,
                  kHeads, c.rows, c.cache_len, kDepth, kSdpaFill, out.get(),
                  workspace.get(), nullptr),
              cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    AttentionInputs reference;
    reference.q = &query.quantized;
    reference.rows = c.rows;
    reference.q_scale = params.q_scale;
    reference.k_scale = params.k_scale;
    reference.v_scale = params.v_scale;
    reference.caches = &caches;
    reference.mask = &mask;
    reference.mask_rows = c.mask_rows;
    reference.fill = kSdpaFill;
    const auto expected = litert::nvidia::quant_reference::Attention(reference);
    const auto actual =
        BitsToFloats(Download<uint16_t>(out, query.bits.size()), c.bf16);
    // The values reach 6.6 (INT8) or 5.7 (INT4): an FP16 result is good to
    // about 0.004 there, a BF16 one to 0.03.
    const auto errors =
        CompareRows(actual, expected, kDepth, c.bf16 ? 0.05f : 0.02f);
    EXPECT_TRUE(RowsMatch(errors))
        << errors.rows_off << " of " << errors.rows
        << " rows are off, the largest error is " << errors.max_error;
  }
}

// Zeroed device memory for the one-launch attention's counters.
CudaAllocation AttentionState(int rows) {
  const size_t bytes = LiteRtNvidiaQuantDecodeAttentionStateBytes(kHeads, rows);
  auto state = AllocateCuda(bytes);
  EXPECT_EQ(cudaMemset(state.get(), 0, bytes), cudaSuccess);
  return state;
}

TEST(QuantDecodeAttentionTest, OneLaunchHasTheTwoLaunchesBits) {
  // With a state the attention is one launch whose last blocks combine the
  // chunks, where the device has the multiprocessors for it (the ring with
  // four rows, the short caches), and the two launches elsewhere: the same
  // bits as the two launches either way, launch after launch (the counters
  // are left zero), for both cache types, masks that hide chunks and rows
  // that see nothing.
  struct Case {
    int bits;
    int cache_len;
    int rows;
    int mask_rows;
    bool bf16;
  };
  uint32_t seed = 900;
  for (const Case c :
       {Case{8, kRing, kGroup, 1, false}, Case{8, kRing, 16, 16, true},
        Case{8, 200, kGroup, 1, false}, Case{4, 2048, kGroup, 1, false},
        Case{4, 2048, 12, 12, true}, Case{4, 37, kGroup, kGroup, false},
        Case{8, 64, 16, 16, false}, Case{4, 32768, kGroup, 1, false}}) {
    SCOPED_TRACE(::testing::Message()
                 << "bits=" << c.bits << " cache_len=" << c.cache_len
                 << " rows=" << c.rows << " mask_rows=" << c.mask_rows
                 << " bf16=" << c.bf16);
    Random random(seed++);
    const LiteRtNvidiaQuantAttentionParams params = {
        0.006f, c.bits == 8 ? 0.039f : 0.3825f, c.bits == 8 ? 0.0518f : 0.7136f,
        c.bits};
    const Caches caches = RandomCaches(c.cache_len, c.bits, &random);
    const auto k = Upload(StoredCache(caches.k, c.bits));
    const auto v = Upload(StoredCache(caches.v, c.bits));
    const size_t result_bytes =
        static_cast<size_t>(kHeads) * c.rows * kDepth * 2;
    auto expected = AllocateCuda(result_bytes);
    auto out = AllocateCuda(result_bytes);
    const size_t workspace_bytes = LiteRtNvidiaDecodeAttentionWorkspaceBytes(
        kHeads, c.rows, c.cache_len, kDepth);
    auto workspace = AllocateCuda(workspace_bytes);
    auto state_workspace = AllocateCuda(workspace_bytes);
    const auto state = AttentionState(c.rows);
    for (int launch = 0; launch < 4; ++launch) {
      SCOPED_TRACE(::testing::Message() << "launch " << launch);
      const Query query = RandomQuery(c.rows, params.q_scale, c.bf16, &random);
      // Mask row r: a prefix of the keys, every key, a scattered third, or
      // none at all.
      std::vector<uint8_t> mask(static_cast<size_t>(c.mask_rows) * c.cache_len,
                                0);
      for (int r = 0; r < c.mask_rows; ++r) {
        uint8_t* row = mask.data() + static_cast<size_t>(r) * c.cache_len;
        switch ((launch + r) % 4) {
          case 0:
            std::fill(row, row + random.Int(1, c.cache_len), 1);
            break;
          case 1:
            std::fill(row, row + c.cache_len, 1);
            break;
          case 2:
            for (int j = c.cache_len / 3; j < c.cache_len; ++j) {
              row[j] = random.Int(0, 2) == 0;
            }
            break;
          default:
            break;
        }
      }
      const auto q = Upload(query.bits);
      const auto mask_device = Upload(mask);
      ASSERT_EQ(LiteRtNvidiaLaunchQuantDecodeAttention(
                    q.get(), c.bf16, k.get(), v.get(), &params,
                    static_cast<const bool*>(mask_device.get()), c.mask_rows,
                    kHeads, c.rows, c.cache_len, kDepth,
                    -std::numeric_limits<float>::infinity(), expected.get(),
                    workspace.get(), nullptr),
                cudaSuccess);
      ASSERT_EQ(LiteRtNvidiaLaunchQuantDecodeAttentionUpdate(
                    q.get(), c.bf16, k.get(), v.get(), /*token=*/nullptr,
                    &params, static_cast<const bool*>(mask_device.get()),
                    c.mask_rows, kHeads, c.rows, c.cache_len, kDepth,
                    -std::numeric_limits<float>::infinity(), out.get(),
                    state_workspace.get(), state.get(), nullptr),
                cudaSuccess);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      ASSERT_EQ(Download<uint16_t>(out, query.bits.size()),
                Download<uint16_t>(expected, query.bits.size()));
    }
    const std::vector<uint32_t> counters =
        Download<uint32_t>(state, static_cast<size_t>(kHeads) * c.rows);
    EXPECT_EQ(counters, std::vector<uint32_t>(counters.size(), 0));
  }
}

TEST(QuantDecodeAttentionTest, WritesTheTokenBeforeItAttends) {
  // A decode step's cache update inside the attention's launches: the caches
  // and the result are those of the cache write followed by the attention,
  // bit for bit, in one launch (with a state) and in two, with positions
  // inside the cache and clamped to it.
  struct Case {
    int bits;
    int cache_len;
    int rows;
    int position;
    bool bf16;
  };
  uint32_t seed = 700;
  for (const Case c :
       {Case{8, kRing, kGroup, 0, false}, Case{8, kRing, kGroup, 317, true},
        Case{8, kRing, 16, 511, false}, Case{8, kRing, kGroup, 4000, false},
        Case{4, 448, kGroup, 77, false}, Case{4, 2048, kGroup, 1234, false},
        Case{4, 2048, kGroup, 1235, true}, Case{4, 2048, kGroup, -3, false},
        Case{8, 1000, kGroup, 999, false}}) {
    SCOPED_TRACE(::testing::Message()
                 << "bits=" << c.bits << " cache_len=" << c.cache_len
                 << " rows=" << c.rows << " position=" << c.position
                 << " bf16=" << c.bf16);
    Random random(seed++);
    const LiteRtNvidiaQuantAttentionParams params = {
        0.006f, c.bits == 8 ? 0.039f : 0.3825f, c.bits == 8 ? 0.0518f : 0.7136f,
        c.bits};
    const Caches caches = RandomCaches(c.cache_len, c.bits, &random);
    const Query query = RandomQuery(c.rows, params.q_scale, c.bf16, &random);
    // The token's rows before their QUANTIZE, past the caches' range too.
    const int limit = c.bits == 8 ? 127 : 7;
    std::vector<float> k_rows(kHeads * kDepth);
    std::vector<float> v_rows(kHeads * kDepth);
    for (float& value : k_rows) {
      value = random.Next() * 1.2f * limit * params.k_scale;
    }
    for (float& value : v_rows) {
      value = random.Next() * 1.2f * limit * params.v_scale;
    }
    const std::vector<uint8_t> mask(c.cache_len, 1);
    const std::vector<int32_t> position = {c.position};
    const auto q = Upload(query.bits);
    const auto k_device = Upload(FloatBits(k_rows, c.bf16));
    const auto v_device = Upload(FloatBits(v_rows, c.bf16));
    const auto position_device = Upload(position);
    const auto mask_device = Upload(mask);
    const std::vector<uint8_t> k_stored = StoredCache(caches.k, c.bits);
    const std::vector<uint8_t> v_stored = StoredCache(caches.v, c.bits);
    const size_t workspace_bytes = LiteRtNvidiaDecodeAttentionWorkspaceBytes(
        kHeads, c.rows, c.cache_len, kDepth);
    ASSERT_TRUE(LiteRtNvidiaQuantDecodeAttentionUpdateSupports(
        c.rows, kDepth, c.bits, c.cache_len));

    // The cache write, then the attention.
    const auto k_written = Upload(k_stored);
    const auto v_written = Upload(v_stored);
    auto expected = AllocateCuda(query.bits.size() * 2);
    auto workspace = AllocateCuda(workspace_bytes);
    ASSERT_EQ(LiteRtNvidiaLaunchQuantCacheWrite(
                  k_written.get(), v_written.get(), k_device.get(),
                  v_device.get(), c.bf16, /*v_token_major=*/false,
                  static_cast<const int32_t*>(position_device.get()),
                  static_cast<const int32_t*>(position_device.get()), c.bits,
                  params.k_scale, params.v_scale, kHeads, c.cache_len, kDepth,
                  /*tokens=*/1, nullptr),
              cudaSuccess);
    ASSERT_EQ(LiteRtNvidiaLaunchQuantDecodeAttention(
                  q.get(), c.bf16, k_written.get(), v_written.get(), &params,
                  static_cast<const bool*>(mask_device.get()), 1, kHeads,
                  c.rows, c.cache_len, kDepth, kSdpaFill, expected.get(),
                  workspace.get(), nullptr),
              cudaSuccess);

    // Both in the attention's launches: one launch, then two.
    const LiteRtNvidiaQuantCacheToken token = {
        k_device.get(), v_device.get(),
        static_cast<const int32_t*>(position_device.get()),
        static_cast<const int32_t*>(position_device.get())};
    const auto state = AttentionState(c.rows);
    for (void* launch_state : {state.get(), static_cast<void*>(nullptr)}) {
      SCOPED_TRACE(::testing::Message()
                   << (launch_state != nullptr ? "one launch" : "two"));
      const auto k_updated = Upload(k_stored);
      const auto v_updated = Upload(v_stored);
      auto out = AllocateCuda(query.bits.size() * 2);
      auto update_workspace = AllocateCuda(workspace_bytes);
      ASSERT_EQ(LiteRtNvidiaLaunchQuantDecodeAttentionUpdate(
                    q.get(), c.bf16, k_updated.get(), v_updated.get(), &token,
                    &params, static_cast<const bool*>(mask_device.get()), 1,
                    kHeads, c.rows, c.cache_len, kDepth, kSdpaFill, out.get(),
                    update_workspace.get(), launch_state, nullptr),
                cudaSuccess);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

      EXPECT_EQ(Download<uint8_t>(k_updated, k_stored.size()),
                Download<uint8_t>(k_written, k_stored.size()));
      EXPECT_EQ(Download<uint8_t>(v_updated, v_stored.size()),
                Download<uint8_t>(v_written, v_stored.size()));
      EXPECT_NE(Download<uint8_t>(k_updated, k_stored.size()), k_stored);
      EXPECT_EQ(Download<uint16_t>(out, query.bits.size()),
                Download<uint16_t>(expected, query.bits.size()));
    }
  }
  // INT4 value rows that do not start on a byte are not written this way.
  EXPECT_FALSE(
      LiteRtNvidiaQuantDecodeAttentionUpdateSupports(kGroup, kDepth, 4, 37));
  EXPECT_TRUE(
      LiteRtNvidiaQuantDecodeAttentionUpdateSupports(kGroup, kDepth, 8, 37));
}

TEST(QuantDecodeAttentionTest, ARowThatSeesNoKeyGetsZero) {
  // With a fill of -inf, which is what the plugin passes, the keys a mask
  // hides stay out of the softmax: the rows that see a key get the
  // decomposition's result and the others zero. Eight rows are two tokens'
  // four query heads, or four tokens' two: row r reads mask row r %
  // mask_rows.
  const float fill = -INFINITY;
  uint32_t seed = 350;
  for (const int mask_rows : {1, 2, 4, 8}) {
    for (const int bits : {8, 4}) {
      SCOPED_TRACE(::testing::Message()
                   << "mask_rows=" << mask_rows << " bits=" << bits);
      constexpr int kRows = 8;
      constexpr int kStep = 300;
      Random random(seed++);
      const LiteRtNvidiaQuantAttentionParams params = {
          0.006f, bits == 8 ? 0.039f : 0.3825f, bits == 8 ? 0.0518f : 0.7136f,
          bits};
      const Caches caches = RandomCaches(kRing, bits, &random);
      const Query query = RandomQuery(kRows, params.q_scale, false, &random);
      // The last mask row sees nothing; with one mask row, no row does.
      std::vector<uint8_t> mask;
      for (int r = 0; r < mask_rows; ++r) {
        std::vector<uint8_t> row(kRing, 0);
        if (r + 1 < mask_rows) {
          std::fill(row.begin(), row.begin() + kStep - r + 1, 1);
        }
        mask.insert(mask.end(), row.begin(), row.end());
      }
      const auto q = Upload(query.bits);
      const auto k = Upload(StoredCache(caches.k, bits));
      const auto v = Upload(StoredCache(caches.v, bits));
      const auto mask_device = Upload(mask);
      auto out = AllocateCuda(query.bits.size() * 2);
      auto workspace = AllocateCuda(LiteRtNvidiaDecodeAttentionWorkspaceBytes(
          kHeads, kRows, kRing, kDepth));
      ASSERT_EQ(
          LiteRtNvidiaLaunchQuantDecodeAttention(
              q.get(), /*q_bf16=*/false, k.get(), v.get(), &params,
              static_cast<const bool*>(mask_device.get()), mask_rows, kHeads,
              kRows, kRing, kDepth, fill, out.get(), workspace.get(), nullptr),
          cudaSuccess);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

      AttentionInputs reference;
      reference.q = &query.quantized;
      reference.rows = kRows;
      reference.q_scale = params.q_scale;
      reference.k_scale = params.k_scale;
      reference.v_scale = params.v_scale;
      reference.caches = &caches;
      reference.mask = &mask;
      reference.mask_rows = mask_rows;
      reference.fill = fill;
      auto expected = litert::nvidia::quant_reference::Attention(reference);
      const auto actual =
          BitsToFloats(Download<uint16_t>(out, query.bits.size()), false);
      for (int h = 0; h < kHeads; ++h) {
        for (int r = 0; r < kRows; ++r) {
          if (r % mask_rows != mask_rows - 1) {
            continue;
          }
          const size_t row = (static_cast<size_t>(h) * kRows + r) * kDepth;
          std::fill_n(expected.begin() + row, kDepth, 0.0f);
          for (int d = 0; d < kDepth; ++d) {
            ASSERT_EQ(actual[row + d], 0.0f) << "head " << h << " row " << r;
          }
        }
      }
      const auto errors = CompareRows(actual, expected, kDepth, 0.02f);
      EXPECT_TRUE(RowsMatch(errors))
          << errors.rows_off << " of " << errors.rows
          << " rows are off, the largest error is " << errors.max_error;
    }
  }
}

// The steps a plugin runs around the tensor-core kernel: the query through
// LiteRtNvidiaLaunchQuantizeQuery, the chunk's keys and values through
// LiteRtNvidiaLaunchQuantizeRows.
struct TiledCase {
  int bits;
  int cache_len;
  int tokens;   // mask rows
  int new_len;  // the chunk's keys after the cache's (0: none)
  int start;    // the position of the chunk's first token
  int steps;    // the tokens of the chunk that are not padding
  bool bf16;
  bool v_token_major;
};

void RunTiled(const TiledCase& c, uint32_t seed) {
  SCOPED_TRACE(::testing::Message()
               << "bits=" << c.bits << " cache_len=" << c.cache_len
               << " tokens=" << c.tokens << " new_len=" << c.new_len
               << " start=" << c.start << " steps=" << c.steps);
  Random random(seed);
  const int rows = kGroup * c.tokens;
  const LiteRtNvidiaQuantAttentionParams params = {
      0.006f, c.bits == 8 ? 0.039f : 0.3825f, c.bits == 8 ? 0.0518f : 0.7136f,
      c.bits};
  const LiteRtNvidiaAttentionShape shape = {kHeads, rows,        c.tokens,
                                            kDepth, c.cache_len, c.new_len};
  ASSERT_TRUE(LiteRtNvidiaTiledQuantAttentionSupports(&shape, c.bits));
  if (!LiteRtNvidiaTiledAttentionAvailable()) {
    GTEST_SKIP() << "the device does not run the tensor-core kernels";
  }
  const Caches caches = RandomCaches(c.cache_len, c.bits, &random);
  const Query query = RandomQuery(rows, params.q_scale, c.bf16, &random);
  const int keys = c.cache_len + c.new_len;
  std::vector<uint8_t> mask;
  if (c.new_len > 0) {
    mask =
        RingPrefillMask(c.cache_len, c.tokens, c.start, c.steps, c.cache_len);
  } else {
    // The chunk is in the cache already: token t sees the keys up to its own.
    mask.assign(static_cast<size_t>(c.tokens) * keys, 0);
    for (int t = 0; t < c.steps; ++t) {
      std::fill_n(mask.begin() + static_cast<size_t>(t) * keys, c.start + t + 1,
                  1);
    }
  }
  const auto q = Upload(query.bits);
  const auto k = Upload(StoredCache(caches.k, c.bits));
  const auto v = Upload(StoredCache(caches.v, c.bits));
  const auto mask_device = Upload(mask);
  const size_t q_count = query.bits.size();
  auto q_half = AllocateCuda(q_count * 2);
  ASSERT_EQ(
      LiteRtNvidiaLaunchQuantizeQuery(q.get(), c.bf16, q_count, params.q_scale,
                                      q_half.get(), nullptr),
      cudaSuccess);
  NewRows rows_new;
  CudaAllocation k_new, v_new;
  if (c.new_len > 0) {
    rows_new = RandomRows(c.new_len, 8, params.k_scale, params.v_scale, c.bf16,
                          c.v_token_major, &random);
    const auto k_float = Upload(rows_new.k_bits);
    const auto v_float = Upload(rows_new.v_bits);
    k_new = AllocateCuda(rows_new.k.size());
    v_new = AllocateCuda(rows_new.v.size());
    ASSERT_EQ(LiteRtNvidiaLaunchQuantizeRows(
                  k_float.get(), v_float.get(), c.bf16, c.v_token_major, kHeads,
                  c.new_len, kDepth, params.k_scale, params.v_scale,
                  k_new.get(), v_new.get(), nullptr),
              cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(Download<int8_t>(k_new, rows_new.k.size()), rows_new.k);
    EXPECT_EQ(Download<int8_t>(v_new, rows_new.v.size()), rows_new.v);
  }
  auto out = AllocateCuda(q_count * 2);
  auto workspace =
      AllocateCuda(LiteRtNvidiaTiledQuantAttentionWorkspaceBytes(&shape));
  const float fill = c.new_len > 0 ? kFusedFill : kSdpaFill;
  ASSERT_EQ(LiteRtNvidiaLaunchTiledQuantAttention(
                &shape, &params, q_half.get(), k.get(), v.get(), k_new.get(),
                v_new.get(), static_cast<const bool*>(mask_device.get()), fill,
                out.get(), workspace.get(), nullptr),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  AttentionInputs reference;
  reference.q = &query.quantized;
  reference.rows = rows;
  reference.q_scale = params.q_scale;
  reference.k_scale = params.k_scale;
  reference.v_scale = params.v_scale;
  reference.caches = &caches;
  reference.k_new = &rows_new.k;
  reference.v_new = &rows_new.v;
  reference.new_len = c.new_len;
  reference.mask = &mask;
  reference.mask_rows = c.tokens;
  reference.fill = fill;
  auto expected = litert::nvidia::quant_reference::Attention(reference);
  // The rows of padding see no key: the kernel gives them zero where the
  // decomposition averages every value (the graphs replace or never read
  // them).
  for (int h = 0; h < kHeads; ++h) {
    for (int r = 0; r < rows; ++r) {
      if (r % c.tokens >= c.steps) {
        std::fill_n(
            expected.begin() + (static_cast<size_t>(h) * rows + r) * kDepth,
            kDepth, 0.0f);
      }
    }
  }
  const auto actual = BitsToFloats(Download<uint16_t>(out, q_count), false);
  // The kernel multiplies FP16 weights and values and sums chains of them in
  // FP16.
  const auto errors = CompareRows(actual, expected, kDepth, 0.02f);
  EXPECT_TRUE(RowsMatch(errors))
      << errors.rows_off << " of " << errors.rows
      << " rows are off, the largest error is " << errors.max_error;
}

TEST(QuantTiledAttentionTest, GlobalLayerOfAPrefillChunk) {
  // odml.sdpa_transposed of a prefill chunk over an INT4 cache that holds the
  // chunk already: 4 query heads x 128 tokens per KV head.
  uint32_t seed = 400;
  for (const TiledCase c :
       {TiledCase{4, 2048, 128, 0, 0, 128, false, false},
        TiledCase{4, 2048, 128, 0, 700, 128, false, false},
        TiledCase{4, 2048, 128, 0, 1900, 100, true, false},  // 28 padded
        TiledCase{4, 1024, 256, 0, 300, 256, false, false},
        TiledCase{8, 1024, 128, 0, 300, 128, false, false},
        // The model's larger chunk: 4 x 1024 rows per KV head.
        TiledCase{4, 2048, 1024, 0, 900, 1000, false, false}}) {
    RunTiled(c, seed++);
  }
}

TEST(QuantTiledAttentionTest, LocalLayerOfAPrefillChunk) {
  // The attention of odml.fused_sdpa_cache_update: the INT8 ring cache, then
  // the chunk's own keys and values, under LiteRT-LM's ring mask.
  uint32_t seed = 500;
  for (const TiledCase c :
       {TiledCase{8, kRing, 128, 128, 0, 128, false, true},
        TiledCase{8, kRing, 128, 128, 200, 128, false, true},
        TiledCase{8, kRing, 128, 128, 1000, 128, true, true},  // a full ring
        TiledCase{8, kRing, 128, 128, 450, 70, false, false},  // 58 padded
        TiledCase{8, kRing, 256, 256, 3000, 256, false, true},
        // The model's larger chunk, twice the ring.
        TiledCase{8, kRing, 1024, 1024, 2000, 1024, false, true}}) {
    RunTiled(c, seed++);
  }
}

}  // namespace
