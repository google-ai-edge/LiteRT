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

#include "litert/vendors/nvidia/trtllm/general_attention.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "cuda_runtime_api.h"
#include "litert/vendors/nvidia/trtllm/general_attention_reference.h"
#include "litert/vendors/nvidia/trtllm/qsa_reference.h"

namespace {

namespace ref = litert::nvidia::general_reference;
using litert::nvidia::qsa_reference::Bf16BitsToFloat;
using litert::nvidia::qsa_reference::FloatToBf16Bits;
using litert::nvidia::qsa_reference::FloatToHalfBits;
using litert::nvidia::qsa_reference::HalfBitsToFloat;

bool HasGpu() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t bytes) : bytes_(std::max<size_t>(bytes, 1)) {
    EXPECT_EQ(cudaMalloc(&data_, bytes_), cudaSuccess);
    EXPECT_EQ(cudaMemset(data_, 0, bytes_), cudaSuccess);
  }
  template <typename T>
  explicit DeviceBuffer(const std::vector<T>& host)
      : DeviceBuffer(host.size() * sizeof(T)) {
    EXPECT_EQ(cudaMemcpy(data_, host.data(), host.size() * sizeof(T),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
  }
  ~DeviceBuffer() { cudaFree(data_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  void* get() const { return data_; }
  template <typename T>
  std::vector<T> Read(size_t count) const {
    std::vector<T> host(count);
    EXPECT_EQ(cudaMemcpy(host.data(), data_, count * sizeof(T),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    return host;
  }

 private:
  void* data_ = nullptr;
  size_t bytes_;
};

// Floats in a tensor of `type`, as bytes, and the values the tensor holds.
std::vector<uint8_t> FloatBytes(const std::vector<float>& values, int32_t type,
                                std::vector<float>* held) {
  std::vector<uint8_t> bytes;
  held->resize(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    if (type == kLiteRtNvidiaGeneralFloat32) {
      uint8_t b[4];
      std::memcpy(b, &values[i], 4);
      bytes.insert(bytes.end(), b, b + 4);
      (*held)[i] = values[i];
    } else {
      const uint16_t bits = type == kLiteRtNvidiaGeneralFloat16
                                ? FloatToHalfBits(values[i])
                                : FloatToBf16Bits(values[i]);
      bytes.push_back(static_cast<uint8_t>(bits & 0xFF));
      bytes.push_back(static_cast<uint8_t>(bits >> 8));
      (*held)[i] = type == kLiteRtNvidiaGeneralFloat16 ? HalfBitsToFloat(bits)
                                                       : Bf16BitsToFloat(bits);
    }
  }
  return bytes;
}

float FromBytes(const std::vector<uint8_t>& bytes, int32_t type, size_t i) {
  if (type == kLiteRtNvidiaGeneralFloat32) {
    float v;
    std::memcpy(&v, &bytes[4 * i], 4);
    return v;
  }
  const uint16_t bits =
      static_cast<uint16_t>(bytes[2 * i] | (bytes[2 * i + 1] << 8));
  return type == kLiteRtNvidiaGeneralFloat16 ? HalfBitsToFloat(bits)
                                             : Bf16BitsToFloat(bits);
}

size_t TypeBytes(int32_t type) {
  return type == kLiteRtNvidiaGeneralFloat32 ? 4 : 2;
}

// A cache tensor's bytes from its unscaled values (`count` of them).
std::vector<uint8_t> CacheBytes(const std::vector<float>& values,
                                int32_t type) {
  std::vector<uint8_t> bytes;
  switch (type) {
    case kLiteRtNvidiaGeneralInt8:
      for (float v : values)
        bytes.push_back(static_cast<uint8_t>(static_cast<int8_t>(v)));
      return bytes;
    case kLiteRtNvidiaGeneralUint8:
      for (float v : values)
        bytes.push_back(static_cast<uint8_t>(static_cast<int32_t>(v) + 128));
      return bytes;
    case kLiteRtNvidiaGeneralInt4:
      for (size_t i = 0; i < values.size(); i += 2) {
        const uint8_t low =
            static_cast<uint8_t>(static_cast<int32_t>(values[i]) & 0xF);
        const uint8_t high =
            static_cast<uint8_t>(static_cast<int32_t>(values[i + 1]) & 0xF);
        bytes.push_back(static_cast<uint8_t>(low | (high << 4)));
      }
      return bytes;
    default: {
      std::vector<float> held;
      return FloatBytes(values, type, &held);
    }
  }
}

// The unscaled value of element i of a cache's bytes.
float CacheValue(const std::vector<uint8_t>& bytes, int32_t type, size_t i) {
  switch (type) {
    case kLiteRtNvidiaGeneralInt8:
      return static_cast<int8_t>(bytes[i]);
    case kLiteRtNvidiaGeneralUint8:
      return static_cast<int32_t>(bytes[i]) - 128;
    case kLiteRtNvidiaGeneralInt4: {
      const int32_t nibble =
          (i & 1) ? (bytes[i / 2] >> 4) : (bytes[i / 2] & 0xF);
      return nibble >= 8 ? nibble - 16 : nibble;
    }
    default:
      return FromBytes(bytes, type, i);
  }
}

int Bits(int32_t type) {
  switch (type) {
    case kLiteRtNvidiaGeneralInt8:
    case kLiteRtNvidiaGeneralUint8:
      return 8;
    case kLiteRtNvidiaGeneralInt4:
      return 4;
    default:
      return 0;
  }
}

struct Case {
  int kv_heads = 2;
  int rows = 4;
  int depth = 64;
  int cache_len = 64;
  int new_len = 0;
  int mask_rows = 1;
  bool bool_mask = true;
  int32_t q_type = kLiteRtNvidiaGeneralBFloat16;
  float q_int8_scale = 0.0f;
  int32_t cache_type = kLiteRtNvidiaGeneralInt8;
  float k_scale = 0.02f;
  float v_scale = 0.03f;
  int32_t new_type = kLiteRtNvidiaGeneralFloat16;
  float new_int8_scale = 0.0f;
  bool new_v_token_major = false;
  float scale = 1.0f;
  float softcap = 0.0f;
  // A mask row that hides every key (bool masks), or -1.
  int hidden_row = -1;
  uint32_t seed = 1;
};

void RunAttention(const Case& c) {
  if (!HasGpu()) GTEST_SKIP() << "no CUDA device";
  std::mt19937 rng(c.seed);
  std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
  const int bits = Bits(c.cache_type);
  const int keys = c.cache_len + c.new_len;
  const size_t cache_count =
      static_cast<size_t>(c.kv_heads) * c.cache_len * c.depth;

  ref::Caches caches;
  caches.heads = c.kv_heads;
  caches.len = c.cache_len;
  caches.depth = c.depth;
  caches.bits = bits;
  caches.k_scale = bits ? c.k_scale : 1.0f;
  caches.v_scale = bits ? c.v_scale : 1.0f;
  const int limit = bits == 4 ? 8 : 128;
  std::uniform_int_distribution<int> ints(-limit, limit - 1);
  for (auto* values : {&caches.k, &caches.v}) {
    values->resize(cache_count);
    for (auto& v : *values)
      v = bits ? static_cast<float>(ints(rng)) : unit(rng);
    if (!bits) {
      // The values a float cache holds.
      std::vector<float> held;
      FloatBytes(*values, c.cache_type, &held);
      *values = held;
    }
  }
  std::vector<float> q(static_cast<size_t>(c.kv_heads) * c.rows * c.depth);
  // Queries scaled so the scores spread over a few units.
  const float q_range =
      3.0f / std::sqrt(static_cast<float>(c.depth)) /
      (bits ? static_cast<float>(limit) * 0.6f * c.k_scale : 0.6f) /
      std::max(c.scale, 1e-3f);
  for (auto& v : q) v = unit(rng) * q_range;
  std::vector<float> q_held;
  const auto q_bytes = FloatBytes(q, c.q_type, &q_held);
  std::vector<float> new_k(static_cast<size_t>(c.kv_heads) * c.new_len *
                           c.depth);
  std::vector<float> new_v(new_k.size());
  const float new_range = bits ? limit * 0.8f * c.k_scale : 1.0f;
  for (auto& v : new_k) v = unit(rng) * new_range;
  for (auto& v : new_v) v = unit(rng) * new_range;
  std::vector<float> new_k_held, new_v_held;
  const auto new_k_bytes = FloatBytes(new_k, c.new_type, &new_k_held);
  // Values as the kernel takes them: rows, or token-major.
  std::vector<float> new_v_input(new_v.size());
  for (int h = 0; h < c.kv_heads; ++h) {
    for (int t = 0; t < c.new_len; ++t) {
      for (int d = 0; d < c.depth; ++d) {
        const size_t rows_index =
            (static_cast<size_t>(h) * c.new_len + t) * c.depth + d;
        const size_t index =
            c.new_v_token_major
                ? (static_cast<size_t>(t) * c.kv_heads + h) * c.depth + d
                : rows_index;
        new_v_input[index] = new_v[rows_index];
      }
    }
  }
  std::vector<float> new_v_input_held;
  const auto new_v_bytes =
      FloatBytes(new_v_input, c.new_type, &new_v_input_held);
  for (int h = 0; h < c.kv_heads; ++h) {
    for (int t = 0; t < c.new_len; ++t) {
      for (int d = 0; d < c.depth; ++d) {
        const size_t rows_index =
            (static_cast<size_t>(h) * c.new_len + t) * c.depth + d;
        const size_t index =
            c.new_v_token_major
                ? (static_cast<size_t>(t) * c.kv_heads + h) * c.depth + d
                : rows_index;
        new_v_held.resize(new_v.size());
        new_v_held[rows_index] = new_v_input_held[index];
      }
    }
  }

  std::vector<float> mask(static_cast<size_t>(c.mask_rows) * keys);
  std::vector<uint8_t> mask_bool(mask.size());
  std::uniform_real_distribution<float> chance(0.0f, 1.0f);
  for (int r = 0; r < c.mask_rows; ++r) {
    for (int j = 0; j < keys; ++j) {
      const size_t i = static_cast<size_t>(r) * keys + j;
      if (c.bool_mask) {
        const bool visible = r != c.hidden_row && chance(rng) < 0.7f;
        mask[i] = visible ? 1.0f : 0.0f;
        mask_bool[i] = visible ? 1 : 0;
      } else {
        const float u = chance(rng);
        mask[i] = u < 0.25f ? -1e9f : (u < 0.5f ? -2.5f * u : 0.0f);
      }
    }
  }

  ref::AttentionCase rc;
  rc.kv_heads = c.kv_heads;
  rc.rows = c.rows;
  rc.depth = c.depth;
  rc.new_len = c.new_len;
  rc.mask_rows = c.mask_rows;
  rc.bool_mask = c.bool_mask;
  rc.q_int8_scale = c.q_int8_scale;
  rc.new_int8_scale = c.new_int8_scale;
  rc.scale = c.scale;
  rc.softcap = c.softcap;
  const auto expected =
      ref::Attention(rc, caches, q_held, new_k_held, new_v_held, mask);

  std::vector<float> v_columns = caches.v;  // already [heads][depth][len]
  DeviceBuffer d_q(q_bytes);
  DeviceBuffer d_k(CacheBytes(caches.k, c.cache_type));
  DeviceBuffer d_v(CacheBytes(v_columns, c.cache_type));
  DeviceBuffer d_new_k(new_k_bytes);
  DeviceBuffer d_new_v(new_v_bytes);
  DeviceBuffer d_mask_bool(mask_bool);
  DeviceBuffer d_mask_float(mask);
  LiteRtNvidiaGeneralAttentionParams p{};
  p.kv_heads = c.kv_heads;
  p.rows = c.rows;
  p.depth = c.depth;
  p.cache_len = c.cache_len;
  p.new_len = c.new_len;
  p.mask_rows = c.mask_rows;
  p.mask_type = c.bool_mask ? kLiteRtNvidiaGeneralMaskBool
                            : kLiteRtNvidiaGeneralMaskFloat;
  p.q = {c.q_type, c.q_int8_scale};
  p.caches = {c.cache_type, c.k_scale, c.v_scale};
  p.new_kv = {c.new_type, c.new_int8_scale};
  p.new_v_token_major = c.new_v_token_major ? 1 : 0;
  p.scale = c.scale;
  p.softcap = c.softcap;
  ASSERT_TRUE(LiteRtNvidiaGeneralAttentionSupports(&p));
  DeviceBuffer workspace(LiteRtNvidiaGeneralAttentionWorkspaceBytes(&p));
  const size_t out_count = static_cast<size_t>(c.kv_heads) * c.rows * c.depth;
  DeviceBuffer d_out(out_count * TypeBytes(c.q_type));
  ASSERT_EQ(
      LiteRtNvidiaLaunchGeneralAttention(
          &p, d_q.get(), d_k.get(), d_v.get(), d_new_k.get(), d_new_v.get(),
          c.bool_mask ? d_mask_bool.get() : d_mask_float.get(), d_out.get(),
          workspace.get(), nullptr),
      cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  const auto out_bytes = d_out.Read<uint8_t>(out_count * TypeBytes(c.q_type));
  // FP32 accumulation, then the output type's rounding.
  const double type_eps =
      c.q_type == kLiteRtNvidiaGeneralFloat32
          ? 1e-6
          : (c.q_type == kLiteRtNvidiaGeneralFloat16 ? 1.0 / 1024 : 1.0 / 128);
  double max_error = 0.0;
  for (size_t i = 0; i < out_count; ++i) {
    const double got = FromBytes(out_bytes, c.q_type, i);
    const double tolerance = 2e-4 * new_range +
                             type_eps * std::fabs(expected[i]) +
                             (bits ? 2e-4 * limit * c.v_scale : 2e-4);
    ASSERT_NEAR(got, expected[i], tolerance) << "element " << i;
    max_error = std::max(max_error, std::fabs(got - expected[i]));
  }
  ::testing::Test::RecordProperty("max_error", std::to_string(max_error));
}

TEST(GeneralAttentionTest, Int8CachesFloatQueryWithScale) {
  Case c;
  c.rows = 4;
  c.depth = 128;
  c.cache_len = 200;
  c.scale = 0.088f;
  RunAttention(c);
}

TEST(GeneralAttentionTest, Int4CachesInt8QueryFloatMaskSoftcap) {
  Case c;
  c.rows = 8;
  c.mask_rows = 4;
  c.depth = 64;
  c.cache_len = 96;
  c.bool_mask = false;
  c.q_type = kLiteRtNvidiaGeneralFloat16;
  c.q_int8_scale = 0.004f;
  c.cache_type = kLiteRtNvidiaGeneralInt4;
  c.k_scale = 0.1f;
  c.v_scale = 0.07f;
  c.softcap = 2.0f;
  c.scale = 3.0f;
  RunAttention(c);
}

TEST(GeneralAttentionTest, Uint8CachesDepth512) {
  Case c;
  c.kv_heads = 1;
  c.rows = 3;
  c.depth = 512;
  c.cache_len = 300;
  c.cache_type = kLiteRtNvidiaGeneralUint8;
  c.q_type = kLiteRtNvidiaGeneralFloat32;
  RunAttention(c);
}

TEST(GeneralAttentionTest, Float16CachesManyRowsDepth96) {
  Case c;
  c.kv_heads = 2;
  c.rows = 40;
  c.mask_rows = 10;
  c.depth = 96;
  c.cache_len = 130;
  c.cache_type = kLiteRtNvidiaGeneralFloat16;
  RunAttention(c);
}

TEST(GeneralAttentionTest, FusedInt4CachesFloatChunkTokenMajor) {
  Case c;
  c.rows = 8;
  c.mask_rows = 4;
  c.new_len = 4;
  c.depth = 64;
  c.cache_len = 64;
  c.cache_type = kLiteRtNvidiaGeneralInt4;
  c.k_scale = 0.125f;
  c.v_scale = 0.0625f;
  c.new_v_token_major = true;
  RunAttention(c);
}

TEST(GeneralAttentionTest, FusedInt8CachesInt8ChunkOtherScaleSoftcapFloatMask) {
  Case c;
  c.rows = 6;
  c.mask_rows = 3;
  c.new_len = 3;
  c.depth = 256;
  c.cache_len = 512;
  c.bool_mask = false;
  c.new_int8_scale = 0.013f;
  c.q_int8_scale = 0.002f;
  c.softcap = 5.0f;
  RunAttention(c);
}

TEST(GeneralAttentionTest, RowSeeingNoKeyAveragesAll) {
  Case c;
  c.rows = 8;
  c.mask_rows = 4;
  c.hidden_row = 2;
  c.cache_len = 70;
  RunAttention(c);
}

TEST(GeneralAttentionTest, LongOddCacheSplits) {
  Case c;
  c.kv_heads = 2;
  c.rows = 8;
  c.depth = 256;
  c.cache_len = 8191;
  c.q_type = kLiteRtNvidiaGeneralFloat16;
  RunAttention(c);
}

TEST(GeneralAttentionTest, LongInt4Cache) {
  Case c;
  c.kv_heads = 4;
  c.rows = 2;
  c.depth = 128;
  c.cache_len = 4100;
  c.cache_type = kLiteRtNvidiaGeneralInt4;
  c.k_scale = 0.05f;
  c.v_scale = 0.05f;
  RunAttention(c);
}

// ---------------------------------------------------------------------------
// Cache writes.
// ---------------------------------------------------------------------------

struct WriteCase {
  int heads = 2;
  int cache_len = 16;
  int depth = 8;
  int tokens = 3;
  int32_t mode = kLiteRtNvidiaGeneralWriteSlice;
  int32_t cache_type = kLiteRtNvidiaGeneralInt8;
  float k_scale = 0.05f;
  float v_scale = 0.03f;
  int32_t src_type = kLiteRtNvidiaGeneralFloat32;
  float src_int8_scale = 0.0f;
  bool token_major = false;
  int32_t k_start = 5;
  int32_t v_start = 5;
  int32_t p0 = 0;
  int32_t p1 = 0;
};

int Quantized(float value, int32_t type, float scale) {
  return type == kLiteRtNvidiaGeneralInt4 ? ref::QuantizeInt4(value, scale)
                                          : ref::QuantizeInt8(value, scale);
}

void RunWrite(const WriteCase& c) {
  if (!HasGpu()) GTEST_SKIP() << "no CUDA device";
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
  const bool quant = Bits(c.cache_type) != 0;
  const size_t count = static_cast<size_t>(c.heads) * c.cache_len * c.depth;
  // Old cache contents.
  std::vector<float> k_old(count), v_old(count);
  std::uniform_int_distribution<int> ints(
      c.cache_type == kLiteRtNvidiaGeneralInt4 ? -8 : -128,
      c.cache_type == kLiteRtNvidiaGeneralInt4 ? 7 : 127);
  for (auto* values : {&k_old, &v_old}) {
    for (auto& v : *values)
      v = quant ? static_cast<float>(ints(rng)) : unit(rng);
    if (!quant) {
      std::vector<float> held;
      FloatBytes(*values, c.cache_type, &held);
      *values = held;
    }
  }
  const size_t src_count = static_cast<size_t>(c.heads) * c.tokens * c.depth;
  std::vector<float> src_k(src_count), src_v(src_count);
  const float range = quant ? 6.0f * c.k_scale : 1.0f;
  for (auto& v : src_k) v = unit(rng) * range;
  for (auto& v : src_v) v = unit(rng) * range;
  std::vector<float> k_held, v_held;
  const auto k_bytes = FloatBytes(src_k, c.src_type, &k_held);
  const auto v_bytes = FloatBytes(src_v, c.src_type, &v_held);

  // Expected caches, unscaled.
  std::vector<float> k_want = k_old, v_want = v_old;
  auto slot_of = [&](int t, int32_t start) -> int {
    if (c.mode == kLiteRtNvidiaGeneralWriteSlice) {
      return std::min(std::max(start, 0), c.cache_len - c.tokens) + t;
    }
    if (c.p0 < 0) return -1;
    const int n = std::min(std::max(c.p1 - c.p0, 0), c.tokens);
    const int first = std::max(0, n - c.cache_len);
    if (t < first || t >= n) return -1;
    return static_cast<int>((static_cast<int64_t>(c.p0) + t) % c.cache_len);
  };
  auto cache_value = [&](float x, float scale) -> float {
    if (c.src_int8_scale > 0.0f) {
      x = ref::QuantizeInt8(x, c.src_int8_scale) * c.src_int8_scale;
    }
    if (quant) return static_cast<float>(Quantized(x, c.cache_type, scale));
    std::vector<float> held;
    FloatBytes({x}, c.cache_type, &held);
    return held[0];
  };
  for (int t = 0; t < c.tokens; ++t) {
    for (int h = 0; h < c.heads; ++h) {
      for (int d = 0; d < c.depth; ++d) {
        const size_t rows_index =
            (static_cast<size_t>(h) * c.tokens + t) * c.depth + d;
        const size_t v_index =
            c.token_major ? (static_cast<size_t>(t) * c.heads + h) * c.depth + d
                          : rows_index;
        const int ks = slot_of(t, c.k_start);
        if (ks >= 0) {
          k_want[(static_cast<size_t>(h) * c.cache_len + ks) * c.depth + d] =
              cache_value(k_held[rows_index], c.k_scale);
        }
        const int vs = slot_of(t, c.v_start);
        if (vs >= 0) {
          v_want[(static_cast<size_t>(h) * c.depth + d) * c.cache_len + vs] =
              cache_value(v_held[v_index], c.v_scale);
        }
      }
    }
  }

  DeviceBuffer d_k(CacheBytes(k_old, c.cache_type));
  DeviceBuffer d_v(CacheBytes(v_old, c.cache_type));
  DeviceBuffer d_src_k(k_bytes);
  DeviceBuffer d_src_v(v_bytes);
  DeviceBuffer d_k_start(std::vector<int32_t>{c.k_start});
  DeviceBuffer d_v_start(std::vector<int32_t>{c.v_start});
  DeviceBuffer d_ring(std::vector<int32_t>{c.p0, c.p1});
  LiteRtNvidiaGeneralWriteParams p{};
  p.heads = c.heads;
  p.cache_len = c.cache_len;
  p.depth = c.depth;
  p.tokens = c.tokens;
  p.mode = c.mode;
  p.caches = {c.cache_type, c.k_scale, c.v_scale};
  p.src = {c.src_type, c.src_int8_scale};
  p.src_v_token_major = c.token_major ? 1 : 0;
  ASSERT_TRUE(LiteRtNvidiaGeneralCacheWriteSupports(&p));
  ASSERT_EQ(LiteRtNvidiaLaunchGeneralCacheWrite(
                &p, d_k.get(), d_v.get(), d_src_k.get(), d_src_v.get(),
                static_cast<const int32_t*>(d_k_start.get()),
                static_cast<const int32_t*>(d_v_start.get()),
                static_cast<const int32_t*>(d_ring.get()), nullptr),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  const size_t bytes = CacheBytes(k_old, c.cache_type).size();
  const auto k_got = d_k.Read<uint8_t>(bytes);
  const auto v_got = d_v.Read<uint8_t>(bytes);
  for (size_t i = 0; i < count; ++i) {
    ASSERT_EQ(CacheValue(k_got, c.cache_type, i), k_want[i]) << "key " << i;
    ASSERT_EQ(CacheValue(v_got, c.cache_type, i), v_want[i]) << "value " << i;
  }
}

TEST(GeneralCacheWriteTest, SliceFloatIntoInt8) {
  WriteCase c;
  RunWrite(c);
}

TEST(GeneralCacheWriteTest, SliceClampsTheStart) {
  WriteCase c;
  c.k_start = 15;
  c.v_start = -3;
  RunWrite(c);
}

TEST(GeneralCacheWriteTest, SliceBf16IntoUint8TokenMajor) {
  WriteCase c;
  c.cache_type = kLiteRtNvidiaGeneralUint8;
  c.src_type = kLiteRtNvidiaGeneralBFloat16;
  c.token_major = true;
  RunWrite(c);
}

TEST(GeneralCacheWriteTest, SliceIntoInt4OddStarts) {
  WriteCase c;
  c.cache_type = kLiteRtNvidiaGeneralInt4;
  c.src_type = kLiteRtNvidiaGeneralFloat16;
  c.k_scale = 0.2f;
  c.v_scale = 0.15f;
  c.k_start = 3;
  c.v_start = 7;
  c.tokens = 4;
  RunWrite(c);
}

TEST(GeneralCacheWriteTest, SliceRequantizesOtherScale) {
  WriteCase c;
  c.src_int8_scale = 0.011f;
  c.cache_type = kLiteRtNvidiaGeneralInt4;
  c.k_scale = 0.09f;
  c.v_scale = 0.08f;
  c.tokens = 1;
  c.k_start = 9;
  c.v_start = 9;
  RunWrite(c);
}

TEST(GeneralCacheWriteTest, SliceIntoFloat16) {
  WriteCase c;
  c.cache_type = kLiteRtNvidiaGeneralFloat16;
  RunWrite(c);
}

TEST(GeneralCacheWriteTest, RingWrapsInt4) {
  WriteCase c;
  c.mode = kLiteRtNvidiaGeneralWriteRing;
  c.cache_type = kLiteRtNvidiaGeneralInt4;
  c.k_scale = 0.2f;
  c.v_scale = 0.2f;
  c.tokens = 5;
  c.p0 = 13;
  c.p1 = 18;
  RunWrite(c);
}

TEST(GeneralCacheWriteTest, RingLongerThanCacheKeepsNewest) {
  WriteCase c;
  c.mode = kLiteRtNvidiaGeneralWriteRing;
  c.cache_len = 8;
  c.tokens = 11;
  c.p0 = 21;
  c.p1 = 31;  // 10 valid tokens, the last 8 written
  c.cache_type = kLiteRtNvidiaGeneralInt4;
  c.k_scale = 0.2f;
  c.v_scale = 0.2f;
  RunWrite(c);
}

TEST(GeneralCacheWriteTest, RingBeforeStartWritesNothing) {
  WriteCase c;
  c.mode = kLiteRtNvidiaGeneralWriteRing;
  c.p0 = -2;
  c.p1 = 1;
  RunWrite(c);
}

}  // namespace
