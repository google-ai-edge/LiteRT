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

#include "litert/vendors/nvidia/trtllm/tiled_attention.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "cuda_runtime_api.h"
#include "driver_types.h"

namespace {

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

// Round to nearest even; the test values stay inside the FP16 range.
uint16_t FloatToHalfBits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000;
  const int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFF) - 112;
  uint32_t mantissa = bits & 0x7FFFFF;
  if (exponent <= 0) {
    if (exponent < -10) {
      return static_cast<uint16_t>(sign);
    }
    mantissa |= 0x800000;
    const uint32_t shift = static_cast<uint32_t>(14 - exponent);
    uint32_t half = mantissa >> shift;
    const uint32_t rest = mantissa & ((1u << shift) - 1);
    const uint32_t halfway = 1u << (shift - 1);
    if (rest > halfway || (rest == halfway && (half & 1))) {
      ++half;
    }
    return static_cast<uint16_t>(sign | half);
  }
  if (exponent >= 31) {
    return static_cast<uint16_t>(sign | 0x7C00);
  }
  uint32_t half =
      sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13);
  const uint32_t rest = mantissa & 0x1FFF;
  if (rest > 0x1000 || (rest == 0x1000 && (half & 1))) {
    ++half;
  }
  return static_cast<uint16_t>(half);
}

float HalfBitsToFloat(uint16_t half) {
  const uint32_t sign = static_cast<uint32_t>(half & 0x8000) << 16;
  const uint32_t exponent = (half >> 10) & 0x1F;
  const uint32_t mantissa = half & 0x3FF;
  if (exponent == 0) {
    const float magnitude = std::ldexp(static_cast<float>(mantissa), -24);
    return (half & 0x8000) ? -magnitude : magnitude;
  }
  uint32_t bits = exponent == 31
                      ? sign | 0x7F800000 | (mantissa << 13)
                      : sign | ((exponent + 112) << 23) | (mantissa << 13);
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

uint16_t FloatToBf16Bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fff + ((bits >> 16) & 1);
  return static_cast<uint16_t>(bits >> 16);
}

float Bf16BitsToFloat(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

enum class MaskKind {
  kNone,    // no mask: every key is visible
  kCausal,  // mask row t sees keys [0, position + t]
  // A ring cache of cache_len slots holds the positions before `position`
  // (slot = position % cache_len); mask row t is token position + t and sees
  // the `window` positions up to itself, in the cache or among the new keys.
  kRing,
};

struct Case {
  LiteRtNvidiaAttentionShape shape;
  MaskKind mask_kind;
  int position;
  int window;
  // Every `hole`-th key is hidden from the rows that see it (0: no holes).
  int hole;
  // This mask row and the ones from `first_padded_mask_row` on see nothing.
  int hidden_mask_row;
  int first_padded_mask_row;
  float fill;
  bool bf16;
};

struct Tensors {
  std::vector<uint16_t> q_bits, k_bits, v_bits, k_new_bits, v_new_bits;
  std::vector<float> q, k, v, k_new, v_new;
  std::vector<uint8_t> mask;
};

Tensors MakeTensors(const Case& test_case) {
  const auto& shape = test_case.shape;
  const int seq = shape.cache_len + shape.new_len;
  uint32_t state = 2463534242u + shape.rows + seq + shape.heads;
  const auto next = [&]() -> float {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return static_cast<float>(state >> 8) / 8388608.0f - 1.0f;  // [-1, 1)
  };
  Tensors tensors;
  tensors.q_bits.resize(static_cast<size_t>(shape.heads) * shape.rows *
                        shape.depth);
  tensors.q.resize(tensors.q_bits.size());
  for (size_t i = 0; i < tensors.q_bits.size(); ++i) {
    const float value = next();
    if (test_case.bf16) {
      tensors.q_bits[i] = FloatToBf16Bits(value);
      // The kernel multiplies queries as FP16.
      tensors.q[i] =
          HalfBitsToFloat(FloatToHalfBits(Bf16BitsToFloat(tensors.q_bits[i])));
    } else {
      tensors.q_bits[i] = FloatToHalfBits(value);
      tensors.q[i] = HalfBitsToFloat(tensors.q_bits[i]);
    }
  }
  const auto fill_cache = [&](int length, float scale,
                              std::vector<uint16_t>* bits,
                              std::vector<float>* values) {
    bits->resize(static_cast<size_t>(shape.heads) * length * shape.depth);
    values->resize(bits->size());
    for (size_t i = 0; i < bits->size(); ++i) {
      (*bits)[i] = FloatToHalfBits(scale * next());
      (*values)[i] = HalfBitsToFloat((*bits)[i]);
    }
  };
  fill_cache(shape.cache_len, 0.5f, &tensors.k_bits, &tensors.k);
  fill_cache(shape.cache_len, 4.0f, &tensors.v_bits, &tensors.v);
  fill_cache(shape.new_len, 0.5f, &tensors.k_new_bits, &tensors.k_new);
  fill_cache(shape.new_len, 4.0f, &tensors.v_new_bits, &tensors.v_new);

  tensors.mask.assign(static_cast<size_t>(shape.mask_rows) * seq,
                      test_case.mask_kind == MaskKind::kNone ? 1 : 0);
  for (int t = 0; t < shape.mask_rows; ++t) {
    if (test_case.mask_kind == MaskKind::kNone ||
        t == test_case.hidden_mask_row ||
        (test_case.first_padded_mask_row >= 0 &&
         t >= test_case.first_padded_mask_row)) {
      continue;
    }
    uint8_t* row = tensors.mask.data() + static_cast<size_t>(t) * seq;
    const auto show = [&](int key) {
      row[key] =
          test_case.hole == 0 || key % test_case.hole != test_case.hole - 1;
    };
    if (test_case.mask_kind == MaskKind::kCausal) {
      for (int j = 0; j <= std::min(test_case.position + t, seq - 1); ++j) {
        show(j);
      }
      continue;
    }
    const int token = test_case.position + t;
    const int oldest = std::max({0, token - test_case.window + 1,
                                 test_case.position - shape.cache_len});
    for (int p = oldest; p < test_case.position; ++p) {
      show(p % shape.cache_len);
    }
    for (int j = std::max(0, t - test_case.window + 1);
         j <= std::min(t, shape.new_len - 1); ++j) {
      show(shape.cache_len + j);
    }
  }
  return tensors;
}

// softmax_j(mask ? q . key_j : fill) . value over all keys, in double.
std::vector<double> Reference(const Case& test_case, const Tensors& tensors,
                              int head, int row) {
  const auto& shape = test_case.shape;
  const int seq = shape.cache_len + shape.new_len;
  const int depth = shape.depth;
  const uint8_t* mask_row =
      tensors.mask.data() + static_cast<size_t>(row % shape.mask_rows) * seq;
  const float* q =
      tensors.q.data() + (static_cast<size_t>(head) * shape.rows + row) * depth;
  const auto cache_row = [&](const std::vector<float>& cache,
                             const std::vector<float>& fresh, int key) {
    return key < shape.cache_len
               ? cache.data() +
                     (static_cast<size_t>(head) * shape.cache_len + key) * depth
               : fresh.data() + (static_cast<size_t>(head) * shape.new_len +
                                 key - shape.cache_len) *
                                    depth;
  };
  std::vector<double> scores(seq);
  double max_score = -INFINITY;
  for (int j = 0; j < seq; ++j) {
    double score = test_case.fill;
    if (mask_row[j]) {
      const float* key = cache_row(tensors.k, tensors.k_new, j);
      score = 0.0;
      for (int d = 0; d < depth; ++d) {
        score += static_cast<double>(q[d]) * key[d];
      }
    }
    scores[j] = score;
    max_score = std::max(max_score, score);
  }
  std::vector<double> out(depth, 0.0);
  double sum = 0.0;
  for (int j = 0; j < seq; ++j) {
    const double p =
        std::isinf(max_score) ? 0.0 : std::exp(scores[j] - max_score);
    if (p == 0.0) continue;
    sum += p;
    const float* value = cache_row(tensors.v, tensors.v_new, j);
    for (int d = 0; d < depth; ++d) {
      out[d] += p * value[d];
    }
  }
  for (double& value : out) {
    value = sum > 0.0 ? value / sum : 0.0;
  }
  return out;
}

void RunCase(const Case& test_case) {
  const auto& shape = test_case.shape;
  SCOPED_TRACE(::testing::Message()
               << "heads=" << shape.heads << " rows=" << shape.rows
               << " mask_rows=" << shape.mask_rows << " depth=" << shape.depth
               << " cache_len=" << shape.cache_len << " new_len="
               << shape.new_len << " position=" << test_case.position
               << " hole=" << test_case.hole << " bf16=" << test_case.bf16);
  ASSERT_TRUE(LiteRtNvidiaTiledAttentionSupports(&shape));
  const Tensors tensors = MakeTensors(test_case);
  const size_t workspace_bytes =
      LiteRtNvidiaTiledAttentionWorkspaceBytes(&shape);
  ASSERT_GT(workspace_bytes, 0u);
  auto q_device = Upload(tensors.q_bits);
  auto k_device = Upload(tensors.k_bits);
  auto v_device = Upload(tensors.v_bits);
  auto k_new_device = Upload(tensors.k_new_bits);
  auto v_new_device = Upload(tensors.v_new_bits);
  auto mask_device = Upload(tensors.mask);
  auto out_device = AllocateCuda(tensors.q_bits.size() * sizeof(uint16_t));
  auto workspace = AllocateCuda(workspace_bytes);
  ASSERT_NE(out_device, nullptr);
  ASSERT_NE(workspace, nullptr);
  ASSERT_EQ(cudaMemset(out_device.get(), 0xff,
                       tensors.q_bits.size() * sizeof(uint16_t)),
            cudaSuccess);
  ASSERT_EQ(LiteRtNvidiaLaunchTiledAttention(
                &shape, q_device.get(), test_case.bf16, k_device.get(),
                v_device.get(), k_new_device.get(), v_new_device.get(),
                test_case.mask_kind == MaskKind::kNone
                    ? nullptr
                    : static_cast<const bool*>(mask_device.get()),
                test_case.fill, out_device.get(), workspace.get(),
                /*stream=*/nullptr),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<uint16_t> out_bits(tensors.q_bits.size());
  ASSERT_EQ(
      cudaMemcpy(out_bits.data(), out_device.get(),
                 out_bits.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
      cudaSuccess);

  const double tolerance = test_case.bf16 ? 1.0e-2 : 1.5e-3;
  for (int head = 0; head < shape.heads; ++head) {
    for (int row = 0; row < shape.rows; ++row) {
      // Every row of the first and last row groups of the first and last
      // heads, and a sample of the others.
      const bool edge = (head == 0 || head == shape.heads - 1) &&
                        (row < 16 || row >= shape.rows - 16);
      if (!edge && (head * shape.rows + row) % 61 != 0) continue;
      const std::vector<double> expected =
          Reference(test_case, tensors, head, row);
      double worst = 0.0;
      double scale = 1.0;
      for (int d = 0; d < shape.depth; ++d) {
        const uint16_t bits =
            out_bits[(static_cast<size_t>(head) * shape.rows + row) *
                         shape.depth +
                     d];
        const double value =
            test_case.bf16 ? Bf16BitsToFloat(bits) : HalfBitsToFloat(bits);
        ASSERT_TRUE(std::isfinite(value))
            << "head=" << head << " row=" << row << " dim=" << d;
        worst = std::max(worst, std::fabs(value - expected[d]));
        scale = std::max(scale, std::fabs(expected[d]));
      }
      EXPECT_LE(worst, tolerance * scale) << "head=" << head << " row=" << row;
    }
  }
}

TEST(TiledAttentionTest, SupportsBlocksOf32768QueryElements) {
  const auto supports = [](LiteRtNvidiaAttentionShape shape) {
    return LiteRtNvidiaTiledAttentionSupports(&shape);
  };
  EXPECT_TRUE(supports({1, 2048, 128, 512, 130816, 0}));
  EXPECT_TRUE(supports({1, 16384, 1024, 512, 2048, 0}));
  EXPECT_TRUE(supports({1, 64, 64, 512, 100, 0}));
  EXPECT_TRUE(supports({8, 2048, 1024, 256, 1152, 1024}));
  EXPECT_TRUE(supports({8, 256, 128, 256, 1152, 128}));
  EXPECT_TRUE(supports({8, 256, 256, 256, 1152, 100}));
  // Rows and mask rows in whole blocks.
  EXPECT_FALSE(supports({1, 2048, 16, 512, 4096, 0}));
  EXPECT_FALSE(supports({1, 96, 96, 512, 4096, 0}));
  EXPECT_FALSE(supports({8, 64, 64, 256, 1152, 64}));
  EXPECT_FALSE(supports({1, 128, 192, 512, 4096, 0}));
  // The new keys start a tile.
  EXPECT_FALSE(supports({8, 256, 128, 256, 1150, 128}));
  EXPECT_FALSE(supports({1, 2048, 128, 128, 4096, 0}));
  EXPECT_FALSE(supports({0, 2048, 128, 512, 4096, 0}));
  EXPECT_FALSE(supports({1, 2048, 128, 512, 0, 128}));
  EXPECT_FALSE(supports({1, 2048, 128, 512, 300000, 0}));
  EXPECT_FALSE(LiteRtNvidiaTiledAttentionSupports(nullptr));
  LiteRtNvidiaAttentionShape unsupported = {1, 96, 96, 512, 4096, 0};
  EXPECT_EQ(LiteRtNvidiaTiledAttentionWorkspaceBytes(&unsupported), 0u);
}

TEST(TiledAttentionTest, RejectsInvalidArguments) {
  uint16_t sentinel = 0;
  bool mask = true;
  LiteRtNvidiaAttentionShape shape = {8, 256, 128, 256, 1152, 128};
  EXPECT_NE(LiteRtNvidiaLaunchTiledAttention(
                &shape, nullptr, false, &sentinel, &sentinel, &sentinel,
                &sentinel, &mask, -1.0e4f, &sentinel, &sentinel, nullptr),
            cudaSuccess);
  EXPECT_NE(LiteRtNvidiaLaunchTiledAttention(
                &shape, &sentinel, false, &sentinel, &sentinel, nullptr,
                &sentinel, &mask, -1.0e4f, &sentinel, &sentinel, nullptr),
            cudaSuccess);
  EXPECT_NE(LiteRtNvidiaLaunchTiledAttention(
                &shape, &sentinel, false, &sentinel, &sentinel, &sentinel,
                &sentinel, &mask, -1.0e4f, &sentinel, nullptr, nullptr),
            cudaSuccess);
  shape.rows = 100;
  EXPECT_NE(LiteRtNvidiaLaunchTiledAttention(
                &shape, &sentinel, false, &sentinel, &sentinel, &sentinel,
                &sentinel, &mask, -1.0e4f, &sentinel, &sentinel, nullptr),
            cudaSuccess);
}

TEST(TiledAttentionTest, SingleHeadCacheMatchesReference) {
  if (!LiteRtNvidiaTiledAttentionAvailable()) {
    GTEST_SKIP() << "Requires compute capability 8.0 or newer and 100 KB of "
                    "shared memory per block.";
  }
  // One block and a partial last tile; several row tiles at the start, the
  // middle and the end of a cache; holes in the mask; no mask.
  RunCase({{1, 64, 64, 512, 100, 0},
           MaskKind::kCausal,
           30,
           0,
           0,
           -1,
           -1,
           -1.0e4f,
           false});
  RunCase({{1, 256, 64, 512, 2100, 0},
           MaskKind::kCausal,
           0,
           0,
           0,
           -1,
           -1,
           -45824.0f,
           false});
  RunCase({{1, 2048, 128, 512, 5000, 0},
           MaskKind::kCausal,
           3333,
           0,
           0,
           -1,
           -1,
           -45824.0f,
           true});
  RunCase({{1, 2048, 128, 512, 5000, 0},
           MaskKind::kCausal,
           4872,
           0,
           7,
           -1,
           -1,
           -INFINITY,
           false});
  RunCase({{1, 128, 128, 512, 700, 0},
           MaskKind::kNone,
           0,
           0,
           0,
           -1,
           -1,
           -1.0e4f,
           false});
}

TEST(TiledAttentionTest, RingCacheAndNewKeysMatchReference) {
  if (!LiteRtNvidiaTiledAttentionAvailable()) {
    GTEST_SKIP() << "Requires compute capability 8.0 or newer and 100 KB of "
                    "shared memory per block.";
  }
  // Eight heads with two query heads each over a ring cache: the first chunk,
  // a cache that has not wrapped, one that has, and a padded chunk.
  RunCase({{8, 256, 128, 256, 1152, 128},
           MaskKind::kRing,
           0,
           1024,
           0,
           -1,
           -1,
           -45824.0f,
           false});
  RunCase({{8, 256, 128, 256, 1152, 128},
           MaskKind::kRing,
           700,
           1024,
           0,
           -1,
           -1,
           -45824.0f,
           true});
  RunCase({{8, 256, 128, 256, 1152, 128},
           MaskKind::kRing,
           3000,
           1024,
           5,
           -1,
           -1,
           -INFINITY,
           false});
  RunCase({{8, 256, 128, 256, 1152, 128},
           MaskKind::kRing,
           3000,
           1024,
           0,
           -1,
           100,
           -45824.0f,
           false});
  RunCase({{2, 2048, 1024, 256, 1152, 1024},
           MaskKind::kRing,
           5000,
           1024,
           0,
           -1,
           -1,
           -45824.0f,
           false});
  // New keys that end inside a tile.
  RunCase({{2, 256, 128, 256, 1152, 100},
           MaskKind::kRing,
           2000,
           1024,
           0,
           -1,
           -1,
           -1.0e4f,
           false});
}

TEST(TiledAttentionTest, RowsWithoutVisibleKeysAverageAllKeys) {
  if (!LiteRtNvidiaTiledAttentionAvailable()) {
    GTEST_SKIP() << "Requires compute capability 8.0 or newer and 100 KB of "
                    "shared memory per block.";
  }
  // With a finite fill a row that sees no key weighs every key equally; the
  // other rows of its block must not be disturbed.
  RunCase({{1, 128, 64, 512, 900, 0},
           MaskKind::kCausal,
           300,
           0,
           0,
           5,
           -1,
           -1.0e4f,
           false});
  RunCase({{4, 256, 128, 256, 1152, 128},
           MaskKind::kRing,
           2000,
           1024,
           0,
           77,
           -1,
           -1.0e4f,
           false});
}

}  // namespace
