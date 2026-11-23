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

#include "litert/vendors/nvidia/trtllm/global_attention.h"

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

constexpr int kDepth = 512;

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
  if (device != nullptr) {
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

bool HasComputeCapability80() {
  int device = 0;
  int major = 0;
  return cudaGetDevice(&device) == cudaSuccess &&
         cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                device) == cudaSuccess &&
         major >= 8;
}

struct Case {
  int rows;
  int mask_rows;
  int seq;
  // Mask row t sees keys [0, position + t] except every `hole`-th one
  // (0: no holes). Rows listed in `hidden_mask_row` see nothing.
  int position;
  int hole;
  int hidden_mask_row;
  float fill;
  bool bf16;
};

// softmax_j(mask ? q . k_j : fill) . v over the whole sequence, in double.
std::vector<double> Reference(const Case& test_case,
                              const std::vector<float>& q,
                              const std::vector<float>& k,
                              const std::vector<float>& v,
                              const std::vector<uint8_t>& mask, int row) {
  const int seq = test_case.seq;
  const uint8_t* mask_row =
      mask.data() + static_cast<size_t>(row % test_case.mask_rows) * seq;
  std::vector<double> scores(seq);
  double max_score = -INFINITY;
  for (int j = 0; j < seq; ++j) {
    double score = test_case.fill;
    if (mask_row[j]) {
      score = 0.0;
      for (int d = 0; d < kDepth; ++d) {
        score += static_cast<double>(q[static_cast<size_t>(row) * kDepth + d]) *
                 k[static_cast<size_t>(j) * kDepth + d];
      }
    }
    scores[j] = score;
    max_score = std::max(max_score, score);
  }
  std::vector<double> out(kDepth, 0.0);
  double sum = 0.0;
  for (int j = 0; j < seq; ++j) {
    const double p =
        std::isinf(max_score) ? 0.0 : std::exp(scores[j] - max_score);
    sum += p;
    for (int d = 0; d < kDepth; ++d) {
      out[d] += p * v[static_cast<size_t>(j) * kDepth + d];
    }
  }
  for (double& value : out) {
    value = sum > 0.0 ? value / sum : 0.0;
  }
  return out;
}

void RunCase(const Case& test_case) {
  SCOPED_TRACE(::testing::Message()
               << "rows=" << test_case.rows
               << " mask_rows=" << test_case.mask_rows
               << " seq=" << test_case.seq << " position=" << test_case.position
               << " hole=" << test_case.hole << " bf16=" << test_case.bf16);
  const int rows = test_case.rows;
  const int seq = test_case.seq;
  uint32_t state = 2463534242u + rows + seq;
  const auto next = [&]() -> float {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return static_cast<float>(state >> 8) / 8388608.0f - 1.0f;  // [-1, 1)
  };
  std::vector<uint16_t> q_bits(static_cast<size_t>(rows) * kDepth);
  std::vector<float> q(q_bits.size());
  for (size_t i = 0; i < q_bits.size(); ++i) {
    const float value = next();
    if (test_case.bf16) {
      q_bits[i] = FloatToBf16Bits(value);
      // The kernel multiplies queries as FP16.
      q[i] = HalfBitsToFloat(FloatToHalfBits(Bf16BitsToFloat(q_bits[i])));
    } else {
      q_bits[i] = FloatToHalfBits(value);
      q[i] = HalfBitsToFloat(q_bits[i]);
    }
  }
  std::vector<uint16_t> k_bits(static_cast<size_t>(seq) * kDepth);
  std::vector<uint16_t> v_bits(k_bits.size());
  std::vector<float> k(k_bits.size());
  std::vector<float> v(k_bits.size());
  for (size_t i = 0; i < k_bits.size(); ++i) {
    k_bits[i] = FloatToHalfBits(0.5f * next());
    v_bits[i] = FloatToHalfBits(4.0f * next());
    k[i] = HalfBitsToFloat(k_bits[i]);
    v[i] = HalfBitsToFloat(v_bits[i]);
  }
  std::vector<uint8_t> mask(static_cast<size_t>(test_case.mask_rows) * seq, 0);
  for (int t = 0; t < test_case.mask_rows; ++t) {
    if (t == test_case.hidden_mask_row) continue;
    for (int j = 0; j <= std::min(test_case.position + t, seq - 1); ++j) {
      mask[static_cast<size_t>(t) * seq + j] =
          test_case.hole == 0 || j % test_case.hole != test_case.hole - 1;
    }
  }

  const size_t workspace_bytes = LiteRtNvidiaGlobalAttentionWorkspaceBytes(
      rows, test_case.mask_rows, seq, kDepth);
  ASSERT_GT(workspace_bytes, 0u);
  auto q_device = Upload(q_bits);
  auto k_device = Upload(k_bits);
  auto v_device = Upload(v_bits);
  auto mask_device = Upload(mask);
  auto out_device = AllocateCuda(q_bits.size() * sizeof(uint16_t));
  auto workspace = AllocateCuda(workspace_bytes);
  ASSERT_NE(out_device, nullptr);
  ASSERT_NE(workspace, nullptr);
  ASSERT_EQ(
      cudaMemset(out_device.get(), 0xff, q_bits.size() * sizeof(uint16_t)),
      cudaSuccess);
  ASSERT_EQ(LiteRtNvidiaLaunchGlobalAttention(
                q_device.get(), test_case.bf16, k_device.get(), v_device.get(),
                static_cast<const bool*>(mask_device.get()),
                test_case.mask_rows, rows, seq, kDepth, test_case.fill,
                out_device.get(), workspace.get(), /*stream=*/nullptr),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<uint16_t> out_bits(q_bits.size());
  ASSERT_EQ(
      cudaMemcpy(out_bits.data(), out_device.get(),
                 out_bits.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
      cudaSuccess);

  // Every row of the first and last blocks and a sample of the others.
  const double tolerance = test_case.bf16 ? 1.0e-2 : 1.5e-3;
  for (int row = 0; row < rows; ++row) {
    if (row >= 32 && row < rows - 32 && row % 37 != 0) continue;
    const std::vector<double> expected =
        Reference(test_case, q, k, v, mask, row);
    double worst = 0.0;
    double scale = 1.0;
    for (int d = 0; d < kDepth; ++d) {
      const uint16_t bits = out_bits[static_cast<size_t>(row) * kDepth + d];
      const double value =
          test_case.bf16 ? Bf16BitsToFloat(bits) : HalfBitsToFloat(bits);
      ASSERT_TRUE(std::isfinite(value)) << "row=" << row << " dim=" << d;
      worst = std::max(worst, std::fabs(value - expected[d]));
      scale = std::max(scale, std::fabs(expected[d]));
    }
    EXPECT_LE(worst, tolerance * scale) << "row=" << row;
  }
}

// A few tokens of several query heads (row = head * tokens + token): the
// token launch against the 16-row launches it stands for, one per 16 / heads
// tokens with the tokens' mask rows repeated per head, bit for bit.
struct TokenCase {
  int heads;
  int tokens;
  int seq;
  // Mask row t sees keys [0, position + t] except every `hole`-th one (0: no
  // holes); `hidden_token`'s row sees nothing. A shared mask is row 0 alone.
  int position;
  int hole;
  int hidden_token;
  bool shared_mask;
  float fill;
  bool bf16;
};

void RunTokenCase(const TokenCase& test_case) {
  SCOPED_TRACE(::testing::Message()
               << "heads=" << test_case.heads << " tokens=" << test_case.tokens
               << " seq=" << test_case.seq << " position=" << test_case.position
               << " hole=" << test_case.hole
               << " hidden=" << test_case.hidden_token << " shared_mask="
               << test_case.shared_mask << " bf16=" << test_case.bf16);
  const int heads = test_case.heads;
  const int tokens = test_case.tokens;
  const int rows = heads * tokens;
  const int chunk = 16 / heads;
  const int seq = test_case.seq;
  const int mask_rows = test_case.shared_mask ? 1 : tokens;
  ASSERT_TRUE(LiteRtNvidiaGlobalAttentionTokensSupports(rows, tokens, mask_rows,
                                                        kDepth));
  uint32_t state = 88172645u + rows + seq + test_case.position;
  const auto next = [&]() -> float {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return static_cast<float>(state >> 8) / 8388608.0f - 1.0f;  // [-1, 1)
  };
  std::vector<uint16_t> q(static_cast<size_t>(rows) * kDepth);
  for (uint16_t& bits : q) {
    bits = test_case.bf16 ? FloatToBf16Bits(next()) : FloatToHalfBits(next());
  }
  std::vector<uint16_t> k(static_cast<size_t>(seq) * kDepth);
  std::vector<uint16_t> v(k.size());
  for (size_t i = 0; i < k.size(); ++i) {
    k[i] = FloatToHalfBits(0.5f * next());
    v[i] = FloatToHalfBits(4.0f * next());
  }
  std::vector<uint8_t> mask(static_cast<size_t>(mask_rows) * seq, 0);
  for (int t = 0; t < mask_rows; ++t) {
    if (t == test_case.hidden_token) continue;
    for (int j = 0; j <= std::min(test_case.position + t, seq - 1); ++j) {
      mask[static_cast<size_t>(t) * seq + j] =
          test_case.hole == 0 || j % test_case.hole != test_case.hole - 1;
    }
  }

  auto k_device = Upload(k);
  auto v_device = Upload(v);
  const auto run = [&](const std::vector<uint16_t>& q_host,
                       const std::vector<uint8_t>& mask_host, int launch_rows,
                       int launch_mask_rows, bool by_token,
                       std::vector<uint16_t>* out) {
    const size_t workspace_bytes =
        by_token ? LiteRtNvidiaGlobalAttentionTokensWorkspaceBytes(
                       launch_rows, tokens, launch_mask_rows, seq, kDepth)
                 : LiteRtNvidiaGlobalAttentionWorkspaceBytes(
                       launch_rows, launch_mask_rows, seq, kDepth);
    ASSERT_GT(workspace_bytes, 0u);
    auto q_device = Upload(q_host);
    auto mask_device = Upload(mask_host);
    auto out_device = AllocateCuda(q_host.size() * sizeof(uint16_t));
    auto workspace = AllocateCuda(workspace_bytes);
    ASSERT_NE(out_device, nullptr);
    ASSERT_NE(workspace, nullptr);
    ASSERT_EQ(
        cudaMemset(out_device.get(), 0xff, q_host.size() * sizeof(uint16_t)),
        cudaSuccess);
    const cudaError_t status =
        by_token
            ? LiteRtNvidiaLaunchGlobalAttentionTokens(
                  q_device.get(), test_case.bf16, k_device.get(),
                  v_device.get(), static_cast<const bool*>(mask_device.get()),
                  launch_mask_rows, tokens, launch_rows, seq, kDepth,
                  test_case.fill, out_device.get(), workspace.get(),
                  /*stream=*/nullptr)
            : LiteRtNvidiaLaunchGlobalAttention(
                  q_device.get(), test_case.bf16, k_device.get(),
                  v_device.get(), static_cast<const bool*>(mask_device.get()),
                  launch_mask_rows, launch_rows, seq, kDepth, test_case.fill,
                  out_device.get(), workspace.get(),
                  /*stream=*/nullptr);
    ASSERT_EQ(status, cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    out->resize(q_host.size());
    ASSERT_EQ(
        cudaMemcpy(out->data(), out_device.get(),
                   out->size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
        cudaSuccess);
  };

  std::vector<uint16_t> out;
  run(q, mask, rows, mask_rows, /*by_token=*/true, &out);
  if (::testing::Test::HasFatalFailure()) return;
  for (int tile = 0; tile < tokens / chunk; ++tile) {
    // The tile's rows as a launch of its own takes them: head-major over the
    // tile's tokens, a mask row per query row (or the one a single token or a
    // shared mask has).
    const int tile_mask_rows = test_case.shared_mask || chunk == 1 ? 1 : 16;
    std::vector<uint16_t> tile_q(static_cast<size_t>(16) * kDepth);
    std::vector<uint8_t> tile_mask(static_cast<size_t>(tile_mask_rows) * seq);
    std::vector<int> source(16);
    for (int r = 0; r < 16; ++r) {
      const int token = tile * chunk + r % chunk;
      source[r] = (r / chunk) * tokens + token;
      std::copy_n(q.begin() + static_cast<size_t>(source[r]) * kDepth, kDepth,
                  tile_q.begin() + static_cast<size_t>(r) * kDepth);
      if (r < tile_mask_rows) {
        std::copy_n(mask.begin() + static_cast<size_t>(token % mask_rows) * seq,
                    seq, tile_mask.begin() + static_cast<size_t>(r) * seq);
      }
    }
    std::vector<uint16_t> tile_out;
    run(tile_q, tile_mask, 16, tile_mask_rows, /*by_token=*/false, &tile_out);
    if (::testing::Test::HasFatalFailure()) return;
    for (int r = 0; r < 16; ++r) {
      EXPECT_EQ(std::memcmp(&out[static_cast<size_t>(source[r]) * kDepth],
                            &tile_out[static_cast<size_t>(r) * kDepth],
                            kDepth * sizeof(uint16_t)),
                0)
          << "tile=" << tile << " row=" << r;
    }
  }
}

TEST(GlobalAttentionTest, SupportsDepth512RowsInMultiplesOf16) {
  EXPECT_TRUE(LiteRtNvidiaGlobalAttentionSupports(16, 1, 512));
  EXPECT_TRUE(LiteRtNvidiaGlobalAttentionSupports(16, 16, 512));
  EXPECT_TRUE(LiteRtNvidiaGlobalAttentionSupports(2048, 128, 512));
  EXPECT_TRUE(LiteRtNvidiaGlobalAttentionSupports(16384, 1024, 512));
  EXPECT_FALSE(LiteRtNvidiaGlobalAttentionSupports(16, 1, 256));
  EXPECT_FALSE(LiteRtNvidiaGlobalAttentionSupports(8, 1, 512));
  EXPECT_FALSE(LiteRtNvidiaGlobalAttentionSupports(24, 1, 512));
  EXPECT_FALSE(LiteRtNvidiaGlobalAttentionSupports(32, 3, 512));
  EXPECT_FALSE(LiteRtNvidiaGlobalAttentionSupports(0, 1, 512));
  EXPECT_FALSE(LiteRtNvidiaGlobalAttentionSupports(16, 0, 512));
  EXPECT_EQ(LiteRtNvidiaGlobalAttentionWorkspaceBytes(8, 1, 1024, 512), 0u);
  EXPECT_EQ(LiteRtNvidiaGlobalAttentionWorkspaceBytes(16, 1, 0, 512), 0u);
}

TEST(GlobalAttentionTest, RejectsInvalidArguments) {
  uint16_t sentinel = 0;
  bool mask = true;
  EXPECT_NE(LiteRtNvidiaLaunchGlobalAttention(
                nullptr, false, &sentinel, &sentinel, &mask, 1, 16, 64, 512,
                -1.0e4f, &sentinel, &sentinel, nullptr),
            cudaSuccess);
  EXPECT_NE(LiteRtNvidiaLaunchGlobalAttention(
                &sentinel, false, &sentinel, &sentinel, &mask, 1, 16, 64, 512,
                -1.0e4f, &sentinel, nullptr, nullptr),
            cudaSuccess);
  EXPECT_NE(LiteRtNvidiaLaunchGlobalAttention(
                &sentinel, false, &sentinel, &sentinel, &mask, 1, 20, 64, 512,
                -1.0e4f, &sentinel, &sentinel, nullptr),
            cudaSuccess);
  EXPECT_NE(LiteRtNvidiaLaunchGlobalAttention(
                &sentinel, false, &sentinel, &sentinel, &mask, 1, 16, 64, 256,
                -1.0e4f, &sentinel, &sentinel, nullptr),
            cudaSuccess);
  EXPECT_NE(LiteRtNvidiaLaunchGlobalAttention(
                &sentinel, false, &sentinel, &sentinel, &mask, 1, 16, 0, 512,
                -1.0e4f, &sentinel, &sentinel, nullptr),
            cudaSuccess);
}

TEST(GlobalAttentionTest, DecodeRowsMatchReference) {
  if (!HasComputeCapability80()) {
    GTEST_SKIP() << "Requires compute capability 8.0 or newer.";
  }
  // One query token (16 heads): a short cache with a partial last tile, a
  // cache walked by several blocks, holes in the mask, -inf and finite fills.
  RunCase({16, 1, 100, 60, 0, -1, -1.0e4f, false});
  RunCase({16, 1, 1000, 700, 7, -1, -INFINITY, false});
  RunCase({16, 1, 40000, 39999, 0, -1, -45824.0f, false});
  RunCase({16, 1, 40000, 12345, 5, -1, -45824.0f, true});
  RunCase({16, 16, 3000, 1000, 3, -1, -1.0e4f, false});
}

TEST(GlobalAttentionTest, PrefillRowsMatchReference) {
  if (!HasComputeCapability80()) {
    GTEST_SKIP() << "Requires compute capability 8.0 or newer.";
  }
  // 4, 8 and 16 tokens (streaming kernel, several row tiles) and 64 and 128
  // tokens (tiled kernel) at the start, the middle and the end of a cache.
  RunCase({64, 4, 3000, 0, 0, -1, -1.0e4f, false});
  RunCase({64, 4, 3000, 1500, 9, -1, -INFINITY, true});
  RunCase({128, 8, 2000, 0, 0, -1, -45824.0f, false});
  RunCase({256, 16, 2100, 1111, 0, -1, -45824.0f, false});
  RunCase({256, 16, 2100, 2084, 11, -1, -INFINITY, false});
  RunCase({1024, 64, 2100, 0, 0, -1, -45824.0f, false});
  RunCase({1024, 64, 2100, 2036, 11, -1, -INFINITY, false});
  RunCase({2048, 128, 5000, 3333, 0, -1, -45824.0f, true});
  RunCase({2048, 128, 5000, 4872, 0, -1, -45824.0f, false});
}

TEST(GlobalAttentionTest, RowsWithoutVisibleKeysAverageTheSequence) {
  if (!HasComputeCapability80()) {
    GTEST_SKIP() << "Requires compute capability 8.0 or newer.";
  }
  // With a finite fill a row that sees no key weighs every key equally; the
  // other rows of its block must not be disturbed.
  RunCase({16, 1, 700, 300, 0, 0, -1.0e4f, false});
  RunCase({64, 4, 900, 300, 0, 2, -1.0e4f, false});
  RunCase({256, 16, 900, 300, 0, 5, -1.0e4f, false});
  RunCase({1024, 64, 900, 300, 0, 5, -1.0e4f, false});
}

TEST(GlobalAttentionTest, TokenTilesSupportAFewTokensOfSeveralHeads) {
  EXPECT_TRUE(LiteRtNvidiaGlobalAttentionTokensSupports(32, 4, 4, 512));
  EXPECT_TRUE(LiteRtNvidiaGlobalAttentionTokensSupports(32, 4, 1, 512));
  EXPECT_TRUE(LiteRtNvidiaGlobalAttentionTokensSupports(64, 4, 4, 512));
  EXPECT_TRUE(LiteRtNvidiaGlobalAttentionTokensSupports(48, 12, 12, 512));
  EXPECT_TRUE(LiteRtNvidiaGlobalAttentionTokensSupports(16, 2, 2, 512));
  // Five tokens of eight heads: the fifth fills half a tile.
  EXPECT_FALSE(LiteRtNvidiaGlobalAttentionTokensSupports(40, 5, 5, 512));
  EXPECT_FALSE(LiteRtNvidiaGlobalAttentionTokensSupports(32, 4, 2, 512));
  EXPECT_FALSE(LiteRtNvidiaGlobalAttentionTokensSupports(32, 4, 4, 256));
  EXPECT_FALSE(LiteRtNvidiaGlobalAttentionTokensSupports(24, 4, 4, 512));
  EXPECT_FALSE(LiteRtNvidiaGlobalAttentionTokensSupports(128, 4, 4, 512));
  EXPECT_FALSE(LiteRtNvidiaGlobalAttentionTokensSupports(32, 0, 1, 512));
  EXPECT_EQ(
      LiteRtNvidiaGlobalAttentionTokensWorkspaceBytes(40, 5, 5, 1024, 512), 0u);
  EXPECT_EQ(LiteRtNvidiaGlobalAttentionTokensWorkspaceBytes(32, 4, 4, 0, 512),
            0u);
  uint16_t sentinel = 0;
  bool mask = true;
  EXPECT_NE(LiteRtNvidiaLaunchGlobalAttentionTokens(
                &sentinel, false, &sentinel, &sentinel, &mask, 5, 5, 40, 64,
                512, -1.0e4f, &sentinel, &sentinel, nullptr),
            cudaSuccess);
  EXPECT_NE(LiteRtNvidiaLaunchGlobalAttentionTokens(
                &sentinel, false, &sentinel, &sentinel, &mask, 4, 4, 32, 64,
                512, -1.0e4f, &sentinel, nullptr, nullptr),
            cudaSuccess);
}

TEST(GlobalAttentionTest, TokenTilesMatchTheirOwnLaunches) {
  if (!HasComputeCapability80()) {
    GTEST_SKIP() << "Requires compute capability 8.0 or newer.";
  }
  // Gemma 4 26B's verify step (8 query heads, 4 of its 5 tokens): a cache of
  // one key tile, of one tile per block, and one the blocks walk; the tokens
  // at the start and the end of the cache and across a 64-key tile boundary
  // (the two row tiles then end at different key tiles).
  RunTokenCase({8, 4, 60, 20, 0, -1, false, -1.0e4f, true});
  RunTokenCase({8, 4, 3000, 0, 0, -1, false, -1.0e4f, true});
  RunTokenCase({8, 4, 3000, 1500, 7, -1, false, -INFINITY, false});
  for (const int position : {8125, 8126, 8127, 8128, 9212}) {
    RunTokenCase({8, 4, 9216, position, 0, -1, false, -1.0e4f, true});
  }
  RunTokenCase({8, 4, 40000, 39996, 0, -1, false, -1.0e4f, true});
  RunTokenCase({8, 4, 40000, 21310, 5, -1, false, -45824.0f, false});
  // A token that sees nothing takes the whole sequence; its tile's other
  // token and the other tile are not disturbed.
  RunTokenCase({8, 4, 3000, 700, 0, 1, false, -1.0e4f, true});
  RunTokenCase({8, 4, 3000, 700, 0, 2, false, -1.0e4f, false});
  // One mask row for every token.
  RunTokenCase({8, 4, 3000, 1000, 3, -1, true, -1.0e4f, true});
  // Other head counts: a token per tile, an odd number of tiles (the last
  // block owns one), tiles of 4 and 8 tokens, and rows that are in order.
  RunTokenCase({16, 4, 3000, 2040, 0, -1, false, -1.0e4f, true});
  RunTokenCase({16, 3, 20000, 12345, 0, -1, false, -1.0e4f, false});
  RunTokenCase({4, 12, 3000, 250, 9, -1, false, -1.0e4f, true});
  RunTokenCase({2, 16, 3000, 2970, 0, -1, false, -INFINITY, false});
  RunTokenCase({1, 48, 3000, 100, 0, -1, false, -1.0e4f, true});
  // One tile: the 16-row launch itself.
  RunTokenCase({8, 2, 3000, 1500, 0, -1, false, -1.0e4f, true});
}

}  // namespace
