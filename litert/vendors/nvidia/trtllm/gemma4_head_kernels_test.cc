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
#include "litert/vendors/nvidia/bytecode.h"
#include "litert/vendors/nvidia/trtllm/decode_attention.h"
#include "litert/vendors/nvidia/trtllm/head_kernels.h"
#include "litert/vendors/nvidia/trtllm/int2_gemv.h"

namespace {

struct CudaDeleter {
  void operator()(void* pointer) const { cudaFree(pointer); }
};
using CudaAllocation = std::unique_ptr<void, CudaDeleter>;

CudaAllocation AllocateCuda(size_t bytes) {
  void* pointer = nullptr;
  const cudaError_t status = cudaMalloc(&pointer, bytes);
  if (status != cudaSuccess) {
    ADD_FAILURE() << "cudaMalloc(" << bytes
                  << ") failed: " << cudaGetErrorString(status);
  }
  return CudaAllocation(pointer);
}

uint16_t FloatToBf16Bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fff + ((bits >> 16) & 1);
  return static_cast<uint16_t>(bits >> 16);
}

float Bf16BitsToFloat(uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

int8_t DecodeInt2(const std::vector<uint8_t>& packed, size_t index) {
  const uint8_t bits = (packed[index / 4] >> (2 * (index % 4))) & 3;
  return static_cast<int8_t>(static_cast<int>(bits & 1) -
                             static_cast<int>(bits & 2));
}

std::vector<uint8_t> MakeWeights(int n, int k) {
  std::vector<uint8_t> weights(static_cast<size_t>(n) * k / 4, 0);
  for (int row = 0; row < n; ++row) {
    for (int column = 0; column < k; ++column) {
      const int8_t value =
          static_cast<int8_t>((7 * row + 3 * column + row / 4) & 3) - 2;
      const size_t index = static_cast<size_t>(row) * k + column;
      weights[index / 4] |= (static_cast<uint8_t>(value) & 3)
                            << (2 * (index % 4));
    }
  }
  return weights;
}

int CurrentComputeCapability() {
  int device = 0;
  cudaDeviceProp properties{};
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
    return 0;
  }
  return properties.major * 10 + properties.minor;
}

TEST(Gemma4HeadKernelsTest, MatchesCpuReference) {
  if (!litert::nvidia::IsInt2GemvComputeCapabilitySupported(
          CurrentComputeCapability())) {
    GTEST_SKIP() << "A native-W2-compatible CUDA GPU is unavailable";
  }

  constexpr float kSoftCap = 30.0f;
  for (const int k : {64, 1536}) {
    constexpr int kN = 128;
    std::vector<float> input(k);
    for (int i = 0; i < k; ++i) {
      input[i] = static_cast<float>((i * 17) % 101 - 50) / 37.0f;
    }
    const auto weights = MakeWeights(kN, k);
    std::vector<uint16_t> scales(kN);
    for (int row = 0; row < kN; ++row) {
      scales[row] =
          FloatToBf16Bits(0.0125f + 0.00025f * static_cast<float>(row % 29));
    }

    auto device_input = AllocateCuda(input.size() * sizeof(float));
    auto device_activation = AllocateCuda(input.size() * sizeof(uint16_t));
    auto device_weights = AllocateCuda(weights.size());
    auto device_scales = AllocateCuda(scales.size() * sizeof(uint16_t));
    auto device_bf16_logits = AllocateCuda(kN * sizeof(uint16_t));
    auto device_fp32_logits = AllocateCuda(kN * sizeof(float));
    ASSERT_TRUE(device_input);
    ASSERT_TRUE(device_activation);
    ASSERT_TRUE(device_weights);
    ASSERT_TRUE(device_scales);
    ASSERT_TRUE(device_bf16_logits);
    ASSERT_TRUE(device_fp32_logits);

    ASSERT_EQ(cudaMemcpy(device_input.get(), input.data(),
                         input.size() * sizeof(float), cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_weights.get(), weights.data(), weights.size(),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(
        cudaMemcpy(device_scales.get(), scales.data(),
                   scales.size() * sizeof(uint16_t), cudaMemcpyHostToDevice),
        cudaSuccess);

    ASSERT_EQ(LiteRtNvidiaLaunchF32ToBf16(
                  static_cast<const float*>(device_input.get()),
                  device_activation.get(), input.size(), /*stream=*/nullptr),
              cudaSuccess);
    ASSERT_EQ(LiteRtNvidiaLaunchBf16Int2PerChannelGemv(
                  device_activation.get(),
                  static_cast<const uint8_t*>(device_weights.get()),
                  device_scales.get(), device_bf16_logits.get(), k, kN,
                  /*stream=*/nullptr),
              cudaSuccess);
    ASSERT_EQ(LiteRtNvidiaLaunchBf16SoftCapToF32(
                  device_bf16_logits.get(),
                  static_cast<float*>(device_fp32_logits.get()), kN, kSoftCap,
                  /*stream=*/nullptr),
              cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<uint16_t> actual_bf16(kN);
    std::vector<float> actual(kN);
    ASSERT_EQ(cudaMemcpy(actual_bf16.data(), device_bf16_logits.get(),
                         actual_bf16.size() * sizeof(uint16_t),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(actual.data(), device_fp32_logits.get(),
                         actual.size() * sizeof(float), cudaMemcpyDeviceToHost),
              cudaSuccess);

    for (int row = 0; row < kN; ++row) {
      float accumulator = 0.0f;
      for (int column = 0; column < k; ++column) {
        accumulator =
            std::fma(Bf16BitsToFloat(FloatToBf16Bits(input[column])),
                     static_cast<float>(DecodeInt2(
                         weights, static_cast<size_t>(row) * k + column)),
                     accumulator);
      }
      const uint16_t expected_bf16 =
          FloatToBf16Bits(accumulator * Bf16BitsToFloat(scales[row]));
      EXPECT_EQ(actual_bf16[row], expected_bf16) << "K=" << k << " row=" << row;
      const float rounded_logit = Bf16BitsToFloat(expected_bf16);
      const float expected = std::tanh(rounded_logit / kSoftCap) * kSoftCap;
      EXPECT_NEAR(actual[row], expected, 5e-5f + 2e-5f * std::abs(expected))
          << "K=" << k << " row=" << row;
    }
  }
}

// Decodes a signed INT4 value of a row-major packed array.
int8_t DecodeInt4(const std::vector<uint8_t>& packed, size_t index) {
  const uint8_t bits = (packed[index / 2] >> (4 * (index % 2))) & 15;
  return static_cast<int8_t>(static_cast<int>(bits ^ 8) - 8);
}

std::vector<uint8_t> MakeInt4Weights(int n, int k) {
  std::vector<uint8_t> weights(static_cast<size_t>(n) * k / 2, 0);
  for (int row = 0; row < n; ++row) {
    for (int column = 0; column < k; ++column) {
      const int8_t value =
          static_cast<int8_t>((5 * row + 3 * column + row / 3) & 15) - 8;
      const size_t index = static_cast<size_t>(row) * k + column;
      weights[index / 2] |= (static_cast<uint8_t>(value) & 15)
                            << (4 * (index % 2));
    }
  }
  return weights;
}

TEST(Gemma4HeadKernelsTest, BatchedRowsMatchCpuReference) {
  if (!litert::nvidia::IsInt2GemvComputeCapabilitySupported(
          CurrentComputeCapability())) {
    GTEST_SKIP() << "A native-W2-compatible CUDA GPU is unavailable";
  }
  // Every row count the batched GEMV takes, in INT2 and INT4, over a short
  // input, Gemma 4 E2B's hidden size and an input long enough for the
  // 256-thread blocks; 136 channels end in a partial block of 8.
  constexpr int kN = 136;
  for (const int bit_width : {2, 4}) {
    for (const int k : {64, 1536, 4096}) {
      const std::vector<uint8_t> weights =
          bit_width == 2 ? MakeWeights(kN, k) : MakeInt4Weights(kN, k);
      std::vector<uint16_t> scales(kN);
      for (int row = 0; row < kN; ++row) {
        scales[row] =
            FloatToBf16Bits(0.0125f + 0.00025f * static_cast<float>(row % 29));
      }
      for (int rows = 1; rows <= kLiteRtNvidiaSubbyteGemvMaxRows; ++rows) {
        SCOPED_TRACE(::testing::Message()
                     << "bits=" << bit_width << " K=" << k << " rows=" << rows);
        std::vector<uint16_t> activation(static_cast<size_t>(rows) * k);
        for (int r = 0; r < rows; ++r) {
          for (int i = 0; i < k; ++i) {
            activation[static_cast<size_t>(r) * k + i] = FloatToBf16Bits(
                static_cast<float>((i * 17 + r * 29) % 101 - 50) / 37.0f);
          }
        }
        auto device_activation =
            AllocateCuda(activation.size() * sizeof(uint16_t));
        auto device_weights = AllocateCuda(weights.size());
        auto device_scales = AllocateCuda(scales.size() * sizeof(uint16_t));
        auto device_output =
            AllocateCuda(static_cast<size_t>(rows) * kN * sizeof(uint16_t));
        ASSERT_TRUE(device_activation && device_weights && device_scales &&
                    device_output);
        ASSERT_EQ(cudaMemcpy(device_activation.get(), activation.data(),
                             activation.size() * sizeof(uint16_t),
                             cudaMemcpyHostToDevice),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpy(device_weights.get(), weights.data(),
                             weights.size(), cudaMemcpyHostToDevice),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpy(device_scales.get(), scales.data(),
                             scales.size() * sizeof(uint16_t),
                             cudaMemcpyHostToDevice),
                  cudaSuccess);
        ASSERT_EQ(LiteRtNvidiaLaunchBf16SubbytePerChannelGemvRows(
                      device_activation.get(),
                      static_cast<const uint8_t*>(device_weights.get()),
                      device_scales.get(), device_output.get(), bit_width, rows,
                      k, kN, /*stream=*/nullptr),
                  cudaSuccess);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        std::vector<uint16_t> actual(static_cast<size_t>(rows) * kN);
        ASSERT_EQ(cudaMemcpy(actual.data(), device_output.get(),
                             actual.size() * sizeof(uint16_t),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        for (int r = 0; r < rows; ++r) {
          for (int row = 0; row < kN; ++row) {
            float accumulator = 0.0f;
            for (int column = 0; column < k; ++column) {
              const size_t index = static_cast<size_t>(row) * k + column;
              const float weight = static_cast<float>(
                  bit_width == 2 ? DecodeInt2(weights, index)
                                 : DecodeInt4(weights, index));
              accumulator =
                  std::fma(Bf16BitsToFloat(
                               activation[static_cast<size_t>(r) * k + column]),
                           weight, accumulator);
            }
            const float expected = accumulator * Bf16BitsToFloat(scales[row]);
            const float got =
                Bf16BitsToFloat(actual[static_cast<size_t>(r) * kN + row]);
            // The device sums in another order; one BF16 ulp of slack.
            EXPECT_NEAR(got, expected, 1e-2f * std::fabs(expected) + 1e-3f)
                << "r=" << r << " row=" << row;
          }
        }
      }
    }
  }
}

// Decode attention over INT8 per-tensor caches (Gemma 4 E4B: two KV heads,
// four query rows each, depth 256), the value cache as rows [S, D] and as the
// model's columns [D, S], against an FP32 reference with the same mask.
TEST(Gemma4HeadKernelsTest, Int8CacheDecodeAttentionMatchesReference) {
  if (!litert::nvidia::IsInt2GemvComputeCapabilitySupported(
          CurrentComputeCapability())) {
    GTEST_SKIP() << "A CUDA GPU with compute capability 8.0 is unavailable";
  }
  constexpr int kHeads = 2, kRows = 4, kDepth = 256, kSeq = 200;
  constexpr float kFill = -1.0e9f;
  ASSERT_TRUE(LiteRtNvidiaDecodeAttentionInt8Supports(kDepth));
  const LiteRtNvidiaInt8CacheParams params_rows = {0.0125f, -3, 0.02f, 5,
                                                   /*value_columns=*/false};
  uint32_t state = 4242u;
  const auto next = [&]() {
    state = state * 1664525u + 1013904223u;
    return state >> 8;
  };
  std::vector<uint16_t> q(static_cast<size_t>(kHeads) * kRows * kDepth);
  for (auto& value : q) {
    value = FloatToBf16Bits(static_cast<float>(next() % 2048) / 1024.0f - 1.0f);
  }
  // k and v_rows as [H, S, D]; v_columns holds the same values as [H, D, S].
  std::vector<int8_t> k(static_cast<size_t>(kHeads) * kSeq * kDepth);
  std::vector<int8_t> v_rows(k.size());
  std::vector<int8_t> v_columns(k.size());
  for (auto& value : k) value = static_cast<int8_t>(next() % 256) - 128;
  for (auto& value : v_rows) value = static_cast<int8_t>(next() % 256) - 128;
  for (int h = 0; h < kHeads; ++h) {
    for (int j = 0; j < kSeq; ++j) {
      for (int d = 0; d < kDepth; ++d) {
        v_columns[(static_cast<size_t>(h) * kDepth + d) * kSeq + j] =
            v_rows[(static_cast<size_t>(h) * kSeq + j) * kDepth + d];
      }
    }
  }
  // Row r sees the first 150 + 10 r keys.
  std::vector<uint8_t> mask(static_cast<size_t>(kRows) * kSeq, 0);
  for (int r = 0; r < kRows; ++r) {
    for (int j = 0; j < 150 + 10 * r; ++j) mask[r * kSeq + j] = 1;
  }
  const size_t workspace_bytes =
      LiteRtNvidiaDecodeAttentionWorkspaceBytes(kHeads, kRows, kSeq, kDepth);
  auto device_q = AllocateCuda(q.size() * sizeof(uint16_t));
  auto device_k = AllocateCuda(k.size());
  auto device_v_rows = AllocateCuda(v_rows.size());
  auto device_v_columns = AllocateCuda(v_columns.size());
  auto device_mask = AllocateCuda(mask.size());
  auto device_out = AllocateCuda(q.size() * sizeof(uint16_t));
  auto device_workspace = AllocateCuda(workspace_bytes);
  ASSERT_TRUE(device_q && device_k && device_v_rows && device_v_columns &&
              device_mask && device_out && device_workspace);
  ASSERT_EQ(cudaMemcpy(device_q.get(), q.data(), q.size() * sizeof(uint16_t),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(
      cudaMemcpy(device_k.get(), k.data(), k.size(), cudaMemcpyHostToDevice),
      cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_v_rows.get(), v_rows.data(), v_rows.size(),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_v_columns.get(), v_columns.data(),
                       v_columns.size(), cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_mask.get(), mask.data(), mask.size(),
                       cudaMemcpyHostToDevice),
            cudaSuccess);

  // FP32 reference.
  std::vector<float> expected(q.size());
  for (int h = 0; h < kHeads; ++h) {
    for (int r = 0; r < kRows; ++r) {
      std::vector<float> scores(kSeq);
      float max_score = -INFINITY;
      for (int j = 0; j < kSeq; ++j) {
        float dot = 0.0f;
        for (int d = 0; d < kDepth; ++d) {
          const float key =
              (static_cast<float>(
                   k[(static_cast<size_t>(h) * kSeq + j) * kDepth + d]) -
               params_rows.k_zero_point) *
              params_rows.k_scale;
          dot += Bf16BitsToFloat(
                     q[(static_cast<size_t>(h) * kRows + r) * kDepth + d]) *
                 key;
        }
        scores[j] = mask[r * kSeq + j] ? dot : kFill;
        max_score = std::max(max_score, scores[j]);
      }
      float sum = 0.0f;
      for (int j = 0; j < kSeq; ++j) {
        scores[j] = std::exp(scores[j] - max_score);
        sum += scores[j];
      }
      for (int d = 0; d < kDepth; ++d) {
        float acc = 0.0f;
        for (int j = 0; j < kSeq; ++j) {
          const float value =
              (static_cast<float>(
                   v_rows[(static_cast<size_t>(h) * kSeq + j) * kDepth + d]) -
               params_rows.v_zero_point) *
              params_rows.v_scale;
          acc += scores[j] / sum * value;
        }
        expected[(static_cast<size_t>(h) * kRows + r) * kDepth + d] = acc;
      }
    }
  }

  for (const bool columns : {false, true}) {
    SCOPED_TRACE(::testing::Message() << "value_columns=" << columns);
    LiteRtNvidiaInt8CacheParams params = params_rows;
    params.value_columns = columns;
    ASSERT_EQ(cudaMemset(device_out.get(), 0, q.size() * sizeof(uint16_t)),
              cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(LiteRtNvidiaLaunchDecodeAttentionInt8(
                  device_q.get(), /*q_bf16=*/true, device_k.get(),
                  columns ? device_v_columns.get() : device_v_rows.get(),
                  &params, static_cast<const bool*>(device_mask.get()),
                  /*mask_rows=*/kRows, kHeads, kRows, kSeq, kDepth, kFill,
                  device_out.get(), device_workspace.get(),
                  /*stream=*/nullptr),
              cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<uint16_t> actual(q.size());
    ASSERT_EQ(
        cudaMemcpy(actual.data(), device_out.get(),
                   actual.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
        cudaSuccess);
    for (size_t i = 0; i < actual.size(); ++i) {
      EXPECT_NEAR(Bf16BitsToFloat(actual[i]), expected[i],
                  1e-2f * std::fabs(expected[i]) + 2e-2f)
          << "i=" << i;
    }
  }
  // Unsupported depth and a missing scale are rejected before any launch.
  LiteRtNvidiaInt8CacheParams params = params_rows;
  EXPECT_EQ(LiteRtNvidiaLaunchDecodeAttentionInt8(
                device_q.get(), true, device_k.get(), device_v_rows.get(),
                &params, nullptr, 1, kHeads, kRows, kSeq, /*depth=*/64, kFill,
                device_out.get(), device_workspace.get(), nullptr),
            cudaErrorInvalidValue);
  params.k_scale = NAN;
  EXPECT_EQ(LiteRtNvidiaLaunchDecodeAttentionInt8(
                device_q.get(), true, device_k.get(), device_v_rows.get(),
                &params, nullptr, 1, kHeads, kRows, kSeq, kDepth, kFill,
                device_out.get(), device_workspace.get(), nullptr),
            cudaErrorInvalidValue);
}

// Gemma 4 E-model geometries with sliding-window masks: a local layer (two
// heads, four rows, depth 256) and E2B's global layers (one head, eight rows,
// depth 512), column value caches. Most chunks are hidden, and the kernel
// must still match the reference, including rows whose window starts and
// ends inside a chunk.
TEST(Gemma4HeadKernelsTest, Int8CacheDecodeAttentionWindowedMasks) {
  if (!litert::nvidia::IsInt2GemvComputeCapabilitySupported(
          CurrentComputeCapability())) {
    GTEST_SKIP() << "A CUDA GPU with compute capability 8.0 is unavailable";
  }
  // With `read` > 0 the kernel is launched through the window entry point:
  // `read` keys from `start` of the full caches (start a multiple of 64 so
  // the visible keys [window_begin, window_begin + window) fit inside).
  struct Case {
    int heads, rows, depth, seq, window_begin, window, start, read;
  };
  const Case cases[] = {{2, 4, 256, 1024, 300, 512, 0, 0},
                        {1, 8, 512, 1024, 70, 512, 0, 0},
                        {1, 8, 512, 200, 0, 200, 0, 0},
                        {2, 3, 128, 320, 100, 64, 0, 0},
                        {2, 4, 256, 2048, 1200, 512, 1152, 576},
                        {1, 8, 512, 1024, 300, 512, 256, 576}};
  constexpr float kFill = -1.0e9f;
  const LiteRtNvidiaInt8CacheParams params = {0.0125f, -3, 0.02f, 5,
                                              /*value_columns=*/true};
  uint32_t state = 777u;
  const auto next = [&]() {
    state = state * 1664525u + 1013904223u;
    return state >> 8;
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(::testing::Message()
                 << "heads=" << c.heads << " rows=" << c.rows
                 << " depth=" << c.depth << " seq=" << c.seq);
    ASSERT_TRUE(LiteRtNvidiaDecodeAttentionInt8Supports(c.depth));
    const size_t q_count = static_cast<size_t>(c.heads) * c.rows * c.depth;
    const size_t cache_count = static_cast<size_t>(c.heads) * c.seq * c.depth;
    std::vector<uint16_t> q(q_count);
    for (auto& value : q) {
      value =
          FloatToBf16Bits(static_cast<float>(next() % 2048) / 1024.0f - 1.0f);
    }
    std::vector<int8_t> k(cache_count), v_rows(cache_count),
        v_columns(cache_count);
    for (auto& value : k) value = static_cast<int8_t>(next() % 256) - 128;
    for (auto& value : v_rows) value = static_cast<int8_t>(next() % 256) - 128;
    for (int h = 0; h < c.heads; ++h) {
      for (int j = 0; j < c.seq; ++j) {
        for (int d = 0; d < c.depth; ++d) {
          v_columns[(static_cast<size_t>(h) * c.depth + d) * c.seq + j] =
              v_rows[(static_cast<size_t>(h) * c.seq + j) * c.depth + d];
        }
      }
    }
    // Row r sees keys [window_begin + r, window_begin + r + window).
    std::vector<uint8_t> mask(static_cast<size_t>(c.rows) * c.seq, 0);
    for (int r = 0; r < c.rows; ++r) {
      for (int j = c.window_begin + r;
           j < std::min(c.seq, c.window_begin + r + c.window); ++j) {
        mask[static_cast<size_t>(r) * c.seq + j] = 1;
      }
    }
    std::vector<float> expected(q_count);
    for (int h = 0; h < c.heads; ++h) {
      for (int r = 0; r < c.rows; ++r) {
        std::vector<float> scores(c.seq);
        float max_score = -INFINITY;
        for (int j = 0; j < c.seq; ++j) {
          float dot = 0.0f;
          for (int d = 0; d < c.depth; ++d) {
            const float key =
                (static_cast<float>(
                     k[(static_cast<size_t>(h) * c.seq + j) * c.depth + d]) -
                 params.k_zero_point) *
                params.k_scale;
            dot += Bf16BitsToFloat(
                       q[(static_cast<size_t>(h) * c.rows + r) * c.depth + d]) *
                   key;
          }
          scores[j] = mask[static_cast<size_t>(r) * c.seq + j] ? dot : kFill;
          max_score = std::max(max_score, scores[j]);
        }
        float sum = 0.0f;
        for (int j = 0; j < c.seq; ++j) {
          scores[j] = std::exp(scores[j] - max_score);
          sum += scores[j];
        }
        for (int d = 0; d < c.depth; ++d) {
          float acc = 0.0f;
          for (int j = 0; j < c.seq; ++j) {
            const float value =
                (static_cast<float>(
                     v_rows[(static_cast<size_t>(h) * c.seq + j) * c.depth +
                            d]) -
                 params.v_zero_point) *
                params.v_scale;
            acc += scores[j] / sum * value;
          }
          expected[(static_cast<size_t>(h) * c.rows + r) * c.depth + d] = acc;
        }
      }
    }
    const size_t workspace_bytes = LiteRtNvidiaDecodeAttentionWorkspaceBytes(
        c.heads, c.rows, c.read == 0 ? c.seq : c.read, c.depth);
    auto device_q = AllocateCuda(q.size() * sizeof(uint16_t));
    auto device_k = AllocateCuda(k.size());
    auto device_v = AllocateCuda(v_columns.size());
    auto device_mask = AllocateCuda(mask.size());
    auto device_out = AllocateCuda(q.size() * sizeof(uint16_t));
    auto device_workspace = AllocateCuda(workspace_bytes);
    ASSERT_TRUE(device_q && device_k && device_v && device_mask && device_out &&
                device_workspace);
    ASSERT_EQ(cudaMemcpy(device_q.get(), q.data(), q.size() * sizeof(uint16_t),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(
        cudaMemcpy(device_k.get(), k.data(), k.size(), cudaMemcpyHostToDevice),
        cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_v.get(), v_columns.data(), v_columns.size(),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_mask.get(), mask.data(), mask.size(),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    if (c.read == 0) {
      ASSERT_EQ(
          LiteRtNvidiaLaunchDecodeAttentionInt8(
              device_q.get(), /*q_bf16=*/true, device_k.get(), device_v.get(),
              &params, static_cast<const bool*>(device_mask.get()),
              /*mask_rows=*/c.rows, c.heads, c.rows, c.seq, c.depth, kFill,
              device_out.get(), device_workspace.get(),
              /*stream=*/nullptr),
          cudaSuccess);
    } else {
      auto device_start = AllocateCuda(sizeof(int32_t));
      ASSERT_TRUE(device_start);
      ASSERT_EQ(cudaMemcpy(device_start.get(), &c.start, sizeof(int32_t),
                           cudaMemcpyHostToDevice),
                cudaSuccess);
      ASSERT_EQ(
          LiteRtNvidiaLaunchDecodeAttentionInt8Window(
              device_q.get(), /*q_bf16=*/true, device_k.get(), device_v.get(),
              &params, static_cast<const bool*>(device_mask.get()),
              /*mask_rows=*/c.rows, c.heads, c.rows, c.seq, c.read,
              static_cast<const int32_t*>(device_start.get()), c.depth, kFill,
              device_out.get(), device_workspace.get(),
              /*stream=*/nullptr),
          cudaSuccess);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<uint16_t> actual(q.size());
    ASSERT_EQ(
        cudaMemcpy(actual.data(), device_out.get(),
                   actual.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
        cudaSuccess);
    for (size_t i = 0; i < actual.size(); ++i) {
      ASSERT_NEAR(Bf16BitsToFloat(actual[i]), expected[i],
                  1e-2f * std::fabs(expected[i]) + 2e-2f)
          << "i=" << i;
    }
  }
}

TEST(Gemma4HeadKernelsTest, BatchedRowsRejectInvalidArguments) {
  if (!litert::nvidia::IsInt2GemvComputeCapabilitySupported(
          CurrentComputeCapability())) {
    GTEST_SKIP() << "A native-W2-compatible CUDA GPU is unavailable";
  }
  auto buffer = AllocateCuda(4096);
  ASSERT_TRUE(buffer);
  const uint8_t* bytes = static_cast<const uint8_t*>(buffer.get());
  EXPECT_EQ(LiteRtNvidiaLaunchBf16SubbytePerChannelGemvRows(
                buffer.get(), bytes, buffer.get(), buffer.get(), 4,
                /*rows=*/0, 64, 8, nullptr),
            cudaErrorInvalidValue);
  EXPECT_EQ(LiteRtNvidiaLaunchBf16SubbytePerChannelGemvRows(
                buffer.get(), bytes, buffer.get(), buffer.get(), 4,
                kLiteRtNvidiaSubbyteGemvMaxRows + 1, 64, 8, nullptr),
            cudaErrorInvalidValue);
  EXPECT_EQ(LiteRtNvidiaLaunchBf16SubbytePerChannelGemvRows(
                buffer.get(), bytes, buffer.get(), buffer.get(), 3,
                /*rows=*/2, 64, 8, nullptr),
            cudaErrorInvalidValue);
}

TEST(Gemma4HeadKernelsTest, RejectsInvalidArguments) {
  EXPECT_EQ(LiteRtNvidiaLaunchF32ToBf16(
                /*input=*/nullptr, /*output=*/nullptr, /*count=*/1,
                /*stream=*/nullptr),
            cudaErrorInvalidValue);
  EXPECT_EQ(LiteRtNvidiaLaunchBf16Int2PerChannelGemv(
                /*activation=*/nullptr, /*packed_weights=*/nullptr,
                /*scales=*/nullptr, /*output=*/nullptr,
                /*input_size=*/15, /*output_size=*/1, /*stream=*/nullptr),
            cudaErrorInvalidValue);
  EXPECT_EQ(LiteRtNvidiaLaunchBf16SoftCapToF32(
                /*input=*/nullptr, /*output=*/nullptr, /*count=*/1,
                /*soft_cap=*/30.0f, /*stream=*/nullptr),
            cudaErrorInvalidValue);
}

}  // namespace
