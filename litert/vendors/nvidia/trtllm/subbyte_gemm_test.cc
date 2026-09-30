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

#include "litert/vendors/nvidia/trtllm/subbyte_gemm.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
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
  if (device != nullptr) {
    EXPECT_EQ(cudaMemcpy(device.get(), host.data(), host.size() * sizeof(T),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
  }
  return device;
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

double Gelu(double x, int gate) {
  if (gate == kLiteRtNvidiaGemmGateGeluTanh) {
    return 0.5 * x *
           (1.0 + std::tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
  }
  return 0.5 * x * (1.0 + std::erf(x * 0.7071067811865476));
}

enum class Rows {
  kUniform,   // every activation in [-scale, scale)
  kOutliers,  // rows of different magnitude, a few activations 40 times larger
  kSparse,    // mostly zeros, some rows all zeros
};

struct Case {
  LiteRtNvidiaGemmShape shape;
  Rows rows;
  float scale;
};

void RunCase(const Case& test_case) {
  const auto& shape = test_case.shape;
  SCOPED_TRACE(::testing::Message()
               << "rows=" << shape.rows << " input_size=" << shape.input_size
               << " output_size=" << shape.output_size << " gate=" << shape.gate
               << " kind=" << static_cast<int>(test_case.rows)
               << " scale=" << test_case.scale);
  ASSERT_TRUE(LiteRtNvidiaSubbyteGemmSupports(&shape));
  const int input = shape.input_size;
  const int output = shape.output_size;
  const int channels = shape.gate != 0 ? 2 * output : output;
  uint32_t state = 2463534242u + shape.rows + input + output;
  const auto next = [&]() -> float {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return static_cast<float>(state >> 8) / 8388608.0f - 1.0f;  // [-1, 1)
  };
  std::vector<uint16_t> activation(static_cast<size_t>(shape.rows) * input);
  for (size_t i = 0; i < activation.size(); ++i) {
    const size_t row = i / input;
    float value = next() * test_case.scale;
    if (test_case.rows == Rows::kOutliers) {
      value *= 1.0f + static_cast<float>(row % 7) * 3.0f;
      if (i % 997 == 0) value *= 40.0f;
    } else if (test_case.rows == Rows::kSparse) {
      if (row % 5 == 3 || i % 11 != 0) value = 0.0f;
    }
    activation[i] = FloatToBf16Bits(value);
  }
  std::vector<uint8_t> weights(static_cast<size_t>(channels) * input / 2);
  for (auto& byte : weights) {
    state = state * 1664525u + 1013904223u;
    byte = static_cast<uint8_t>(state >> 24);
  }
  std::vector<uint16_t> scales(channels);
  for (auto& scale : scales) {
    scale = FloatToBf16Bits(0.002f + 0.004f * std::fabs(next()));
  }

  std::vector<uint8_t> tiled(LiteRtNvidiaSubbyteGemmTiledWeightBytes(&shape));
  ASSERT_GE(tiled.size(), weights.size());
  ASSERT_TRUE(
      LiteRtNvidiaSubbyteGemmTileWeights(&shape, weights.data(), tiled.data()));

  const size_t workspace_bytes = LiteRtNvidiaSubbyteGemmWorkspaceBytes(&shape);
  ASSERT_GT(workspace_bytes, 0u);
  auto activation_device = Upload(activation);
  auto weights_device = Upload(tiled);
  auto scales_device = Upload(scales);
  const size_t output_bytes =
      static_cast<size_t>(shape.rows) * output * sizeof(uint16_t);
  auto output_device = AllocateCuda(output_bytes);
  auto workspace = AllocateCuda(workspace_bytes);
  ASSERT_NE(output_device, nullptr);
  ASSERT_NE(workspace, nullptr);
  ASSERT_EQ(cudaMemset(output_device.get(), 0xff, output_bytes), cudaSuccess);
  ASSERT_EQ(LiteRtNvidiaLaunchBf16Int4Gemm(
                &shape, activation_device.get(),
                static_cast<const uint8_t*>(weights_device.get()),
                scales_device.get(), output_device.get(), workspace.get(),
                /*stream=*/nullptr),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<uint16_t> actual(static_cast<size_t>(shape.rows) * output);
  ASSERT_EQ(cudaMemcpy(actual.data(), output_device.get(), output_bytes,
                       cudaMemcpyDeviceToHost),
            cudaSuccess);

  // The first and last rows of the first and last row tiles and a sample of
  // the others.
  for (int row = 0; row < shape.rows; ++row) {
    if (row != 0 && row != 127 && row != shape.rows - 128 &&
        row != shape.rows - 1 && row % 89 != 0) {
      continue;
    }
    const auto dot = [&](int channel) {
      const uint8_t* packed =
          weights.data() + static_cast<size_t>(channel) * input / 2;
      double sum = 0.0;
      for (int k = 0; k < input; ++k) {
        const int nibble = (packed[k / 2] >> ((k & 1) * 4)) & 15;
        sum += static_cast<double>(Bf16BitsToFloat(
                   activation[static_cast<size_t>(row) * input + k])) *
               ((nibble ^ 8) - 8);
      }
      return sum * Bf16BitsToFloat(scales[channel]);
    };
    // The error of a row against the rounding error BF16 results have anyway:
    // the FP16 sums over 128 input dims add less than that.
    double error = 0.0;
    double rounding = 0.0;
    double norm = 0.0;
    for (int n = 0; n < output; ++n) {
      const double expected =
          shape.gate != 0 ? Gelu(dot(n), shape.gate) * dot(output + n) : dot(n);
      const double value =
          Bf16BitsToFloat(actual[static_cast<size_t>(row) * output + n]);
      ASSERT_TRUE(std::isfinite(value)) << "row=" << row << " channel=" << n;
      const double rounded =
          Bf16BitsToFloat(FloatToBf16Bits(static_cast<float>(expected)));
      error += (value - expected) * (value - expected);
      rounding += (rounded - expected) * (rounded - expected);
      norm += expected * expected;
    }
    if (norm == 0.0) {
      EXPECT_EQ(error, 0.0) << "row=" << row;
      continue;
    }
    EXPECT_LE(std::sqrt(error),
              1.5 * std::sqrt(rounding) + 1e-4 * std::sqrt(norm))
        << "row=" << row << " relative error " << std::sqrt(error / norm)
        << ", of rounding " << std::sqrt(rounding / norm);
  }
}

TEST(SubbyteGemmTest, SupportsRowsAndInputsInMultiplesOf128) {
  const auto supports = [](LiteRtNvidiaGemmShape shape) {
    return LiteRtNvidiaSubbyteGemmSupports(&shape);
  };
  EXPECT_TRUE(supports({1024, 3840, 15360, kLiteRtNvidiaGemmGateNone}));
  EXPECT_TRUE(supports({128, 15360, 3840, kLiteRtNvidiaGemmGateNone}));
  EXPECT_TRUE(supports({1024, 1536, 6144, kLiteRtNvidiaGemmGateGeluTanh}));
  EXPECT_TRUE(supports({128, 256, 2, kLiteRtNvidiaGemmGateGeluErf}));
  EXPECT_FALSE(supports({1, 3840, 15360, kLiteRtNvidiaGemmGateNone}));
  EXPECT_FALSE(supports({100, 3840, 15360, kLiteRtNvidiaGemmGateNone}));
  EXPECT_FALSE(supports({128, 3872, 15360, kLiteRtNvidiaGemmGateNone}));
  EXPECT_FALSE(supports({128, 3840, 15361, kLiteRtNvidiaGemmGateNone}));
  EXPECT_FALSE(supports({128, 3840, 0, kLiteRtNvidiaGemmGateNone}));
  EXPECT_FALSE(supports({128, 3840, 15360, 3}));
  EXPECT_FALSE(LiteRtNvidiaSubbyteGemmSupports(nullptr));
  LiteRtNvidiaGemmShape unsupported = {100, 3840, 15360, 0};
  EXPECT_EQ(LiteRtNvidiaSubbyteGemmWorkspaceBytes(&unsupported), 0u);
  EXPECT_EQ(LiteRtNvidiaSubbyteGemmTiledWeightBytes(&unsupported), 0u);
  EXPECT_EQ(LiteRtNvidiaSubbyteGemmBlocks(&unsupported), 0);
}

TEST(SubbyteGemmTest, CountsBlocks) {
  LiteRtNvidiaGemmShape shape = {1024, 3840, 3840, kLiteRtNvidiaGemmGateNone};
  EXPECT_EQ(LiteRtNvidiaSubbyteGemmBlocks(&shape), 8 * 30);
  shape = {128, 3840, 130, kLiteRtNvidiaGemmGateNone};
  EXPECT_EQ(LiteRtNvidiaSubbyteGemmBlocks(&shape), 2);
  shape = {256, 3840, 15360, kLiteRtNvidiaGemmGateGeluTanh};
  EXPECT_EQ(LiteRtNvidiaSubbyteGemmBlocks(&shape), 2 * 240);
}

TEST(SubbyteGemmTest, LargeProductsFillTheDevice) {
  int device = 0;
  int multiprocessors = 0;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  ASSERT_EQ(cudaDeviceGetAttribute(&multiprocessors,
                                   cudaDevAttrMultiProcessorCount, device),
            cudaSuccess);
  ASSERT_GT(multiprocessors, 0);
  // Three blocks per two multiprocessors, in column tiles of one row tile.
  const int tiles = (multiprocessors * 3 + 1) / 2;
  LiteRtNvidiaGemmShape shape = {128, 256, tiles * 128,
                                 kLiteRtNvidiaGemmGateNone};
  EXPECT_TRUE(LiteRtNvidiaSubbyteGemmFillsDevice(&shape));
  shape.output_size -= 128;
  EXPECT_FALSE(LiteRtNvidiaSubbyteGemmFillsDevice(&shape));
  shape.rows = 256;
  EXPECT_TRUE(LiteRtNvidiaSubbyteGemmFillsDevice(&shape));
  shape.rows = 100;
  EXPECT_FALSE(LiteRtNvidiaSubbyteGemmFillsDevice(&shape));
  EXPECT_FALSE(LiteRtNvidiaSubbyteGemmFillsDevice(nullptr));
}

TEST(SubbyteGemmTest, TilesWeightsInTheOrderBlocksReadThem) {
  constexpr int kInput = 256;
  constexpr int kRowBytes = kInput / 2;
  const auto fill = [](std::vector<uint8_t>& weights) {
    for (size_t i = 0; i < weights.size(); ++i) {
      weights[i] = static_cast<uint8_t>(i * 131 + (i >> 8) * 17 + 1);
    }
  };
  {
    // Two column tiles, the second of two channels and 126 rows of zeros.
    const LiteRtNvidiaGemmShape shape = {128, kInput, 130,
                                         kLiteRtNvidiaGemmGateNone};
    std::vector<uint8_t> weights(130 * kRowBytes);
    fill(weights);
    std::vector<uint8_t> tiled(LiteRtNvidiaSubbyteGemmTiledWeightBytes(&shape),
                               0xa5);
    ASSERT_EQ(tiled.size(), 2u * 4 * 128 * 32);
    ASSERT_TRUE(LiteRtNvidiaSubbyteGemmTileWeights(&shape, weights.data(),
                                                   tiled.data()));
    for (int tile = 0; tile < 2; ++tile) {
      for (int chunk = 0; chunk < 4; ++chunk) {
        for (int row = 0; row < 128; ++row) {
          for (int i = 0; i < 32; ++i) {
            const int channel = tile * 128 + row;
            const uint8_t expected =
                channel < 130 ? weights[channel * kRowBytes + chunk * 32 + i]
                              : 0;
            ASSERT_EQ(tiled[((tile * 4 + chunk) * 128 + row) * 32 + i],
                      expected)
                << "tile=" << tile << " chunk=" << chunk << " row=" << row
                << " i=" << i;
          }
        }
      }
    }
  }
  {
    // With a gate, rows [0, 64) of a tile are channels of the gate projection
    // and rows [64, 128) the same channels of the up projection.
    const LiteRtNvidiaGemmShape shape = {128, kInput, 66,
                                         kLiteRtNvidiaGemmGateGeluTanh};
    std::vector<uint8_t> weights(2 * 66 * kRowBytes);
    fill(weights);
    std::vector<uint8_t> tiled(LiteRtNvidiaSubbyteGemmTiledWeightBytes(&shape),
                               0xa5);
    ASSERT_EQ(tiled.size(), 2u * 4 * 128 * 32);
    ASSERT_TRUE(LiteRtNvidiaSubbyteGemmTileWeights(&shape, weights.data(),
                                                   tiled.data()));
    for (int tile = 0; tile < 2; ++tile) {
      for (int chunk = 0; chunk < 4; ++chunk) {
        for (int row = 0; row < 128; ++row) {
          for (int i = 0; i < 32; ++i) {
            const int channel = tile * 64 + row % 64;
            const int projection = row / 64;
            const uint8_t expected =
                channel < 66 ? weights[(projection * 66 + channel) * kRowBytes +
                                       chunk * 32 + i]
                             : 0;
            ASSERT_EQ(tiled[((tile * 4 + chunk) * 128 + row) * 32 + i],
                      expected)
                << "tile=" << tile << " chunk=" << chunk << " row=" << row
                << " i=" << i;
          }
        }
      }
    }
  }
  const LiteRtNvidiaGemmShape shape = {128, kInput, 130,
                                       kLiteRtNvidiaGemmGateNone};
  uint8_t byte = 0;
  EXPECT_FALSE(LiteRtNvidiaSubbyteGemmTileWeights(&shape, nullptr, &byte));
  EXPECT_FALSE(LiteRtNvidiaSubbyteGemmTileWeights(&shape, &byte, nullptr));
}

TEST(SubbyteGemmTest, RejectsInvalidArguments) {
  uint16_t sentinel = 0;
  uint8_t weights = 0;
  LiteRtNvidiaGemmShape shape = {128, 256, 128, kLiteRtNvidiaGemmGateNone};
  EXPECT_NE(LiteRtNvidiaLaunchBf16Int4Gemm(&shape, nullptr, &weights, &sentinel,
                                           &sentinel, &sentinel, nullptr),
            cudaSuccess);
  EXPECT_NE(
      LiteRtNvidiaLaunchBf16Int4Gemm(&shape, &sentinel, nullptr, &sentinel,
                                     &sentinel, &sentinel, nullptr),
      cudaSuccess);
  EXPECT_NE(
      LiteRtNvidiaLaunchBf16Int4Gemm(&shape, &sentinel, &weights, &sentinel,
                                     &sentinel, nullptr, nullptr),
      cudaSuccess);
  shape.rows = 100;
  EXPECT_NE(
      LiteRtNvidiaLaunchBf16Int4Gemm(&shape, &sentinel, &weights, &sentinel,
                                     &sentinel, &sentinel, nullptr),
      cudaSuccess);
}

TEST(SubbyteGemmTest, MatchesReference) {
  if (!LiteRtNvidiaSubbyteGemmAvailable()) {
    GTEST_SKIP() << "Requires compute capability 8.0 or newer and 81 KB of "
                    "shared memory per block.";
  }
  // One block of one pair of chunks; several row and column tiles with a
  // partial last column tile; Gemma 4 12B's shapes.
  RunCase({{128, 128, 128, kLiteRtNvidiaGemmGateNone}, Rows::kUniform, 1.0f});
  RunCase({{256, 1152, 330, kLiteRtNvidiaGemmGateNone}, Rows::kUniform, 1.0f});
  RunCase({{128, 896, 2050, kLiteRtNvidiaGemmGateNone}, Rows::kOutliers, 1.0f});
  RunCase(
      {{1024, 3840, 4096, kLiteRtNvidiaGemmGateNone}, Rows::kOutliers, 1.0f});
  RunCase(
      {{128, 15360, 3840, kLiteRtNvidiaGemmGateNone}, Rows::kUniform, 3.0f});
}

TEST(SubbyteGemmTest, ScalesRowsOfAnyMagnitude) {
  if (!LiteRtNvidiaSubbyteGemmAvailable()) {
    GTEST_SKIP() << "Requires compute capability 8.0 or newer and 81 KB of "
                    "shared memory per block.";
  }
  // Activations far outside the FP16 range, tiny ones, and rows of zeros.
  RunCase(
      {{128, 1024, 256, kLiteRtNvidiaGemmGateNone}, Rows::kOutliers, 3.0e6f});
  RunCase(
      {{128, 1024, 256, kLiteRtNvidiaGemmGateNone}, Rows::kUniform, 1.0e-9f});
  RunCase({{256, 1024, 256, kLiteRtNvidiaGemmGateNone}, Rows::kSparse, 2.0f});
}

TEST(SubbyteGemmTest, GatedProjectionsMatchReference) {
  if (!LiteRtNvidiaSubbyteGemmAvailable()) {
    GTEST_SKIP() << "Requires compute capability 8.0 or newer and 81 KB of "
                    "shared memory per block.";
  }
  RunCase(
      {{128, 512, 64, kLiteRtNvidiaGemmGateGeluTanh}, Rows::kUniform, 1.0f});
  RunCase({{256, 1536, 6144, kLiteRtNvidiaGemmGateGeluTanh},
           Rows::kOutliers,
           1.0f});
  RunCase(
      {{128, 1536, 250, kLiteRtNvidiaGemmGateGeluErf}, Rows::kUniform, 2.0f});
  RunCase(
      {{256, 768, 512, kLiteRtNvidiaGemmGateGeluTanh}, Rows::kSparse, 2.0f});
}

}  // namespace
