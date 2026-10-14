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

#include "litert/vendors/nvidia/trtllm/int4_block_gemv.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "cuda_runtime_api.h"

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

// Round-to-nearest-even FP32 -> FP16 for normal values.
uint16_t FloatToHalfBits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000u;
  int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xff) - 127 + 15;
  uint32_t mantissa = bits & 0x7fffffu;
  if (exponent <= 0 || exponent >= 31) {
    ADD_FAILURE() << "FP16 test value out of the normal range: " << value;
    return static_cast<uint16_t>(sign);
  }
  uint32_t half = (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13);
  const uint32_t remainder = mantissa & 0x1fffu;
  if (remainder > 0x1000u || (remainder == 0x1000u && (half & 1u))) {
    ++half;
  }
  return static_cast<uint16_t>(sign | half);
}

float HalfBitsToFloat(uint16_t value) {
  const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
  const uint32_t exponent = (value >> 10) & 0x1fu;
  const uint32_t mantissa = value & 0x3ffu;
  uint32_t bits = sign;
  if (exponent != 0) {
    bits |= ((exponent + 127 - 15) << 23) | (mantissa << 13);
  } else if (mantissa != 0) {
    float result = std::ldexp(static_cast<float>(mantissa), -24);
    return (value & 0x8000u) ? -result : result;
  }
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

int8_t DecodeInt4(const std::vector<uint8_t>& packed, size_t index) {
  const uint8_t bits = (packed[index / 2] >> (4 * (index % 2))) & 0xf;
  return static_cast<int8_t>(static_cast<int>(bits ^ 8) - 8);
}

// Row-major [n, k] INT4 values, the even element in the low nibble.
std::vector<uint8_t> MakeInt4Weights(int n, int k, int seed) {
  std::vector<uint8_t> weights(static_cast<size_t>(n) * k / 2, 0);
  for (int row = 0; row < n; ++row) {
    for (int column = 0; column < k; ++column) {
      const int value = ((5 * row + 3 * column + seed + row / 3) % 16) - 8;
      const size_t index = static_cast<size_t>(row) * k + column;
      weights[index / 2] |= (static_cast<uint8_t>(value) & 0xf)
                            << (4 * (index % 2));
    }
  }
  return weights;
}

// FP16 scales [n, k / 32] in the range of Gemma 4 26B's blockwise scales.
std::vector<uint16_t> MakeScales(int n, int k, int seed) {
  const int blocks = k / 32;
  std::vector<uint16_t> scales(static_cast<size_t>(n) * blocks);
  for (int row = 0; row < n; ++row) {
    for (int b = 0; b < blocks; ++b) {
      scales[static_cast<size_t>(row) * blocks + b] = FloatToHalfBits(
          0.003f +
          0.0002f * static_cast<float>((row * 7 + b * 13 + seed) % 97));
    }
  }
  return scales;
}

float Gelu(float x, int gate) {
  if (gate == 1) {
    return 0.5f * x *
           (1.0f +
            std::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
  }
  return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752f));
}

// y[r][n] of the blockwise GEMV, summing each block first as the kernel's
// reference contract.
float Reference(const std::vector<uint16_t>& activation,
                const std::vector<uint8_t>& weights,
                const std::vector<uint16_t>& scales, int r, int row, int k) {
  const int blocks = k / 32;
  float total = 0.0f;
  for (int b = 0; b < blocks; ++b) {
    float partial = 0.0f;
    for (int c = b * 32; c < (b + 1) * 32; ++c) {
      partial =
          std::fma(Bf16BitsToFloat(activation[static_cast<size_t>(r) * k + c]),
                   static_cast<float>(
                       DecodeInt4(weights, static_cast<size_t>(row) * k + c)),
                   partial);
    }
    total += partial *
             HalfBitsToFloat(scales[static_cast<size_t>(row) * blocks + b]);
  }
  return total;
}

TEST(Int4BlockGemvTest, MatchesCpuReference) {
  // Every row count, over the input sizes of the 26B model's projections
  // (the expert down projection of 22 blocks runs on one warp) and a short
  // odd case; 44 channels end in a partial tile of 4.
  constexpr int kN = 44;
  for (const int k : {64, 704, 2112, 2816}) {
    const std::vector<uint8_t> weights = MakeInt4Weights(kN, k, 1);
    const std::vector<uint16_t> scales = MakeScales(kN, k, 1);
    for (int rows = 1; rows <= kLiteRtNvidiaInt4BlockGemvMaxRows; ++rows) {
      SCOPED_TRACE(::testing::Message() << "K=" << k << " rows=" << rows);
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
      ASSERT_EQ(cudaMemcpy(device_weights.get(), weights.data(), weights.size(),
                           cudaMemcpyHostToDevice),
                cudaSuccess);
      ASSERT_EQ(
          cudaMemcpy(device_scales.get(), scales.data(),
                     scales.size() * sizeof(uint16_t), cudaMemcpyHostToDevice),
          cudaSuccess);
      ASSERT_EQ(LiteRtNvidiaLaunchBf16Int4BlockGemvRows(
                    device_activation.get(),
                    static_cast<const uint8_t*>(device_weights.get()),
                    device_scales.get(), device_output.get(), rows, k, kN,
                    /*gate=*/0, /*stream=*/nullptr),
                cudaSuccess);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      std::vector<uint16_t> actual(static_cast<size_t>(rows) * kN);
      ASSERT_EQ(
          cudaMemcpy(actual.data(), device_output.get(),
                     actual.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
          cudaSuccess);
      for (int r = 0; r < rows; ++r) {
        for (int row = 0; row < kN; ++row) {
          const float expected =
              Reference(activation, weights, scales, r, row, k);
          const float got =
              Bf16BitsToFloat(actual[static_cast<size_t>(r) * kN + row]);
          EXPECT_NEAR(got, expected, 1e-2f * std::fabs(expected) + 1e-3f)
              << "r=" << r << " row=" << row;
        }
      }
    }
  }
}

TEST(Int4BlockGemvTest, GatedFeedForwardMatchesCpuReference) {
  // Gate rows then up rows; gelu(gate) * up in both GELU forms.
  constexpr int kN = 40;
  constexpr int kK = 2816;
  const std::vector<uint8_t> weights = MakeInt4Weights(2 * kN, kK, 3);
  const std::vector<uint16_t> scales = MakeScales(2 * kN, kK, 3);
  for (const int gate : {1, 2}) {
    for (const int rows : {1, 3, 8}) {
      SCOPED_TRACE(::testing::Message() << "gate=" << gate << " rows=" << rows);
      std::vector<uint16_t> activation(static_cast<size_t>(rows) * kK);
      for (int r = 0; r < rows; ++r) {
        for (int i = 0; i < kK; ++i) {
          activation[static_cast<size_t>(r) * kK + i] = FloatToBf16Bits(
              static_cast<float>((i * 23 + r * 31) % 89 - 44) / 41.0f);
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
      ASSERT_EQ(cudaMemcpy(device_weights.get(), weights.data(), weights.size(),
                           cudaMemcpyHostToDevice),
                cudaSuccess);
      ASSERT_EQ(
          cudaMemcpy(device_scales.get(), scales.data(),
                     scales.size() * sizeof(uint16_t), cudaMemcpyHostToDevice),
          cudaSuccess);
      ASSERT_EQ(LiteRtNvidiaLaunchBf16Int4BlockGemvRows(
                    device_activation.get(),
                    static_cast<const uint8_t*>(device_weights.get()),
                    device_scales.get(), device_output.get(), rows, kK, kN,
                    gate, /*stream=*/nullptr),
                cudaSuccess);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      std::vector<uint16_t> actual(static_cast<size_t>(rows) * kN);
      ASSERT_EQ(
          cudaMemcpy(actual.data(), device_output.get(),
                     actual.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
          cudaSuccess);
      for (int r = 0; r < rows; ++r) {
        for (int row = 0; row < kN; ++row) {
          const float g = Reference(activation, weights, scales, r, row, kK);
          const float u =
              Reference(activation, weights, scales, r, kN + row, kK);
          const float expected = Gelu(g, gate) * u;
          const float got =
              Bf16BitsToFloat(actual[static_cast<size_t>(r) * kN + row]);
          EXPECT_NEAR(got, expected, 1e-2f * std::fabs(expected) + 1e-3f)
              << "r=" << r << " row=" << row;
        }
      }
    }
  }
}

TEST(Int4BlockGemvTest, RejectsInvalidArguments) {
  uint8_t byte = 0;
  EXPECT_EQ(
      LiteRtNvidiaLaunchBf16Int4BlockGemvRows(&byte, &byte, &byte, &byte,
                                              /*rows=*/0, 64, 8, 0, nullptr),
      cudaErrorInvalidValue);
  EXPECT_EQ(
      LiteRtNvidiaLaunchBf16Int4BlockGemvRows(&byte, &byte, &byte, &byte, 1,
                                              /*input_size=*/48, 8, 0, nullptr),
      cudaErrorInvalidValue);
  EXPECT_EQ(LiteRtNvidiaLaunchBf16Int4BlockGemvRows(
                &byte, &byte, &byte, &byte, 1, 64, 8, /*gate=*/3, nullptr),
            cudaErrorInvalidValue);
  EXPECT_EQ(LiteRtNvidiaLaunchBf16Int4BlockGemvRows(
                &byte, &byte, &byte, &byte,
                kLiteRtNvidiaInt4BlockGemvMaxRows + 1, 64, 8, 0, nullptr),
            cudaErrorInvalidValue);
}

}  // namespace
