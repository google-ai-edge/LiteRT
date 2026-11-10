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

// The per-channel GEMV with its fused RMS norm against a double-precision
// reference, and against the plain kernel over rows normed on the host.

#include "litert/vendors/nvidia/trtllm/int2_gemv.h"

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

int DecodeInt4(const std::vector<uint8_t>& packed, size_t index) {
  const int nibble = (packed[index / 2] >> (4 * (index % 2))) & 15;
  return (nibble ^ 8) - 8;
}

// A deterministic pseudo-random float in [-1, 1).
struct Random {
  uint32_t state;
  float Next() {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return static_cast<float>(state >> 8) / 8388608.0f - 1.0f;
  }
};

struct Case {
  int rows;
  int input_size;
  int output_size;
};

TEST(Int2GemvTest, NormedPerChannelGemvMatchesReference) {
  const Case cases[] = {
      {1, 256, 40}, {3, 3840, 72}, {8, 4096, 24}, {2, 8192, 16}};
  for (const Case& c : cases) {
    Random random{2463534242u +
                  static_cast<uint32_t>(c.rows * 7 + c.input_size)};
    std::vector<uint16_t> activation(static_cast<size_t>(c.rows) *
                                     c.input_size);
    for (size_t i = 0; i < activation.size(); ++i) {
      // Rows of different magnitude: the norm's factor differs per row.
      activation[i] = FloatToBf16Bits(
          random.Next() * (1.0f + static_cast<float>(i / c.input_size)));
    }
    std::vector<uint8_t> weights(static_cast<size_t>(c.output_size) *
                                 c.input_size / 2);
    for (auto& byte : weights) {
      byte = static_cast<uint8_t>(random.state >> 24);
      random.Next();
    }
    std::vector<uint16_t> scales(c.output_size);
    for (auto& scale : scales)
      scale = FloatToBf16Bits(0.002f + 0.004f * std::fabs(random.Next()));
    std::vector<float> norm_scale(c.input_size);
    for (auto& value : norm_scale) value = 1.0f + 0.5f * random.Next();
    const float eps = 1e-6f;
    const float multiplier = 1.5f;

    auto device_activation = AllocateCuda(activation.size() * 2);
    auto device_weights = AllocateCuda(weights.size());
    auto device_scales = AllocateCuda(scales.size() * 2);
    auto device_norm = AllocateCuda(norm_scale.size() * 4);
    auto device_output =
        AllocateCuda(static_cast<size_t>(c.rows) * c.output_size * 2);
    ASSERT_EQ(cudaMemcpy(device_activation.get(), activation.data(),
                         activation.size() * 2, cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_weights.get(), weights.data(), weights.size(),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_scales.get(), scales.data(), scales.size() * 2,
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_norm.get(), norm_scale.data(),
                         norm_scale.size() * 4, cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(
        LiteRtNvidiaLaunchBf16SubbytePerChannelGemvRowsNormed(
            device_activation.get(),
            static_cast<const uint8_t*>(device_weights.get()),
            device_scales.get(), static_cast<const float*>(device_norm.get()),
            eps, multiplier, device_output.get(), 4, c.rows, c.input_size,
            c.output_size, nullptr),
        cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<uint16_t> output(static_cast<size_t>(c.rows) * c.output_size);
    ASSERT_EQ(cudaMemcpy(output.data(), device_output.get(), output.size() * 2,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);

    // The plain kernel over the rows normed on the host in double, rounded to
    // BF16: the fused kernel skips that rounding, so it is closer to the
    // reference than the plain path is.
    std::vector<uint16_t> normed(activation.size());
    std::vector<double> factor(c.rows);
    for (int r = 0; r < c.rows; ++r) {
      double squares = 0.0;
      for (int k = 0; k < c.input_size; ++k) {
        const double x = Bf16BitsToFloat(
            activation[static_cast<size_t>(r) * c.input_size + k]);
        squares += x * x;
      }
      factor[r] = 1.0 / std::sqrt(squares / c.input_size + eps) * multiplier;
      for (int k = 0; k < c.input_size; ++k) {
        const double x = Bf16BitsToFloat(
            activation[static_cast<size_t>(r) * c.input_size + k]);
        normed[static_cast<size_t>(r) * c.input_size + k] =
            FloatToBf16Bits(static_cast<float>(x * factor[r] * norm_scale[k]));
      }
    }
    for (int r = 0; r < c.rows; ++r) {
      double error = 0.0, norm = 0.0, plain_error = 0.0;
      for (int n = 0; n < c.output_size; ++n) {
        double expected = 0.0, plain = 0.0;
        for (int k = 0; k < c.input_size; ++k) {
          const int w =
              DecodeInt4(weights, static_cast<size_t>(n) * c.input_size + k);
          const double x = Bf16BitsToFloat(
              activation[static_cast<size_t>(r) * c.input_size + k]);
          expected += w * x * norm_scale[k];
          plain += w * static_cast<double>(Bf16BitsToFloat(
                           normed[static_cast<size_t>(r) * c.input_size + k]));
        }
        expected *= factor[r] * Bf16BitsToFloat(scales[n]);
        plain *= Bf16BitsToFloat(scales[n]);
        const double got =
            Bf16BitsToFloat(output[static_cast<size_t>(r) * c.output_size + n]);
        error += (got - expected) * (got - expected);
        plain_error += (plain - expected) * (plain - expected);
        norm += expected * expected;
      }
      // BF16 output rounding is about 1.7e-3 of a row's norm.
      EXPECT_LE(std::sqrt(error), 3.0e-3 * std::sqrt(norm))
          << "rows=" << c.rows << " input=" << c.input_size << " row=" << r;
      EXPECT_LE(std::sqrt(error),
                std::sqrt(plain_error) + 3.0e-3 * std::sqrt(norm))
          << "the fused norm is no worse than the host-normed plain path";
    }
  }
}

TEST(Int2GemvTest, NormedLauncherRefusesOtherWidthsAndFallsBackWithoutScale) {
  const int rows = 1, input_size = 256, output_size = 8;
  std::vector<uint16_t> activation(input_size, FloatToBf16Bits(0.5f));
  std::vector<uint8_t> weights(
      static_cast<size_t>(output_size) * input_size / 2, 0x5a);
  std::vector<uint16_t> scales(output_size, FloatToBf16Bits(0.01f));
  auto device_activation = AllocateCuda(activation.size() * 2);
  auto device_weights = AllocateCuda(weights.size());
  auto device_scales = AllocateCuda(scales.size() * 2);
  auto device_norm = AllocateCuda(input_size * 4);
  auto device_output = AllocateCuda(output_size * 2);
  ASSERT_EQ(cudaMemcpy(device_activation.get(), activation.data(),
                       activation.size() * 2, cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_weights.get(), weights.data(), weights.size(),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_scales.get(), scales.data(), scales.size() * 2,
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  // INT2 weights have no fused norm.
  EXPECT_EQ(LiteRtNvidiaLaunchBf16SubbytePerChannelGemvRowsNormed(
                device_activation.get(),
                static_cast<const uint8_t*>(device_weights.get()),
                device_scales.get(),
                static_cast<const float*>(device_norm.get()), 1e-6f, 1.0f,
                device_output.get(), 2, rows, input_size, output_size, nullptr),
            cudaErrorInvalidValue);
  // A negative epsilon is no norm.
  EXPECT_EQ(LiteRtNvidiaLaunchBf16SubbytePerChannelGemvRowsNormed(
                device_activation.get(),
                static_cast<const uint8_t*>(device_weights.get()),
                device_scales.get(),
                static_cast<const float*>(device_norm.get()), -1.0f, 1.0f,
                device_output.get(), 4, rows, input_size, output_size, nullptr),
            cudaErrorInvalidValue);
  // Without a scale vector the plain product runs.
  EXPECT_EQ(LiteRtNvidiaLaunchBf16SubbytePerChannelGemvRowsNormed(
                device_activation.get(),
                static_cast<const uint8_t*>(device_weights.get()),
                device_scales.get(), nullptr, 1e-6f, 1.0f, device_output.get(),
                2, rows, input_size, output_size, nullptr),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

}  // namespace
