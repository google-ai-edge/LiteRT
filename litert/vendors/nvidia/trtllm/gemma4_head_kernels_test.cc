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
