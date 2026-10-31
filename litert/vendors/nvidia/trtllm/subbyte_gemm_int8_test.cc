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

// The int8 tensor-core GEMM against a double-precision reference: its error
// is the int8 quantization of the activations (a block of 128 dims scaled by
// its largest magnitude) on top of the BF16 output rounding, and the outlier
// dims (the 32 largest over the rows) do not coarsen it.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "litert/vendors/nvidia/trtllm/subbyte_gemm.h"

namespace {

struct CudaDeleter {
  void operator()(void* pointer) const { cudaFree(pointer); }
};
using CudaBuffer = std::unique_ptr<void, CudaDeleter>;

CudaBuffer AllocateCuda(size_t bytes) {
  void* pointer = nullptr;
  if (cudaMalloc(&pointer, bytes) != cudaSuccess) return nullptr;
  return CudaBuffer(pointer);
}

template <typename T>
CudaBuffer Upload(const std::vector<T>& host) {
  auto device = AllocateCuda(host.size() * sizeof(T));
  if (device != nullptr) {
    EXPECT_EQ(cudaMemcpy(device.get(), host.data(), host.size() * sizeof(T),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
  }
  return device;
}

uint16_t FloatToBf16Bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t rounding = ((bits >> 16) & 1u) + 0x7fffu;
  return static_cast<uint16_t>((bits + rounding) >> 16);
}

float Bf16BitsToFloat(uint16_t bits) {
  const uint32_t widened = static_cast<uint32_t>(bits) << 16;
  float value;
  std::memcpy(&value, &widened, sizeof(value));
  return value;
}

double Gelu(double x, int gate) {
  if (gate == kLiteRtNvidiaGemmGateGeluTanh) {
    return 0.5 * x *
           (1.0 + std::tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
  }
  return 0.5 * x * (1.0 + std::erf(x * 0.7071067811865476));
}

enum class Rows {
  kUniform,  // every activation in [-scale, scale), rows of different magnitude
  kChannels,  // sixteen input dims forty times larger in every row (massive
              // activations)
  kSparse,    // mostly zeros, some rows all zeros
  kHuge,  // uniform rows, three of them with an activation past fp16's reach
};

struct Case {
  LiteRtNvidiaGemmShape shape;
  Rows rows;
  bool norm;
};

void RunCase(const Case& test_case) {
  const auto& shape = test_case.shape;
  SCOPED_TRACE(::testing::Message()
               << "rows=" << shape.rows << " input_size=" << shape.input_size
               << " output_size=" << shape.output_size << " gate=" << shape.gate
               << " kind=" << static_cast<int>(test_case.rows)
               << " norm=" << test_case.norm);
  ASSERT_TRUE(LiteRtNvidiaInt8GemmSupports(&shape));
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
    const int k = static_cast<int>(i % input);
    float value = next() * (1.0f + static_cast<float>(row % 7) * 3.0f);
    if (test_case.rows == Rows::kChannels && k % 241 == 7) value *= 40.0f;
    if (test_case.rows == Rows::kSparse && (row % 5 == 3 || i % 11 != 0))
      value = 0.0f;
    if (test_case.rows == Rows::kHuge) {
      // In rows the check samples: the fp16 outlier path would overflow on
      // them.
      if (row == 0 && k == 5) value = 7.0e4f;
      if (row == 89 && k == 1000) value = -3.0e5f;
      if (row == shape.rows - 1 && k == 77) value = 1.2e5f;
    }
    activation[i] = FloatToBf16Bits(value);
  }
  std::vector<uint8_t> weights(static_cast<size_t>(channels) * input / 2);
  for (auto& byte : weights) {
    state = state * 1664525u + 1013904223u;
    byte = static_cast<uint8_t>(state >> 24);
  }
  std::vector<uint16_t> scales(channels);
  for (auto& scale : scales)
    scale = FloatToBf16Bits(0.002f + 0.004f * std::fabs(next()));
  std::vector<float> norm_scale(input);
  for (auto& s : norm_scale) s = 0.5f + std::fabs(next());
  const float norm_eps = 1e-6f;
  const float norm_multiplier = 1.0f;

  const size_t workspace_bytes = LiteRtNvidiaInt8GemmWorkspaceBytes(&shape);
  ASSERT_GT(workspace_bytes, 0u);
  auto activation_device = Upload(activation);
  auto weights_device = Upload(weights);
  auto scales_device = Upload(scales);
  auto norm_device = Upload(norm_scale);
  const size_t output_bytes =
      static_cast<size_t>(shape.rows) * output * sizeof(uint16_t);
  auto output_device = AllocateCuda(output_bytes);
  auto workspace = AllocateCuda(workspace_bytes);
  ASSERT_NE(output_device, nullptr);
  ASSERT_NE(workspace, nullptr);
  ASSERT_EQ(cudaMemset(output_device.get(), 0xff, output_bytes), cudaSuccess);
  ASSERT_EQ(LiteRtNvidiaLaunchBf16Int4GemmInt8Normed(
                &shape, activation_device.get(),
                test_case.norm ? static_cast<const float*>(norm_device.get())
                               : nullptr,
                norm_eps, norm_multiplier,
                static_cast<const uint8_t*>(weights_device.get()),
                scales_device.get(), output_device.get(), workspace.get(),
                /*stream=*/nullptr),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<uint16_t> actual(static_cast<size_t>(shape.rows) * output);
  ASSERT_EQ(cudaMemcpy(actual.data(), output_device.get(), output_bytes,
                       cudaMemcpyDeviceToHost),
            cudaSuccess);

  // A sample of rows: the first and last of the first and last tiles, and every
  // 89th.
  for (int row = 0; row < shape.rows; ++row) {
    if (row != 0 && row != 127 && row != shape.rows - 128 &&
        row != shape.rows - 1 && row % 89 != 0)
      continue;
    double factor = 1.0;
    if (test_case.norm) {
      double squares = 0.0;
      for (int k = 0; k < input; ++k) {
        const double a =
            Bf16BitsToFloat(activation[static_cast<size_t>(row) * input + k]);
        squares += a * a;
      }
      factor = 1.0 / std::sqrt(squares / input + norm_eps) * norm_multiplier;
    }
    const auto dot = [&](int channel) {
      const uint8_t* packed =
          weights.data() + static_cast<size_t>(channel) * input / 2;
      double sum = 0.0;
      for (int k = 0; k < input; ++k) {
        const int nibble = (packed[k / 2] >> ((k & 1) * 4)) & 15;
        double a =
            Bf16BitsToFloat(activation[static_cast<size_t>(row) * input + k]);
        if (test_case.norm) a *= factor * norm_scale[k];
        sum += a * ((nibble ^ 8) - 8);
      }
      return sum * Bf16BitsToFloat(scales[channel]);
    };
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
    // The int8 activations add a few times the BF16 rounding of the results
    // (measured: 4e-3 of the results' RMS on uniform rows, 2.3e-3 with the
    // outlier channels taken out, 9e-3 for a gated product with a fused norm,
    // against 1.7e-3 of rounding alone). This bounds the kernel's error; the
    // model's quality under it is the LiveCodeBench evaluation's to show.
    EXPECT_LE(std::sqrt(error),
              6.0 * std::sqrt(rounding) + 4e-3 * std::sqrt(norm))
        << "row=" << row << " relative error " << std::sqrt(error / norm)
        << ", of rounding " << std::sqrt(rounding / norm);
  }
}

TEST(SubbyteGemmInt8Test, Supports) {
  const auto supports = [](LiteRtNvidiaGemmShape shape) {
    return LiteRtNvidiaInt8GemmSupports(&shape);
  };
  EXPECT_TRUE(supports({1024, 3840, 8192, kLiteRtNvidiaGemmGateNone, 0}));
  EXPECT_TRUE(supports({128, 15360, 3840, kLiteRtNvidiaGemmGateGeluTanh, 0}));
  EXPECT_FALSE(supports(
      {1024, 3840, 8192, kLiteRtNvidiaGemmGateNone, 32}));  // blockwise scales
  EXPECT_FALSE(supports({1024, 3904, 8192, kLiteRtNvidiaGemmGateNone,
                         0}));  // not a multiple of 128
  EXPECT_FALSE(supports({1024, 16512, 8192, kLiteRtNvidiaGemmGateNone,
                         0}));  // past the selection's buffer
  EXPECT_FALSE(supports({100, 3840, 8192, kLiteRtNvidiaGemmGateNone, 0}));
  LiteRtNvidiaGemmShape blockwise = {1024, 3840, 8192,
                                     kLiteRtNvidiaGemmGateNone, 32};
  EXPECT_EQ(LiteRtNvidiaInt8GemmWorkspaceBytes(&blockwise), 0u);
}

TEST(SubbyteGemmInt8Test, UniformRows) {
  RunCase(
      {{256, 1536, 768, kLiteRtNvidiaGemmGateNone, 0}, Rows::kUniform, false});
  RunCase({{128, 3840, 512, kLiteRtNvidiaGemmGateGeluTanh, 0},
           Rows::kUniform,
           false});
  RunCase({{128, 4096, 384, kLiteRtNvidiaGemmGateGeluErf, 0},
           Rows::kUniform,
           true});
}

TEST(SubbyteGemmInt8Test, OutliersPastFp16Reach) {
  RunCase({{256, 1536, 768, kLiteRtNvidiaGemmGateNone, 0}, Rows::kHuge, false});
  RunCase(
      {{128, 3840, 512, kLiteRtNvidiaGemmGateGeluTanh, 0}, Rows::kHuge, false});
}

TEST(SubbyteGemmInt8Test, OutlierChannelsAreExact) {
  RunCase(
      {{256, 3840, 512, kLiteRtNvidiaGemmGateNone, 0}, Rows::kChannels, false});
  RunCase({{128, 2560, 384, kLiteRtNvidiaGemmGateGeluTanh, 0},
           Rows::kChannels,
           true});
}

TEST(SubbyteGemmInt8Test, SparseRows) {
  RunCase(
      {{256, 1536, 256, kLiteRtNvidiaGemmGateNone, 0}, Rows::kSparse, false});
}

}  // namespace
