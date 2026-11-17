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

// Times the launches of a statically quantized 3B model's block
// (trtllm/quant_linear.h) at the model's shapes, for one row (a decode step)
// and for a prefill chunk: the time of each launch alone, back to back on one
// stream, synchronized once. It prints; it asserts only that the launches
// succeed.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <gtest/gtest.h>
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "litert/vendors/nvidia/trtllm/quant_linear.h"

namespace {

// Device memory filled with a byte pattern.
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t bytes, int value = 0x35) {
    EXPECT_EQ(cudaMalloc(&data_, bytes), cudaSuccess);
    EXPECT_EQ(cudaMemset(data_, value, bytes), cudaSuccess);
  }
  ~DeviceBuffer() { cudaFree(data_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  template <typename T>
  T* as() const {
    return static_cast<T*>(data_);
  }

 private:
  void* data_ = nullptr;
};

// Microseconds per call of `launch`, over `iterations` calls.
template <typename Launch>
double MicrosecondsPerLaunch(int iterations, Launch launch) {
  for (int i = 0; i < 20; ++i) EXPECT_EQ(launch(), cudaSuccess);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  const auto begin = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) EXPECT_EQ(launch(), cudaSuccess);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  const std::chrono::duration<double, std::micro> elapsed =
      std::chrono::steady_clock::now() - begin;
  return elapsed.count() / iterations;
}

struct Product {
  const char* name;
  int layout;
  int bit_width;
  int rows;
  int columns;
  int segments;
  int requant;
  bool lut;
};

TEST(QuantLinearBenchmark, BlockLaunches) {
  constexpr Product kProducts[] = {
      {"q, k, v projections (INT4 3072x2048)", kLiteRtNvidiaQuantLayoutRows, 4,
       3072, 2048, 1, kLiteRtNvidiaQuantRequantFp32, false},
      {"output projection (INT4 2048x2048)", kLiteRtNvidiaQuantLayoutRows, 4,
       2048, 2048, 1, kLiteRtNvidiaQuantRequantFp32, false},
      {"MLP up-1 (INT2 2 x 4096x1024)", kLiteRtNvidiaQuantLayoutRows, 2, 8192,
       1024, 2, kLiteRtNvidiaQuantRequantDouble, false},
      {"MLP up-2 (INT2 32 x 256x1024, SiLU)", kLiteRtNvidiaQuantLayoutBlocks, 2,
       32768, 256, 32, kLiteRtNvidiaQuantRequantDouble, true},
      {"MLP down-1 (INT2 32 x 1024x256)", kLiteRtNvidiaQuantLayoutBlocks, 2,
       8192, 1024, 32, kLiteRtNvidiaQuantRequantDouble, false},
      {"MLP down-2 (INT2 2 x 1024x4096)", kLiteRtNvidiaQuantLayoutRows, 2, 2048,
       4096, 2, kLiteRtNvidiaQuantRequantDouble, false},
      {"head (INT2 262144x2048, logits)", kLiteRtNvidiaQuantLayoutRows, 2,
       262144, 2048, 1, kLiteRtNvidiaQuantRequantLogits, false},
  };
  for (const int m : {1, 128}) {
    std::printf("rows per launch: %d\n", m);
    double block = 0.0;
    for (const Product& product : kProducts) {
      const bool head = product.requant == kLiteRtNvidiaQuantRequantLogits;
      if (head && m != 1) continue;
      LiteRtNvidiaQuantLinearParams params = {};
      params.layout = product.layout;
      params.bit_width = product.bit_width;
      params.rows = product.rows;
      params.columns = product.columns;
      params.segments = product.segments;
      params.swap0 = 1;
      params.swap1 = 1;
      params.activation = kLiteRtNvidiaQuantInt8;
      params.requant = product.requant;
      params.lut = product.lut ? 1 : 0;
      params.ratio = 0.4;
      ASSERT_TRUE(LiteRtNvidiaQuantLinearSupports(&params));
      const size_t weight_bytes = static_cast<size_t>(product.rows) *
                                  product.columns * product.bit_width / 8;
      const DeviceBuffer activation(static_cast<size_t>(m) * product.segments *
                                    product.columns);
      const DeviceBuffer weights(weight_bytes, 0x5A);
      const DeviceBuffer multipliers(product.rows * sizeof(float), 0);
      const DeviceBuffer lut(256);
      const DeviceBuffer row_scale(m * sizeof(float), 0);
      const DeviceBuffer output(static_cast<size_t>(m) * product.rows *
                                (head ? sizeof(float) : 1));
      const double microseconds = MicrosecondsPerLaunch(m == 1 ? 500 : 20, [&] {
        return LiteRtNvidiaLaunchQuantLinear(
            &params, m, activation.as<void>(), weights.as<uint8_t>(),
            multipliers.as<float>(), lut.as<int8_t>(), row_scale.as<float>(),
            output.as<void>(), nullptr);
      });
      std::printf("  %-44s %9.1f us  (%.1f MB of weights)\n", product.name,
                  microseconds, weight_bytes / 1048576.0);
      if (!head) block += microseconds;
    }
    {
      LiteRtNvidiaQuantGlueParams params = {};
      params.width = 2048;
      params.x_type = kLiteRtNvidiaQuantFp16;
      params.branch = 1;
      params.norm = 1;
      params.quantize = kLiteRtNvidiaQuantGlueStatic;
      params.z_type = kLiteRtNvidiaQuantFp16;
      params.branch_scale = 0.05f;
      params.c_x = 0.8f;
      params.c_y = 0.6f;
      params.quantize_scale = 30.0f;
      const size_t count = static_cast<size_t>(m) * params.width;
      const DeviceBuffer x(count * 2, 0x3C);
      const DeviceBuffer y(count);
      const DeviceBuffer weights(params.width * sizeof(float), 0);
      const DeviceBuffer x_out(count * 2);
      const DeviceBuffer q(count);
      const double microseconds = MicrosecondsPerLaunch(m == 1 ? 500 : 50, [&] {
        return LiteRtNvidiaLaunchQuantGlue(
            &params, m, x.as<void>(), y.as<int8_t>(), weights.as<float>(),
            weights.as<float>(), x_out.as<void>(), nullptr, q.as<int8_t>(),
            nullptr, nullptr);
      });
      std::printf("  %-44s %9.1f us  (two per block)\n",
                  "glue (residual, norm, QUANTIZE)", microseconds);
      block += 2 * microseconds;
    }
    {
      LiteRtNvidiaQuantQkvRopeParams params = {};
      params.tokens = m;
      params.q_heads = 8;
      params.k_heads = 2;
      params.v_heads = 2;
      params.depth = 256;
      params.q_scale = params.k_scale = params.v_scale = 0.1f;
      params.epsilon = 1e-6f;
      params.q_multiplier = 0.125f;
      const size_t count = static_cast<size_t>(m) * 12 * params.depth;
      const DeviceBuffer qkv(count);
      const DeviceBuffer positions(m * sizeof(int32_t), 0);
      const DeviceBuffer weights(params.depth * sizeof(float), 0);
      const DeviceBuffer q(count * 2);
      const DeviceBuffer k(count * 2);
      const DeviceBuffer v(count * 2);
      const double microseconds = MicrosecondsPerLaunch(m == 1 ? 500 : 50, [&] {
        return LiteRtNvidiaLaunchQuantQkvRope(
            &params, qkv.as<int8_t>(), positions.as<int32_t>(),
            weights.as<float>(), weights.as<float>(), weights.as<float>(),
            weights.as<float>(), q.as<void>(), k.as<void>(), v.as<void>(),
            nullptr);
      });
      std::printf("  %-44s %9.1f us\n", "head norms and rotation",
                  microseconds);
      block += microseconds;
    }
    std::printf("  one block without its attention: %.1f us\n", block);
  }
}

}  // namespace
