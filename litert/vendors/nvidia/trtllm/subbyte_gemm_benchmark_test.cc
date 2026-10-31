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

// The INT4 GEMMs on the shapes of the Gemma 4 prefill: time per launch,
// throughput, a checksum of the output (so two builds of a kernel can be told
// identical or not without a model run) and the error against double
// precision on a sample of rows.
//   bazel test //litert/vendors/nvidia/trtllm:subbyte_gemm_benchmark_test \
//     --test_output=all   (needs a GPU)
// GEMM_BENCH_ITERS=N for more samples; GEMM_VARIANT=int8 runs the int8
// tensor-core kernel on the per-channel shapes; GEMM_OUTLIERS=channel gives
// sixteen input dims forty times larger (massive activations), GEMM_OUTLIERS=1
// one activation in 997 forty times larger.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

CudaBuffer Allocate(size_t bytes) {
  void* pointer = nullptr;
  if (cudaMalloc(&pointer, bytes) != cudaSuccess) return nullptr;
  return CudaBuffer(pointer);
}

uint16_t FloatToBf16Bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t rounding = ((bits >> 16) & 1u) + 0x7fffu;
  return static_cast<uint16_t>((bits + rounding) >> 16);
}

uint16_t FloatToHalfBits(float value) {
  // Values here are small positives: a plain conversion suffices.
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const int exponent = static_cast<int>((bits >> 23) & 0xff) - 127 + 15;
  const uint32_t mantissa = (bits >> 13) & 0x3ff;
  if (exponent <= 0) return 0;
  return static_cast<uint16_t>((exponent << 10) | mantissa);
}

struct Shape {
  const char* name;
  LiteRtNvidiaGemmShape shape;
};

// The 12B's prefill products at the 1,024-token batch: the fused QKV, the
// output projection, the gated MLP and the down projection; the E4B's and
// E2B's MLPs for the small families.
const Shape kShapes[] = {
    {"12b qkv 1024x3840->8192",
     {1024, 3840, 8192, kLiteRtNvidiaGemmGateNone, 32}},
    {"12b o 1024x4096->3840",
     {1024, 4096, 3840, kLiteRtNvidiaGemmGateNone, 32}},
    {"12b gate+up 1024x3840->15360 gelu",
     {1024, 3840, 15360, kLiteRtNvidiaGemmGateGeluTanh, 32}},
    {"12b down 1024x15360->3840",
     {1024, 15360, 3840, kLiteRtNvidiaGemmGateNone, 32}},
    {"12b qkv per-channel", {1024, 3840, 8192, kLiteRtNvidiaGemmGateNone, 0}},
    {"12b gate+up per-channel",
     {1024, 3840, 15360, kLiteRtNvidiaGemmGateGeluTanh, 0}},
    {"e4b gate+up 1024x2560->10240 gelu",
     {1024, 2560, 10240, kLiteRtNvidiaGemmGateGeluTanh, 32}},
    {"e2b gate+up 1024x1536->6144 gelu",
     {1024, 1536, 6144, kLiteRtNvidiaGemmGateGeluTanh, 32}},
    {"12b down 128 rows", {128, 15360, 3840, kLiteRtNvidiaGemmGateNone, 32}},
};

uint64_t Checksum(const std::vector<uint16_t>& values) {
  uint64_t hash = 1469598103934665603ull;
  for (const uint16_t value : values) {
    hash ^= value;
    hash *= 1099511628211ull;
  }
  return hash;
}

TEST(SubbyteGemmBenchmark, PrefillShapes) {
  int device = 0;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  cudaDeviceProp properties{};
  ASSERT_EQ(cudaGetDeviceProperties(&properties, device), cudaSuccess);
  int shared_optin = 0;
  int shared_per_sm = 0;
  cudaDeviceGetAttribute(&shared_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                         device);
  cudaDeviceGetAttribute(&shared_per_sm,
                         cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);
  std::printf(
      "device %s: sm_%d%d, %d multiprocessors, %d KB shared per block "
      "(opt-in), %d KB per multiprocessor, %d registers per block, clock %d "
      "MHz, memory clock %d MHz, bus %d bits\n",
      properties.name, properties.major, properties.minor,
      properties.multiProcessorCount, shared_optin / 1024, shared_per_sm / 1024,
      properties.regsPerBlock, properties.clockRate / 1000,
      properties.memoryClockRate / 1000, properties.memoryBusWidth);
  const char* iters_env = std::getenv("GEMM_BENCH_ITERS");
  const int iters = iters_env != nullptr ? std::atoi(iters_env) : 20;
  std::printf("%-40s %10s %10s %10s %18s\n", "shape", "us/launch", "TFLOPS",
              "blocks", "checksum");
  for (const Shape& entry : kShapes) {
    const LiteRtNvidiaGemmShape& shape = entry.shape;
    ASSERT_TRUE(LiteRtNvidiaSubbyteGemmSupports(&shape)) << entry.name;
    const int input = shape.input_size;
    const int output = shape.output_size;
    const int channels = shape.gate != 0 ? 2 * output : output;
    uint32_t state = 2463534242u + shape.rows + input + output;
    const auto next = [&]() -> float {
      state ^= state << 13;
      state ^= state >> 17;
      state ^= state << 5;
      return static_cast<float>(state >> 8) / 8388608.0f - 1.0f;
    };
    std::vector<uint16_t> activation(static_cast<size_t>(shape.rows) * input);
    for (size_t i = 0; i < activation.size(); ++i) {
      const size_t row = i / input;
      float value = next() * (1.0f + static_cast<float>(row % 7));
      const char* outliers = std::getenv("GEMM_OUTLIERS");
      if (outliers != nullptr && std::strcmp(outliers, "channel") == 0) {
        if ((i % input) % 241 == 7) value *= 40.0f;
      } else if (outliers != nullptr && i % 997 == 0) {
        value *= 40.0f;
      }
      activation[i] = FloatToBf16Bits(value);
    }
    std::vector<uint8_t> weights(static_cast<size_t>(channels) * input / 2);
    for (auto& byte : weights) {
      state = state * 1664525u + 1013904223u;
      byte = static_cast<uint8_t>(state >> 24);
    }
    const bool blockwise = shape.scale_block == 32;
    const int blocks = blockwise ? input / 32 : 1;
    std::vector<uint16_t> scales(static_cast<size_t>(channels) * blocks);
    for (auto& scale : scales) {
      const float value = 0.002f + 0.004f * std::fabs(next());
      scale = blockwise ? FloatToHalfBits(value) : FloatToBf16Bits(value);
    }
    ASSERT_EQ(LiteRtNvidiaSubbyteGemmWeightBytes(&shape), weights.size());
    auto activation_device = Allocate(activation.size() * 2);
    auto weights_device = Allocate(weights.size());
    auto scales_device = Allocate(scales.size() * 2);
    const size_t output_bytes = static_cast<size_t>(shape.rows) * output * 2;
    auto output_device = Allocate(output_bytes);
    auto workspace = Allocate(LiteRtNvidiaSubbyteGemmWorkspaceBytes(&shape));
    ASSERT_TRUE(activation_device && weights_device && scales_device &&
                output_device && workspace);
    ASSERT_EQ(cudaMemcpy(activation_device.get(), activation.data(),
                         activation.size() * 2, cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(weights_device.get(), weights.data(), weights.size(),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(scales_device.get(), scales.data(), scales.size() * 2,
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    const char* variant = std::getenv("GEMM_VARIANT");
    const bool int8 = variant != nullptr && std::strcmp(variant, "int8") == 0;
    if (int8 && !LiteRtNvidiaInt8GemmSupports(&shape)) continue;
    auto workspace8 =
        Allocate(int8 ? LiteRtNvidiaInt8GemmWorkspaceBytes(&shape) : 16);
    ASSERT_TRUE(workspace8);
    const auto launch = [&]() {
      if (int8) {
        return LiteRtNvidiaLaunchBf16Int4GemmInt8Normed(
            &shape, activation_device.get(), /*positions=*/nullptr, nullptr,
            0.0f, 1.0f, static_cast<const uint8_t*>(weights_device.get()),
            scales_device.get(), output_device.get(), workspace8.get(),
            nullptr);
      }
      return LiteRtNvidiaLaunchBf16Int4Gemm(
          &shape, activation_device.get(),
          static_cast<const uint8_t*>(weights_device.get()),
          scales_device.get(), output_device.get(), workspace.get(), nullptr);
    };
    for (int i = 0; i < 3; ++i) ASSERT_EQ(launch(), cudaSuccess) << entry.name;
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) launch();
    cudaEventRecord(stop);
    ASSERT_EQ(cudaEventSynchronize(stop), cudaSuccess);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    const double us = ms * 1000.0 / iters;
    const double flops =
        2.0 * shape.rows * static_cast<double>(channels) * input;
    std::vector<uint16_t> result(static_cast<size_t>(shape.rows) * output);
    ASSERT_EQ(cudaMemcpy(result.data(), output_device.get(), output_bytes,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    // The error against double precision on a sample of rows and channels,
    // relative to the results' RMS (the BF16 output rounding alone is
    // about 1.7e-3).
    double err = 0.0, norm = 0.0;
    for (int row = 0; row < shape.rows; row += 97) {
      for (int n = 0; n < output; n += 37) {
        const auto dot = [&](int channel) {
          const uint8_t* packed =
              weights.data() + static_cast<size_t>(channel) * input / 2;
          double sum = 0.0, block_sum = 0.0;
          for (int k = 0; k < input; ++k) {
            const int nibble = (packed[k / 2] >> ((k & 1) * 4)) & 15;
            const uint32_t w =
                static_cast<uint32_t>(
                    activation[static_cast<size_t>(row) * input + k])
                << 16;
            float a;
            std::memcpy(&a, &w, 4);
            block_sum += static_cast<double>(a) * ((nibble ^ 8) - 8);
            if (blockwise && (k + 1) % 32 == 0) {
              const uint16_t sb =
                  scales[static_cast<size_t>(channel) * blocks + k / 32];
              sum += block_sum * std::ldexp(1.0 + (sb & 0x3ff) / 1024.0,
                                            ((sb >> 10) & 0x1f) - 15);
              block_sum = 0.0;
            }
          }
          if (blockwise) return sum;
          const uint32_t w = static_cast<uint32_t>(scales[channel]) << 16;
          float sv;
          std::memcpy(&sv, &w, 4);
          return block_sum * sv;
        };
        const double x = dot(n);
        const double gelu =
            0.5 * x *
            (1.0 + std::tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
        const double expected = shape.gate != 0 ? gelu * dot(output + n) : x;
        const uint32_t w =
            static_cast<uint32_t>(result[static_cast<size_t>(row) * output + n])
            << 16;
        float v;
        std::memcpy(&v, &w, 4);
        err += (v - expected) * (v - expected);
        norm += expected * expected;
      }
    }
    std::printf("%-40s %10.1f %10.1f %10d %18llx  rel.err %.2e\n", entry.name,
                us, flops / us * 1e-6, LiteRtNvidiaSubbyteGemmBlocks(&shape),
                static_cast<unsigned long long>(Checksum(result)),
                std::sqrt(err / norm));
  }
}

}  // namespace
