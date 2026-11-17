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

#include "litert/vendors/nvidia/trtllm/quant_gemm.h"

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
#include "litert/vendors/nvidia/trtllm/quant_linear.h"
#include "litert/vendors/nvidia/trtllm/quant_linear_reference.h"

namespace {

namespace ref = litert::nvidia::quant_reference;

struct CudaDeleter {
  void operator()(void* pointer) const { cudaFree(pointer); }
};
using CudaAllocation = std::unique_ptr<void, CudaDeleter>;

CudaAllocation AllocateCuda(size_t bytes) {
  void* pointer = nullptr;
  const cudaError_t status = cudaMalloc(&pointer, std::max<size_t>(bytes, 16));
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

class Random {
 public:
  explicit Random(uint32_t seed) : state_(seed) {}
  uint32_t Bits() {
    state_ = state_ * 1664525u + 1013904223u;
    return state_ >> 8;
  }
  // Uniform in [-1, 1].
  float Unit() {
    return static_cast<float>(Bits() & 0xFFFF) / 65535.0f * 2.0f - 1.0f;
  }

 private:
  uint32_t state_;
};

// How the activations of a case are filled.
enum class Values {
  kRandom,
  // Every activation -128 and every weight field the most negative one: the
  // largest sums, which saturate the results.
  kExtreme,
  // Random values with multipliers and a ratio that are powers of two: a
  // result is its sum over a power of two, so that many are ties (n + 0.5),
  // which the requantization rounds away from zero.
  kTies,
};

struct Product {
  const char* name;
  LiteRtNvidiaQuantLinearParams params;
  int m;
  Values values = Values::kRandom;
};

// The bytes of an activation as the kernels read it, `count` values.
std::vector<uint8_t> ActivationBytes(const LiteRtNvidiaQuantLinearParams& p,
                                     size_t count, Values values,
                                     Random* random) {
  std::vector<uint8_t> bytes;
  if (p.activation == kLiteRtNvidiaQuantInt8) {
    bytes.resize(count);
    for (auto& byte : bytes) {
      byte = values == Values::kExtreme ? 0x80
                                        : static_cast<uint8_t>(random->Bits());
    }
    return bytes;
  }
  // Floats past the INT8 range on both sides, so that some saturate.
  const float range = 140.0f / p.quantize_scale;
  const size_t width = p.activation == kLiteRtNvidiaQuantFp32 ? 4 : 2;
  bytes.resize(count * width);
  for (size_t i = 0; i < count; ++i) {
    const float value =
        values == Values::kExtreme ? -range : random->Unit() * range;
    if (p.activation == kLiteRtNvidiaQuantFp32) {
      std::memcpy(bytes.data() + i * 4, &value, 4);
    } else {
      const uint16_t bits = p.activation == kLiteRtNvidiaQuantFp16
                                ? ref::HalfBits(value)
                                : ref::Bf16Bits(value);
      std::memcpy(bytes.data() + i * 2, &bits, 2);
    }
  }
  return bytes;
}

// Runs a product on the row kernels and on the GEMM and expects the same
// bytes.
void ExpectRowKernelsBytes(const Product& product, uint32_t seed) {
  SCOPED_TRACE(::testing::Message() << product.name << " m=" << product.m);
  const LiteRtNvidiaQuantLinearParams& p = product.params;
  ASSERT_TRUE(LiteRtNvidiaQuantLinearSupports(&p));
  ASSERT_TRUE(LiteRtNvidiaQuantGemmSupports(&p, product.m));
  Random random(seed);
  const size_t inputs = static_cast<size_t>(product.m) * p.segments * p.columns;
  const std::vector<uint8_t> activation =
      ActivationBytes(p, inputs, product.values, &random);
  std::vector<uint8_t> weights(static_cast<size_t>(p.rows) * p.columns *
                               p.bit_width / 8);
  for (auto& byte : weights) {
    // INT4 -8 in both fields, INT2 -2 (value -1.5) in all four.
    byte = product.values == Values::kExtreme
               ? (p.bit_width == 4 ? 0x88 : 0xAA)
               : static_cast<uint8_t>(random.Bits());
  }
  // Multipliers that spread the results over the INT8 range, a few of them
  // past it.
  std::vector<float> multipliers(p.rows);
  const float typical =
      p.requant == kLiteRtNvidiaQuantRequantFp32
          ? 70.0f / (340.0f * std::sqrt(static_cast<float>(p.columns)))
          : 140.0f / (165.0f * std::sqrt(static_cast<float>(p.columns)) *
                      static_cast<float>(p.ratio));
  for (auto& multiplier : multipliers) {
    multiplier = typical * (0.25f + 1.5f * (random.Unit() * 0.5f + 0.5f));
    if (product.values == Values::kTies) {
      multiplier = std::ldexp(1.0f, std::ilogb(multiplier));
    }
  }
  std::vector<int8_t> lut(256);
  for (auto& entry : lut) entry = static_cast<int8_t>(random.Bits());

  const auto activation_device = Upload(activation);
  const auto weights_device = Upload(weights);
  const auto multipliers_device = Upload(multipliers);
  const auto lut_device = Upload(lut);
  const size_t outputs = static_cast<size_t>(product.m) * p.rows;
  auto expected_device = AllocateCuda(outputs);
  auto actual_device = AllocateCuda(outputs);
  ASSERT_EQ(cudaMemset(expected_device.get(), 0x55, outputs), cudaSuccess);
  ASSERT_EQ(cudaMemset(actual_device.get(), 0x33, outputs), cudaSuccess);
  auto workspace =
      AllocateCuda(LiteRtNvidiaQuantGemmWorkspaceBytes(&p, product.m));
  const int8_t* lut_pointer =
      p.lut != 0 ? static_cast<const int8_t*>(lut_device.get()) : nullptr;
  ASSERT_EQ(
      LiteRtNvidiaLaunchQuantLinear(
          &p, product.m, activation_device.get(),
          static_cast<const uint8_t*>(weights_device.get()),
          static_cast<const float*>(multipliers_device.get()), lut_pointer,
          /*row_scale=*/nullptr, expected_device.get(), nullptr),
      cudaSuccess);
  ASSERT_EQ(LiteRtNvidiaLaunchQuantGemm(
                &p, product.m, activation_device.get(),
                static_cast<const uint8_t*>(weights_device.get()),
                static_cast<const float*>(multipliers_device.get()),
                lut_pointer, actual_device.get(), workspace.get(), nullptr),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<int8_t> expected(outputs);
  std::vector<int8_t> actual(outputs);
  ASSERT_EQ(cudaMemcpy(expected.data(), expected_device.get(), outputs,
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(actual.data(), actual_device.get(), outputs,
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  size_t different = 0;
  size_t first = outputs;
  for (size_t i = 0; i < outputs; ++i) {
    if (expected[i] != actual[i]) {
      if (different == 0) first = i;
      ++different;
    }
  }
  EXPECT_EQ(different, 0u) << "first at activation row " << first / p.rows
                           << ", output " << first % p.rows
                           << ": the row kernels "
                           << static_cast<int>(expected[first]) << ", the GEMM "
                           << static_cast<int>(actual[first]);
  if (product.values != Values::kExtreme) {
    // The results are spread over the range, not one value or saturated.
    size_t saturated = 0;
    for (const int8_t value : expected) {
      saturated += value == 127 || value == -128 ? 1 : 0;
    }
    if (p.lut == 0) {
      EXPECT_LT(saturated, outputs / 2);
      EXPECT_GT(saturated, 0u);
    }
    EXPECT_NE(*std::min_element(expected.begin(), expected.end()),
              *std::max_element(expected.begin(), expected.end()));
  }
}

LiteRtNvidiaQuantLinearParams Params(int bit_width, int rows, int columns,
                                     int segments, int swap0, int swap1,
                                     int activation, int lut) {
  LiteRtNvidiaQuantLinearParams p = {};
  p.layout = kLiteRtNvidiaQuantLayoutRows;
  p.bit_width = bit_width;
  p.rows = rows;
  p.columns = columns;
  p.segments = segments;
  p.swap0 = swap0;
  p.swap1 = swap1;
  p.activation = activation;
  p.requant = bit_width == 4 ? kLiteRtNvidiaQuantRequantFp32
                             : kLiteRtNvidiaQuantRequantDouble;
  p.lut = lut;
  // fl16(1 / scale) of an activation scale of 0.0163.
  p.quantize_scale = ref::RoundToFp16(1.0f / 0.0163f);
  p.ratio = 1.3172;
  return p;
}

// A statically quantized model's products, at the rows of a prefill chunk.
constexpr int kInt8 = kLiteRtNvidiaQuantInt8;
constexpr int kFp16 = kLiteRtNvidiaQuantFp16;

TEST(QuantGemmTest, ProjectionsHaveRowKernelsBytes) {
  // q, k and v as one product of INT4 weights over a float activation, and
  // the output projection over INT8.
  uint32_t seed = 100;
  for (const Product& product :
       {Product{"qkv fp16", Params(4, 3072, 2048, 1, 1, 1, kFp16, 0), 128},
        Product{"qkv bf16",
                Params(4, 3072, 2048, 1, 1, 1, kLiteRtNvidiaQuantBf16, 0), 256},
        Product{"qkv fp32",
                Params(4, 3072, 2048, 1, 1, 1, kLiteRtNvidiaQuantFp32, 0), 128},
        Product{"out", Params(4, 2048, 2048, 1, 1, 1, kInt8, 0), 384},
        Product{"out, a chunk", Params(4, 2048, 2048, 1, 1, 1, kInt8, 0), 1024},
        Product{"out, extreme", Params(4, 2048, 2048, 1, 1, 1, kInt8, 0), 128,
                Values::kExtreme}}) {
    ExpectRowKernelsBytes(product, seed++);
  }
}

TEST(QuantGemmTest, FactorizedMlpStagesHaveRowKernelsBytes) {
  // The four stages of a factorized MLP with centered INT2 weights: up-1 over
  // the halves of a float or an INT8 activation with its regrouping, up-2 over
  // 32 runs with the SiLU table, down-1 over 32 runs with its regrouping, and
  // down-2 over two halves.
  uint32_t seed = 200;
  for (const Product& product :
       {Product{"up-1 fp16", Params(2, 8192, 1024, 2, 2, 32, kFp16, 0), 128},
        Product{"up-1", Params(2, 8192, 1024, 2, 2, 32, kInt8, 0), 256},
        Product{"up-2", Params(2, 32768, 256, 32, 1, 1, kInt8, 1), 128},
        Product{"up-2, three tiles", Params(2, 32768, 256, 32, 1, 1, kInt8, 1),
                384},
        Product{"down-1", Params(2, 8192, 1024, 32, 32, 2, kInt8, 0), 256},
        Product{"down-2", Params(2, 2048, 4096, 2, 1, 1, kInt8, 0), 128},
        Product{"down-2, a chunk", Params(2, 2048, 4096, 2, 1, 1, kInt8, 0),
                1024},
        Product{"down-2, extreme", Params(2, 2048, 4096, 2, 1, 1, kInt8, 0),
                128, Values::kExtreme},
        Product{"up-2, extreme", Params(2, 32768, 256, 32, 1, 1, kInt8, 1), 128,
                Values::kExtreme}}) {
    ExpectRowKernelsBytes(product, seed++);
  }
}

TEST(QuantGemmTest, TiesRoundAsRowKernelsDo) {
  // Results that are exactly n + 0.5, and results a rounding error away from
  // one: the doubles decide them, as they do for the row kernels.
  uint32_t seed = 400;
  for (double ratio : {1.0, 0.5, 2.0, 1.0000001, 0.99999994}) {
    Product product = {"down-1 ties",
                       Params(2, 8192, 1024, 32, 32, 2, kInt8, 0), 128,
                       Values::kTies};
    product.params.ratio = ratio;
    ExpectRowKernelsBytes(product, seed++);
    Product fp32 = {"out ties", Params(4, 2048, 2048, 1, 1, 1, kInt8, 0), 128,
                    Values::kTies};
    ExpectRowKernelsBytes(fp32, seed++);
  }
}

TEST(QuantGemmTest, RegroupedRunsShorterThanATileHaveRowKernelsBytes) {
  // Runs of 64 outputs: a block's results are two runs of a row's results
  // that the regrouping puts apart, so they leave as single bytes instead of
  // the rows of 16-byte vectors every product above stores.
  ExpectRowKernelsBytes(Product{"up-1, runs of 64",
                                Params(2, 8192, 1024, 2, 2, 64, kInt8, 0), 256},
                        500);
}

TEST(QuantGemmTest, TakesWholeTilesOfRowsLayout) {
  LiteRtNvidiaQuantLinearParams p = Params(4, 2048, 2048, 1, 1, 1, kInt8, 0);
  EXPECT_TRUE(LiteRtNvidiaQuantGemmSupports(&p, 128));
  EXPECT_TRUE(LiteRtNvidiaQuantGemmSupports(&p, 1024));
  // A decode step and rows that are not whole tiles stay on the row kernels.
  EXPECT_FALSE(LiteRtNvidiaQuantGemmSupports(&p, 1));
  EXPECT_FALSE(LiteRtNvidiaQuantGemmSupports(&p, 100));
  EXPECT_FALSE(LiteRtNvidiaQuantGemmSupports(&p, 192));
  EXPECT_EQ(LiteRtNvidiaQuantGemmWorkspaceBytes(&p, 128), 0u);
  p.activation = kFp16;
  EXPECT_EQ(LiteRtNvidiaQuantGemmWorkspaceBytes(&p, 128), 128u * 2048);
  p.columns = 2000;  // not whole chunks of inputs
  EXPECT_FALSE(LiteRtNvidiaQuantGemmSupports(&p, 128));
  p.columns = 2048;
  p.rows = 2056;  // not whole tiles of outputs
  EXPECT_FALSE(LiteRtNvidiaQuantGemmSupports(&p, 128));
  // The head's logits are not INT8 results.
  LiteRtNvidiaQuantLinearParams head =
      Params(2, 262144, 2048, 1, 1, 1, kInt8, 0);
  head.requant = kLiteRtNvidiaQuantRequantLogits;
  EXPECT_FALSE(LiteRtNvidiaQuantGemmSupports(&head, 128));
  // The blocks layout too.
  LiteRtNvidiaQuantLinearParams blocks =
      Params(2, 32768, 256, 32, 1, 1, kInt8, 1);
  blocks.layout = kLiteRtNvidiaQuantLayoutBlocks;
  EXPECT_FALSE(LiteRtNvidiaQuantGemmSupports(&blocks, 128));
}

TEST(QuantGemmTest, UnalignedWeightsGoToRowKernels) {
  // Weights that do not start on 16 bytes: the same bytes all the same.
  const LiteRtNvidiaQuantLinearParams p =
      Params(4, 256, 256, 1, 1, 1, kInt8, 0);
  const int m = 128;
  Random random(300);
  std::vector<uint8_t> activation(static_cast<size_t>(m) * p.columns);
  for (auto& byte : activation) byte = static_cast<uint8_t>(random.Bits());
  std::vector<uint8_t> weights(static_cast<size_t>(p.rows) * p.columns / 2 + 8);
  for (auto& byte : weights) byte = static_cast<uint8_t>(random.Bits());
  const std::vector<float> multipliers(p.rows, 0.002f);
  const auto activation_device = Upload(activation);
  const auto weights_device = Upload(weights);
  const auto multipliers_device = Upload(multipliers);
  const size_t outputs = static_cast<size_t>(m) * p.rows;
  auto expected_device = AllocateCuda(outputs);
  auto actual_device = AllocateCuda(outputs);
  const uint8_t* shifted =
      static_cast<const uint8_t*>(weights_device.get()) + 8;
  ASSERT_EQ(LiteRtNvidiaLaunchQuantLinear(
                &p, m, activation_device.get(), shifted,
                static_cast<const float*>(multipliers_device.get()), nullptr,
                nullptr, expected_device.get(), nullptr),
            cudaSuccess);
  ASSERT_EQ(LiteRtNvidiaLaunchQuantGemm(
                &p, m, activation_device.get(), shifted,
                static_cast<const float*>(multipliers_device.get()), nullptr,
                actual_device.get(), nullptr, nullptr),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<int8_t> expected(outputs);
  std::vector<int8_t> actual(outputs);
  ASSERT_EQ(cudaMemcpy(expected.data(), expected_device.get(), outputs,
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(actual.data(), actual_device.get(), outputs,
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(expected, actual);
}

}  // namespace
