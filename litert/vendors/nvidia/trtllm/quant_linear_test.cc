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

#include "litert/vendors/nvidia/trtllm/quant_linear.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "litert/vendors/nvidia/trtllm/quant_linear_reference.h"

namespace {

namespace ref = litert::nvidia::quant_reference;

// Device memory holding a copy of host values.
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t bytes) : bytes_(bytes) {
    EXPECT_EQ(cudaMalloc(&data_, bytes), cudaSuccess);
    EXPECT_EQ(cudaMemset(data_, 0xA5, bytes), cudaSuccess);
  }
  template <typename T>
  explicit DeviceBuffer(const std::vector<T>& values)
      : DeviceBuffer(values.size() * sizeof(T)) {
    EXPECT_EQ(cudaMemcpy(data_, values.data(), bytes_, cudaMemcpyHostToDevice),
              cudaSuccess);
  }
  ~DeviceBuffer() { cudaFree(data_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  template <typename T>
  T* as() const {
    return static_cast<T*>(data_);
  }
  template <typename T>
  std::vector<T> Download() const {
    std::vector<T> values(bytes_ / sizeof(T));
    EXPECT_EQ(cudaMemcpy(values.data(), data_, bytes_, cudaMemcpyDeviceToHost),
              cudaSuccess);
    return values;
  }

 private:
  void* data_ = nullptr;
  size_t bytes_ = 0;
};

// A small deterministic generator.
class Random {
 public:
  explicit Random(uint64_t seed) : state_(seed * 0x9E3779B97F4A7C15ull + 1) {}
  uint32_t Next() {
    state_ ^= state_ << 13;
    state_ ^= state_ >> 7;
    state_ ^= state_ << 17;
    return static_cast<uint32_t>(state_ >> 32);
  }
  // Uniform in [low, high].
  int Int(int low, int high) {
    return low +
           static_cast<int>(Next() % static_cast<uint32_t>(high - low + 1));
  }
  float Float(float low, float high) {
    return low + (high - low) * (Next() >> 8) / 16777216.0f;
  }

 private:
  uint64_t state_;
};

std::vector<uint8_t> RandomBytes(Random& random, size_t count) {
  std::vector<uint8_t> bytes(count);
  for (auto& byte : bytes) byte = static_cast<uint8_t>(random.Next());
  return bytes;
}

std::vector<int8_t> RandomInt8(Random& random, size_t count, int low = -128,
                               int high = 127) {
  std::vector<int8_t> values(count);
  for (auto& value : values) value = static_cast<int8_t>(random.Int(low, high));
  return values;
}

std::vector<float> RandomFloats(Random& random, size_t count, float low,
                                float high) {
  std::vector<float> values(count);
  for (auto& value : values) value = random.Float(low, high);
  return values;
}

std::vector<float> ToFloats(const std::vector<int8_t>& values) {
  return std::vector<float>(values.begin(), values.end());
}

// Runs the kernel on INT8 activations and compares every output with the
// reference.
void ExpectLinearMatches(const LiteRtNvidiaQuantLinearParams& params, int m,
                         const std::vector<int8_t>& activation,
                         const std::vector<uint8_t>& weights,
                         const std::vector<float>& multipliers,
                         const std::vector<int8_t>& lut = {}) {
  ASSERT_TRUE(LiteRtNvidiaQuantLinearSupports(&params));
  const DeviceBuffer device_activation(activation);
  const DeviceBuffer device_weights(weights);
  const DeviceBuffer device_multipliers(multipliers);
  const DeviceBuffer device_lut(lut.empty() ? std::vector<int8_t>(256) : lut);
  const DeviceBuffer device_output(static_cast<size_t>(m) * params.rows);
  ASSERT_EQ(
      LiteRtNvidiaLaunchQuantLinear(
          &params, m, device_activation.as<void>(),
          device_weights.as<uint8_t>(), device_multipliers.as<float>(),
          device_lut.as<int8_t>(), nullptr, device_output.as<void>(), nullptr),
      cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  const auto actual = device_output.Download<int8_t>();
  const auto expected = ref::Linear(params, m, ToFloats(activation), weights,
                                    multipliers, lut, {});
  size_t mismatches = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    if (actual[i] != static_cast<int>(expected[i]) && ++mismatches <= 5) {
      ADD_FAILURE() << "output " << i << ": " << static_cast<int>(actual[i])
                    << " expected " << expected[i];
    }
  }
  EXPECT_EQ(mismatches, 0u) << "of " << actual.size();
}

LiteRtNvidiaQuantLinearParams RowsParams(int bit_width, int rows, int columns,
                                         int segments) {
  LiteRtNvidiaQuantLinearParams params = {};
  params.layout = kLiteRtNvidiaQuantLayoutRows;
  params.bit_width = bit_width;
  params.rows = rows;
  params.columns = columns;
  params.segments = segments;
  params.swap0 = 1;
  params.swap1 = 1;
  params.activation = kLiteRtNvidiaQuantInt8;
  params.requant = bit_width == 4 ? kLiteRtNvidiaQuantRequantFp32
                                  : kLiteRtNvidiaQuantRequantDouble;
  params.ratio = 1.0;
  return params;
}

// The q, k and v projections: three INT4 matrices of one width, one launch.
// Multipliers of 1/2, 1/4 and 1/8 put many results on a tie, which XNNPACK's
// requantization rounds to even; large ones saturate.
TEST(QuantLinearTest, Int4RowsMatchXnnpackRequantization) {
  Random random(1);
  for (const int columns : {64, 2048}) {
    SCOPED_TRACE(columns);
    const int rows = columns == 64 ? 96 : 3072;
    auto params = RowsParams(4, rows, columns, 1);
    const auto weights =
        RandomBytes(random, static_cast<size_t>(rows) * columns / 2);
    std::vector<float> multipliers(rows);
    for (int r = 0; r < rows; ++r) {
      switch (r % 5) {
        case 0:
          multipliers[r] = 0.5f / (1 << random.Int(0, 2));
          break;
        case 1:
          multipliers[r] = random.Float(0.5f, 4.0f);
          break;
        default:
          multipliers[r] = random.Float(0.001f, 0.02f);
      }
    }
    for (const int m : {1, 3}) {
      ExpectLinearMatches(params, m,
                          RandomInt8(random, static_cast<size_t>(m) * columns),
                          weights, multipliers);
    }
  }
}

// The QUANTIZE in front of a projection runs in the kernel: every float type
// gives what the model's FLOAT16 tensor would.
// Two rows of 256 inputs, and one of 2,048 (a decode step's row: the CTA
// quantizes it once for its warps) with rows that leave a CTA partly empty.
TEST(QuantLinearTest, FloatActivationsAreQuantizedAsFloat16) {
  Random random(2);
  const float scale = 0.04118f;
  struct Shape {
    int rows, columns, m;
  };
  for (const Shape shape : {Shape{32, 256, 2}, Shape{40, 2048, 1}}) {
    for (const int bit_width : {4, 2}) {
      for (const int type : {kLiteRtNvidiaQuantFp16, kLiteRtNvidiaQuantBf16,
                             kLiteRtNvidiaQuantFp32}) {
        SCOPED_TRACE(testing::Message() << bit_width << " bits, type " << type
                                        << ", " << shape.columns << " inputs");
        auto params = RowsParams(bit_width, shape.rows, shape.columns, 1);
        params.activation = type;
        params.quantize_scale = ref::RoundToFp16(1.0f / scale);
        params.ratio = 0.37;
        const auto weights =
            RandomBytes(random, static_cast<size_t>(shape.rows) *
                                    shape.columns * bit_width / 8);
        const auto multipliers =
            RandomFloats(random, shape.rows, 0.002f, 0.03f);
        // Values past the INT8 range too, and exact ties of the quantization.
        const size_t count = static_cast<size_t>(shape.m) * shape.columns;
        std::vector<float> values = RandomFloats(random, count, -6.0f, 6.0f);
        for (int i = 0; i < 16; ++i) {
          values[i] = (i - 8 + 0.5f) / params.quantize_scale;
        }
        std::vector<uint8_t> raw;
        std::vector<float> as_fp16(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
          if (type == kLiteRtNvidiaQuantFp32) {
            as_fp16[i] = ref::RoundToFp16(values[i]);
          } else if (type == kLiteRtNvidiaQuantBf16) {
            const uint16_t bits = ref::Bf16Bits(values[i]);
            values[i] = ref::Bf16Value(bits);
            as_fp16[i] = ref::RoundToFp16(values[i]);
          } else {
            as_fp16[i] = ref::RoundToFp16(values[i]);
          }
        }
        if (type == kLiteRtNvidiaQuantFp32) {
          raw.resize(values.size() * sizeof(float));
          std::memcpy(raw.data(), values.data(), raw.size());
        } else {
          std::vector<uint16_t> halves(values.size());
          for (size_t i = 0; i < values.size(); ++i) {
            halves[i] = type == kLiteRtNvidiaQuantBf16
                            ? ref::Bf16Bits(values[i])
                            : ref::HalfBits(values[i]);
          }
          raw.resize(halves.size() * sizeof(uint16_t));
          std::memcpy(raw.data(), halves.data(), raw.size());
        }
        ASSERT_TRUE(LiteRtNvidiaQuantLinearSupports(&params));
        const DeviceBuffer device_activation(raw);
        const DeviceBuffer device_weights(weights);
        const DeviceBuffer device_multipliers(multipliers);
        const DeviceBuffer device_output(static_cast<size_t>(shape.m) *
                                         shape.rows);
        ASSERT_EQ(
            LiteRtNvidiaLaunchQuantLinear(
                &params, shape.m, device_activation.as<void>(),
                device_weights.as<uint8_t>(), device_multipliers.as<float>(),
                nullptr, nullptr, device_output.as<void>(), nullptr),
            cudaSuccess);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const auto actual = device_output.Download<int8_t>();
        const auto expected =
            ref::Linear(params, shape.m, as_fp16, weights, multipliers, {}, {});
        for (size_t i = 0; i < actual.size(); ++i) {
          ASSERT_EQ(actual[i], static_cast<int>(expected[i])) << i;
        }
      }
    }
  }
}

// The first and the last stage of a factorized MLP: two centered INT2 matrices
// over the halves of the input, the outputs regrouped for the next stage.
// Scales that are powers of two put results on a tie, which TFLite rounds away
// from zero.
TEST(QuantLinearTest, CenteredInt2RowsMatchTfliteBlockwiseKernel) {
  Random random(3);
  struct Shape {
    int rows, columns, segments, swap0, swap1;
  };
  for (const Shape shape : {Shape{64, 48, 2, 2, 4}, Shape{8192, 1024, 2, 2, 32},
                            Shape{2048, 4096, 2, 1, 1}}) {
    SCOPED_TRACE(shape.rows);
    auto params = RowsParams(2, shape.rows, shape.columns, shape.segments);
    params.swap0 = shape.swap0;
    params.swap1 = shape.swap1;
    params.ratio =
        static_cast<double>(0.02716f) / static_cast<double>(0.06822f);
    const auto weights = RandomBytes(
        random, static_cast<size_t>(shape.rows) * shape.columns / 4);
    std::vector<float> multipliers(shape.rows);
    for (int r = 0; r < shape.rows; ++r) {
      multipliers[r] = r % 4 == 0 ? 0.0625f : random.Float(0.01f, 0.08f);
    }
    for (const int m : {1, 2}) {
      ExpectLinearMatches(
          params, m,
          RandomInt8(random,
                     static_cast<size_t>(m) * shape.segments * shape.columns),
          weights, multipliers);
    }
    // A ratio of 1/2 with scales of 1/16: half-integers before the rounding.
    params.ratio = 0.5;
    ExpectLinearMatches(
        params, 1,
        RandomInt8(random, static_cast<size_t>(shape.segments) * shape.columns,
                   -8, 8),
        weights, multipliers);
  }
}

// The middle stages of a factorized MLP: a centered INT2 matrix per block as
// BATCH_MATMUL stores it, [blocks, inputs, outputs], with the SiLU between them
// as a table and the outputs regrouped for the last stage.
TEST(QuantLinearTest, CenteredInt2BlocksMatchTfliteBlockwiseKernel) {
  Random random(4);
  struct Shape {
    int blocks, columns, outputs, swap0, swap1, lut;
  };
  for (const Shape shape :
       {Shape{4, 64, 48, 1, 1, 1}, Shape{4, 160, 32, 4, 2, 0},
        Shape{32, 256, 1024, 1, 1, 1}, Shape{32, 1024, 256, 32, 2, 0}}) {
    SCOPED_TRACE(testing::Message() << shape.columns << "x" << shape.outputs);
    LiteRtNvidiaQuantLinearParams params = {};
    params.layout = kLiteRtNvidiaQuantLayoutBlocks;
    params.bit_width = 2;
    params.rows = shape.blocks * shape.outputs;
    params.columns = shape.columns;
    params.segments = shape.blocks;
    params.swap0 = shape.swap0;
    params.swap1 = shape.swap1;
    params.activation = kLiteRtNvidiaQuantInt8;
    params.requant = kLiteRtNvidiaQuantRequantDouble;
    params.lut = shape.lut;
    params.ratio = static_cast<double>(0.06822f) / static_cast<double>(0.114f);
    const auto weights = RandomBytes(
        random, static_cast<size_t>(params.rows) * shape.columns / 4);
    std::vector<float> multipliers(params.rows);
    for (int r = 0; r < params.rows; ++r) {
      multipliers[r] = r % 4 == 0 ? 0.125f : random.Float(0.02f, 0.09f);
    }
    std::vector<int8_t> lut;
    if (shape.lut != 0) lut = RandomInt8(random, 256);
    for (const int m : {1, 3}) {
      ExpectLinearMatches(params, m,
                          RandomInt8(random, static_cast<size_t>(m) *
                                                 shape.blocks * shape.columns),
                          weights, multipliers, lut);
    }
  }
}

// The vocabulary head: FLOAT16 logits from the row's dynamic scale, then the
// soft cap in FP32.
TEST(QuantLinearTest, LogitsMatchTfliteDynamicRangeKernel) {
  Random random(5);
  constexpr int kRows = 4096;
  constexpr int kColumns = 2048;
  constexpr int kM = 2;
  auto params = RowsParams(2, kRows, kColumns, 1);
  params.requant = kLiteRtNvidiaQuantRequantLogits;
  const auto weights =
      RandomBytes(random, static_cast<size_t>(kRows) * kColumns / 4);
  const auto multipliers = RandomFloats(random, kRows, 0.02f, 0.025f);
  const auto activation =
      RandomInt8(random, static_cast<size_t>(kM) * kColumns, -127, 127);
  const std::vector<float> row_scale = {0.0123f, 0.0456f};
  for (const bool capped : {false, true}) {
    SCOPED_TRACE(capped);
    params.cap_in = capped ? 1.0f / 30.0f : 0.0f;
    params.cap_out = capped ? 30.0f : 0.0f;
    const DeviceBuffer device_activation(activation);
    const DeviceBuffer device_weights(weights);
    const DeviceBuffer device_multipliers(multipliers);
    const DeviceBuffer device_row_scale(row_scale);
    const DeviceBuffer device_output(static_cast<size_t>(kM) * kRows *
                                     sizeof(float));
    ASSERT_EQ(LiteRtNvidiaLaunchQuantLinear(
                  &params, kM, device_activation.as<void>(),
                  device_weights.as<uint8_t>(), device_multipliers.as<float>(),
                  nullptr, device_row_scale.as<float>(),
                  device_output.as<void>(), nullptr),
              cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto actual = device_output.Download<float>();
    const auto expected = ref::Linear(params, kM, ToFloats(activation), weights,
                                      multipliers, {}, row_scale);
    for (size_t i = 0; i < actual.size(); ++i) {
      if (capped) {
        // The device's tanh and the host's differ in the last bits.
        ASSERT_NEAR(actual[i], expected[i],
                    1e-5f * std::fabs(expected[i]) + 1e-6f)
            << i;
      } else {
        ASSERT_EQ(actual[i], expected[i]) << i;
      }
    }
  }
}

struct GlueCase {
  int width;
  int x_type;
  int branch;
  int norm;
  int quantize;
  int z_type;
};

// The float steps between two products: the residual update behind an INT8
// projection result, the next norm and its quantization, each rounded to
// FLOAT16 where the graph holds one.
TEST(QuantLinearTest, GlueMatchesReference) {
  Random random(6);
  for (const GlueCase c :
       {GlueCase{2048, kLiteRtNvidiaQuantFp16, 1, 1,
                 kLiteRtNvidiaQuantGlueStatic, kLiteRtNvidiaQuantFp16},
        GlueCase{2048, kLiteRtNvidiaQuantFp32, 0, 1,
                 kLiteRtNvidiaQuantGlueStatic, kLiteRtNvidiaQuantFp16},
        GlueCase{2048, kLiteRtNvidiaQuantFp16, 1, 1,
                 kLiteRtNvidiaQuantGlueDynamic, kLiteRtNvidiaQuantFp32},
        GlueCase{96, kLiteRtNvidiaQuantBf16, 1, 0, kLiteRtNvidiaQuantGlueStatic,
                 kLiteRtNvidiaQuantFp16},
        GlueCase{4096, kLiteRtNvidiaQuantFp16, 0, 0, kLiteRtNvidiaQuantGlueNone,
                 kLiteRtNvidiaQuantFp32}}) {
    SCOPED_TRACE(testing::Message()
                 << "width " << c.width << " type " << c.x_type << " branch "
                 << c.branch << " norm " << c.norm << " quantize "
                 << c.quantize);
    constexpr int kM = 3;
    LiteRtNvidiaQuantGlueParams params = {};
    params.width = c.width;
    params.x_type = c.x_type;
    params.branch = c.branch;
    params.norm = c.norm;
    params.quantize = c.quantize;
    params.z_type = c.z_type;
    params.branch_scale = ref::RoundToFp16(0.05521f);
    params.branch_eps = 1e-6f;
    params.c_x = ref::RoundToFp16(0.796f);
    params.c_y = ref::RoundToFp16(0.6055f);
    params.norm_eps = 1e-6f;
    params.quantize_scale = ref::RoundToFp16(1.0f / 0.02716f);
    ASSERT_TRUE(LiteRtNvidiaQuantGlueSupports(&params));
    const size_t total = static_cast<size_t>(kM) * c.width;
    std::vector<float> x = RandomFloats(random, total, -3.0f, 3.0f);
    // The last row is all zero: its dynamic scale is 1.
    if (c.branch == 0) std::fill(x.end() - c.width, x.end(), 0.0f);
    std::vector<uint8_t> raw_x;
    std::vector<float> x_fp16(total);
    if (c.x_type == kLiteRtNvidiaQuantFp32) {
      raw_x.resize(total * sizeof(float));
      std::memcpy(raw_x.data(), x.data(), raw_x.size());
      for (size_t i = 0; i < total; ++i) x_fp16[i] = ref::RoundToFp16(x[i]);
    } else {
      std::vector<uint16_t> halves(total);
      for (size_t i = 0; i < total; ++i) {
        if (c.x_type == kLiteRtNvidiaQuantBf16) {
          halves[i] = ref::Bf16Bits(x[i]);
          x_fp16[i] = ref::RoundToFp16(ref::Bf16Value(halves[i]));
        } else {
          halves[i] = ref::HalfBits(x[i]);
          x_fp16[i] = ref::HalfValue(halves[i]);
        }
      }
      raw_x.resize(total * sizeof(uint16_t));
      std::memcpy(raw_x.data(), halves.data(), raw_x.size());
    }
    const auto y = RandomInt8(random, total);
    const auto branch_weights = RandomFloats(random, c.width, 0.5f, 2.0f);
    const auto norm_weights = RandomFloats(random, c.width, 0.5f, 2.0f);
    const DeviceBuffer device_x(raw_x);
    const DeviceBuffer device_y(y);
    const DeviceBuffer device_branch_weights(branch_weights);
    const DeviceBuffer device_norm_weights(norm_weights);
    const DeviceBuffer device_x_out(total * sizeof(uint16_t));
    const DeviceBuffer device_z(total * sizeof(float));
    const DeviceBuffer device_q(total);
    const DeviceBuffer device_row_scale(kM * sizeof(float));
    const bool quantized = c.quantize != kLiteRtNvidiaQuantGlueNone;
    ASSERT_EQ(
        LiteRtNvidiaLaunchQuantGlue(
            &params, kM, device_x.as<void>(), device_y.as<int8_t>(),
            device_branch_weights.as<float>(), device_norm_weights.as<float>(),
            device_x_out.as<void>(), device_z.as<void>(),
            quantized ? device_q.as<int8_t>() : nullptr,
            device_row_scale.as<float>(), nullptr),
        cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto expected =
        ref::Glue(params, kM, x_fp16, y, branch_weights, norm_weights);
    const auto x_out = device_x_out.Download<uint16_t>();
    const auto q = device_q.Download<int8_t>();
    for (size_t i = 0; i < total; ++i) {
      ASSERT_EQ(x_out[i], ref::HalfBits(expected.x[i])) << "x' " << i;
      if (quantized) {
        ASSERT_EQ(q[i], expected.q[i]) << "q " << i;
      }
    }
    if (c.z_type == kLiteRtNvidiaQuantFp32) {
      const auto z = device_z.Download<float>();
      for (size_t i = 0; i < total; ++i) {
        ASSERT_EQ(z[i], expected.z[i]) << "z " << i;
      }
    } else {
      const auto z = device_z.Download<uint16_t>();
      for (size_t i = 0; i < total; ++i) {
        ASSERT_EQ(z[i], ref::HalfBits(expected.z[i])) << "z " << i;
      }
    }
    if (c.quantize == kLiteRtNvidiaQuantGlueDynamic) {
      const auto row_scale = device_row_scale.Download<float>();
      for (int row = 0; row < kM; ++row) {
        EXPECT_EQ(row_scale[row], expected.row_scale[row]) << row;
      }
    }
  }
}

struct GlueLinearCase {
  int x_type;
  int branch;
  int bit_width;
  int rows;
  int columns;
  int segments;
  int swap0;
  int swap1;
  bool lut;
};

// The glue and the product behind it in one launch: the glue's x', z and
// the product's INT8 results of the two launches, against the references.
TEST(QuantLinearTest, GlueLinearMatchesTheTwoLaunches) {
  Random random(9);
  // The q, k and v projections behind a residual update; the first block's,
  // behind the norm of a FLOAT32 input; the first stage of a factorized MLP
  // over the halves, regrouped; a product with a table.
  for (const GlueLinearCase c :
       {GlueLinearCase{kLiteRtNvidiaQuantFp16, 1, 4, 3072, 2048, 1, 1, 1,
                       false},
        GlueLinearCase{kLiteRtNvidiaQuantFp32, 0, 4, 256, 2048, 1, 1, 1, false},
        GlueLinearCase{kLiteRtNvidiaQuantFp16, 1, 2, 1024, 1024, 2, 2, 8,
                       false},
        GlueLinearCase{kLiteRtNvidiaQuantBf16, 1, 2, 512, 256, 1, 1, 1,
                       true}}) {
    for (const int m : {1, 3}) {
      SCOPED_TRACE(testing::Message()
                   << c.bit_width << " bits, " << c.rows << " x " << c.columns
                   << ", branch " << c.branch << ", m " << m);
      const int width = c.segments * c.columns;
      LiteRtNvidiaQuantGlueParams glue = {};
      glue.width = width;
      glue.x_type = c.x_type;
      glue.branch = c.branch;
      glue.norm = 1;
      glue.quantize = kLiteRtNvidiaQuantGlueStatic;
      glue.z_type = kLiteRtNvidiaQuantFp16;
      glue.branch_scale = ref::RoundToFp16(0.05521f);
      glue.branch_eps = 1e-6f;
      glue.c_x = ref::RoundToFp16(0.796f);
      glue.c_y = ref::RoundToFp16(0.6055f);
      glue.norm_eps = 1e-6f;
      glue.quantize_scale = ref::RoundToFp16(1.0f / 0.02716f);
      auto params = RowsParams(c.bit_width, c.rows, c.columns, c.segments);
      params.swap0 = c.swap0;
      params.swap1 = c.swap1;
      params.lut = c.lut ? 1 : 0;
      params.ratio = 0.37;
      ASSERT_TRUE(LiteRtNvidiaQuantGlueLinearSupports(&glue, &params));
      const size_t total = static_cast<size_t>(m) * width;
      const auto x = RandomFloats(random, total, -3.0f, 3.0f);
      std::vector<uint8_t> raw_x;
      std::vector<float> x_fp16(total);
      if (c.x_type == kLiteRtNvidiaQuantFp32) {
        raw_x.resize(total * sizeof(float));
        std::memcpy(raw_x.data(), x.data(), raw_x.size());
        for (size_t i = 0; i < total; ++i) x_fp16[i] = ref::RoundToFp16(x[i]);
      } else {
        std::vector<uint16_t> halves(total);
        for (size_t i = 0; i < total; ++i) {
          halves[i] = c.x_type == kLiteRtNvidiaQuantBf16 ? ref::Bf16Bits(x[i])
                                                         : ref::HalfBits(x[i]);
          x_fp16[i] = c.x_type == kLiteRtNvidiaQuantBf16
                          ? ref::RoundToFp16(ref::Bf16Value(halves[i]))
                          : ref::HalfValue(halves[i]);
        }
        raw_x.resize(total * sizeof(uint16_t));
        std::memcpy(raw_x.data(), halves.data(), raw_x.size());
      }
      const auto y = RandomInt8(random, total);
      const auto branch_weights = RandomFloats(random, width, 0.5f, 2.0f);
      const auto norm_weights = RandomFloats(random, width, 0.5f, 2.0f);
      const auto weights = RandomBytes(
          random, static_cast<size_t>(c.rows) * c.columns * c.bit_width / 8);
      const auto multipliers =
          c.bit_width == 4 ? RandomFloats(random, c.rows, 0.0005f, 0.004f)
                           : RandomFloats(random, c.rows, 0.001f, 0.01f);
      const auto lut = RandomInt8(random, 256);
      const DeviceBuffer device_x(raw_x);
      const DeviceBuffer device_y(y);
      const DeviceBuffer device_branch_weights(branch_weights);
      const DeviceBuffer device_norm_weights(norm_weights);
      const DeviceBuffer device_weights(weights);
      const DeviceBuffer device_multipliers(multipliers);
      const DeviceBuffer device_lut(lut);
      const DeviceBuffer device_x_out(total * sizeof(uint16_t));
      const DeviceBuffer device_z(total * sizeof(uint16_t));
      const DeviceBuffer device_output(static_cast<size_t>(m) * c.rows);
      ASSERT_EQ(
          LiteRtNvidiaLaunchQuantGlueLinear(
              &glue, &params, m, device_x.as<void>(), device_y.as<int8_t>(),
              device_branch_weights.as<float>(),
              device_norm_weights.as<float>(), device_weights.as<uint8_t>(),
              device_multipliers.as<float>(), device_lut.as<int8_t>(),
              device_x_out.as<void>(), device_z.as<void>(),
              device_output.as<void>(), nullptr),
          cudaSuccess);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      const auto expected_glue =
          ref::Glue(glue, m, x_fp16, y, branch_weights, norm_weights);
      const auto expected = ref::Linear(
          params, m,
          std::vector<float>(expected_glue.q.begin(), expected_glue.q.end()),
          weights, multipliers, c.lut ? lut : std::vector<int8_t>(), {});
      const auto x_out = device_x_out.Download<uint16_t>();
      const auto z = device_z.Download<uint16_t>();
      for (size_t i = 0; i < total; ++i) {
        ASSERT_EQ(x_out[i], ref::HalfBits(expected_glue.x[i])) << "x' " << i;
        ASSERT_EQ(z[i], ref::HalfBits(expected_glue.z[i])) << "z " << i;
      }
      const auto actual = device_output.Download<int8_t>();
      for (size_t i = 0; i < actual.size(); ++i) {
        ASSERT_EQ(actual[i], static_cast<int>(expected[i])) << "output " << i;
      }
      // The two launches give the same bytes.
      const DeviceBuffer device_q(total);
      const DeviceBuffer device_two(static_cast<size_t>(m) * c.rows);
      ASSERT_EQ(LiteRtNvidiaLaunchQuantGlue(
                    &glue, m, device_x.as<void>(), device_y.as<int8_t>(),
                    device_branch_weights.as<float>(),
                    device_norm_weights.as<float>(), nullptr, nullptr,
                    device_q.as<int8_t>(), nullptr, nullptr),
                cudaSuccess);
      ASSERT_EQ(LiteRtNvidiaLaunchQuantLinear(&params, m, device_q.as<void>(),
                                              device_weights.as<uint8_t>(),
                                              device_multipliers.as<float>(),
                                              device_lut.as<int8_t>(), nullptr,
                                              device_two.as<void>(), nullptr),
                cudaSuccess);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      EXPECT_EQ(device_two.Download<int8_t>(), actual);
    }
  }
}

// A factorized MLP's four products on one row in one launch, against the host
// reference of the four: a 3B model's shapes (the halves of 1,024 inputs ->
// 8,192 rows regrouped into 32 blocks -> 1,024 a block through a table -> 256 a
// block regrouped into halves -> 2,048) and small ones whose rows are not a
// multiple of 16 bytes (CTAs that own no rows of a stage). Three launches on
// one state, which each leaves zero, then the four launches that run without a
// state; then the same with the glue in front (a residual update and the norm
// of a FLOAT16 x), whose x' and z are checked too.
TEST(QuantLinearTest, FactorizedMlpMatchesTheFourProducts) {
  struct Shape {
    int c, r1, blocks, n, d, r4;
  };
  for (const Shape shape :
       {Shape{1024, 4096, 32, 1024, 128, 1024}, Shape{32, 64, 4, 48, 16, 32}}) {
    SCOPED_TRACE(shape.c);
    Random random(41 + shape.c);
    LiteRtNvidiaQuantFactorizedMlpParams params = {};
    params.stages[0] = RowsParams(2, 2 * shape.r1, shape.c, 2);
    params.stages[0].swap0 = 2;
    params.stages[0].swap1 = shape.blocks;
    params.stages[0].ratio = 0.21;
    params.stages[1] = RowsParams(2, shape.blocks * shape.n,
                                  2 * shape.r1 / shape.blocks, shape.blocks);
    params.stages[1].lut = 1;
    params.stages[1].ratio = 0.33;
    params.stages[2] =
        RowsParams(2, shape.blocks * 2 * shape.d, shape.n, shape.blocks);
    params.stages[2].swap0 = shape.blocks;
    params.stages[2].swap1 = 2;
    params.stages[2].ratio = 0.27;
    params.stages[3] = RowsParams(2, 2 * shape.r4, shape.blocks * shape.d, 2);
    params.stages[3].ratio = 0.19;
    ASSERT_TRUE(LiteRtNvidiaQuantFactorizedMlpSupports(&params));
    const auto x = RandomInt8(random, 2 * shape.c);
    const auto lut = RandomInt8(random, 256);
    std::vector<std::vector<uint8_t>> weights;
    std::vector<std::vector<float>> multipliers;
    std::vector<float> expected = ToFloats(x);
    for (const auto& stage : params.stages) {
      weights.push_back(RandomBytes(
          random, static_cast<size_t>(stage.rows) * stage.columns / 4));
      multipliers.push_back(RandomFloats(random, stage.rows, 0.002f, 0.01f));
      expected =
          ref::Linear(stage, 1, expected, weights.back(), multipliers.back(),
                      stage.lut != 0 ? lut : std::vector<int8_t>(), {});
    }
    const DeviceBuffer device_x(x);
    const DeviceBuffer device_lut(lut);
    const DeviceBuffer w0(weights[0]), w1(weights[1]), w2(weights[2]),
        w3(weights[3]);
    const DeviceBuffer m0(multipliers[0]), m1(multipliers[1]),
        m2(multipliers[2]), m3(multipliers[3]);
    const uint8_t* const device_weights[4] = {
        w0.as<uint8_t>(), w1.as<uint8_t>(), w2.as<uint8_t>(), w3.as<uint8_t>()};
    const float* const device_multipliers[4] = {m0.as<float>(), m1.as<float>(),
                                                m2.as<float>(), m3.as<float>()};
    const DeviceBuffer workspace(
        LiteRtNvidiaQuantFactorizedMlpWorkspaceBytes(&params));
    const DeviceBuffer state(
        std::vector<uint8_t>(LiteRtNvidiaQuantFactorizedMlpStateBytes(), 0));
    const DeviceBuffer output(static_cast<size_t>(params.stages[3].rows));
    for (int launch = 0; launch < 4; ++launch) {
      SCOPED_TRACE(launch);
      ASSERT_EQ(LiteRtNvidiaLaunchQuantFactorizedMlp(
                    &params, device_x.as<int8_t>(), device_weights,
                    device_multipliers, device_lut.as<int8_t>(),
                    launch < 3 ? state.as<void>() : nullptr,
                    workspace.as<void>(), output.as<int8_t>(), nullptr),
                cudaSuccess);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      const auto actual = output.Download<int8_t>();
      size_t mismatches = 0;
      for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != static_cast<int>(expected[i]) && ++mismatches <= 5) {
          ADD_FAILURE() << "output " << i << ": " << static_cast<int>(actual[i])
                        << " expected " << expected[i];
        }
      }
      EXPECT_EQ(mismatches, 0u) << "of " << actual.size();
      for (const uint8_t byte : state.Download<uint8_t>()) EXPECT_EQ(byte, 0);
    }

    LiteRtNvidiaQuantGlueParams glue = {};
    glue.width = 2 * shape.c;
    glue.x_type = kLiteRtNvidiaQuantFp16;
    glue.branch = 1;
    glue.norm = 1;
    glue.quantize = kLiteRtNvidiaQuantGlueStatic;
    glue.z_type = kLiteRtNvidiaQuantFp16;
    glue.branch_scale = ref::RoundToFp16(0.05521f);
    glue.branch_eps = 1e-6f;
    glue.c_x = ref::RoundToFp16(0.796f);
    glue.c_y = ref::RoundToFp16(0.6055f);
    glue.norm_eps = 1e-6f;
    glue.quantize_scale = ref::RoundToFp16(1.0f / 0.02716f);
    ASSERT_TRUE(LiteRtNvidiaQuantGlueFactorizedMlpSupports(&glue, &params));
    std::vector<uint16_t> halves(glue.width);
    std::vector<float> x_fp16(glue.width);
    for (int i = 0; i < glue.width; ++i) {
      halves[i] = ref::HalfBits(random.Float(-3.0f, 3.0f));
      x_fp16[i] = ref::HalfValue(halves[i]);
    }
    const auto y = RandomInt8(random, glue.width);
    const auto branch_weights = RandomFloats(random, glue.width, 0.5f, 2.0f);
    const auto norm_weights = RandomFloats(random, glue.width, 0.5f, 2.0f);
    const auto expected_glue =
        ref::Glue(glue, 1, x_fp16, y, branch_weights, norm_weights);
    std::vector<float> glued(expected_glue.q.begin(), expected_glue.q.end());
    for (int s = 0; s < 4; ++s) {
      glued = ref::Linear(
          params.stages[s], 1, glued, weights[s], multipliers[s],
          params.stages[s].lut != 0 ? lut : std::vector<int8_t>(), {});
    }
    const DeviceBuffer device_halves(halves);
    const DeviceBuffer device_y(y);
    const DeviceBuffer device_branch_weights(branch_weights);
    const DeviceBuffer device_norm_weights(norm_weights);
    const DeviceBuffer x_out(glue.width * sizeof(uint16_t));
    const DeviceBuffer z_out(glue.width * sizeof(uint16_t));
    for (int launch = 0; launch < 2; ++launch) {
      SCOPED_TRACE(testing::Message() << "glue, launch " << launch);
      ASSERT_EQ(
          LiteRtNvidiaLaunchQuantGlueFactorizedMlp(
              &glue, &params, device_halves.as<void>(), device_y.as<int8_t>(),
              device_branch_weights.as<float>(),
              device_norm_weights.as<float>(), device_weights,
              device_multipliers, device_lut.as<int8_t>(),
              launch == 0 ? state.as<void>() : nullptr, workspace.as<void>(),
              x_out.as<void>(), z_out.as<void>(), output.as<int8_t>(), nullptr),
          cudaSuccess);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      const auto x_values = x_out.Download<uint16_t>();
      const auto z_values = z_out.Download<uint16_t>();
      for (int i = 0; i < glue.width; ++i) {
        ASSERT_EQ(x_values[i], ref::HalfBits(expected_glue.x[i])) << "x' " << i;
        ASSERT_EQ(z_values[i], ref::HalfBits(expected_glue.z[i])) << "z " << i;
      }
      const auto actual = output.Download<int8_t>();
      for (size_t i = 0; i < actual.size(); ++i) {
        ASSERT_EQ(actual[i], static_cast<int>(glued[i])) << "output " << i;
      }
      for (const uint8_t byte : state.Download<uint8_t>()) EXPECT_EQ(byte, 0);
    }
  }
}

// The head norms and the rotation between the projections and the attention.
// Values are bit-identical; rotated queries and keys differ from the host
// where the device's sine or cosine rounds to the other FLOAT16 neighbour.
TEST(QuantLinearTest, QkvRopeMatchesReference) {
  Random random(7);
  struct Heads {
    int q, k, v;
  };
  // A cached block, a block that reads another block's cache, and a block
  // that only writes its cache.
  for (const Heads heads : {Heads{8, 2, 2}, Heads{8, 0, 0}, Heads{0, 2, 2}}) {
    SCOPED_TRACE(testing::Message() << heads.q << " " << heads.k);
    LiteRtNvidiaQuantQkvRopeParams params = {};
    params.tokens = 5;
    params.q_heads = heads.q;
    params.k_heads = heads.k;
    params.v_heads = heads.v;
    params.depth = 256;
    params.q_scale = ref::RoundToFp16(0.1162f);
    params.k_scale = ref::RoundToFp16(0.1423f);
    params.v_scale = ref::RoundToFp16(0.1423f);
    params.epsilon = 1e-6f;
    params.q_multiplier = 0.125f;
    ASSERT_TRUE(LiteRtNvidiaQuantQkvRopeSupports(&params));
    const int total_heads = heads.q + heads.k + heads.v;
    const auto qkv = RandomInt8(random, static_cast<size_t>(params.tokens) *
                                            total_heads * params.depth);
    const std::vector<int32_t> positions = {0, 1, 511, 4097, 32770};
    const auto q_norm = RandomFloats(random, params.depth, 0.5f, 2.0f);
    const auto k_norm = RandomFloats(random, params.depth, 0.5f, 2.0f);
    const auto v_norm = RandomFloats(random, params.depth, 0.5f, 2.0f);
    // A proportion of 1/4: the pairs past the first 32 keep their values.
    auto inverse = ref::InverseTimescales(params.depth, 1.0f, 1000000.0f);
    if (heads.q == 0) std::fill(inverse.begin() + 32, inverse.end(), 0.0f);
    const size_t q_count =
        static_cast<size_t>(heads.q) * params.tokens * params.depth;
    const size_t kv_count =
        static_cast<size_t>(heads.k) * params.tokens * params.depth;
    const DeviceBuffer device_qkv(qkv);
    const DeviceBuffer device_positions(positions);
    const DeviceBuffer device_q_norm(q_norm);
    const DeviceBuffer device_k_norm(k_norm);
    const DeviceBuffer device_v_norm(v_norm);
    const DeviceBuffer device_inverse(inverse);
    const DeviceBuffer device_q((q_count + 1) * sizeof(uint16_t));
    const DeviceBuffer device_k((kv_count + 1) * sizeof(uint16_t));
    const DeviceBuffer device_v((kv_count + 1) * sizeof(uint16_t));
    ASSERT_EQ(LiteRtNvidiaLaunchQuantQkvRope(
                  &params, device_qkv.as<int8_t>(),
                  device_positions.as<int32_t>(), device_q_norm.as<float>(),
                  device_k_norm.as<float>(), device_v_norm.as<float>(),
                  device_inverse.as<float>(), device_q.as<void>(),
                  device_k.as<void>(), device_v.as<void>(), nullptr),
              cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto expected =
        ref::QkvRope(params, qkv, positions, q_norm, k_norm, v_norm, inverse);
    const auto expect_rotated = [](const std::vector<uint16_t>& actual,
                                   const std::vector<float>& reference,
                                   size_t count) {
      size_t different = 0;
      for (size_t i = 0; i < count; ++i) {
        if (actual[i] == ref::HalfBits(reference[i])) continue;
        ++different;
        ASSERT_NEAR(ref::HalfValue(actual[i]), reference[i],
                    4e-3f * std::max(1.0f, std::fabs(reference[i])))
            << i;
      }
      EXPECT_LE(different, count / 100) << "of " << count;
    };
    expect_rotated(device_q.Download<uint16_t>(), expected.q, q_count);
    expect_rotated(device_k.Download<uint16_t>(), expected.k, kv_count);
    const auto v = device_v.Download<uint16_t>();
    for (size_t i = 0; i < kv_count; ++i) {
      ASSERT_EQ(v[i], ref::HalfBits(expected.v[i])) << i;
    }
  }
}

TEST(QuantLinearTest, RejectsUnsupportedShapes) {
  auto params = RowsParams(4, 96, 64, 1);
  EXPECT_TRUE(LiteRtNvidiaQuantLinearSupports(&params));
  params.rows = 100;  // not whole CTAs of eight rows
  EXPECT_FALSE(LiteRtNvidiaQuantLinearSupports(&params));
  params = RowsParams(4, 96, 60, 1);  // not whole 32-bit words
  EXPECT_FALSE(LiteRtNvidiaQuantLinearSupports(&params));
  params = RowsParams(2, 96, 64, 1);
  params.requant =
      kLiteRtNvidiaQuantRequantFp32;  // INT2 requantizes in doubles
  EXPECT_FALSE(LiteRtNvidiaQuantLinearSupports(&params));
  params = RowsParams(2, 96, 64, 1);
  params.swap0 = 5;
  EXPECT_FALSE(LiteRtNvidiaQuantLinearSupports(&params));
  params = RowsParams(4, 96, 64, 1);
  EXPECT_EQ(LiteRtNvidiaLaunchQuantLinear(&params, 1, nullptr, nullptr, nullptr,
                                          nullptr, nullptr, nullptr, nullptr),
            cudaErrorInvalidValue);
  LiteRtNvidiaQuantGlueParams glue = {};
  glue.width = 8192;
  glue.x_type = kLiteRtNvidiaQuantFp16;
  glue.z_type = kLiteRtNvidiaQuantFp16;
  EXPECT_FALSE(LiteRtNvidiaQuantGlueSupports(&glue));
  glue.width = 2048;
  glue.norm = 1;
  glue.quantize = kLiteRtNvidiaQuantGlueStatic;
  glue.quantize_scale = 1.0f;
  params = RowsParams(4, 96, 2048, 1);
  EXPECT_TRUE(LiteRtNvidiaQuantGlueLinearSupports(&glue, &params));
  glue.quantize = kLiteRtNvidiaQuantGlueDynamic;  // the head's: two launches
  EXPECT_FALSE(LiteRtNvidiaQuantGlueLinearSupports(&glue, &params));
  glue.quantize = kLiteRtNvidiaQuantGlueStatic;
  glue.width = 4096;
  params = RowsParams(4, 96, 4096, 1);  // rows of 512 words
  EXPECT_FALSE(LiteRtNvidiaQuantGlueLinearSupports(&glue, &params));
  LiteRtNvidiaQuantQkvRopeParams rope = {};
  rope.tokens = 1;
  rope.q_heads = 8;
  rope.depth = 255;
  rope.q_scale = 1.0f;
  EXPECT_FALSE(LiteRtNvidiaQuantQkvRopeSupports(&rope));
}

}  // namespace
