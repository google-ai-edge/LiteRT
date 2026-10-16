/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensor/backends/ynnpack/arithmetic.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/runners/ynnpack/runner.h"
#include "tensor/tensor.h"
#include "tensor/utils/matchers.h"

namespace litert::tensor {
namespace {

using ::testing::FloatNear;
using ::testing::Pointwise;

using YnnTensor = Tensor<YnnpackMixinTag>;

int BitsOf(Type type) {
  switch (type) {
    case Type::kI2:
      return 2;
    case Type::kI4:
      return 4;
    default:
      return 8;
  }
}

// Packs signed `values` as `type` (int8, int4 or int2), low bits first.
std::shared_ptr<Buffer> PackWeights(Type type,
                                    absl::Span<const int8_t> values) {
  std::shared_ptr<OwningCpuBuffer> buffer;
  switch (type) {
    case Type::kI2:
      buffer = OwningCpuBuffer::Allocate<Type::kI2>(values.size());
      break;
    case Type::kI4:
      buffer = OwningCpuBuffer::Allocate<Type::kI4>(values.size());
      break;
    default:
      buffer = OwningCpuBuffer::Allocate<Type::kI8>(values.size());
      break;
  }
  const int bits = BitsOf(type);
  const int per_byte = 8 / bits;
  const uint8_t mask = static_cast<uint8_t>((1 << bits) - 1);
  std::vector<uint8_t> packed((values.size() + per_byte - 1) / per_byte, 0);
  for (size_t i = 0; i < values.size(); ++i) {
    const uint8_t v = static_cast<uint8_t>(values[i]) & mask;
    packed[i / per_byte] |= static_cast<uint8_t>(v << (bits * (i % per_byte)));
  }
  size_t i = 0;
  for (std::byte& b : *buffer) {
    b = i < packed.size() ? static_cast<std::byte>(packed[i]) : std::byte{0};
    ++i;
  }
  return buffer;
}

struct QuantizedFcCase {
  Type type;
  // Number of output channels.
  int n;
  // 0 for per-channel quantization.
  int block_size;
  bool has_bias;
  FusedActivation activation;
};

std::string CaseName(const QuantizedFcCase& c) {
  return absl::StrCat("type=", BitsOf(c.type), "bit n=", c.n,
                      " block_size=", c.block_size, " has_bias=", c.has_bias,
                      " relu=", c.activation == kActRelu);
}

// Runs a fully connected op with symmetrically quantized weights and checks it
// against a float reference.
void CheckQuantizedFullyConnected(const QuantizedFcCase& c) {
  constexpr int kM = 3;
  constexpr int kK = 16;
  const int n_channels = c.n;

  std::vector<float> input_data(kM * kK);
  for (int i = 0; i < kM * kK; ++i) {
    input_data[i] = std::sin(0.37f * i) * 2.0f;
  }
  std::vector<float> bias_data(n_channels);
  for (int n = 0; n < n_channels; ++n) {
    bias_data[n] = 0.25f * n - 0.5f;
  }

  // Integer weights spanning the range of the type.
  const int bits = BitsOf(c.type);
  const int lowest = -(1 << (bits - 1));
  const int range = 1 << bits;
  std::vector<int8_t> quantized(n_channels * kK);
  for (int i = 0; i < quantized.size(); ++i) {
    quantized[i] = static_cast<int8_t>(lowest + (i * 7 + 3) % range);
  }

  const int scales_per_row = c.block_size == 0 ? 1 : kK / c.block_size;
  std::vector<float> scales(n_channels * scales_per_row);
  for (int i = 0; i < scales.size(); ++i) {
    scales[i] = (0.5f + 0.125f * (i % 5)) / (range / 2);
  }
  std::shared_ptr<Quantization> quantization;
  if (c.block_size == 0) {
    quantization = std::make_shared<PerChannelAffineQuantization>(
        scales, std::vector<int64_t>(n_channels, 0),
        /*quantized_dimension=*/0);
  } else {
    quantization = std::make_shared<BlockwiseQuantization>(
        scales, std::vector<int64_t>(scales.size(), 0), c.block_size,
        /*quantized_dimension=*/0);
  }

  YnnTensor input({.name = "input", .type = Type::kFP32, .shape = {kM, kK}});
  YnnTensor weights({.name = "weights",
                     .type = c.type,
                     .shape = {n_channels, kK},
                     .buffer = PackWeights(c.type, quantized),
                     .quantization = quantization});
  std::optional<YnnTensor> bias;
  if (c.has_bias) {
    bias = YnnTensor({.name = "bias",
                      .type = Type::kFP32,
                      .shape = {n_channels},
                      .buffer = bias_data});
  }
  YnnTensor output = FullyConnected(input, weights, bias, c.activation);

  // Float reference.
  std::vector<float> expected(kM * n_channels);
  float max_abs = 0.0f;
  for (int m = 0; m < kM; ++m) {
    for (int n = 0; n < n_channels; ++n) {
      float sum = c.has_bias ? bias_data[n] : 0.0f;
      for (int k = 0; k < kK; ++k) {
        const int scale_index =
            c.block_size == 0 ? n : n * scales_per_row + k / c.block_size;
        sum += input_data[m * kK + k] * quantized[n * kK + k] *
               scales[scale_index];
      }
      if (c.activation == kActRelu) {
        sum = std::max(sum, 0.0f);
      }
      expected[m * n_channels + n] = sum;
      max_abs = std::max(max_abs, std::abs(sum));
    }
  }

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto runner, YnnpackRunner::Create({output}));
  ASSERT_THAT(runner.SetInputAsCopy(input, input_data), IsOk());
  ASSERT_THAT(runner.Run(), IsOk());

  // The input is dynamically quantized to int8, hence the loose tolerance.
  EXPECT_THAT(runner.ReadOutputAs<float>(output),
              IsOkAndHolds(Pointwise(FloatNear(2e-2f * max_abs), expected)));
}

TEST(ArithmeticYnnpackTest, FullyConnectedWithQuantizedWeightsWorks) {
  for (const QuantizedFcCase& c : {
           // Per-channel quantization.
           QuantizedFcCase{Type::kI8, 8, 0, false, kActNone},
           QuantizedFcCase{Type::kI8, 8, 0, true, kActNone},
           QuantizedFcCase{Type::kI4, 8, 0, true, kActRelu},
           QuantizedFcCase{Type::kI2, 8, 0, true, kActNone},
           // Blockwise quantization.
           QuantizedFcCase{Type::kI8, 8, 8, false, kActNone},
           QuantizedFcCase{Type::kI8, 8, 4, true, kActNone},
           QuantizedFcCase{Type::kI4, 8, 8, false, kActNone},
           QuantizedFcCase{Type::kI4, 8, 4, true, kActRelu},
           QuantizedFcCase{Type::kI2, 8, 8, true, kActNone},
           // Sub-byte channels that don't fill whole bytes once transposed use
           // the dequantizing lowering.
           QuantizedFcCase{Type::kI8, 5, 0, true, kActNone},
           QuantizedFcCase{Type::kI4, 5, 0, true, kActNone},
           QuantizedFcCase{Type::kI4, 5, 8, true, kActNone},
       }) {
    SCOPED_TRACE(CaseName(c));
    CheckQuantizedFullyConnected(c);
  }
}

// Zero points of a quantized test matrix.
enum class ZeroPoints { kNone, kSingle, kPerScale };

// A quantized rank 2 constant [rows, cols], together with its dequantized
// values for the float references.
struct QuantizedMatrix {
  Type type;
  int rows;
  int cols;
  std::vector<int8_t> values;
  std::shared_ptr<Quantization> quantization;
  // Row-major, [rows, cols].
  std::vector<float> dequantized;

  float At(int r, int c) const { return dequantized[r * cols + c]; }

  YnnTensor ToTensor() const {
    return YnnTensor({.name = "weights",
                      .type = type,
                      .shape = {rows, cols},
                      .buffer = PackWeights(type, values),
                      .quantization = quantization});
  }
};

// Builds a quantized [rows, cols] matrix whose channels are
// `quantized_dimension`. `block_size` 0 means per-channel quantization, and
// `per_tensor` uses a single scale instead.
QuantizedMatrix MakeQuantizedMatrix(
    Type type, int rows, int cols, int quantized_dimension, int block_size,
    bool per_tensor = false, ZeroPoints zero_points = ZeroPoints::kNone) {
  QuantizedMatrix m{.type = type, .rows = rows, .cols = cols};
  const int bits = BitsOf(type);
  const int lowest = -(1 << (bits - 1));
  const int range = 1 << bits;
  m.values.resize(rows * cols);
  for (int i = 0; i < m.values.size(); ++i) {
    m.values[i] = static_cast<int8_t>(lowest + (i * 7 + 3) % range);
  }

  const int channels = quantized_dimension == 0 ? rows : cols;
  const int blocked = quantized_dimension == 0 ? cols : rows;
  const int num_blocks = block_size == 0 ? 1 : blocked / block_size;
  const int num_scales = per_tensor ? 1 : channels * num_blocks;
  std::vector<float> scales(num_scales);
  for (int i = 0; i < num_scales; ++i) {
    scales[i] = (0.5f + 0.125f * (i % 5)) / (range / 2);
  }
  std::vector<int64_t> zps;
  switch (zero_points) {
    case ZeroPoints::kNone:
      break;
    case ZeroPoints::kSingle:
      zps = {1};
      break;
    case ZeroPoints::kPerScale:
      for (int i = 0; i < num_scales; ++i) {
        zps.push_back(i % 3 - 1);
      }
      break;
  }

  m.dequantized.resize(rows * cols);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      const int channel = quantized_dimension == 0 ? r : c;
      const int position = quantized_dimension == 0 ? c : r;
      int index = 0;
      if (!per_tensor) {
        index = block_size == 0 ? channel
                                : channel * num_blocks + position / block_size;
      }
      const int64_t zp = zps.empty()       ? 0
                         : zps.size() == 1 ? zps[0]
                                           : zps[index];
      m.dequantized[r * cols + c] =
          (m.values[r * cols + c] - zp) * scales[index];
    }
  }

  if (block_size == 0) {
    m.quantization = std::make_shared<PerChannelAffineQuantization>(
        scales, zps, quantized_dimension);
  } else {
    m.quantization = std::make_shared<BlockwiseQuantization>(
        scales, zps, block_size, quantized_dimension);
  }
  return m;
}

std::vector<float> MakeInput(int size) {
  std::vector<float> data(size);
  for (int i = 0; i < size; ++i) {
    data[i] = std::sin(0.37f * i) * 2.0f;
  }
  return data;
}

// Returns `a[rows, K] . b` where `b(k, n)` returns the element of the
// right-hand side, and the max absolute value of the result.
template <class B>
std::pair<std::vector<float>, float> MatMul(absl::Span<const float> a, int rows,
                                            int k_size, int n_size, B b) {
  std::vector<float> result(rows * n_size);
  float max_abs = 0.f;
  for (int m = 0; m < rows; ++m) {
    for (int n = 0; n < n_size; ++n) {
      float sum = 0.f;
      for (int k = 0; k < k_size; ++k) {
        sum += a[m * k_size + k] * b(k, n);
      }
      result[m * n_size + n] = sum;
      max_abs = std::max(max_abs, std::abs(sum));
    }
  }
  return {result, max_abs};
}

// The input of the quantized dots is dynamically quantized to int8.
auto QuantizedDotNear(absl::Span<const float> expected, float max_abs) {
  return IsOkAndHolds(
      Pointwise(FloatNear(2e-2f * max_abs + 1e-5f),
                std::vector<float>(expected.begin(), expected.end())));
}

// The fully connected input may have any rank; the rows are all but its last
// dimension.
TEST(ArithmeticYnnpackTest, QuantizedFullyConnectedWithRank3Input) {
  constexpr int kK = 16;
  constexpr int kN = 8;
  for (const int block_size : {0, 8}) {
    SCOPED_TRACE(absl::StrCat("block_size=", block_size));
    const QuantizedMatrix w = MakeQuantizedMatrix(
        Type::kI4, kN, kK, /*quantized_dimension=*/0, block_size);
    YnnTensor input(
        {.name = "input", .type = Type::kFP32, .shape = {2, 3, kK}});
    YnnTensor output = FullyConnected(input, w.ToTensor());

    const std::vector<float> input_data = MakeInput(2 * 3 * kK);
    const auto [expected, max_abs] = MatMul(
        input_data, 2 * 3, kK, kN, [&](int k, int n) { return w.At(n, k); });

    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto runner,
                                    YnnpackRunner::Create({output}));
    ASSERT_THAT(runner.SetInputAsCopy(input, input_data), IsOk());
    ASSERT_THAT(runner.Run(), IsOk());
    EXPECT_THAT(runner.ReadOutputAs<float>(output),
                QuantizedDotNear(expected, max_abs));
  }
}

// A decode graph (1 row) is reshaped into a prefill graph (several rows).
TEST(ArithmeticYnnpackTest, QuantizedFullyConnectedAfterReshapeInput) {
  constexpr int kK = 16;
  constexpr int kN = 8;
  for (const int block_size : {0, 8}) {
    SCOPED_TRACE(absl::StrCat("block_size=", block_size));
    const QuantizedMatrix w = MakeQuantizedMatrix(
        Type::kI4, kN, kK, /*quantized_dimension=*/0, block_size);
    YnnTensor input({.name = "input", .type = Type::kFP32, .shape = {1, kK}});
    YnnTensor output = FullyConnected(input, w.ToTensor());

    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto runner,
                                    YnnpackRunner::Create({output}));
    for (const int rows : {1, 5}) {
      SCOPED_TRACE(absl::StrCat("rows=", rows));
      const std::vector<float> input_data = MakeInput(rows * kK);
      const auto [expected, max_abs] = MatMul(
          input_data, rows, kK, kN, [&](int k, int n) { return w.At(n, k); });
      ASSERT_THAT(runner.ReshapeInput(input, {rows, kK}), IsOk());
      ASSERT_THAT(runner.SetInputAsCopy(input, input_data), IsOk());
      ASSERT_THAT(runner.Run(), IsOk());
      EXPECT_THAT(runner.ReadOutputAs<float>(output),
                  QuantizedDotNear(expected, max_abs));
    }
  }
}

struct QuantizationCase {
  std::string name;
  Type type;
  int n;
  int block_size;
  bool per_tensor;
  ZeroPoints zero_points;
};

// Per-tensor (F4) and asymmetric (F5) weights, on the quantized dot (N = 8) and
// on the dequantizing lowering (int4 with N = 5, which doesn't fill whole bytes
// once transposed).
TEST(ArithmeticYnnpackTest, FullyConnectedWithPerTensorAndAsymmetricWeights) {
  constexpr int kM = 3;
  constexpr int kK = 16;
  for (const QuantizationCase& c : {
           QuantizationCase{"per-tensor i8", Type::kI8, 8, 0, true,
                            ZeroPoints::kNone},
           QuantizationCase{"per-tensor i4", Type::kI4, 8, 0, true,
                            ZeroPoints::kNone},
           QuantizationCase{"per-tensor i4 dequantized", Type::kI4, 5, 0, true,
                            ZeroPoints::kNone},
           QuantizationCase{"per-tensor asymmetric i8", Type::kI8, 8, 0, true,
                            ZeroPoints::kSingle},
           QuantizationCase{"asymmetric single zp i8", Type::kI8, 8, 0, false,
                            ZeroPoints::kSingle},
           QuantizationCase{"asymmetric per-channel zp i8", Type::kI8, 8, 0,
                            false, ZeroPoints::kPerScale},
           QuantizationCase{"asymmetric per-channel zp i4", Type::kI4, 8, 0,
                            false, ZeroPoints::kPerScale},
           QuantizationCase{"asymmetric i4 dequantized", Type::kI4, 5, 0, false,
                            ZeroPoints::kPerScale},
           QuantizationCase{"asymmetric blockwise i4 dequantized", Type::kI4, 8,
                            8, false, ZeroPoints::kPerScale},
       }) {
    SCOPED_TRACE(c.name);
    const QuantizedMatrix w =
        MakeQuantizedMatrix(c.type, c.n, kK, /*quantized_dimension=*/0,
                            c.block_size, c.per_tensor, c.zero_points);
    YnnTensor input({.name = "input", .type = Type::kFP32, .shape = {kM, kK}});
    YnnTensor output = FullyConnected(input, w.ToTensor());

    const std::vector<float> input_data = MakeInput(kM * kK);
    const auto [expected, max_abs] = MatMul(
        input_data, kM, kK, c.n, [&](int k, int n) { return w.At(n, k); });

    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto runner,
                                    YnnpackRunner::Create({output}));
    ASSERT_THAT(runner.SetInputAsCopy(input, input_data), IsOk());
    ASSERT_THAT(runner.Run(), IsOk());
    EXPECT_THAT(runner.ReadOutputAs<float>(output),
                QuantizedDotNear(expected, max_abs));
  }
}

struct BatchMatMulCase {
  std::string name;
  Type type;
  int block_size;
  bool adj_y;
};

// A quantized rank 2 right-hand side of a batch matmul goes through the
// quantized dot (F6): [N, K] with `adj_y` and channels on dimension 0, [K, N]
// otherwise with channels on dimension 1 (F2 for blockwise weights).
TEST(ArithmeticYnnpackTest, BatchMatMulWithQuantizedRhs) {
  constexpr int kK = 16;
  constexpr int kN = 8;
  for (const BatchMatMulCase& c : {
           BatchMatMulCase{"per-channel i8 adj_y", Type::kI8, 0, true},
           BatchMatMulCase{"per-channel i4 adj_y", Type::kI4, 0, true},
           BatchMatMulCase{"blockwise i4 adj_y", Type::kI4, 8, true},
           BatchMatMulCase{"per-channel i8", Type::kI8, 0, false},
           BatchMatMulCase{"per-channel i4", Type::kI4, 0, false},
           BatchMatMulCase{"blockwise i8", Type::kI8, 4, false},
           BatchMatMulCase{"blockwise i4", Type::kI4, 8, false},
       }) {
    SCOPED_TRACE(c.name);
    const QuantizedMatrix w =
        c.adj_y ? MakeQuantizedMatrix(c.type, kN, kK, /*quantized_dimension=*/0,
                                      c.block_size)
                : MakeQuantizedMatrix(c.type, kK, kN, /*quantized_dimension=*/1,
                                      c.block_size);
    YnnTensor lhs({.name = "lhs", .type = Type::kFP32, .shape = {2, 3, kK}});
    YnnTensor output = BatchMatMul(lhs, w.ToTensor(), /*adj_x=*/false, c.adj_y);

    const std::vector<float> lhs_data = MakeInput(2 * 3 * kK);
    const auto [expected, max_abs] =
        MatMul(lhs_data, 2 * 3, kK, kN,
               [&](int k, int n) { return c.adj_y ? w.At(n, k) : w.At(k, n); });

    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto runner,
                                    YnnpackRunner::Create({output}));
    ASSERT_THAT(runner.SetInputAsCopy(lhs, lhs_data), IsOk());
    ASSERT_THAT(runner.Run(), IsOk());
    EXPECT_THAT(runner.ReadOutputAs<float>(output),
                QuantizedDotNear(expected, max_abs));
  }
}

struct TransposeCase {
  std::string name;
  Type type;
  int n;
  int block_size;
};

// The Tensor API keeps the quantization of a transposed tensor (F1). The
// backend transposes the integers and the batch matmul consuming them uses the
// quantized dot. Sub-byte weights (int4) whose transposed innermost dimension
// doesn't fill whole bytes (odd N) are transposed as fp32 instead.
TEST(ArithmeticYnnpackTest, TransposedQuantizedWeightsIntoBatchMatMul) {
  constexpr int kK = 16;
  for (const TransposeCase& c : {
           TransposeCase{"per-channel i8", Type::kI8, 8, 0},
           TransposeCase{"blockwise i8", Type::kI8, 8, 8},
           TransposeCase{"per-channel i8 odd N", Type::kI8, 5, 0},
           TransposeCase{"per-channel i4", Type::kI4, 8, 0},
           TransposeCase{"blockwise i4", Type::kI4, 8, 8},
           TransposeCase{"per-channel i4 odd N fp32 fallback", Type::kI4, 5, 0},
       }) {
    SCOPED_TRACE(c.name);
    const QuantizedMatrix w = MakeQuantizedMatrix(
        c.type, c.n, kK, /*quantized_dimension=*/0, c.block_size);
    YnnTensor lhs({.name = "lhs", .type = Type::kFP32, .shape = {3, kK}});
    // [N, K] -> [K, N].
    YnnTensor transposed = Transpose(w.ToTensor(), std::vector<int>{1, 0});
    YnnTensor output = BatchMatMul(lhs, transposed);

    const std::vector<float> lhs_data = MakeInput(3 * kK);
    const auto [expected, max_abs] =
        MatMul(lhs_data, 3, kK, c.n, [&](int k, int n) { return w.At(n, k); });

    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto runner,
                                    YnnpackRunner::Create({output}));
    ASSERT_THAT(runner.SetInputAsCopy(lhs, lhs_data), IsOk());
    ASSERT_THAT(runner.Run(), IsOk());
    EXPECT_THAT(runner.ReadOutputAs<float>(output),
                QuantizedDotNear(expected, max_abs));
  }
}

// Consumers that aren't dots read the dequantized values of transposed weights
// (F1), whether the integers were transposed or the fp32 values (int4 with odd
// N).
TEST(ArithmeticYnnpackTest, TransposedQuantizedWeightsIntoNonDotConsumer) {
  constexpr int kK = 16;
  for (const Type type : {Type::kI4, Type::kI8}) {
    for (const int n_channels : {8, 5}) {
      for (const int block_size : {0, 8}) {
        for (const ZeroPoints zp : {ZeroPoints::kNone, ZeroPoints::kPerScale}) {
          SCOPED_TRACE(absl::StrCat(
              "bits=", BitsOf(type), " n=", n_channels,
              " block_size=", block_size,
              " zero_points=", zp == ZeroPoints::kNone ? "none" : "per-scale"));
          const QuantizedMatrix w = MakeQuantizedMatrix(
              type, n_channels, kK, /*quantized_dimension=*/0, block_size,
              /*per_tensor=*/false, zp);
          // [N, K] -> [K, N].
          YnnTensor transposed =
              Transpose(w.ToTensor(), std::vector<int>{1, 0});
          YnnTensor x(
              {.name = "x", .type = Type::kFP32, .shape = {kK, n_channels}});
          YnnTensor output = Add(Dequantize(transposed), x);

          const std::vector<float> x_data = MakeInput(kK * n_channels);
          std::vector<float> expected(kK * n_channels);
          for (int k = 0; k < kK; ++k) {
            for (int n = 0; n < n_channels; ++n) {
              expected[k * n_channels + n] =
                  x_data[k * n_channels + n] + w.At(n, k);
            }
          }

          LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto runner,
                                          YnnpackRunner::Create({output}));
          ASSERT_THAT(runner.SetInputAsCopy(x, x_data), IsOk());
          ASSERT_THAT(runner.Run(), IsOk());
          EXPECT_THAT(runner.ReadOutputAs<float>(output),
                      IsOkAndHolds(Pointwise(FloatNear(1e-5f), expected)));
        }
      }
    }
  }
}

// [K, N] weights with their channels on dimension 1 (F2) that can't use the
// quantized dot (here because of `adj_x`) are dequantized.
TEST(ArithmeticYnnpackTest, QuantizedWeightsWithChannelsOnDim1Dequantized) {
  constexpr int kM = 3;
  constexpr int kK = 16;
  constexpr int kN = 8;
  for (const int block_size : {0, 4, 8}) {
    SCOPED_TRACE(absl::StrCat("block_size=", block_size));
    const QuantizedMatrix w = MakeQuantizedMatrix(
        Type::kI4, kK, kN, /*quantized_dimension=*/1, block_size,
        /*per_tensor=*/false, ZeroPoints::kPerScale);
    // [K, M], transposed by `adj_x`.
    YnnTensor lhs({.name = "lhs", .type = Type::kFP32, .shape = {kK, kM}});
    YnnTensor output =
        BatchMatMul(lhs, w.ToTensor(), /*adj_x=*/true, /*adj_y=*/false);

    const std::vector<float> lhs_data = MakeInput(kK * kM);
    std::vector<float> lhs_t(kM * kK);
    for (int k = 0; k < kK; ++k) {
      for (int m = 0; m < kM; ++m) {
        lhs_t[m * kK + k] = lhs_data[k * kM + m];
      }
    }
    const auto [expected, max_abs] =
        MatMul(lhs_t, kM, kK, kN, [&](int k, int n) { return w.At(k, n); });

    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto runner,
                                    YnnpackRunner::Create({output}));
    ASSERT_THAT(runner.SetInputAsCopy(lhs, lhs_data), IsOk());
    ASSERT_THAT(runner.Run(), IsOk());
    EXPECT_THAT(
        runner.ReadOutputAs<float>(output),
        IsOkAndHolds(Pointwise(FloatNear(1e-4f * max_abs + 1e-5f), expected)));
  }
}

// A quantized constant consumed both by a quantized dot, which reads its
// integer data, and by an op that reads its dequantized values.
TEST(ArithmeticYnnpackTest, QuantizedWeightsSharedByDotAndNonDotConsumer) {
  constexpr int kM = 3;
  constexpr int kK = 16;
  constexpr int kN = 8;
  for (const Type type : {Type::kI4, Type::kI8}) {
    for (const int block_size : {0, 8}) {
      SCOPED_TRACE(
          absl::StrCat("bits=", BitsOf(type), " block_size=", block_size));
      const QuantizedMatrix w = MakeQuantizedMatrix(
          type, kN, kK, /*quantized_dimension=*/0, block_size);
      YnnTensor weights = w.ToTensor();
      YnnTensor input(
          {.name = "input", .type = Type::kFP32, .shape = {kM, kK}});
      YnnTensor x({.name = "x", .type = Type::kFP32, .shape = {kN, kK}});
      YnnTensor dot = FullyConnected(input, weights);
      YnnTensor sum = Add(Dequantize(weights), x);

      const std::vector<float> input_data = MakeInput(kM * kK);
      const auto [expected_dot, max_abs] = MatMul(
          input_data, kM, kK, kN, [&](int k, int n) { return w.At(n, k); });
      const std::vector<float> x_data = MakeInput(kN * kK);
      std::vector<float> expected_sum(kN * kK);
      for (int i = 0; i < expected_sum.size(); ++i) {
        expected_sum[i] = x_data[i] + w.dequantized[i];
      }

      LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto runner,
                                      YnnpackRunner::Create({dot, sum}));
      ASSERT_THAT(runner.SetInputAsCopy(input, input_data), IsOk());
      ASSERT_THAT(runner.SetInputAsCopy(x, x_data), IsOk());
      ASSERT_THAT(runner.Run(), IsOk());
      EXPECT_THAT(runner.ReadOutputAs<float>(dot),
                  QuantizedDotNear(expected_dot, max_abs));
      EXPECT_THAT(runner.ReadOutputAs<float>(sum),
                  IsOkAndHolds(Pointwise(FloatNear(1e-5f), expected_sum)));
    }
  }
}

}  // namespace
}  // namespace litert::tensor
