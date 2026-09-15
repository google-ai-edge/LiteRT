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

#include "tensor/examples/gemma4/helpers/float_activation_fully_connected.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/runners/xnnpack/runner.h"
#include "tensor/tensor.h"
#include "tensor/utils/matchers.h"

namespace litert::tensor::examples::gemma4 {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using XnnTensor = Tensor<XnnpackMixinTag>;

XnnTensor MakeWeights(Type type, const std::vector<int8_t>& values,
                      std::shared_ptr<Quantization> quantization) {
  std::shared_ptr<Buffer> buffer;
  if (type == Type::kI4) {
    std::vector<uint8_t> packed(values.size() / 2);
    for (size_t i = 0; i < packed.size(); ++i) {
      packed[i] = (static_cast<uint8_t>(values[2 * i]) & 0xF) |
                  ((static_cast<uint8_t>(values[2 * i + 1]) & 0xF) << 4);
    }
    buffer = OwningCpuBuffer::Copy<Type::kU8>(packed);
  } else {
    buffer = OwningCpuBuffer::Copy<Type::kI8>(values);
  }
  return XnnTensor({.name = "weights",
                    .type = type,
                    .shape = {3, 8},
                    .buffer = buffer,
                    .quantization = std::move(quantization)});
}

std::shared_ptr<PerChannelAffineQuantization> MakeQuantization() {
  return std::make_shared<PerChannelAffineQuantization>(
      std::vector<float>{0.03125f, 0.015625f, 0.0078125f},
      std::vector<int64_t>{0}, /*quantized_dimension=*/0);
}

class FloatActivationFullyConnectedTest
    : public ::testing::TestWithParam<Type> {};

// Models a buffer whose size metadata overstates the CPU-accessible span.
class OverreportedSizeBuffer : public SpanCpuBuffer {
 public:
  OverreportedSizeBuffer(const std::vector<uint8_t>& bytes,
                         size_t reported_size)
      : SpanCpuBuffer(bytes), reported_size_(reported_size) {}

  absl::StatusOr<size_t> ByteSize() const override { return reported_size_; }

 private:
  size_t reported_size_;
};

TEST_P(FloatActivationFullyConnectedTest,
       MatchesDequantizedDotWithoutInputRounding) {
  std::vector<int8_t> values = {
      0, 7, 0,  0, 0, 0,  0, 0,  -8, 7, -3, 2,
      0, 1, -1, 5, 4, -6, 2, -8, 7,  0, 1,  -2,
  };
  if (GetParam() == Type::kI8) {
    for (int8_t& value : values) value *= 15;
  }
  auto quantization = MakeQuantization();
  XnnTensor weights = MakeWeights(GetParam(), values, quantization);
  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {1, 2, 8}});
  XnnTensor output = FloatActivationFullyConnected(input, weights);
  ASSERT_THAT(output.GetStatus(), IsOk());
  EXPECT_THAT(output.GetShape(), ElementsAre(1, 2, 3));
  EXPECT_EQ(output.GetType(), Type::kFP32);
  // The small second channel is lost by INT8 dynamic input quantization,
  // while the first output isolates its contribution to the dot product.
  const std::vector<float> input_values = {
      1.234567f, 0.00091f,   -0.314159f, 2.003141f, -1.10101f, 0.271828f,
      0.987654f, -0.777777f, -1.876543f, -0.00113f, 0.123456f, -2.12345f,
      1.414213f, -0.333333f, 0.618034f,  0.707107f,
  };
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(XnnpackRunner runner,
                                  XnnpackRunner::Create({output}));
  ASSERT_THAT(runner.SetInput(input, input_values), IsOk());
  ASSERT_THAT(runner.Run(), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto actual,
                                  runner.ReadOutputAs<float>(output));
  ASSERT_EQ(actual.size(), 6);
  for (int batch = 0; batch < 2; ++batch) {
    for (int channel = 0; channel < 3; ++channel) {
      double expected = 0.0;
      for (int i = 0; i < 8; ++i) {
        expected += static_cast<double>(input_values[batch * 8 + i]) *
                    values[channel * 8 + i] * quantization->scales[channel];
      }
      EXPECT_NEAR(actual.data()[batch * 3 + channel], expected, 1e-5)
          << "batch=" << batch << " channel=" << channel;
    }
  }
}

TEST_P(FloatActivationFullyConnectedTest, RejectsTruncatedConstantBuffer) {
  const size_t required_bytes = GetParam() == Type::kI4 ? 12 : 24;
  XnnTensor weights =
      MakeWeights(GetParam(), std::vector<int8_t>(24), MakeQuantization());
  weights.SetBuffer(OwningCpuBuffer::Copy<Type::kU8>(
      std::vector<uint8_t>(required_bytes - 1)));
  XnnTensor input({.type = Type::kFP32, .shape = {1, 8}});
  EXPECT_THAT(FloatActivationFullyConnected(input, weights).GetStatus(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       ::testing::HasSubstr("weight buffer is too small")));
}

TEST_P(FloatActivationFullyConnectedTest, RejectsNullLockedConstantData) {
  const size_t required_bytes = GetParam() == Type::kI4 ? 12 : 24;
  XnnTensor weights =
      MakeWeights(GetParam(), std::vector<int8_t>(24), MakeQuantization());
  weights.SetBuffer(std::make_shared<SpanCpuBuffer>(nullptr, required_bytes));
  XnnTensor input({.type = Type::kFP32, .shape = {1, 8}});
  EXPECT_THAT(FloatActivationFullyConnected(input, weights).GetStatus(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       ::testing::HasSubstr("could not lock weight data")));
}

TEST_P(FloatActivationFullyConnectedTest, RejectsShortLockWithLargerMetadata) {
  const size_t required_bytes = GetParam() == Type::kI4 ? 12 : 24;
  std::vector<uint8_t> bytes(required_bytes - 1);
  XnnTensor weights =
      MakeWeights(GetParam(), std::vector<int8_t>(24), MakeQuantization());
  weights.SetBuffer(
      std::make_shared<OverreportedSizeBuffer>(bytes, required_bytes));
  XnnTensor input({.type = Type::kFP32, .shape = {1, 8}});
  EXPECT_THAT(
      FloatActivationFullyConnected(input, weights).GetStatus(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               ::testing::HasSubstr("locked weight span is too small")));
}

TEST_P(FloatActivationFullyConnectedTest,
       RejectsBufferTruncatedBeforeLowering) {
  const size_t required_bytes = GetParam() == Type::kI4 ? 12 : 24;
  XnnTensor weights =
      MakeWeights(GetParam(), std::vector<int8_t>(24), MakeQuantization());
  XnnTensor input({.type = Type::kFP32, .shape = {1, 8}});
  XnnTensor output = FloatActivationFullyConnected(input, weights);
  ASSERT_THAT(output.GetStatus(), IsOk());
  // Tensor handles share metadata, so graph lowering must recheck storage
  // changed after the expression passed its initial validation.
  weights.SetBuffer(OwningCpuBuffer::Copy<Type::kU8>(
      std::vector<uint8_t>(required_bytes - 1)));
  EXPECT_THAT(XnnpackRunner::Create({output}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       ::testing::HasSubstr("weight buffer is too small")));
}

INSTANTIATE_TEST_SUITE_P(QuantizedWeights, FloatActivationFullyConnectedTest,
                         ::testing::Values(Type::kI4, Type::kI8));

TEST(FloatActivationFullyConnectedValidationTest, RejectsOddInt4InputChannels) {
  // Native FP32 qc4w kernels do not handle odd input channels consistently
  // across architectures. All supported Gemma model dimensions are even.
  XnnTensor weights(
      {.type = Type::kI4,
       .shape = {3, 3},
       .buffer =
           OwningCpuBuffer::Copy<Type::kU8>({0xE1, 0xC3, 0xA5, 0x87, 0x00}),
       .quantization = std::make_shared<PerChannelAffineQuantization>(
           std::vector<float>{0.25f, 0.5f, 0.125f}, std::vector<int64_t>{0},
           0)});
  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {3}});
  XnnTensor output = FloatActivationFullyConnected(input, weights);
  EXPECT_THAT(output.GetStatus(), StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(FloatActivationFullyConnectedValidationTest, RejectsInvalidInputAndShape) {
  XnnTensor weights =
      MakeWeights(Type::kI4, std::vector<int8_t>(24), MakeQuantization());
  for (const auto& init : {
           TensorInit{.type = Type::kI8, .shape = {1, 8}},
           TensorInit{.type = Type::kFP32, .shape = {}},
           TensorInit{.type = Type::kFP32, .shape = {1, 7}},
       }) {
    EXPECT_THAT(
        FloatActivationFullyConnected(XnnTensor(init), weights).GetStatus(),
        StatusIs(absl::StatusCode::kInvalidArgument));
  }
  XnnTensor input({.type = Type::kFP32, .shape = {8}});
  XnnTensor output = FloatActivationFullyConnected(input, weights);
  EXPECT_THAT(output.GetStatus(), IsOk());
  EXPECT_THAT(output.GetShape(), ElementsAre(3));
}

TEST(FloatActivationFullyConnectedValidationTest, RejectsInvalidQuantization) {
  XnnTensor input({.type = Type::kFP32, .shape = {1, 8}});
  for (int variant = 0; variant < 5; ++variant) {
    auto quantization = MakeQuantization();
    switch (variant) {
      case 0:
        quantization->scales.resize(1);
        break;
      case 1:
        quantization->quantized_dimension = 1;
        break;
      case 2:
        quantization->zero_points[0] = 1;
        break;
      case 3:
        quantization->scales[0] = 0.0f;
        break;
      case 4:
        quantization->scales[0] = std::numeric_limits<float>::quiet_NaN();
        break;
    }
    XnnTensor weights =
        MakeWeights(Type::kI4, std::vector<int8_t>(24), quantization);
    EXPECT_THAT(FloatActivationFullyConnected(input, weights).GetStatus(),
                StatusIs(absl::StatusCode::kInvalidArgument))
        << variant;
  }
}

}  // namespace
}  // namespace litert::tensor::examples::gemma4
