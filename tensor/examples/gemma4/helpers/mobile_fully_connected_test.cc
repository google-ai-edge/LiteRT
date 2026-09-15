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

#include "tensor/examples/gemma4/helpers/mobile_fully_connected.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "tensor/backends/xnnpack/arithmetic.h"
#include "tensor/runners/xnnpack/runner.h"
#include "tensor/utils/matchers.h"

namespace litert::tensor::examples::gemma4 {
namespace {
using XnnTensor = Tensor<XnnpackMixinTag>;

class MobileFullyConnectedTest : public testing::TestWithParam<Type> {};

TEST_P(MobileFullyConnectedTest, StaticScalesRoundingAndSaturation) {
  constexpr int channels = 32;
  XnnTensor input({.type = Type::kFP32, .shape = {3, channels}});
  std::vector<int8_t> integer_weights(2 * channels);
  for (int i = 0; i < channels; ++i) {
    integer_weights[i] = i % 7 - 3;
    integer_weights[channels + i] = i % 5 - 2;
  }
  std::shared_ptr<Buffer> buffer;
  if (GetParam() == Type::kI4) {
    std::vector<uint8_t> packed(channels);
    for (int i = 0; i < 2 * channels; i += 2) {
      packed[i / 2] =
          (integer_weights[i] & 15) | ((integer_weights[i + 1] & 15) << 4);
    }
    buffer = OwningCpuBuffer::Copy<Type::kU8>(packed);
  } else {
    buffer = OwningCpuBuffer::Copy<Type::kI8>(integer_weights);
  }
  const std::vector<float> weight_scales = {0.25f, 0.5f};
  XnnTensor weight(
      {.name = "linear.weight",
       .type = GetParam(),
       .shape = {2, channels},
       .buffer = buffer,
       .quantization = std::make_shared<PerChannelAffineQuantization>(
           weight_scales, std::vector<int64_t>{0}, 0)});
  absl::flat_hash_map<std::string, XnnTensor> weights;
  weights.emplace(
      "linear.input_scale",
      XnnTensor({.type = Type::kFP32, .shape = {}, .buffer = 0.25f}));
  weights.emplace(
      "linear.output_scale",
      XnnTensor({.type = Type::kFP32, .shape = {}, .buffer = 0.5f}));
  auto output = MobileFullyConnected(input, weight, &weights);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto runner, XnnpackRunner::Create({output}));
  std::vector<float> values(3 * channels);
  for (int i = 0; i < channels; ++i) {
    values[i] = (i % 13 - 6) * 0.31f;
    values[channels + i] = integer_weights[i] >= 0 ? 100.0f : -100.0f;
    values[2 * channels + i] = -values[channels + i];
  }
  ASSERT_THAT(runner.SetInput(input, values), IsOk());
  ASSERT_THAT(runner.Run(), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto actual,
                                  runner.ReadOutputAs<float>(output));
  ASSERT_EQ(actual.size(), 6);
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 2; ++col) {
      int32_t accumulator = 0;
      for (int k = 0; k < channels; ++k) {
        int q = std::clamp(static_cast<int>(std::nearbyint(
                               values[row * channels + k] / 0.25f)),
                           -128, 127);
        accumulator += q * integer_weights[col * channels + k];
      }
      float quantized = std::clamp(
          std::nearbyint(accumulator * 0.25f * weight_scales[col] / 0.5f),
          -128.0f, 127.0f);
      EXPECT_FLOAT_EQ(actual.data()[row * 2 + col], quantized * 0.5f);
    }
  }
}

TEST_P(MobileFullyConnectedTest, ExactHalfwayOutputsRoundToEven) {
  constexpr int channels = 32;
  constexpr int output_channels = 32;
  constexpr float input_scale = 0.25f;
  constexpr float weight_scale = 0.5f;
  constexpr float output_scale = 0.25f;

  // Only two input channels contribute. The first produces small signed odd
  // accumulators, so requantization lands exactly at +/-0.5, +/-1.5, ... .
  // The second exercises both INT8 saturation limits. All scales are powers
  // of two, eliminating scale approximation as a reason to miss a tie.
  std::vector<int8_t> integer_weights(output_channels * channels, 0);
  for (int col = 0; col < output_channels; ++col) {
    if (col < 16) {
      integer_weights[col * channels] = col - 8;
    } else {
      integer_weights[col * channels + 1] = col % 2 == 0 ? 7 : -7;
    }
  }
  std::shared_ptr<Buffer> buffer;
  if (GetParam() == Type::kI4) {
    std::vector<uint8_t> packed(integer_weights.size() / 2);
    for (size_t i = 0; i < integer_weights.size(); i += 2) {
      packed[i / 2] =
          (integer_weights[i] & 15) | ((integer_weights[i + 1] & 15) << 4);
    }
    buffer = OwningCpuBuffer::Copy<Type::kU8>(packed);
  } else {
    buffer = OwningCpuBuffer::Copy<Type::kI8>(integer_weights);
  }
  XnnTensor weight(
      {.name = "linear.weight",
       .type = GetParam(),
       .shape = {output_channels, channels},
       .buffer = buffer,
       .quantization = std::make_shared<PerChannelAffineQuantization>(
           std::vector<float>(output_channels, weight_scale),
           std::vector<int64_t>{0}, 0)});
  absl::flat_hash_map<std::string, XnnTensor> weights;
  weights.emplace(
      "linear.input_scale",
      XnnTensor({.type = Type::kFP32, .shape = {}, .buffer = input_scale}));
  weights.emplace(
      "linear.output_scale",
      XnnTensor({.type = Type::kFP32, .shape = {}, .buffer = output_scale}));

  // Separate single-row and multi-row runtimes exercise decode and prefill
  // kernel selection. References use integer arithmetic, independent of the
  // platform floating-point rounding mode.
  for (int rows : {1, 6}) {
    SCOPED_TRACE(testing::Message() << "rows=" << rows);
    XnnTensor input({.type = Type::kFP32, .shape = {rows, channels}});
    auto output = MobileFullyConnected(input, weight, &weights);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto runner,
                                    XnnpackRunner::Create({output}));
    std::vector<float> values(rows * channels, 0.0f);
    for (int row = 0; row < rows; ++row) {
      const int signed_odd = row % 2 == 0 ? 2 * row + 1 : -(2 * row + 1);
      values[row * channels] = signed_odd * input_scale;
      values[row * channels + 1] = 127 * input_scale;
    }
    ASSERT_THAT(runner.SetInput(input, values), IsOk());
    ASSERT_THAT(runner.Run(), IsOk());
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto actual,
                                    runner.ReadOutputAs<float>(output));
    ASSERT_EQ(actual.size(), rows * output_channels);
    for (int row = 0; row < rows; ++row) {
      const int signed_odd = row % 2 == 0 ? 2 * row + 1 : -(2 * row + 1);
      for (int col = 0; col < output_channels; ++col) {
        const int accumulator = signed_odd * integer_weights[col * channels] +
                                127 * integer_weights[col * channels + 1];
        const int magnitude = std::abs(accumulator);
        const int rounded_magnitude = magnitude / 2 + (magnitude % 4 == 3);
        const int rounded =
            accumulator < 0 ? -rounded_magnitude : rounded_magnitude;
        const float expected = std::clamp(rounded, -128, 127) * output_scale;
        EXPECT_FLOAT_EQ(actual.data()[row * output_channels + col], expected)
            << "row=" << row << " col=" << col << " accumulator=" << accumulator
            << " scaled_accumulator=" << accumulator * 0.5f;
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(WeightTypes, MobileFullyConnectedTest,
                         testing::Values(Type::kI4, Type::kI8));

TEST(MobileFullyConnectedValidationTest, RejectsPartialScaleMetadata) {
  XnnTensor input({.type = Type::kFP32, .shape = {1, 2}});
  XnnTensor weight(
      {.name = "linear.weight", .type = Type::kFP32, .shape = {2, 2}});
  absl::flat_hash_map<std::string, XnnTensor> weights;
  weights.emplace(
      "linear.input_scale",
      XnnTensor({.type = Type::kFP32, .shape = {}, .buffer = 1.0f}));
  EXPECT_FALSE(MobileFullyConnected(input, weight, &weights).GetStatus().ok());
}

}  // namespace
}  // namespace litert::tensor::examples::gemma4
