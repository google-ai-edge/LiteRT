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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_cpu_options.h"
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/backends/tflite/tflite_flatbuffer_conversion.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/experimental/gemma4/bundle_matched_ops.h"
#include "tensor/examples/experimental/gemma4/feed_forward_network.h"
#include "tensor/examples/experimental/gemma4/mobile_fully_connected.h"
#include "tensor/examples/experimental/gemma4/rope.h"
#include "tensor/runners/litert/litert_dynamic_runner.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"
#include "tensor/utils/matchers.h"
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"

namespace litert::tensor::examples::experimental::gemma4::cpu {
namespace {
using TfLiteTensor = Tensor<TfLiteMixinTag>;

// Run the serialized Tensor API operations through the same fully delegated
// LiteRT CPU backend as the example. The buffers outlive the compiled model.
class OpRunner {
 public:
  static absl::StatusOr<std::unique_ptr<OpRunner>> Create(
      std::vector<TensorHandle> outputs,
      const std::vector<std::pair<TensorHandle, Shape>>& signatures = {}) {
    std::unique_ptr<OpRunner> result(new OpRunner);
    ModelFactory factory;
    for (int i = 0; i < outputs.size(); ++i) {
      LRT_TENSOR_RETURN_IF_ERROR(outputs[i].GetStatus());
      outputs[i].SetName(absl::StrCat("output_", i));
    }
    for (const auto& [tensor, shape] : signatures) {
      LRT_TENSOR_RETURN_IF_ERROR(factory.SetShapeSignature(tensor, shape));
    }
    LRT_TENSOR_RETURN_IF_ERROR(factory.AddSignature(outputs, "test"));
    LRT_TENSOR_ASSIGN_OR_RETURN(result->bytes_, factory.CreateFlatbuffer());
    LITERT_ASSIGN_OR_RETURN(auto env, Environment::Create({}));
    result->env_.emplace(std::move(env));
    LITERT_ASSIGN_OR_RETURN(auto options, Options::Create());
    LITERT_RETURN_IF_ERROR(
        options.SetHardwareAccelerators(HwAccelerators::kCpu));
    LITERT_ASSIGN_OR_RETURN(CpuOptions & cpu, options.GetCpuOptions());
    LITERT_RETURN_IF_ERROR(cpu.SetNumThreads(1));
    LITERT_RETURN_IF_ERROR(cpu.SetXNNPackFlags(
        TFLITE_XNNPACK_DELEGATE_FLAG_QS8 |
        TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS));
    LITERT_RETURN_IF_ERROR(cpu.SetHintFullyDelegatedToSingleDelegate(true));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        auto model,
        LitertDynamicRunner::Create(
            *result->env_,
            absl::Span<const uint8_t>(
                reinterpret_cast<const uint8_t*>(result->bytes_.data()),
                result->bytes_.size()),
            options));
    result->runner_.emplace(std::move(model));
    return result;
  }

  absl::Status SetInput(const TensorHandle& input,
                        const std::vector<float>& values) {
    return runner_->SetInput(
        "test", std::string(input.GetName()),
        absl::Span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(values.data()),
            values.size() * sizeof(float)));
  }
  absl::Status Resize(const TensorHandle& input, const Shape& shape) {
    return runner_->ResizeInput("test", std::string(input.GetName()), shape,
                                ResizeMode::kStrict);
  }
  absl::Status Run() { return runner_->Run("test"); }
  template <class U>
  absl::StatusOr<LockedBufferSpan<const U>> ReadOutputAs(
      const TensorHandle& output) {
    LRT_TENSOR_ASSIGN_OR_RETURN(
        auto tensor, runner_->GetOutput("test", std::string(output.GetName())));
    return tensor.GetBufferPtr()->Lock().template As<const U>();
  }

 private:
  std::optional<Environment> env_;
  std::vector<char> bytes_;
  std::optional<LitertDynamicRunner> runner_;
};

TfLiteTensor Floats(Shape shape, const std::vector<float>& values) {
  return TfLiteTensor({.type = Type::kFP32,
                       .shape = std::move(shape),
                       .buffer = OwningCpuBuffer::Copy<Type::kFP32>(values)});
}

class MobileFullyConnectedTest : public testing::TestWithParam<Type> {};

TEST_P(MobileFullyConnectedTest, StaticScalesRoundingAndSaturation) {
  constexpr int channels = 32;
  TfLiteTensor input(
      {.name = "input", .type = Type::kFP32, .shape = {3, channels}});
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
  TfLiteTensor weight(
      {.name = "linear.weight",
       .type = GetParam(),
       .shape = {2, channels},
       .buffer = buffer,
       .quantization = std::make_shared<PerChannelAffineQuantization>(
           weight_scales, std::vector<int64_t>{0}, 0)});
  absl::flat_hash_map<std::string, TfLiteTensor> weights;
  weights.emplace(
      "linear.input_scale",
      TfLiteTensor({.type = Type::kFP32, .shape = {}, .buffer = 0.25f}));
  weights.emplace(
      "linear.output_scale",
      TfLiteTensor({.type = Type::kFP32, .shape = {}, .buffer = 0.5f}));
  auto output = MobileFullyConnected(input, weight, &weights);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto runner, OpRunner::Create({output}));
  std::vector<float> values(3 * channels);
  for (int i = 0; i < channels; ++i) {
    values[i] = (i % 13 - 6) * 0.31f;
    values[channels + i] = integer_weights[i] >= 0 ? 100.0f : -100.0f;
    values[2 * channels + i] = -values[channels + i];
  }
  ASSERT_THAT(runner->SetInput(input, values), IsOk());
  ASSERT_THAT(runner->Run(), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto actual,
                                  runner->ReadOutputAs<float>(output));
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
  TfLiteTensor weight(
      {.name = "linear.weight",
       .type = GetParam(),
       .shape = {output_channels, channels},
       .buffer = buffer,
       .quantization = std::make_shared<PerChannelAffineQuantization>(
           std::vector<float>(output_channels, weight_scale),
           std::vector<int64_t>{0}, 0)});
  absl::flat_hash_map<std::string, TfLiteTensor> weights;
  weights.emplace(
      "linear.input_scale",
      TfLiteTensor({.type = Type::kFP32, .shape = {}, .buffer = input_scale}));
  weights.emplace(
      "linear.output_scale",
      TfLiteTensor({.type = Type::kFP32, .shape = {}, .buffer = output_scale}));

  // Separate single-row and multi-row runtimes exercise decode and prefill
  // kernel selection. References use integer arithmetic, independent of the
  // platform floating-point rounding mode.
  for (int rows : {1, 6}) {
    SCOPED_TRACE(testing::Message() << "rows=" << rows);
    TfLiteTensor input(
        {.name = "input", .type = Type::kFP32, .shape = {rows, channels}});
    auto output = MobileFullyConnected(input, weight, &weights);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto runner, OpRunner::Create({output}));
    std::vector<float> values(rows * channels, 0.0f);
    for (int row = 0; row < rows; ++row) {
      const int signed_odd = row % 2 == 0 ? 2 * row + 1 : -(2 * row + 1);
      values[row * channels] = signed_odd * input_scale;
      values[row * channels + 1] = 127 * input_scale;
    }
    ASSERT_THAT(runner->SetInput(input, values), IsOk());
    ASSERT_THAT(runner->Run(), IsOk());
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto actual,
                                    runner->ReadOutputAs<float>(output));
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
  TfLiteTensor input({.name = "input", .type = Type::kFP32, .shape = {1, 2}});
  TfLiteTensor weight(
      {.name = "linear.weight", .type = Type::kFP32, .shape = {2, 2}});
  absl::flat_hash_map<std::string, TfLiteTensor> weights;
  weights.emplace("linear.input_scale", Floats({}, {1.0f}));
  EXPECT_FALSE(MobileFullyConnected(input, weight, &weights).GetStatus().ok());
  weights.clear();
  weights.emplace("linear.output_scale", Floats({}, {1.0f}));
  EXPECT_FALSE(MobileFullyConnected(input, weight, &weights).GetStatus().ok());
}

TEST(MobileFullyConnectedValidationTest,
     RejectsNonScalarAndInvalidActivationScales) {
  for (float value : {0.0f, -1.0f, std::numeric_limits<float>::infinity(),
                      std::numeric_limits<float>::quiet_NaN()}) {
    EXPECT_FALSE(ReadMobileActivationScale(Floats({}, {value})).ok());
  }
  EXPECT_FALSE(ReadMobileActivationScale(Floats({1, 1}, {1.0f})).ok());
  EXPECT_FALSE(ReadMobileActivationScale(Floats({2}, {1.0f, 2.0f})).ok());
  EXPECT_TRUE(ReadMobileActivationScale(Floats({}, {0.25f})).ok());
  EXPECT_TRUE(ReadMobileActivationScale(Floats({1}, {0.25f})).ok());
}

TEST(MobileFullyConnectedValidationTest,
     RejectsStaticInt2WithoutActivationScales) {
  TfLiteTensor input({.name = "input", .type = Type::kFP32, .shape = {1, 32}});
  TfLiteTensor weight(
      {.name = "linear.weight", .type = Type::kI2, .shape = {32, 32}});
  EXPECT_FALSE(MobileFullyConnected(input, weight).GetStatus().ok());
  const absl::flat_hash_map<std::string, TfLiteTensor> weights;
  EXPECT_FALSE(MobileFullyConnected(input, weight, &weights).GetStatus().ok());
}

TEST(Gemma4OpsTest, PrimitiveRmsNormWithAndWithoutLearnedScale) {
  TfLiteTensor input(
      {.name = "input", .type = Type::kFP32, .shape = {1, 2, 4}});
  const std::vector<float> values{1, -2, 3, -4, 0, 0.01f, -0.03f, 0.02f};
  const std::vector<float> scales{0.5f, 2, 1, -1};
  const auto eps = Floats({}, {1e-6f});
  auto plain =
      PrimitiveRmsNorm(input, TfLiteTensor(TensorHandle::Invalid()), eps);
  auto learned = PrimitiveRmsNorm(input, Floats({4}, scales), eps);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto runner,
                                  OpRunner::Create({plain, learned}));
  ASSERT_THAT(runner->SetInput(input, values), IsOk());
  ASSERT_THAT(runner->Run(), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto a, runner->ReadOutputAs<float>(plain));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto b, runner->ReadOutputAs<float>(learned));
  for (int row = 0; row < 2; ++row) {
    double sum = 0;
    for (int k = 0; k < 4; ++k)
      sum += double(values[row * 4 + k]) * values[row * 4 + k];
    for (int k = 0; k < 4; ++k) {
      const float expected = values[row * 4 + k] / std::sqrt(sum / 4 + 1e-6);
      EXPECT_NEAR(a.data()[row * 4 + k], expected, 2e-6f);
      EXPECT_NEAR(b.data()[row * 4 + k], expected * scales[k], 2e-6f);
    }
  }
}

TEST(Gemma4OpsTest, FeedForwardUsesGeluGateAndPreservesUnquantizedActivations) {
  TfLiteTensor input(
      {.name = "input", .type = Type::kFP32, .shape = {1, 2, 2}});
  const std::vector<float> values{0.3f, -0.7f, 1.2f, 0.4f};
  const std::vector<float> gate{1, 2, -0.5f, 1, 0.3f, -0.2f};
  const std::vector<float> up{0.5f, 1, 2, -1, 0.2f, 0.3f};
  const std::vector<float> down{1, 0.5f, -0.3f, -1, 2, 0.7f};
  auto output = FeedForwardNetwork(input, Floats({3, 2}, gate),
                                   Floats({3, 2}, up), Floats({2, 3}, down));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto runner, OpRunner::Create({output}));
  ASSERT_THAT(runner->SetInput(input, values), IsOk());
  ASSERT_THAT(runner->Run(), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto actual,
                                  runner->ReadOutputAs<float>(output));
  for (int row = 0; row < 2; ++row) {
    double hidden[3];
    for (int i = 0; i < 3; ++i) {
      const double g =
          gate[i * 2] * values[row * 2] + gate[i * 2 + 1] * values[row * 2 + 1];
      const double u =
          up[i * 2] * values[row * 2] + up[i * 2 + 1] * values[row * 2 + 1];
      hidden[i] = u * 0.5 * g *
                  (1 + std::tanh(std::sqrt(2.0 / std::acos(-1.0)) *
                                 (g + 0.044715 * g * g * g)));
    }
    for (int col = 0; col < 2; ++col) {
      double expected = 0;
      for (int i = 0; i < 3; ++i) expected += hidden[i] * down[col * 3 + i];
      EXPECT_NEAR(actual.data()[row * 2 + col], expected, 2e-5f);
    }
  }
}

TEST(Gemma4OpsTest, DynamicRopeResizesRepeatedlyAtNonzeroPositions) {
  TfLiteTensor input(
      {.name = "input", .type = Type::kFP32, .shape = {1, 2, 4, 4}});
  TfLiteTensor cos({.name = "cos", .type = Type::kFP32, .shape = {1, 1, 4, 4}});
  TfLiteTensor sin({.name = "sin", .type = Type::kFP32, .shape = {1, 1, 4, 4}});
  auto output = RotaryEmbedding(input, cos, sin, true);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto runner, OpRunner::Create({output}, {{input, {1, 2, -1, 4}},
                                               {cos, {1, 1, -1, 4}},
                                               {sin, {1, 1, -1, 4}}}));
  for (int rows : {4, 1, 3, 2, 4}) {
    SCOPED_TRACE(rows);
    std::vector<float> values(2 * rows * 4), cosine(rows * 4), sine(rows * 4);
    for (int i = 0; i < values.size(); ++i) values[i] = (i % 17 - 8) * 0.13f;
    for (int r = 0; r < rows; ++r) {
      for (int k = 0; k < 4; ++k) {
        const float angle = (7 + r) * (k % 2 == 0 ? 1.0f : 0.01f);
        cosine[r * 4 + k] = std::cos(angle);
        sine[r * 4 + k] = std::sin(angle);
      }
    }
    ASSERT_THAT(runner->Resize(input, {1, 2, rows, 4}), IsOk());
    ASSERT_THAT(runner->Resize(cos, {1, 1, rows, 4}), IsOk());
    ASSERT_THAT(runner->Resize(sin, {1, 1, rows, 4}), IsOk());
    ASSERT_THAT(runner->SetInput(input, values), IsOk());
    ASSERT_THAT(runner->SetInput(cos, cosine), IsOk());
    ASSERT_THAT(runner->SetInput(sin, sine), IsOk());
    ASSERT_THAT(runner->Run(), IsOk());
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto actual,
                                    runner->ReadOutputAs<float>(output));
    ASSERT_EQ(actual.size(), values.size());
    for (int h = 0; h < 2; ++h) {
      for (int r = 0; r < rows; ++r) {
        for (int k = 0; k < 4; ++k) {
          const int i = (h * rows + r) * 4 + k;
          const float rotated =
              values[(h * rows + r) * 4 + (k + 2) % 4] * (k < 2 ? -1 : 1);
          EXPECT_NEAR(actual.data()[i],
                      values[i] * cosine[r * 4 + k] + rotated * sine[r * 4 + k],
                      1e-6f);
        }
      }
    }
  }
}

TEST(Gemma4OpsTest, GlobalPleSplitsAfterStaticRequantizationWithDynamicRows) {
  constexpr int layers = 3, width = 2, channels = 32;
  const std::string prefix = "model.per_layer_model_projection";
  TfLiteTensor input(
      {.name = "input", .type = Type::kFP32, .shape = {1, 4, channels}});
  std::vector<int8_t> codes(layers * width * channels);
  for (int i = 0; i < codes.size(); ++i) codes[i] = i % 9 - 4;
  TfLiteTensor weight(
      {.name = prefix + ".weight",
       .type = Type::kI8,
       .shape = {layers * width, channels},
       .buffer = OwningCpuBuffer::Copy<Type::kI8>(codes),
       .quantization = std::make_shared<PerChannelAffineQuantization>(
           std::vector<float>{0.125f, 0.25f, 0.5f, 0.125f, 0.25f, 0.5f},
           std::vector<int64_t>{0}, 0)});
  absl::flat_hash_map<std::string, TfLiteTensor> weights{
      {prefix + ".input_scale", Floats({}, {0.25f})},
      {prefix + ".output_scale", Floats({}, {0.5f})}};
  auto parts = BundlePerLayerProjectionParts(input, weight, weights, layers,
                                             width, true);
  std::vector<TensorHandle> outputs(parts.begin(), parts.end());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto runner, OpRunner::Create(outputs, {{input, {1, -1, channels}}}));
  const float scales[] = {0.125f, 0.25f, 0.5f, 0.125f, 0.25f, 0.5f};
  for (int rows : {4, 1, 3}) {
    std::vector<float> values(rows * channels);
    for (int i = 0; i < values.size(); ++i) values[i] = (i % 19 - 9) * 0.31f;
    ASSERT_THAT(runner->Resize(input, {1, rows, channels}), IsOk());
    ASSERT_THAT(runner->SetInput(input, values), IsOk());
    ASSERT_THAT(runner->Run(), IsOk());
    for (int layer = 0; layer < layers; ++layer) {
      LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
          auto actual, runner->ReadOutputAs<float>(parts[layer]));
      ASSERT_EQ(actual.size(), rows * width);
      for (int row = 0; row < rows; ++row) {
        for (int k = 0; k < width; ++k) {
          int acc = 0;
          const int channel = layer * width + k;
          for (int j = 0; j < channels; ++j) {
            acc += std::clamp(
                       int(std::nearbyint(values[row * channels + j] / 0.25f)),
                       -128, 127) *
                   codes[channel * channels + j];
          }
          const float expected =
              std::clamp(std::nearbyint(acc * 0.25f * scales[channel] / 0.5f),
                         -128.0f, 127.0f) *
              0.5f;
          EXPECT_FLOAT_EQ(actual.data()[row * width + k], expected);
        }
      }
    }
  }
  weights.erase(prefix + ".output_scale");
  EXPECT_FALSE(
      BundlePerLayerProjectionParts(input, weight, weights, layers, width)[0]
          .GetStatus()
          .ok());
}

}  // namespace
}  // namespace litert::tensor::examples::experimental::gemma4::cpu
