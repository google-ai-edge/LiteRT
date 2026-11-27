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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_cpu_options.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/backends/tflite/tflite_flatbuffer_conversion.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/experimental/gemma4/active_graph_context.h"
#include "tensor/examples/experimental/gemma4/attention.h"
#include "tensor/examples/experimental/gemma4/bundle_matched_ops.h"
#include "tensor/examples/experimental/gemma4/feed_forward_network.h"
#include "tensor/examples/experimental/gemma4/gemma4_graph.h"
#include "tensor/examples/experimental/gemma4/kv_cache.h"
#include "tensor/examples/experimental/gemma4/mobile_fully_connected.h"
#include "tensor/examples/experimental/gemma4/rope.h"
#include "tensor/examples/experimental/gemma4/transformer.h"
#include "tensor/examples/gemma4/gemma4_config.h"
#include "tensor/runners/litert/litert_dynamic_runner.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"
#include "tensor/utils/matchers.h"
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"

namespace litert::tensor::examples::experimental::gemma4::cpu {
namespace {

using ::absl_testing::StatusIs;
using ::litert::tensor::examples::gemma4::Config;
using TfLiteTensor = Tensor<TfLiteMixinTag>;

// Runs serialized Tensor API operations through the same fully delegated
// LiteRT CPU backend as the example. The buffers outlive the compiled model.
class OpRunner {
 public:
  OpRunner(const OpRunner&) = delete;
  OpRunner& operator=(const OpRunner&) = delete;

  static absl::StatusOr<std::unique_ptr<OpRunner>> Create(
      std::vector<TensorHandle> outputs,
      const std::vector<std::pair<TensorHandle, Shape>>& signatures = {}) {
    std::unique_ptr<OpRunner> result(new OpRunner());
    ModelFactory factory;
    for (int i = 0; i < static_cast<int>(outputs.size()); ++i) {
      LRT_TENSOR_RETURN_IF_ERROR(outputs[i].GetStatus());
      outputs[i].SetName(absl::StrCat("output_", i));
    }
    for (const auto& [tensor, shape] : signatures) {
      LRT_TENSOR_RETURN_IF_ERROR(factory.SetShapeSignature(tensor, shape));
    }
    LRT_TENSOR_RETURN_IF_ERROR(
        factory.AddSignature(std::move(outputs), "test"));
    LRT_TENSOR_ASSIGN_OR_RETURN(result->bytes_, factory.CreateFlatbuffer());
    LITERT_ASSIGN_OR_RETURN(Environment env, Environment::Create({}));
    result->env_.emplace(std::move(env));
    LITERT_ASSIGN_OR_RETURN(Options options, Options::Create());
    LITERT_RETURN_IF_ERROR(
        options.SetHardwareAccelerators(HwAccelerators::kCpu));
    LITERT_ASSIGN_OR_RETURN(CpuOptions & cpu, options.GetCpuOptions());
    LITERT_RETURN_IF_ERROR(cpu.SetNumThreads(1));
    LITERT_RETURN_IF_ERROR(cpu.SetXNNPackFlags(
        TFLITE_XNNPACK_DELEGATE_FLAG_QS8 |
        TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS));
    LITERT_RETURN_IF_ERROR(cpu.SetHintFullyDelegatedToSingleDelegate(true));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        LitertDynamicRunner model,
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
                        absl::Span<const float> values) {
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
        TensorHandle tensor,
        runner_->GetOutput("test", std::string(output.GetName())));
    return tensor.GetBufferPtr()->Lock().template As<const U>();
  }

 private:
  OpRunner() = default;

  std::optional<Environment> env_;
  std::vector<char> bytes_;
  std::optional<LitertDynamicRunner> runner_;
};

TfLiteTensor Floats(Shape shape, const std::vector<float>& values) {
  return TfLiteTensor({.type = Type::kFP32,
                       .shape = std::move(shape),
                       .buffer = OwningCpuBuffer::Copy<Type::kFP32>(values)});
}

Config MinimalLayerConfig() {
  Config config = Config::E2B();
  config.vocab_size = 8;
  config.embed_dim = 4;
  config.hidden_dim = 8;
  config.head_dim = 4;
  config.global_key_size = 4;
  config.num_heads = 1;
  config.num_kv_heads = 1;
  config.num_layers = 1;
  config.per_layer_input_dim = 2;
  config.sliding_window_size = 4;
  config.attention_pattern_size = 1;
  config.frac_shared_layers = 0.0f;
  return config;
}

class MobileFullyConnectedTest : public testing::TestWithParam<Type> {};

TEST_P(MobileFullyConnectedTest, StaticScalesRoundingAndSaturation) {
  constexpr int kChannels = 32;
  TfLiteTensor input(
      {.name = "input", .type = Type::kFP32, .shape = {3, kChannels}});
  std::vector<int8_t> integer_weights(2 * kChannels);
  for (int i = 0; i < kChannels; ++i) {
    integer_weights[i] = i % 7 - 3;
    integer_weights[kChannels + i] = i % 5 - 2;
  }
  const std::vector<float> weight_scales = {0.25f, 0.5f};
  TfLiteTensor weight(
      {.name = "linear.weight",
       .type = GetParam(),
       .shape = {2, kChannels},
       .buffer = OwningCpuBuffer::CopyAs(GetParam(), integer_weights),
       .quantization = std::make_shared<PerChannelAffineQuantization>(
           weight_scales, std::vector<int64_t>{0}, 0)});
  absl::flat_hash_map<std::string, TfLiteTensor> weights;
  weights.emplace(
      "linear.input_scale",
      TfLiteTensor({.type = Type::kFP32, .shape = {}, .buffer = 0.25f}));
  weights.emplace(
      "linear.output_scale",
      TfLiteTensor({.type = Type::kFP32, .shape = {}, .buffer = 0.5f}));
  const TfLiteTensor output = MobileFullyConnected(input, weight, &weights);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::unique_ptr<OpRunner> runner,
                                  OpRunner::Create({output}));
  std::vector<float> values(3 * kChannels);
  for (int i = 0; i < kChannels; ++i) {
    values[i] = (i % 13 - 6) * 0.31f;
    values[kChannels + i] = integer_weights[i] >= 0 ? 100.0f : -100.0f;
    values[2 * kChannels + i] = -values[kChannels + i];
  }
  ASSERT_THAT(runner->SetInput(input, values), IsOk());
  ASSERT_THAT(runner->Run(), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const float> actual,
                                  runner->ReadOutputAs<float>(output));
  ASSERT_EQ(actual.size(), 6);
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 2; ++col) {
      int32_t accumulator = 0;
      for (int k = 0; k < kChannels; ++k) {
        const int quantized_input =
            std::clamp(static_cast<int>(
                           std::nearbyint(values[row * kChannels + k] / 0.25f)),
                       -128, 127);
        accumulator += quantized_input * integer_weights[col * kChannels + k];
      }
      const float quantized = std::clamp(
          std::nearbyint(accumulator * 0.25f * weight_scales[col] / 0.5f),
          -128.0f, 127.0f);
      EXPECT_FLOAT_EQ(actual.data()[row * 2 + col], quantized * 0.5f)
          << "row=" << row << " col=" << col;
    }
  }
}

TEST_P(MobileFullyConnectedTest, ExactHalfwayOutputsRoundToEven) {
  constexpr int kChannels = 32;
  constexpr int kOutputChannels = 32;
  constexpr float kInputScale = 0.25f;
  constexpr float kWeightScale = 0.5f;
  constexpr float kOutputScale = 0.25f;

  // Only two input channels contribute. The first produces small signed odd
  // accumulators, so requantization lands exactly at +/-0.5, +/-1.5, ... .
  // The second exercises both INT8 saturation limits. All scales are powers
  // of two, eliminating scale approximation as a reason to miss a tie.
  std::vector<int8_t> integer_weights(kOutputChannels * kChannels, 0);
  for (int col = 0; col < kOutputChannels; ++col) {
    if (col < 16) {
      integer_weights[col * kChannels] = col - 8;
    } else {
      integer_weights[col * kChannels + 1] = col % 2 == 0 ? 7 : -7;
    }
  }
  TfLiteTensor weight(
      {.name = "linear.weight",
       .type = GetParam(),
       .shape = {kOutputChannels, kChannels},
       .buffer = OwningCpuBuffer::CopyAs(GetParam(), integer_weights),
       .quantization = std::make_shared<PerChannelAffineQuantization>(
           std::vector<float>(kOutputChannels, kWeightScale),
           std::vector<int64_t>{0}, 0)});
  absl::flat_hash_map<std::string, TfLiteTensor> weights;
  weights.emplace(
      "linear.input_scale",
      TfLiteTensor({.type = Type::kFP32, .shape = {}, .buffer = kInputScale}));
  weights.emplace(
      "linear.output_scale",
      TfLiteTensor({.type = Type::kFP32, .shape = {}, .buffer = kOutputScale}));

  // Separate single-row and multi-row runtimes exercise decode and prefill
  // kernel selection. References use integer arithmetic, independent of the
  // platform floating-point rounding mode.
  for (int rows : {1, 6}) {
    SCOPED_TRACE(testing::Message() << "rows=" << rows);
    TfLiteTensor input(
        {.name = "input", .type = Type::kFP32, .shape = {rows, kChannels}});
    const TfLiteTensor output = MobileFullyConnected(input, weight, &weights);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::unique_ptr<OpRunner> runner,
                                    OpRunner::Create({output}));
    std::vector<float> values(rows * kChannels, 0.0f);
    for (int row = 0; row < rows; ++row) {
      const int signed_odd = row % 2 == 0 ? 2 * row + 1 : -(2 * row + 1);
      values[row * kChannels] = signed_odd * kInputScale;
      values[row * kChannels + 1] = 127 * kInputScale;
    }
    ASSERT_THAT(runner->SetInput(input, values), IsOk());
    ASSERT_THAT(runner->Run(), IsOk());
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const float> actual,
                                    runner->ReadOutputAs<float>(output));
    ASSERT_EQ(actual.size(), rows * kOutputChannels);
    for (int row = 0; row < rows; ++row) {
      const int signed_odd = row % 2 == 0 ? 2 * row + 1 : -(2 * row + 1);
      for (int col = 0; col < kOutputChannels; ++col) {
        const int accumulator = signed_odd * integer_weights[col * kChannels] +
                                127 * integer_weights[col * kChannels + 1];
        const int magnitude = std::abs(accumulator);
        const int rounded_magnitude = magnitude / 2 + (magnitude % 4 == 3);
        const int rounded =
            accumulator < 0 ? -rounded_magnitude : rounded_magnitude;
        const float expected = std::clamp(rounded, -128, 127) * kOutputScale;
        EXPECT_FLOAT_EQ(actual.data()[row * kOutputChannels + col], expected)
            << "row=" << row << " col=" << col << " accumulator=" << accumulator
            << " scaled_accumulator=" << accumulator * 0.5f;
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(WeightTypes, MobileFullyConnectedTest,
                         testing::Values(Type::kI4, Type::kI8),
                         [](const testing::TestParamInfo<Type>& info) {
                           return info.param == Type::kI4 ? "Int4" : "Int8";
                         });

TEST(MobileFullyConnectedValidationTest, RejectsInputScaleWithoutOutputScale) {
  TfLiteTensor input({.name = "input", .type = Type::kFP32, .shape = {1, 2}});
  TfLiteTensor weight(
      {.name = "linear.weight", .type = Type::kFP32, .shape = {2, 2}});
  absl::flat_hash_map<std::string, TfLiteTensor> weights;
  weights.emplace("linear.input_scale", Floats({}, {1.0f}));

  EXPECT_THAT(MobileFullyConnected(input, weight, &weights).GetStatus(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(MobileFullyConnectedValidationTest, RejectsOutputScaleWithoutInputScale) {
  TfLiteTensor input({.name = "input", .type = Type::kFP32, .shape = {1, 2}});
  TfLiteTensor weight(
      {.name = "linear.weight", .type = Type::kFP32, .shape = {2, 2}});
  absl::flat_hash_map<std::string, TfLiteTensor> weights;
  weights.emplace("linear.output_scale", Floats({}, {1.0f}));

  EXPECT_THAT(MobileFullyConnected(input, weight, &weights).GetStatus(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(MobileFullyConnectedValidationTest,
     ReadsRankZeroAndLengthOneActivationScales) {
  EXPECT_THAT(ReadMobileActivationScale(Floats({}, {0.25f})),
              IsOkAndHolds(0.25f));
  EXPECT_THAT(ReadMobileActivationScale(Floats({1}, {0.25f})),
              IsOkAndHolds(0.25f));
}

struct InvalidScaleCase {
  std::string name;
  Shape shape;
  std::vector<float> values;
};

class InvalidActivationScaleTest
    : public testing::TestWithParam<InvalidScaleCase> {};

TEST_P(InvalidActivationScaleTest, RejectsInvalidActivationScale) {
  const InvalidScaleCase& tc = GetParam();

  EXPECT_THAT(ReadMobileActivationScale(Floats(tc.shape, tc.values)),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

INSTANTIATE_TEST_SUITE_P(
    InvalidScales, InvalidActivationScaleTest,
    testing::Values(
        InvalidScaleCase{"Zero", {}, {0.0f}},
        InvalidScaleCase{"Negative", {}, {-1.0f}},
        InvalidScaleCase{
            "Infinity", {}, {std::numeric_limits<float>::infinity()}},
        InvalidScaleCase{"NaN", {}, {std::numeric_limits<float>::quiet_NaN()}},
        InvalidScaleCase{"RankTwo", {1, 1}, {1.0f}},
        InvalidScaleCase{"LengthTwo", {2}, {1.0f, 2.0f}}),
    [](const testing::TestParamInfo<InvalidScaleCase>& info) {
      return info.param.name;
    });

TEST(MobileFullyConnectedValidationTest, RejectsStaticInt2WithoutWeightMap) {
  TfLiteTensor input({.name = "input", .type = Type::kFP32, .shape = {1, 32}});
  TfLiteTensor weight(
      {.name = "linear.weight", .type = Type::kI2, .shape = {32, 32}});

  EXPECT_THAT(MobileFullyConnected(input, weight).GetStatus(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(MobileFullyConnectedValidationTest,
     RejectsStaticInt2WithMissingOrUnsuffixedScales) {
  TfLiteTensor input({.name = "input", .type = Type::kFP32, .shape = {1, 32}});
  TfLiteTensor weight(
      {.name = "linear.weight", .type = Type::kI2, .shape = {32, 32}});
  TfLiteTensor unsuffixed_weight(
      {.name = "linear_no_suffix", .type = Type::kI2, .shape = {32, 32}});
  const absl::flat_hash_map<std::string, TfLiteTensor> weights;

  EXPECT_THAT(MobileFullyConnected(input, weight, &weights).GetStatus(),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      MobileFullyConnected(input, unsuffixed_weight, &weights).GetStatus(),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(MobileFullyConnectedValidationTest,
     AcceptsStaticInt2WithValidActivationScales) {
  constexpr int kChannels = 32;
  TfLiteTensor input(
      {.name = "input", .type = Type::kFP32, .shape = {1, kChannels}});
  std::vector<int8_t> integer_weights(2 * kChannels);
  for (int i = 0; i < static_cast<int>(integer_weights.size()); ++i) {
    integer_weights[i] = static_cast<int8_t>((i % 4) - 2);
  }
  TfLiteTensor weight(
      {.name = "linear.weight",
       .type = Type::kI2,
       .shape = {2, kChannels},
       .buffer = OwningCpuBuffer::CopyAs(Type::kI2, integer_weights),
       .quantization = std::make_shared<PerChannelAffineQuantization>(
           std::vector<float>{0.25f, 0.5f}, std::vector<int64_t>{0}, 0)});
  absl::flat_hash_map<std::string, TfLiteTensor> weights;
  weights.emplace("linear.input_scale", Floats({}, {0.25f}));
  weights.emplace("linear.output_scale", Floats({}, {0.5f}));

  const TfLiteTensor output = MobileFullyConnected(input, weight, &weights);
  ASSERT_THAT(output.GetStatus(), IsOk());
  EXPECT_EQ(output.GetType(), Type::kFP32);
  EXPECT_EQ(output.GetShape(), (Shape{1, 2}));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::unique_ptr<OpRunner> runner,
                                  OpRunner::Create({output}));
  std::vector<float> values(kChannels);
  for (int i = 0; i < kChannels; ++i) {
    values[i] = (i % 9 - 4) * 0.25f;
  }
  ASSERT_THAT(runner->SetInput(input, values), IsOk());
  ASSERT_THAT(runner->Run(), IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const float> actual,
                                  runner->ReadOutputAs<float>(output));
  ASSERT_EQ(actual.size(), 2);
  const std::vector<float> weight_scales = {0.25f, 0.5f};
  for (int col = 0; col < 2; ++col) {
    int32_t accumulator = 0;
    for (int k = 0; k < kChannels; ++k) {
      const int quantized_input = std::clamp(
          static_cast<int>(std::nearbyint(values[k] / 0.25f)), -128, 127);
      accumulator += quantized_input * integer_weights[col * kChannels + k];
    }
    const float quantized = std::clamp(
        std::nearbyint(accumulator * 0.25f * weight_scales[col] / 0.5f),
        -128.0f, 127.0f);
    EXPECT_FLOAT_EQ(actual.data()[col], quantized * 0.5f) << "col=" << col;
  }
}

TEST(MobileFullyConnectedValidationTest,
     RejectsUnquantizedWeightWithStaticScales) {
  TfLiteTensor fp32_input(
      {.name = "input", .type = Type::kFP32, .shape = {1, 32}});
  TfLiteTensor unquantized_weight(
      {.name = "linear.weight", .type = Type::kI8, .shape = {2, 32}});
  absl::flat_hash_map<std::string, TfLiteTensor> weights;
  weights.emplace("linear.input_scale", Floats({}, {0.25f}));
  weights.emplace("linear.output_scale", Floats({}, {0.5f}));

  EXPECT_THAT(MobileFullyConnected(fp32_input, unquantized_weight, &weights)
                  .GetStatus(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(MobileFullyConnectedValidationTest, RejectsNonFp32InputWithStaticScales) {
  TfLiteTensor int8_input(
      {.name = "int8_input", .type = Type::kI8, .shape = {1, 32}});
  TfLiteTensor quantized_weight(
      {.name = "linear.weight",
       .type = Type::kI8,
       .shape = {2, 32},
       .quantization = std::make_shared<PerChannelAffineQuantization>(
           std::vector<float>{0.25f, 0.25f}, std::vector<int64_t>{0}, 0)});
  absl::flat_hash_map<std::string, TfLiteTensor> weights;
  weights.emplace("linear.input_scale", Floats({}, {0.25f}));
  weights.emplace("linear.output_scale", Floats({}, {0.5f}));

  EXPECT_THAT(
      MobileFullyConnected(int8_input, quantized_weight, &weights).GetStatus(),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(Gemma4OpsTest, PrimitiveRmsNormComputesPlainAndScaledOutputs) {
  TfLiteTensor input(
      {.name = "input", .type = Type::kFP32, .shape = {1, 2, 4}});
  const std::vector<float> values{1, -2, 3, -4, 0, 0.01f, -0.03f, 0.02f};
  const std::vector<float> scales{0.5f, 2, 1, -1};
  const TfLiteTensor eps = Floats({}, {1e-6f});

  const TfLiteTensor plain =
      PrimitiveRmsNorm(input, TfLiteTensor(TensorHandle::Invalid()), eps);
  const TfLiteTensor learned =
      PrimitiveRmsNorm(input, Floats({4}, scales), eps);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::unique_ptr<OpRunner> runner,
                                  OpRunner::Create({plain, learned}));
  ASSERT_THAT(runner->SetInput(input, values), IsOk());
  ASSERT_THAT(runner->Run(), IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const float> plain_output,
                                  runner->ReadOutputAs<float>(plain));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const float> learned_output,
                                  runner->ReadOutputAs<float>(learned));
  ASSERT_EQ(plain_output.size(), 8);
  ASSERT_EQ(learned_output.size(), 8);
  for (int row = 0; row < 2; ++row) {
    double sum = 0.0;
    for (int k = 0; k < 4; ++k) {
      const double value = static_cast<double>(values[row * 4 + k]);
      sum += value * value;
    }
    for (int k = 0; k < 4; ++k) {
      const float expected = values[row * 4 + k] / std::sqrt(sum / 4 + 1e-6);
      EXPECT_NEAR(plain_output.data()[row * 4 + k], expected, 2e-6f)
          << "row=" << row << " k=" << k;
      EXPECT_NEAR(learned_output.data()[row * 4 + k], expected * scales[k],
                  2e-6f)
          << "row=" << row << " k=" << k;
    }
  }
}

TEST(Gemma4OpsTest, PrimitiveRmsNormPropagatesErrorScale) {
  TfLiteTensor input(
      {.name = "input", .type = Type::kFP32, .shape = {1, 2, 4}});
  const TfLiteTensor eps = Floats({}, {1e-6f});
  const TfLiteTensor error_scale(absl::FailedPreconditionError("bad scale"));

  EXPECT_THAT(PrimitiveRmsNorm(input, error_scale, eps).GetStatus(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(Gemma4OpsTest, PrimitiveRmsNormRejectsScalarInput) {
  const TfLiteTensor scalar(
      {.name = "scalar", .type = Type::kFP32, .shape = {}});
  const TfLiteTensor eps = Floats({}, {1e-6f});

  EXPECT_THAT(
      PrimitiveRmsNorm(scalar, TfLiteTensor(TensorHandle::Invalid()), eps)
          .GetStatus(),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(Gemma4OpsTest, FeedForwardUsesGeluGateAndPreservesUnquantizedActivations) {
  TfLiteTensor input(
      {.name = "input", .type = Type::kFP32, .shape = {1, 2, 2}});
  const std::vector<float> values{0.3f, -0.7f, 1.2f, 0.4f};
  const std::vector<float> gate{1, 2, -0.5f, 1, 0.3f, -0.2f};
  const std::vector<float> up{0.5f, 1, 2, -1, 0.2f, 0.3f};
  const std::vector<float> down{1, 0.5f, -0.3f, -1, 2, 0.7f};

  const TfLiteTensor output = FeedForwardNetwork(
      input, Floats({3, 2}, gate), Floats({3, 2}, up), Floats({2, 3}, down));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::unique_ptr<OpRunner> runner,
                                  OpRunner::Create({output}));
  ASSERT_THAT(runner->SetInput(input, values), IsOk());
  ASSERT_THAT(runner->Run(), IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const float> actual,
                                  runner->ReadOutputAs<float>(output));
  ASSERT_EQ(actual.size(), 4);
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
      double expected = 0.0;
      for (int i = 0; i < 3; ++i) {
        expected += hidden[i] * down[col * 3 + i];
      }
      EXPECT_NEAR(actual.data()[row * 2 + col], expected, 2e-5f)
          << "row=" << row << " col=" << col;
    }
  }
}

TEST(Gemma4OpsTest, DynamicRopeResizesRepeatedlyAtNonzeroPositions) {
  TfLiteTensor input(
      {.name = "input", .type = Type::kFP32, .shape = {1, 2, 4, 4}});
  TfLiteTensor cos({.name = "cos", .type = Type::kFP32, .shape = {1, 1, 4, 4}});
  TfLiteTensor sin({.name = "sin", .type = Type::kFP32, .shape = {1, 1, 4, 4}});
  const TfLiteTensor output =
      SplitHalfRoPE(input, cos, sin, RopeLeadingDims::kDynamic);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OpRunner> runner,
      OpRunner::Create({output}, {{input, {1, 2, -1, 4}},
                                  {cos, {1, 1, -1, 4}},
                                  {sin, {1, 1, -1, 4}}}));

  for (int rows : {4, 1, 3, 2, 4}) {
    SCOPED_TRACE(absl::StrCat("rows=", rows));
    std::vector<float> values(2 * rows * 4);
    std::vector<float> cosine(rows * 4);
    std::vector<float> sine(rows * 4);
    for (int i = 0; i < static_cast<int>(values.size()); ++i) {
      values[i] = (i % 17 - 8) * 0.13f;
    }
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

    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const float> actual,
                                    runner->ReadOutputAs<float>(output));
    ASSERT_EQ(actual.size(), 2 * rows * 4);
    for (int h = 0; h < 2; ++h) {
      for (int r = 0; r < rows; ++r) {
        for (int k = 0; k < 4; ++k) {
          const int i = (h * rows + r) * 4 + k;
          const float rotated =
              values[(h * rows + r) * 4 + (k + 2) % 4] * (k < 2 ? -1 : 1);
          EXPECT_NEAR(actual.data()[i],
                      values[i] * cosine[r * 4 + k] + rotated * sine[r * 4 + k],
                      1e-6f)
              << "h=" << h << " r=" << r << " k=" << k;
        }
      }
    }
  }
}

class SplitHalfRopeModeTest : public testing::TestWithParam<RopeLeadingDims> {};

TEST_P(SplitHalfRopeModeTest, PropagatesErrorTensorsOnAllInputs) {
  const RopeLeadingDims mode = GetParam();
  TfLiteTensor valid(
      {.name = "valid", .type = Type::kFP32, .shape = {1, 2, 4, 4}});
  TfLiteTensor error(absl::FailedPreconditionError("bad tensor"));

  EXPECT_THAT(SplitHalfRoPE(error, valid, valid, mode).GetStatus(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(SplitHalfRoPE(valid, error, valid, mode).GetStatus(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(SplitHalfRoPE(valid, valid, error, mode).GetStatus(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_P(SplitHalfRopeModeTest, RejectsInvalidHandlesOnAllInputs) {
  const RopeLeadingDims mode = GetParam();
  TfLiteTensor valid(
      {.name = "valid", .type = Type::kFP32, .shape = {1, 2, 4, 4}});
  TfLiteTensor invalid(TensorHandle::Invalid());

  EXPECT_THAT(SplitHalfRoPE(invalid, valid, valid, mode).GetStatus(),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(SplitHalfRoPE(valid, invalid, valid, mode).GetStatus(),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(SplitHalfRoPE(valid, valid, invalid, mode).GetStatus(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_P(SplitHalfRopeModeTest, RejectsScalarAndInvalidLastDimension) {
  const RopeLeadingDims mode = GetParam();
  TfLiteTensor valid(
      {.name = "valid", .type = Type::kFP32, .shape = {1, 2, 4, 4}});
  TfLiteTensor scalar({.name = "scalar", .type = Type::kFP32, .shape = {}});

  EXPECT_THAT(SplitHalfRoPE(scalar, valid, valid, mode).GetStatus(),
              StatusIs(absl::StatusCode::kInvalidArgument));
  for (int bad_dim : {0, -2, 1, 3}) {
    SCOPED_TRACE(absl::StrCat("bad_dim=", bad_dim));
    TfLiteTensor bad_input({.name = "bad_input",
                            .type = Type::kFP32,
                            .shape = {1, 1, 2, bad_dim}});
    EXPECT_THAT(SplitHalfRoPE(bad_input, valid, valid, mode).GetStatus(),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

INSTANTIATE_TEST_SUITE_P(
    LeadingDimModes, SplitHalfRopeModeTest,
    testing::Values(RopeLeadingDims::kStatic, RopeLeadingDims::kDynamic),
    [](const testing::TestParamInfo<RopeLeadingDims>& info) {
      return info.param == RopeLeadingDims::kDynamic ? "Dynamic" : "Static";
    });

TEST(Gemma4OpsTest, GlobalPleSplitsAfterStaticRequantizationWithDynamicRows) {
  constexpr int kLayers = 3;
  constexpr int kWidth = 2;
  constexpr int kChannels = 32;
  constexpr absl::string_view kPrefix = "model.per_layer_model_projection";
  const std::vector<float> scales = {0.125f, 0.25f, 0.5f, 0.125f, 0.25f, 0.5f};
  TfLiteTensor input(
      {.name = "input", .type = Type::kFP32, .shape = {1, 4, kChannels}});
  std::vector<int8_t> codes(kLayers * kWidth * kChannels);
  for (int i = 0; i < static_cast<int>(codes.size()); ++i) {
    codes[i] = i % 9 - 4;
  }
  TfLiteTensor weight(
      {.name = absl::StrCat(kPrefix, ".weight"),
       .type = Type::kI8,
       .shape = {kLayers * kWidth, kChannels},
       .buffer = OwningCpuBuffer::Copy<Type::kI8>(codes),
       .quantization = std::make_shared<PerChannelAffineQuantization>(
           scales, std::vector<int64_t>{0}, 0)});
  absl::flat_hash_map<std::string, TfLiteTensor> weights{
      {absl::StrCat(kPrefix, ".input_scale"), Floats({}, {0.25f})},
      {absl::StrCat(kPrefix, ".output_scale"), Floats({}, {0.5f})}};

  const std::vector<TfLiteTensor> parts = BundlePerLayerProjectionParts(
      input, weight, weights, kLayers, kWidth, PleRowMode::kDynamic);
  std::vector<TensorHandle> outputs(parts.begin(), parts.end());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OpRunner> runner,
      OpRunner::Create(outputs, {{input, {1, -1, kChannels}}}));

  for (int rows : {4, 1, 3}) {
    SCOPED_TRACE(absl::StrCat("rows=", rows));
    std::vector<float> values(rows * kChannels);
    for (int i = 0; i < static_cast<int>(values.size()); ++i) {
      values[i] = (i % 19 - 9) * 0.31f;
    }
    ASSERT_THAT(runner->Resize(input, {1, rows, kChannels}), IsOk());
    ASSERT_THAT(runner->SetInput(input, values), IsOk());
    ASSERT_THAT(runner->Run(), IsOk());
    for (int layer = 0; layer < kLayers; ++layer) {
      LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
          LockedBufferSpan<const float> actual,
          runner->ReadOutputAs<float>(parts[layer]));
      ASSERT_EQ(actual.size(), rows * kWidth);
      for (int row = 0; row < rows; ++row) {
        for (int k = 0; k < kWidth; ++k) {
          int acc = 0;
          const int channel = layer * kWidth + k;
          for (int j = 0; j < kChannels; ++j) {
            acc += std::clamp(static_cast<int>(std::nearbyint(
                                  values[row * kChannels + j] / 0.25f)),
                              -128, 127) *
                   codes[channel * kChannels + j];
          }
          const float expected =
              std::clamp(std::nearbyint(acc * 0.25f * scales[channel] / 0.5f),
                         -128.0f, 127.0f) *
              0.5f;
          EXPECT_FLOAT_EQ(actual.data()[row * kWidth + k], expected)
              << "layer=" << layer << " row=" << row << " k=" << k;
        }
      }
    }
  }
}

TEST(Gemma4OpsTest, GlobalPlePropagatesInputWeightAndFcErrors) {
  constexpr int kLayers = 3;
  constexpr int kWidth = 2;
  constexpr int kChannels = 32;
  constexpr absl::string_view kPrefix = "model.per_layer_model_projection";
  TfLiteTensor input(
      {.name = "input", .type = Type::kFP32, .shape = {1, 4, kChannels}});
  TfLiteTensor weight(
      {.name = absl::StrCat(kPrefix, ".weight"),
       .type = Type::kI8,
       .shape = {kLayers * kWidth, kChannels},
       .quantization = std::make_shared<PerChannelAffineQuantization>(
           std::vector<float>(kLayers * kWidth, 0.25f), std::vector<int64_t>{0},
           0)});
  absl::flat_hash_map<std::string, TfLiteTensor> weights{
      {absl::StrCat(kPrefix, ".input_scale"), Floats({}, {0.25f})},
      {absl::StrCat(kPrefix, ".output_scale"), Floats({}, {0.5f})}};
  TfLiteTensor err(absl::FailedPreconditionError("injected error"));

  EXPECT_THAT(
      BundlePerLayerProjectionParts(err, weight, weights, kLayers, kWidth)[0]
          .GetStatus(),
      StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(
      BundlePerLayerProjectionParts(input, err, weights, kLayers, kWidth)[0]
          .GetStatus(),
      StatusIs(absl::StatusCode::kFailedPrecondition));

  // Unquantized INT8 weight passes initial shape/dtype checks but fails inside
  // MobileFullyConnected.
  TfLiteTensor unquantized_weight({.name = absl::StrCat(kPrefix, ".weight"),
                                   .type = Type::kI8,
                                   .shape = {kLayers * kWidth, kChannels}});
  EXPECT_THAT(BundlePerLayerProjectionParts(input, unquantized_weight, weights,
                                            kLayers, kWidth)[0]
                  .GetStatus(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(Gemma4OpsTest, GlobalPleRejectsInvalidDimensionsDtypeRankNameAndScales) {
  constexpr int kLayers = 3;
  constexpr int kWidth = 2;
  constexpr int kChannels = 32;
  constexpr absl::string_view kPrefix = "model.per_layer_model_projection";
  TfLiteTensor input(
      {.name = "input", .type = Type::kFP32, .shape = {1, 4, kChannels}});
  TfLiteTensor weight(
      {.name = absl::StrCat(kPrefix, ".weight"),
       .type = Type::kI8,
       .shape = {kLayers * kWidth, kChannels},
       .quantization = std::make_shared<PerChannelAffineQuantization>(
           std::vector<float>(kLayers * kWidth, 0.25f), std::vector<int64_t>{0},
           0)});
  absl::flat_hash_map<std::string, TfLiteTensor> weights{
      {absl::StrCat(kPrefix, ".input_scale"), Floats({}, {0.25f})},
      {absl::StrCat(kPrefix, ".output_scale"), Floats({}, {0.5f})}};

  EXPECT_THAT(
      BundlePerLayerProjectionParts(input, weight, weights, 0, kWidth)[0]
          .GetStatus(),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      BundlePerLayerProjectionParts(input, weight, weights, kLayers, 0)[0]
          .GetStatus(),
      StatusIs(absl::StatusCode::kInvalidArgument));

  TfLiteTensor rank1_input(
      {.name = "rank1", .type = Type::kFP32, .shape = {kChannels}});
  EXPECT_THAT(
      BundlePerLayerProjectionParts(rank1_input, weight, weights, kLayers,
                                    kWidth, PleRowMode::kDynamic)[0]
          .GetStatus(),
      StatusIs(absl::StatusCode::kInvalidArgument));

  TfLiteTensor wrong_dtype_weight({.name = absl::StrCat(kPrefix, ".weight"),
                                   .type = Type::kFP32,
                                   .shape = {kLayers * kWidth, kChannels}});
  EXPECT_THAT(BundlePerLayerProjectionParts(input, wrong_dtype_weight, weights,
                                            kLayers, kWidth)[0]
                  .GetStatus(),
              StatusIs(absl::StatusCode::kInvalidArgument));

  TfLiteTensor wrong_name_weight({.name = "other.weight",
                                  .type = Type::kI8,
                                  .shape = {kLayers * kWidth, kChannels}});
  EXPECT_THAT(BundlePerLayerProjectionParts(input, wrong_name_weight, weights,
                                            kLayers, kWidth)[0]
                  .GetStatus(),
              StatusIs(absl::StatusCode::kInvalidArgument));

  weights.erase(absl::StrCat(kPrefix, ".output_scale"));
  EXPECT_THAT(
      BundlePerLayerProjectionParts(input, weight, weights, kLayers, kWidth)[0]
          .GetStatus(),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

struct AttentionTestInputs {
  Config config = MinimalLayerConfig();
  TfLiteTensor input =
      TfLiteTensor({.name = "input", .type = Type::kFP32, .shape = {1, 2, 4}});
  TfLiteTensor cos =
      TfLiteTensor({.name = "cos", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  TfLiteTensor sin =
      TfLiteTensor({.name = "sin", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  TfLiteTensor key_cache = MakeInt8KeyCache<TfLiteMixinTag>("k", 1, 4, 0.25f);
  TfLiteTensor value_cache = MakeInt8KeyCache<TfLiteMixinTag>("v", 1, 4, 0.25f);
  TfLiteTensor eps = Floats({}, {1e-6f});
  absl::flat_hash_map<std::string, TfLiteTensor> weights;
  ActiveGraphContext context = [] {
    ActiveGraphContext ctx;
    ctx.owners = {0};
    ctx.layers.resize(1);
    return ctx;
  }();
};

TEST(AttentionContractTest, RejectsOutOfBoundsLayerContext) {
  AttentionTestInputs s;
  ActiveGraphContext empty_context;

  EXPECT_THAT(Attention(s.input, s.cos, s.sin, s.key_cache, s.value_cache,
                        s.config, s.weights, "layer0",
                        Config::LayerType::kGlobal, s.eps, empty_context)
                  .output.GetStatus(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(AttentionContractTest, RejectsInvalidKvOwnerIndices) {
  AttentionTestInputs s;
  for (int bad_owner : {-1, 1}) {
    SCOPED_TRACE(absl::StrCat("bad_owner=", bad_owner));
    ActiveGraphContext bad_context;
    bad_context.owners = {bad_owner};
    bad_context.layers.resize(1);

    EXPECT_THAT(Attention(s.input, s.cos, s.sin, s.key_cache, s.value_cache,
                          s.config, s.weights, "layer0",
                          Config::LayerType::kGlobal, s.eps, bad_context)
                    .output.GetStatus(),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

TEST(AttentionContractTest, RejectsUnauthoredSharedKvOwner) {
  AttentionTestInputs s;
  ActiveGraphContext ctx;
  ctx.owners = {0, 0};
  ctx.layers.resize(2);
  ctx.layer = 1;

  EXPECT_THAT(
      Attention(s.input, s.cos, s.sin, s.key_cache, s.value_cache, s.config,
                s.weights, "layer1", Config::LayerType::kGlobal, s.eps, ctx)
          .output.GetStatus(),
      StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(AttentionContractTest, RejectsRankLessThanTwoInput) {
  AttentionTestInputs s;
  TfLiteTensor rank1({.name = "rank1", .type = Type::kFP32, .shape = {4}});

  EXPECT_THAT(Attention(rank1, s.cos, s.sin, s.key_cache, s.value_cache,
                        s.config, s.weights, "layer0",
                        Config::LayerType::kGlobal, s.eps, s.context)
                  .output.GetStatus(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(AttentionContractTest, RejectsUnquantizedKvCacheMetadata) {
  AttentionTestInputs s;
  TfLiteTensor unquantized(
      {.name = "u", .type = Type::kI8, .shape = {1, 1, 1, 4}});

  EXPECT_THAT(Attention(s.input, s.cos, s.sin, unquantized, s.value_cache,
                        s.config, s.weights, "layer0",
                        Config::LayerType::kGlobal, s.eps, s.context)
                  .output.GetStatus(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(AttentionContractTest, RejectsBatchSizeGreaterThanOne) {
  AttentionTestInputs s;
  TfLiteTensor batch2({.name = "b2", .type = Type::kFP32, .shape = {2, 2, 4}});

  EXPECT_THAT(Attention(batch2, s.cos, s.sin, s.key_cache, s.value_cache,
                        s.config, s.weights, "layer0",
                        Config::LayerType::kGlobal, s.eps, s.context)
                  .output.GetStatus(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(AttentionContractTest, RejectsMultiKvHeadConfig) {
  AttentionTestInputs s;
  s.config.num_kv_heads = 2;

  EXPECT_THAT(Attention(s.input, s.cos, s.sin, s.key_cache, s.value_cache,
                        s.config, s.weights, "layer0",
                        Config::LayerType::kGlobal, s.eps, s.context)
                  .output.GetStatus(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(AttentionContractTest, RejectsInvalidSoftCapValues) {
  for (float bad_cap : {0.0f, -1.0f, std::numeric_limits<float>::quiet_NaN()}) {
    SCOPED_TRACE(absl::StrCat("bad_cap=", bad_cap));
    AttentionTestInputs s;
    s.config.attn_logits_soft_cap = bad_cap;

    EXPECT_THAT(Attention(s.input, s.cos, s.sin, s.key_cache, s.value_cache,
                          s.config, s.weights, "layer0",
                          Config::LayerType::kGlobal, s.eps, s.context)
                    .output.GetStatus(),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

TEST(AttentionContractTest, PropagatesInputProjectionRopeNormAndQuantErrors) {
  // Error input tensor.
  {
    AttentionTestInputs s;
    TfLiteTensor err(absl::FailedPreconditionError("bad input"));
    EXPECT_THAT(Attention(err, s.cos, s.sin, s.key_cache, s.value_cache,
                          s.config, s.weights, "layer0",
                          Config::LayerType::kGlobal, s.eps, s.context)
                    .output.GetStatus(),
                StatusIs(absl::StatusCode::kFailedPrecondition));
  }
  // Failing q_proj, k_proj, and v_proj (static INT2 without activation scales).
  for (absl::string_view proj : {"q_proj", "k_proj", "v_proj"}) {
    SCOPED_TRACE(proj);
    AttentionTestInputs s;
    s.weights.emplace(
        absl::StrCat("layer0.", proj, ".weight"),
        TfLiteTensor({.name = absl::StrCat("layer0.", proj, ".weight"),
                      .type = Type::kI2,
                      .shape = {4, 4}}));
    EXPECT_THAT(Attention(s.input, s.cos, s.sin, s.key_cache, s.value_cache,
                          s.config, s.weights, "layer0",
                          Config::LayerType::kGlobal, s.eps, s.context)
                    .output.GetStatus(),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
  // Failing Q RoPE (error cos tensor).
  {
    AttentionTestInputs s;
    TfLiteTensor err(absl::FailedPreconditionError("bad cos"));
    EXPECT_THAT(Attention(s.input, err, s.sin, s.key_cache, s.value_cache,
                          s.config, s.weights, "layer0",
                          Config::LayerType::kGlobal, s.eps, s.context)
                    .output.GetStatus(),
                StatusIs(absl::StatusCode::kFailedPrecondition));
  }
  // Failing K RoPE (error k_norm.weight).
  {
    AttentionTestInputs s;
    s.weights.emplace("layer0.k_norm.weight",
                      TfLiteTensor(absl::FailedPreconditionError("bad knorm")));
    EXPECT_THAT(Attention(s.input, s.cos, s.sin, s.key_cache, s.value_cache,
                          s.config, s.weights, "layer0",
                          Config::LayerType::kGlobal, s.eps, s.context)
                    .output.GetStatus(),
                StatusIs(absl::StatusCode::kFailedPrecondition));
  }
  // Failing V RMSNorm (error eps on V when q_norm/k_norm are fine is covered
  // via scalar V reshape mismatch; here inject an invalid scale quantization
  // on key_cache and value_cache to hit QuantizeInt8Kv error branches).
  {
    AttentionTestInputs s;
    TfLiteTensor bad_k(
        {.name = "bad_k",
         .type = Type::kI8,
         .shape = {1, 1, 1, 4},
         .quantization = std::make_shared<PerChannelAffineQuantization>(
             std::vector<float>{0.0f}, std::vector<int64_t>{0}, 0)});
    EXPECT_THAT(Attention(s.input, s.cos, s.sin, bad_k, s.value_cache, s.config,
                          s.weights, "layer0", Config::LayerType::kGlobal,
                          s.eps, s.context)
                    .output.GetStatus(),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
  {
    AttentionTestInputs s;
    TfLiteTensor bad_v(
        {.name = "bad_v",
         .type = Type::kI8,
         .shape = {1, 1, 1, 4},
         .quantization = std::make_shared<PerChannelAffineQuantization>(
             std::vector<float>{0.0f}, std::vector<int64_t>{0}, 0)});
    EXPECT_THAT(Attention(s.input, s.cos, s.sin, s.key_cache, bad_v, s.config,
                          s.weights, "layer0", Config::LayerType::kGlobal,
                          s.eps, s.context)
                    .output.GetStatus(),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

TEST(AttentionContractTest,
     AuthorsPositiveSoftCapAndPopulatesOwnerContextHandles) {
  AttentionTestInputs s;
  s.config.attn_logits_soft_cap = 50.0f;

  const AttentionOutput<TfLiteMixinTag> out = Attention(
      s.input, s.cos, s.sin, s.key_cache, s.value_cache, s.config, s.weights,
      "layer0", Config::LayerType::kGlobal, s.eps, s.context);

  ASSERT_THAT(out.output.GetStatus(), IsOk());
  EXPECT_EQ(out.output.GetShape(), (Shape{1, 2, 4}));
  EXPECT_EQ(out.key_cache.GetShape(), (Shape{1, 1, 2, 4}));
  EXPECT_EQ(out.value_cache.GetShape(), (Shape{1, 1, 2, 4}));
  EXPECT_EQ(s.context.layers[0].owner, 0);
  EXPECT_TRUE(s.context.layers[0].global);
  EXPECT_THAT(s.context.layers[0].past_key.GetStatus(), IsOk());
  EXPECT_THAT(s.context.layers[0].new_key.GetStatus(), IsOk());
  EXPECT_THAT(s.context.global_mask.GetStatus(), IsOk());
  EXPECT_EQ(s.context.global_mask.GetShape(), (Shape{1, 1, 2, 4}));
}

TEST(TransformerLayerTest, AuthorsLayerWithAndWithoutPerLayerInput) {
  AttentionTestInputs s;
  TfLiteTensor ple({.name = "ple0", .type = Type::kFP32, .shape = {1, 2, 2}});

  const TransformerLayerOutput<TfLiteMixinTag> with_ple =
      TransformerLayer(s.input, s.cos, s.sin, s.key_cache, s.value_cache, ple,
                       s.config, s.weights, /*layer_idx=*/0, s.eps, s.context);

  ASSERT_THAT(with_ple.output.GetStatus(), IsOk());
  EXPECT_EQ(with_ple.output.GetShape(), (Shape{1, 2, 4}));
  EXPECT_EQ(with_ple.key_cache.GetShape(), (Shape{1, 1, 2, 4}));
  EXPECT_EQ(with_ple.value_cache.GetShape(), (Shape{1, 1, 2, 4}));
}

TEST(TransformerLayerTest, PropagatesAttentionFfnAndPerLayerInputErrors) {
  // Attention error (empty context).
  {
    AttentionTestInputs s;
    ActiveGraphContext empty_ctx;
    EXPECT_THAT(
        TransformerLayer(s.input, s.cos, s.sin, s.key_cache, s.value_cache,
                         TfLiteTensor(TensorHandle::Invalid()), s.config,
                         s.weights, /*layer_idx=*/0, s.eps, empty_ctx)
            .output.GetStatus(),
        StatusIs(absl::StatusCode::kInvalidArgument));
  }
  // FFN error (INT2 gate_proj without activation scales).
  {
    AttentionTestInputs s;
    s.weights.emplace(
        "model.layers.0.mlp.gate_proj.weight",
        TfLiteTensor({.name = "model.layers.0.mlp.gate_proj.weight",
                      .type = Type::kI2,
                      .shape = {8, 4}}));
    EXPECT_THAT(
        TransformerLayer(s.input, s.cos, s.sin, s.key_cache, s.value_cache,
                         TfLiteTensor(TensorHandle::Invalid()), s.config,
                         s.weights, /*layer_idx=*/0, s.eps, s.context)
            .output.GetStatus(),
        StatusIs(absl::StatusCode::kInvalidArgument));
  }
  // Error PLE tensor.
  {
    AttentionTestInputs s;
    TfLiteTensor ple_error(absl::FailedPreconditionError("bad ple"));
    EXPECT_THAT(TransformerLayer(s.input, s.cos, s.sin, s.key_cache,
                                 s.value_cache, ple_error, s.config, s.weights,
                                 /*layer_idx=*/0, s.eps, s.context)
                    .output.GetStatus(),
                StatusIs(absl::StatusCode::kFailedPrecondition));
  }
  // Scalar PLE tensor.
  {
    AttentionTestInputs s;
    TfLiteTensor scalar_ple({.name = "s", .type = Type::kFP32, .shape = {}});
    EXPECT_THAT(TransformerLayer(s.input, s.cos, s.sin, s.key_cache,
                                 s.value_cache, scalar_ple, s.config, s.weights,
                                 /*layer_idx=*/0, s.eps, s.context)
                    .output.GetStatus(),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
  // Failing PLE gate and projection linear layers.
  for (absl::string_view ple_weight :
       {"model.layers.0.per_layer_input_gate.weight",
        "model.layers.0.per_layer_projection.weight"}) {
    SCOPED_TRACE(ple_weight);
    AttentionTestInputs s;
    TfLiteTensor ple({.name = "ple0", .type = Type::kFP32, .shape = {1, 2, 2}});
    s.weights.emplace(std::string(ple_weight),
                      TfLiteTensor({.name = std::string(ple_weight),
                                    .type = Type::kI2,
                                    .shape = {2, 4}}));
    EXPECT_THAT(
        TransformerLayer(s.input, s.cos, s.sin, s.key_cache, s.value_cache, ple,
                         s.config, s.weights, /*layer_idx=*/0, s.eps, s.context)
            .output.GetStatus(),
        StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

Gemma4Inputs<TfLiteMixinTag> MakeMinimalGraphInputs() {
  Gemma4Inputs<TfLiteMixinTag> inputs;
  inputs.embedded_input =
      TfLiteTensor({.name = "emb", .type = Type::kFP32, .shape = {1, 2, 4}});
  inputs.rope_global_cos =
      TfLiteTensor({.name = "gc", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  inputs.rope_global_sin =
      TfLiteTensor({.name = "gs", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  inputs.rope_local_cos = inputs.rope_global_cos;
  inputs.rope_local_sin = inputs.rope_global_sin;
  inputs.key_caches.push_back(
      MakeInt8KeyCache<TfLiteMixinTag>("k", 1, 4, 0.25f));
  inputs.value_caches.push_back(
      MakeInt8KeyCache<TfLiteMixinTag>("v", 1, 4, 0.25f));
  inputs.per_layer_token_embeddings.push_back(
      TfLiteTensor({.name = "ple0", .type = Type::kFP32, .shape = {1, 2, 2}}));
  return inputs;
}

TEST(BuildGemma4GraphTest, RejectsNonPositiveLayerCount) {
  Config config = MinimalLayerConfig();
  config.num_layers = 0;
  Gemma4Inputs<TfLiteMixinTag> inputs = MakeMinimalGraphInputs();
  ActiveGraphContext ctx;

  EXPECT_THAT(BuildGemma4Graph(inputs, config, ctx).logits.GetStatus(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(BuildGemma4GraphTest, PropagatesErrorEmbeddedInput) {
  const Config config = MinimalLayerConfig();
  Gemma4Inputs<TfLiteMixinTag> inputs = MakeMinimalGraphInputs();
  inputs.embedded_input =
      TfLiteTensor(absl::FailedPreconditionError("bad emb"));
  ActiveGraphContext ctx;

  EXPECT_THAT(BuildGemma4Graph(inputs, config, ctx).logits.GetStatus(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(BuildGemma4GraphTest,
     AuthorsTiedEmbeddingFallbackAndPerLayerPleProjection) {
  const Config config = MinimalLayerConfig();
  Gemma4Inputs<TfLiteMixinTag> inputs = MakeMinimalGraphInputs();
  inputs.bundle_dynamic_qd8_head = false;
  ActiveGraphContext ctx;

  const Gemma4Outputs<TfLiteMixinTag> out =
      BuildGemma4Graph(inputs, config, ctx);

  ASSERT_THAT(out.logits.GetStatus(), IsOk());
  EXPECT_EQ(out.logits.GetShape(), (Shape{1, 2, config.vocab_size}));
  EXPECT_EQ(out.key_caches.size(), 1);
  EXPECT_EQ(out.value_caches.size(), 1);
  EXPECT_THAT(ctx.global_mask.GetStatus(), IsOk());
}

TEST(BuildGemma4GraphTest,
     SharedNonOwnerLayerAcceptsShortKeyValueCacheMetadataVectors) {
  Config config = MinimalLayerConfig();
  config.num_layers = 2;
  config.frac_shared_layers = 0.5f;
  Gemma4Inputs<TfLiteMixinTag> inputs = MakeMinimalGraphInputs();
  // Provide cache metadata only for owner layer 0; layer 1 shares layer 0's KV.
  inputs.per_layer_token_embeddings.push_back(
      TfLiteTensor({.name = "ple1", .type = Type::kFP32, .shape = {1, 2, 2}}));
  inputs.bundle_dynamic_qd8_head = false;
  ActiveGraphContext ctx;

  const Gemma4Outputs<TfLiteMixinTag> out =
      BuildGemma4Graph(inputs, config, ctx);

  ASSERT_THAT(out.logits.GetStatus(), IsOk());
  EXPECT_EQ(out.logits.GetShape(), (Shape{1, 2, config.vocab_size}));
  EXPECT_EQ(out.key_caches.size(), 2);
  EXPECT_EQ(ctx.layers[1].owner, 0);
}

TEST(BuildGemma4GraphTest, RejectsMissingAndInvalidBundleSoftcapReciprocal) {
  const Config config = MinimalLayerConfig();
  {
    Gemma4Inputs<TfLiteMixinTag> inputs = MakeMinimalGraphInputs();
    inputs.bundle_dynamic_qd8_head = true;
    ActiveGraphContext ctx;
    EXPECT_THAT(BuildGemma4Graph(inputs, config, ctx).logits.GetStatus(),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
  {
    Gemma4Inputs<TfLiteMixinTag> inputs = MakeMinimalGraphInputs();
    inputs.bundle_dynamic_qd8_head = true;
    inputs.weights.emplace("matched.logits_softcap_reciprocal",
                           Floats({}, {-0.1f}));
    ActiveGraphContext ctx;
    EXPECT_THAT(BuildGemma4Graph(inputs, config, ctx).logits.GetStatus(),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

TEST(BuildGemma4GraphTest, PropagatesPerLayerPleFinalNormAndHeadErrors) {
  const Config config = MinimalLayerConfig();
  // Error token PLE entry.
  {
    Gemma4Inputs<TfLiteMixinTag> inputs = MakeMinimalGraphInputs();
    inputs.per_layer_token_embeddings[0] =
        TfLiteTensor(absl::FailedPreconditionError("bad tok ple"));
    ActiveGraphContext ctx;
    EXPECT_THAT(BuildGemma4Graph(inputs, config, ctx).logits.GetStatus(),
                StatusIs(absl::StatusCode::kFailedPrecondition));
  }
  // Failing per-layer PLE projection (INT2 without activation scales).
  {
    Gemma4Inputs<TfLiteMixinTag> inputs = MakeMinimalGraphInputs();
    inputs.weights.emplace(
        "model.layers.0.per_layer_model_projection.weight",
        TfLiteTensor(
            {.name = "model.layers.0.per_layer_model_projection.weight",
             .type = Type::kI2,
             .shape = {2, 4}}));
    ActiveGraphContext ctx;
    EXPECT_THAT(BuildGemma4Graph(inputs, config, ctx).logits.GetStatus(),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
  // Failing final RMSNorm scale.
  {
    Gemma4Inputs<TfLiteMixinTag> inputs = MakeMinimalGraphInputs();
    inputs.weights.emplace("model.norm.weight",
                           TfLiteTensor(absl::FailedPreconditionError("norm")));
    ActiveGraphContext ctx;
    EXPECT_THAT(BuildGemma4Graph(inputs, config, ctx).logits.GetStatus(),
                StatusIs(absl::StatusCode::kFailedPrecondition));
  }
  // Error lm_head weight tensor.
  {
    Gemma4Inputs<TfLiteMixinTag> inputs = MakeMinimalGraphInputs();
    inputs.weights.emplace("lm_head.weight",
                           TfLiteTensor(absl::FailedPreconditionError("head")));
    ActiveGraphContext ctx;
    EXPECT_THAT(BuildGemma4Graph(inputs, config, ctx).logits.GetStatus(),
                StatusIs(absl::StatusCode::kFailedPrecondition));
  }
  // Invalid INT2 lm_head without activation scales when bundle_dynamic_qd8_head
  // is false.
  {
    Gemma4Inputs<TfLiteMixinTag> inputs = MakeMinimalGraphInputs();
    inputs.bundle_dynamic_qd8_head = false;
    inputs.weights.emplace(
        "lm_head.weight",
        TfLiteTensor(
            {.name = "lm_head.weight", .type = Type::kI2, .shape = {8, 4}}));
    ActiveGraphContext ctx;
    EXPECT_THAT(BuildGemma4Graph(inputs, config, ctx).logits.GetStatus(),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

TEST(KvCacheQuantizationTest, CloneKvQuantizationRejectsMissingQuantization) {
  const TensorHandle unquantized(
      {.name = "cache", .type = Type::kI8, .shape = {1, 1, 1, 4}});

  EXPECT_THAT(CloneKvQuantization(unquantized),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(KvCacheQuantizationTest, CloneKvQuantizationReturnsIndependentCopy) {
  const TfLiteTensor cache =
      MakeInt8KeyCache<TfLiteMixinTag>("cache", /*rows=*/1, /*dim=*/4, 0.0625f);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<PerChannelAffineQuantization> cloned,
      CloneKvQuantization(cache));

  ASSERT_NE(cloned, nullptr);
  EXPECT_NE(cloned.get(), cache.GetQuantization().get());
  EXPECT_THAT(cloned->scales, testing::ElementsAre(0.0625f));
}

TEST(KvCacheQuantizationTest, QuantizeInt8KvRejectsMissingQuantization) {
  const TfLiteTensor input(
      {.name = "k", .type = Type::kFP32, .shape = {1, 1, 1, 4}});
  const TfLiteTensor unquantized(
      {.name = "cache", .type = Type::kI8, .shape = {1, 1, 1, 4}});

  EXPECT_THAT(QuantizeInt8Kv(input, unquantized).GetStatus(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace litert::tensor::examples::experimental::gemma4::cpu
