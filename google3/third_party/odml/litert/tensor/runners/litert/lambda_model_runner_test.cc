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

#include "third_party/odml/litert/tensor/runners/litert/lambda_model_runner.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "testing/base/public/gunit.h"
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "third_party/odml/litert/tensor/arithmetic.h"
#include "third_party/odml/litert/tensor/backends/tflite/arithmetic_tflite.h"
#include "third_party/odml/litert/tensor/backends/tflite/tflite_flatbuffer_conversion.h"
#include "third_party/odml/litert/tensor/datatypes.h"
#include "third_party/odml/litert/tensor/runners/litert/litert_dynamic_runner.h"
#include "third_party/odml/litert/tensor/tensor.h"

namespace litert::tensor {
namespace {

TEST(LambdaModelRunnerTest, SimpleLambda) {
  LITERT_ASSIGN_OR_ABORT(auto env, Environment::Create({}));
  LITERT_ASSIGN_OR_ABORT(auto options, Options::Create());
  options.SetHardwareAccelerators(HwAccelerators::kCpu);

  auto runner = CreateLambdaRunner(
      env, options,
      {{"x", Tensor<TfLiteMixinTag>(
                 {.name = "x", .type = Type::kFP32, .shape = {1}})}},
      [](const auto& inputs) {
        Tensor y = Add(inputs.at("x"), 1.0f);
        return TensorsMap{{"y", y}};
      });

  std::vector<float> input_data = {2.0f};
  auto x_tensor = Create("x", Type::kFP32, {1}, std::move(input_data));
  EXPECT_TRUE(runner.SetInput("x", x_tensor).ok());
  EXPECT_TRUE(runner.Run().ok());

  auto y_or = runner.GetOutput("y");
  ASSERT_TRUE(y_or.ok());
  auto y_tensor = std::move(*y_or);

  auto buffer_or = y_tensor.GetBuffer();
  ASSERT_TRUE(buffer_or.ok());
  auto locked_span = buffer_or->Lock();
  const float* data = reinterpret_cast<const float*>(locked_span.data());
  EXPECT_EQ(data[0], 3.0f);
}

TEST(LambdaModelRunnerTest, StaticRunner) {
  LITERT_ASSIGN_OR_ABORT(auto env, Environment::Create({}));
  LITERT_ASSIGN_OR_ABORT(auto options, Options::Create());
  options.SetHardwareAccelerators(HwAccelerators::kCpu);

  Tensor<TfLiteMixinTag> x({.name = "x", .type = Type::kFP32, .shape = {1}});
  Tensor<TfLiteMixinTag> y = Add(x, 1.0f);
  y.SetName("y");

  TensorsMap inputs = {{"x", x}};
  TensorsMap outputs = {{"y", y}};

  auto runner = CreateStaticRunner(env, options, inputs, outputs);

  std::vector<float> input_data = {2.0f};
  auto x_tensor = Create("x", Type::kFP32, {1}, std::move(input_data));
  EXPECT_TRUE(runner.SetInput("x", x_tensor).ok());
  EXPECT_TRUE(runner.Run().ok());

  auto y_or = runner.GetOutput("y");
  ASSERT_TRUE(y_or.ok());
  auto y_tensor = std::move(*y_or);

  auto buffer_or = y_tensor.GetBuffer();
  ASSERT_TRUE(buffer_or.ok());
  auto locked_span = buffer_or->Lock();
  const float* data = reinterpret_cast<const float*>(locked_span.data());
  EXPECT_EQ(data[0], 3.0f);
}

TEST(LitertDynamicRunnerTest, CreateFromBufferAndBinaryInput) {
  LITERT_ASSIGN_OR_ABORT(auto env, Environment::Create({}));
  LITERT_ASSIGN_OR_ABORT(auto options, Options::Create());
  options.SetHardwareAccelerators(HwAccelerators::kCpu);

  Tensor<TfLiteMixinTag> x({.name = "x", .type = Type::kFP32, .shape = {1}});
  Tensor<TfLiteMixinTag> y = Add(x, 1.0f);
  y.SetName("y");

  std::vector<char> model_buffer;
  ASSERT_TRUE(Save(std::vector<Tensor<TfLiteMixinTag>>{y}, model_buffer).ok());

  absl::Span<const uint8_t> buffer_span(
      reinterpret_cast<const uint8_t*>(model_buffer.data()),
      model_buffer.size());

  auto runner_or = LitertDynamicRunner::Create(env, buffer_span, options);
  ASSERT_TRUE(runner_or.ok());
  auto runner = std::move(*runner_or);

  std::vector<float> input_data = {2.0f};
  absl::Span<const uint8_t> input_span(
      reinterpret_cast<const uint8_t*>(input_data.data()),
      input_data.size() * sizeof(float));

  EXPECT_TRUE(runner.SetInput("x", input_span).ok());
  EXPECT_TRUE(runner.Run().ok());

  auto y_or = runner.GetOutput("y");
  ASSERT_TRUE(y_or.ok());
  auto y_tensor = std::move(*y_or);

  auto buffer_or = y_tensor.GetBuffer();
  ASSERT_TRUE(buffer_or.ok());
  auto locked_span = buffer_or->Lock();
  const float* data = reinterpret_cast<const float*>(locked_span.data());
  EXPECT_EQ(data[0], 3.0f);
}

}  // namespace
}  // namespace litert::tensor
