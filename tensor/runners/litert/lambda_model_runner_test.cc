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

#include "tensor/runners/litert/lambda_model_runner.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/backends/tflite/tflite_flatbuffer_conversion.h"
#include "tensor/datatypes.h"
#include "tensor/runners/litert/feedback_loop_config.h"
#include "tensor/runners/litert/litert_dynamic_runner.h"
#include "tensor/tensor.h"

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

// Concatenation exercises the same empty/growing/shrinking history boundary
// as chunked attention, with an output whose size really changes after resize.
TEST(LitertDynamicRunnerTest, ActiveHistoryAcrossSignatures) {
  LITERT_ASSIGN_OR_ABORT(auto env, Environment::Create({}));
  LITERT_ASSIGN_OR_ABORT(auto options, Options::Create());
  ASSERT_TRUE(options.SetHardwareAccelerators(HwAccelerators::kCpu));
  ModelFactory factory;
  for (const auto& [name, rows] : std::vector<std::pair<std::string, int>>{
           {"prefill", 4}, {"decode", 1}}) {
    Tensor<TfLiteMixinTag> past(
        {.name = "past", .type = Type::kFP32, .shape = {1, 1}});
    Tensor<TfLiteMixinTag> chunk(
        {.name = "chunk", .type = Type::kFP32, .shape = {rows, 1}});
    auto result = Concatenation({past, chunk}, 0);
    result.SetName("result");
    ASSERT_TRUE(factory.AddSignature({result}, name).ok());
  }
  auto model = factory.CreateFlatbuffer();
  ASSERT_TRUE(model.ok()) << model.status();
  auto runner_or = LitertDynamicRunner::Create(
      env,
      absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(model->data()),
                                model->size()),
      options);
  ASSERT_TRUE(runner_or.ok()) << runner_or.status();
  if (!runner_or.ok()) return;
  auto runner = std::move(*runner_or);
  alignas(64) float history[128] = {};
  for (int i = 0; i < 128; ++i) history[i] = static_cast<float>(i);
  for (int length : {0, 3, 32, 1, 0}) {
    for (const auto& [name, rows] : std::vector<std::pair<std::string, int>>{
             {"prefill", 4}, {"decode", 1}}) {
      const std::vector<int> shape{length, 1};
      ASSERT_TRUE(runner.ResizeInput(name, "past", shape, false).ok());
      EXPECT_FALSE(runner.Run(name).ok());
      // NOLINTBEGIN(misc-include-cleaner)
      RankedTensorType type(ElementType::Float32,
                            Layout(Dimensions(shape.begin(), shape.end())));
      LITERT_ASSIGN_OR_ABORT(
          auto input, TensorBuffer::CreateFromHostMemory(env, type, history,
                                                         sizeof(history)));
      // NOLINTEND(misc-include-cleaner)
      ASSERT_TRUE(runner.SetInputBuffer(name, "past", std::move(input)).ok());
      auto handle = runner.GetInput(name, "past");
      ASSERT_TRUE(handle.ok());
      if (!handle.ok()) return;
      EXPECT_EQ(handle->GetShape(), shape);
      {
        auto history_lock = handle->GetBufferPtr()->Lock();
        EXPECT_EQ(history_lock.data(),
                  reinterpret_cast<const std::byte*>(history));
      }
      std::vector<float> chunk(rows, 100.0f);
      ASSERT_TRUE(
          runner
              .SetInput(name, "chunk",
                        absl::Span<const uint8_t>(
                            reinterpret_cast<const uint8_t*>(chunk.data()),
                            chunk.size() * sizeof(float)))
              .ok());
      auto status = runner.Run(name);
      ASSERT_TRUE(status.ok()) << status;
      auto result = runner.GetOutput(name, "result");
      ASSERT_TRUE(result.ok()) << result.status();
      if (!result.ok()) return;
      EXPECT_EQ(result->GetShape(), (Shape{length + rows, 1}));
      auto data = result->GetBufferPtr()->Lock().As<const float>();
      ASSERT_EQ(data.size(), length + rows);
      for (int i = 0; i < length; ++i) EXPECT_EQ(data.data()[i], history[i]);
      for (int i = 0; i < rows; ++i) EXPECT_EQ(data.data()[length + i], 100.0f);
    }
  }
  EXPECT_FALSE(runner.ResizeInput("prefill", "past", {-1, 1}, false).ok());
  EXPECT_FALSE(runner.ResizeInput("missing", "past", {2, 1}, false).ok());
  EXPECT_FALSE(runner.ResizeInput("prefill", size_t{999}, {2, 1}, false).ok());
  // A rejected resize leaves the last successful binding usable.
  EXPECT_TRUE(runner.Run("prefill").ok());
}

TEST(LitertDynamicRunnerTest, ResizeCopyInputAndValidateBorrowedShape) {
  LITERT_ASSIGN_OR_ABORT(auto env, Environment::Create({}));
  LITERT_ASSIGN_OR_ABORT(auto options, Options::Create());
  ASSERT_TRUE(options.SetHardwareAccelerators(HwAccelerators::kCpu));
  Tensor<TfLiteMixinTag> x({.name = "x", .type = Type::kFP32, .shape = {1}});
  auto y = Add(x, 1.0f);
  y.SetName("y");
  std::vector<char> model;
  ASSERT_TRUE(Save(std::vector<Tensor<TfLiteMixinTag>>{y}, model).ok());
  auto runner_or = LitertDynamicRunner::Create(
      env,
      absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(model.data()),
                                model.size()),
      options);
  ASSERT_TRUE(runner_or.ok());
  if (!runner_or.ok()) return;
  auto runner = std::move(*runner_or);
  EXPECT_FALSE(runner.ResizeInput("x", {3}).ok());  // Static signature.
  ASSERT_TRUE(runner.ResizeInput("x", {3}, false).ok());
  alignas(64) float data[32] = {1, 2, 3};
  // NOLINTBEGIN(misc-include-cleaner)
  LITERT_ASSIGN_OR_ABORT(
      auto wrong,
      TensorBuffer::CreateFromHostMemory(
          env, RankedTensorType(ElementType::Float32, Layout(Dimensions{1})),
          data, sizeof(data)));
  // NOLINTEND(misc-include-cleaner)
  EXPECT_FALSE(runner.SetInputBuffer("x", std::move(wrong)).ok());
  ASSERT_TRUE(runner
                  .SetInput("x", absl::Span<const uint8_t>(
                                     reinterpret_cast<const uint8_t*>(data),
                                     3 * sizeof(float)))
                  .ok());
  ASSERT_TRUE(runner.Run().ok());
  auto result = runner.GetOutput("y");
  ASSERT_TRUE(result.ok());
  if (!result.ok()) return;
  EXPECT_EQ(result->GetShape(), (Shape{3}));
  auto output = result->GetBufferPtr()->Lock().As<const float>();
  ASSERT_EQ(output.size(), 3);
  EXPECT_EQ(output.data()[0], 2);
  EXPECT_EQ(output.data()[2], 4);
  ASSERT_TRUE(runner.RegisterFeedbackLoop("x", "y").ok());
  EXPECT_FALSE(runner.ResizeInput("x", {2}, false).ok());
}

TEST(LitertDynamicRunnerTest, FeedbackLoopTest) {
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

  std::vector<FeedbackLoopConfig> feedback_loops = {
      {.input_name = "x", .output_name = "y"}};

  auto runner_or =
      LitertDynamicRunner::Create(env, buffer_span, options, feedback_loops);
  ASSERT_TRUE(runner_or.ok());
  auto runner = std::move(*runner_or);

  // Initial input
  std::vector<float> input_data = {0.0f};
  absl::Span<const uint8_t> input_span(
      reinterpret_cast<const uint8_t*>(input_data.data()),
      input_data.size() * sizeof(float));

  EXPECT_TRUE(runner.SetInput("x", input_span).ok());

  // Run 1: y = 0 + 1 = 1
  const void* y_ptr_1 = nullptr;
  EXPECT_TRUE(runner.Run().ok());
  {
    auto y_or = runner.GetOutput("y");
    ASSERT_TRUE(y_or.ok());
    auto y_tensor = std::move(*y_or);
    auto buffer_or = y_tensor.GetBuffer();
    ASSERT_TRUE(buffer_or.ok());
    auto locked_span = buffer_or->Lock();
    y_ptr_1 = locked_span.data();
    const float* data = reinterpret_cast<const float*>(y_ptr_1);
    EXPECT_EQ(data[0], 1.0f);
  }

  // Run 2: y = 1 + 1 = 2
  EXPECT_TRUE(runner.Run().ok());
  {
    auto y_or = runner.GetOutput("y");
    ASSERT_TRUE(y_or.ok());
    auto y_tensor = std::move(*y_or);
    auto buffer_or = y_tensor.GetBuffer();
    ASSERT_TRUE(buffer_or.ok());
    auto locked_span = buffer_or->Lock();
    const float* data = reinterpret_cast<const float*>(locked_span.data());
    EXPECT_EQ(data[0], 2.0f);

    // Verify zero-copy: input of Run 2 is the same buffer as output of Run 1
    auto x_or = runner.GetInput("x");
    ASSERT_TRUE(x_or.ok());
    auto x_tensor = std::move(*x_or);
    auto x_buffer_or = x_tensor.GetBuffer();
    ASSERT_TRUE(x_buffer_or.ok());
    auto x_locked_span = x_buffer_or->Lock();
    const void* x_ptr_2 = x_locked_span.data();
    EXPECT_EQ(x_ptr_2, y_ptr_1);
  }

  // Run 3: y = 2 + 1 = 3
  EXPECT_TRUE(runner.Run().ok());
  {
    auto y_or = runner.GetOutput("y");
    ASSERT_TRUE(y_or.ok());
    auto y_tensor = std::move(*y_or);
    auto buffer_or = y_tensor.GetBuffer();
    ASSERT_TRUE(buffer_or.ok());
    auto locked_span = buffer_or->Lock();
    const float* data = reinterpret_cast<const float*>(locked_span.data());
    EXPECT_EQ(data[0], 3.0f);
  }

  // Reset and verify we can start again with new input
  EXPECT_TRUE(runner.Reset().ok());

  std::vector<float> input_data2 = {10.0f};
  absl::Span<const uint8_t> input_span2(
      reinterpret_cast<const uint8_t*>(input_data2.data()),
      input_data2.size() * sizeof(float));
  EXPECT_TRUE(runner.SetInput("x", input_span2).ok());

  // Run 1 after reset: y = 10 + 1 = 11
  EXPECT_TRUE(runner.Run().ok());
  {
    auto y_or = runner.GetOutput("y");
    ASSERT_TRUE(y_or.ok());
    auto y_tensor = std::move(*y_or);
    auto buffer_or = y_tensor.GetBuffer();
    ASSERT_TRUE(buffer_or.ok());
    auto locked_span = buffer_or->Lock();
    const float* data = reinterpret_cast<const float*>(locked_span.data());
    EXPECT_EQ(data[0], 11.0f);
  }
}

TEST(LambdaModelRunnerTest, FeedbackLoopTest) {
  LITERT_ASSIGN_OR_ABORT(auto env, Environment::Create({}));
  LITERT_ASSIGN_OR_ABORT(auto options, Options::Create());
  options.SetHardwareAccelerators(HwAccelerators::kCpu);

  std::vector<FeedbackLoopConfig> feedback_loops = {
      {.input_name = "x", .output_name = "y"}};

  auto runner = CreateLambdaRunner(
      env, options,
      TensorsMap{{"x", Tensor<TfLiteMixinTag>(
                           {.name = "x", .type = Type::kFP32, .shape = {1}})}},
      [](const auto& inputs) {
        Tensor y = Add(inputs.at("x"), 1.0f);
        return TensorsMap{{"y", y}};
      },
      feedback_loops);

  // Initial input
  std::vector<float> input_data = {0.0f};
  auto x_tensor = Create("x", Type::kFP32, {1}, std::move(input_data));
  EXPECT_TRUE(runner.SetInput("x", x_tensor).ok());

  // Run 1: y = 0 + 1 = 1
  const void* y_ptr_1 = nullptr;
  EXPECT_TRUE(runner.Run().ok());
  {
    auto y_or = runner.GetOutput("y");
    ASSERT_TRUE(y_or.ok());
    auto y_tensor = std::move(*y_or);
    auto buffer_or = y_tensor.GetBuffer();
    ASSERT_TRUE(buffer_or.ok());
    auto locked_span = buffer_or->Lock();
    y_ptr_1 = locked_span.data();
    const float* data = reinterpret_cast<const float*>(y_ptr_1);
    EXPECT_EQ(data[0], 1.0f);
  }

  // Run 2: y = 1 + 1 = 2
  EXPECT_TRUE(runner.Run().ok());
  {
    auto y_or = runner.GetOutput("y");
    ASSERT_TRUE(y_or.ok());
    auto y_tensor = std::move(*y_or);
    auto buffer_or = y_tensor.GetBuffer();
    ASSERT_TRUE(buffer_or.ok());
    auto locked_span = buffer_or->Lock();
    const float* data = reinterpret_cast<const float*>(locked_span.data());
    EXPECT_EQ(data[0], 2.0f);

    // Verify zero-copy: input of Run 2 is the same buffer as output of Run 1
    auto x_or = runner.GetInput("x");
    ASSERT_TRUE(x_or.ok());
    auto x_tensor = std::move(*x_or);
    auto x_buffer_or = x_tensor.GetBuffer();
    ASSERT_TRUE(x_buffer_or.ok());
    auto x_locked_span = x_buffer_or->Lock();
    const void* x_ptr_2 = x_locked_span.data();
    EXPECT_EQ(x_ptr_2, y_ptr_1);
  }

  // Reset and verify we can start again with new input
  EXPECT_TRUE(runner.Reset().ok());

  std::vector<float> input_data2 = {10.0f};
  auto x_tensor2 = Create("x", Type::kFP32, {1}, std::move(input_data2));
  EXPECT_TRUE(runner.SetInput("x", x_tensor2).ok());

  // Run 1 after reset: y = 10 + 1 = 11
  EXPECT_TRUE(runner.Run().ok());
  {
    auto y_or = runner.GetOutput("y");
    ASSERT_TRUE(y_or.ok());
    auto y_tensor = std::move(*y_or);
    auto buffer_or = y_tensor.GetBuffer();
    ASSERT_TRUE(buffer_or.ok());
    auto locked_span = buffer_or->Lock();
    const float* data = reinterpret_cast<const float*>(locked_span.data());
    EXPECT_EQ(data[0], 11.0f);
  }
}

}  // namespace
}  // namespace litert::tensor
