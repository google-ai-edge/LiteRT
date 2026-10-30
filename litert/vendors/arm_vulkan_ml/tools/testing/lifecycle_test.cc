// Copyright 2026 The LiteRT Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_arm_vulkan_ml_options.h"
#include "tflite/schema/schema_generated.h"

namespace litert::arm_vulkan_ml {
namespace {
// Small exact ADD models avoid dataset dependencies and large simultaneous
// emulator instances. Constants and expected sums are independent of lowering.
std::string WriteModel(bool quantized, bool softmax = false) {
  tflite::ModelT model;
  model.version = 3;
  model.buffers.push_back(std::make_unique<tflite::BufferT>());
  model.buffers.push_back(std::make_unique<tflite::BufferT>());
  auto& bytes = model.buffers[1]->data;
  if (quantized) bytes.assign(4, 2);
  else {
    const float constants[] = {2, 2, 2, 2};
    bytes.resize(sizeof(constants));
    std::memcpy(bytes.data(), constants, sizeof(constants));
  }
  auto graph = std::make_unique<tflite::SubGraphT>();
  for (int i = 0; i < 3; ++i) {
    auto tensor = std::make_unique<tflite::TensorT>();
    tensor->name = "tensor_" + std::to_string(i);
    tensor->shape = {1, 4};
    tensor->type = quantized ? tflite::TensorType_INT8 : tflite::TensorType_FLOAT32;
    tensor->buffer = i == 1 ? 1 : 0;
    if (quantized) {
      tensor->quantization = std::make_unique<tflite::QuantizationParametersT>();
      tensor->quantization->scale = {1.0f};
      tensor->quantization->zero_point = {0};
      if (softmax) {
        tensor->quantization->scale = {i == 2 ? 1.0f / 256 : .125f};
        tensor->quantization->zero_point = {i == 2 ? -128 : 17};
      }
    }
    graph->tensors.push_back(std::move(tensor));
  }
  auto opcode = std::make_unique<tflite::OperatorCodeT>();
  opcode->builtin_code = softmax ? tflite::BuiltinOperator_SOFTMAX : tflite::BuiltinOperator_ADD;
  opcode->version = quantized ? 2 : 1;
  model.operator_codes.push_back(std::move(opcode));
  auto op = std::make_unique<tflite::OperatorT>();
  op->inputs = softmax ? std::vector<int32_t>{0} : std::vector<int32_t>{0, 1};
  op->outputs = {2};
  if (softmax) {
    tflite::SoftmaxOptionsT options;
    options.beta = .7f;
    op->builtin_options.Set(options);
  } else {
    op->builtin_options.Set(tflite::AddOptionsT{});
  }
  graph->operators.push_back(std::move(op));
  graph->inputs = {0}; graph->outputs = {2};
  model.subgraphs.push_back(std::move(graph));
  flatbuffers::FlatBufferBuilder builder;
  tflite::FinishModelBuffer(builder, tflite::Model::Pack(builder, &model));
  const auto path = testing::TempDir() + (softmax ? "/softmax_int8.tflite" : quantized ? "/lifecycle_int8.tflite" : "/lifecycle_fp32.tflite");
  std::ofstream file(path, std::ios::binary);
  file.write(reinterpret_cast<const char*>(builder.GetBufferPointer()), builder.GetSize());
  EXPECT_TRUE(file.good());
  return path;
}

template <typename T>
Expected<std::vector<T>> RunInference(CompiledModel& model, const std::vector<T>& input) {
  LITERT_ASSIGN_OR_RETURN(auto inputs, model.CreateInputBuffers());
  LITERT_ASSIGN_OR_RETURN(auto outputs, model.CreateOutputBuffers());
  LITERT_RETURN_IF_ERROR(inputs[0].Write<T>(Span<const T>(input.data(), input.size())));
  LITERT_RETURN_IF_ERROR(model.Run(inputs, outputs));
  std::vector<T> output(input.size());
  LITERT_RETURN_IF_ERROR(outputs[0].Read<T>(Span<T>(output.data(), output.size())));
  return output;
}

template <typename T>
void CheckLifecycle(Environment& environment, const std::string& path) {
  auto create = [&]() -> Expected<CompiledModel> {
    LITERT_ASSIGN_OR_RETURN(auto options, Options::Create());
    LITERT_RETURN_IF_ERROR(options.SetHardwareAccelerators(HwAccelerators::kNpu));
    LITERT_ASSIGN_OR_RETURN(auto& arm, options.GetOptions<ArmVulkanMLOptions>());
    LITERT_RETURN_IF_ERROR(arm.SetEnableJustInTime(true));
    return CompiledModel::Create(environment, path, options);
  };
  auto first = create();
  ASSERT_TRUE(first) << first.Error().Message();
  auto accelerated = first->IsFullyAccelerated();
  ASSERT_TRUE(accelerated); ASSERT_TRUE(*accelerated);
  const std::vector<T> input{-3, 0, 1, 5}, expected{-1, 2, 3, 7};
  for (int i = 0; i < 3; ++i) {
    auto actual = RunInference(*first, input);
    ASSERT_TRUE(actual); EXPECT_EQ(*actual, expected);
  }
  {
    auto second = create();
    ASSERT_TRUE(second) << second.Error().Message();
    accelerated = second->IsFullyAccelerated();
    ASSERT_TRUE(accelerated); ASSERT_TRUE(*accelerated);
    auto actual = RunInference(*second, input);
    ASSERT_TRUE(actual); EXPECT_EQ(*actual, expected);
    actual = RunInference(*first, std::vector<T>{1, 2, 3, 4});
    ASSERT_TRUE(actual); EXPECT_EQ(*actual, (std::vector<T>{3, 4, 5, 6}));
    actual = RunInference(*second, input);
    ASSERT_TRUE(actual); EXPECT_EQ(*actual, expected);
  }
  auto actual = RunInference(*first, input);
  ASSERT_TRUE(actual); EXPECT_EQ(*actual, expected);
}

TEST(LifecycleTest, RepeatedAndSimultaneousCompiledModelsOwnTheirResources) {
  if (!std::getenv("LITERT_ARM_VULKAN_ML_RUN_INTEGRATION_TESTS")) {
    GTEST_SKIP() << "Opt in after configuring a native device or Arm emulator";
  }
  const char* compiler = std::getenv("ARM_TEST_COMPILER");
  const char* dispatch = std::getenv("ARM_TEST_DISPATCH");
  ASSERT_NE(compiler, nullptr); ASSERT_NE(dispatch, nullptr);
  std::vector<EnvironmentOptions::Option> options{
      {EnvironmentOptions::Tag::kCompilerPluginLibraryDir,
       std::filesystem::absolute(compiler).parent_path().string()},
      {EnvironmentOptions::Tag::kDispatchLibraryDir,
       std::filesystem::absolute(dispatch).parent_path().string()}};
  auto environment = Environment::Create(EnvironmentOptions(options));
  ASSERT_TRUE(environment);
  CheckLifecycle<float>(*environment, WriteModel(false));
  CheckLifecycle<int8_t>(*environment, WriteModel(true));
}

TEST(LifecycleTest, QuantizedSoftmaxOptionsReachTheJitCompiler) {
  if (!std::getenv("LITERT_ARM_VULKAN_ML_RUN_INTEGRATION_TESTS")) {
    GTEST_SKIP() << "Opt in after configuring a native device or Arm emulator";
  }
  const char* compiler = std::getenv("ARM_TEST_COMPILER");
  const char* dispatch = std::getenv("ARM_TEST_DISPATCH");
  ASSERT_NE(compiler, nullptr);
  ASSERT_NE(dispatch, nullptr);
  std::vector<EnvironmentOptions::Option> environment_options{
      {EnvironmentOptions::Tag::kCompilerPluginLibraryDir,
       std::filesystem::absolute(compiler).parent_path().string()},
      {EnvironmentOptions::Tag::kDispatchLibraryDir,
       std::filesystem::absolute(dispatch).parent_path().string()}};
  auto environment = Environment::Create(EnvironmentOptions(environment_options));
  ASSERT_TRUE(environment);
  const auto path = WriteModel(true, true);
  for (auto lut : {kLrtArmVulkanMLSoftmaxLutAuto, kLrtArmVulkanMLSoftmaxLutInt8}) {
    auto options = Options::Create();
    ASSERT_TRUE(options);
    ASSERT_TRUE(options->SetHardwareAccelerators(HwAccelerators::kNpu));
    auto arm = options->GetOptions<ArmVulkanMLOptions>();
    ASSERT_TRUE(arm);
    ASSERT_TRUE(arm->SetEnableJustInTime(true));
    ASSERT_TRUE(arm->SetSoftmaxLutType(lut));
    auto model = CompiledModel::Create(*environment, path, *options);
    ASSERT_TRUE(model) << model.Error().Message();
    auto accelerated = model->IsFullyAccelerated();
    ASSERT_TRUE(accelerated);
    ASSERT_TRUE(*accelerated);
    for (const std::vector<int8_t>& input : {
        std::vector<int8_t>{-37, -37, -37, -37}, {-3, 0, 1, 5},
        {-128, -128, -128, 127}}) {
      auto actual = RunInference(*model, input);
      ASSERT_TRUE(actual) << actual.Error().Message();
      double exponential[4], sum = 0;
      for (int i = 0; i < 4; ++i) {
        exponential[i] = std::exp(static_cast<double>(input[i]) * .125 * .7f);
        sum += exponential[i];
      }
      for (int i = 0; i < 4; ++i) {
        int expected = std::min(255, static_cast<int>(std::floor(256 * exponential[i] / sum + .5))) - 128;
        EXPECT_LE(std::abs(static_cast<int>((*actual)[i]) - expected), 1);
      }
    }
  }
}

TEST(LifecycleTest, ScalarAndComplexBufferLayoutsReachTheDispatchApi) {
  if (!std::getenv("LITERT_ARM_VULKAN_ML_RUN_INTEGRATION_TESTS"))
    GTEST_SKIP() << "Opt in after configuring a native device or Arm emulator";
  const char* compiler = std::getenv("ARM_TEST_COMPILER");
  const char* dispatch = std::getenv("ARM_TEST_DISPATCH");
  ASSERT_NE(compiler, nullptr); ASSERT_NE(dispatch, nullptr);
  std::vector<EnvironmentOptions::Option> environment_options{
      {EnvironmentOptions::Tag::kCompilerPluginLibraryDir,
       std::filesystem::absolute(compiler).parent_path().string()},
      {EnvironmentOptions::Tag::kDispatchLibraryDir,
       std::filesystem::absolute(dispatch).parent_path().string()}};
  auto environment = Environment::Create(EnvironmentOptions(environment_options));
  ASSERT_TRUE(environment);
  for (bool complex : {false, true}) {
    tflite::ModelT object;
    object.version = 3;
    object.buffers.push_back(std::make_unique<tflite::BufferT>());
    auto graph = std::make_unique<tflite::SubGraphT>();
    for (int i = 0; i < 2; ++i) {
      auto tensor = std::make_unique<tflite::TensorT>();
      tensor->name = "tensor_" + std::to_string(i);
      tensor->shape = i == 1 && !complex ? std::vector<int32_t>{} : std::vector<int32_t>{2, 3};
      tensor->type = complex ? (i == 0 ? tflite::TensorType_COMPLEX64 : tflite::TensorType_FLOAT32)
                             : (i == 0 ? tflite::TensorType_FLOAT32 : tflite::TensorType_INT32);
      graph->tensors.push_back(std::move(tensor));
    }
    auto opcode = std::make_unique<tflite::OperatorCodeT>();
    opcode->builtin_code = complex ? tflite::BuiltinOperator_IMAG : tflite::BuiltinOperator_RANK;
    object.operator_codes.push_back(std::move(opcode));
    auto operation = std::make_unique<tflite::OperatorT>();
    operation->inputs = {0}; operation->outputs = {1};
    graph->operators.push_back(std::move(operation));
    graph->inputs = {0}; graph->outputs = {1};
    object.subgraphs.push_back(std::move(graph));
    flatbuffers::FlatBufferBuilder builder;
    tflite::FinishModelBuffer(builder, tflite::Model::Pack(builder, &object));
    const auto path = testing::TempDir() + (complex ? "/imag.tflite" : "/rank.tflite");
    {
      std::ofstream file(path, std::ios::binary);
      file.write(reinterpret_cast<const char*>(builder.GetBufferPointer()), builder.GetSize());
      ASSERT_TRUE(file.good());
    }
    auto options = Options::Create(); ASSERT_TRUE(options);
    ASSERT_TRUE(options->SetHardwareAccelerators(HwAccelerators::kNpu));
    auto arm = options->GetOptions<ArmVulkanMLOptions>(); ASSERT_TRUE(arm);
    ASSERT_TRUE(arm->SetEnableJustInTime(true));
    auto model = CompiledModel::Create(*environment, path, *options);
    ASSERT_TRUE(model) << model.Error().Message();
    auto accelerated = model->IsFullyAccelerated(); ASSERT_TRUE(accelerated); ASSERT_TRUE(*accelerated);
    auto inputs = model->CreateInputBuffers(); ASSERT_TRUE(inputs);
    auto outputs = model->CreateOutputBuffers(); ASSERT_TRUE(outputs);
    const std::vector<float> values = complex ? std::vector<float>{1, 2, 3, -4, 5, 6, -7, 8, 9, 10, 11, -12}
                                              : std::vector<float>{1, 2, 3, 4, 5, 6};
    // COMPLEX64 storage consists of interleaved FP32 real/imaginary values.
    ASSERT_TRUE((*inputs)[0].Write<float>(Span<const float>(values.data(), values.size())));
    for (int run = 0; run < 3; ++run) {
      ASSERT_TRUE(model->Run(*inputs, *outputs));
      if (complex) {
        std::vector<float> actual(6);
        ASSERT_TRUE((*outputs)[0].Read<float>(Span<float>(actual.data(), actual.size())));
        EXPECT_EQ(actual, (std::vector<float>{2, -4, 6, 8, 10, -12}));
      } else {
        int32_t rank = -1;
        ASSERT_TRUE((*outputs)[0].Read<int32_t>(Span<int32_t>(&rank, 1)));
        EXPECT_EQ(rank, 2);
      }
    }
  }
}
}  // namespace
}  // namespace litert::arm_vulkan_ml
