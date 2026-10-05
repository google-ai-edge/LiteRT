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

#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/c/options/litert_cpu_options.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "litert/runtime/op_resolver.h"
#include "litert/test/common.h"
#include "tflite/schema/schema_generated.h"

namespace litert {
namespace {

TEST(SelectiveCompiledModelTest, OnlyRegistersSelectedOpsAndVersions) {
  auto resolver = internal::CreateOpResolver(false);
  ASSERT_TRUE(resolver);
  EXPECT_NE((*resolver)->FindOp(tflite::BuiltinOperator_ADD, 1), nullptr);
  EXPECT_EQ((*resolver)->FindOp(tflite::BuiltinOperator_ADD, 999), nullptr);
  EXPECT_EQ((*resolver)->FindOp(tflite::BuiltinOperator_MUL, 1), nullptr);
}

TEST(SelectiveCompiledModelTest, RunsRepeatedlyWithReusableBuffers) {
  auto env = Environment::Create({});
  ASSERT_TRUE(env);
  auto model = CompiledModel::Create(
      *env, testing::GetTestFilePath("simple_model.tflite"),
      HwAccelerators::kCpu);
  ASSERT_TRUE(model);
  auto inputs = model->CreateInputBuffers();
  auto outputs = model->CreateOutputBuffers();
  ASSERT_TRUE(inputs);
  ASSERT_TRUE(outputs);
  ASSERT_EQ(inputs->size(), 2);
  ASSERT_EQ(outputs->size(), 1);
  for (float scale : {1.0f, 2.0f}) {
    const float lhs[] = {scale, 2 * scale};
    const float rhs[] = {10 * scale, 20 * scale};
    ASSERT_TRUE((*inputs)[0].Write<float>(lhs));
    ASSERT_TRUE((*inputs)[1].Write<float>(rhs));
    ASSERT_TRUE(model->Run(*inputs, *outputs));
    float result[2];
    ASSERT_TRUE((*outputs)[0].Read<float>(result));
    EXPECT_FLOAT_EQ(result[0], 11 * scale);
    EXPECT_FLOAT_EQ(result[1], 22 * scale);
  }
}

TEST(SelectiveCompiledModelTest, RejectsModelWithUnregisteredOp) {
  auto env = Environment::Create({});
  ASSERT_TRUE(env);
  EXPECT_FALSE(CompiledModel::Create(
      *env, testing::GetTestFilePath("simple_mul_op.tflite"),
      HwAccelerators::kCpu));
}

TEST(SelectiveCompiledModelTest, RejectsUnavailableReferenceMode) {
  auto env = Environment::Create({});
  ASSERT_TRUE(env);
  auto options = Options::Create();
  ASSERT_TRUE(options);
  ASSERT_TRUE(options->SetHardwareAccelerators(HwAccelerators::kCpu));
  auto cpu = options->GetCpuOptions();
  ASSERT_TRUE(cpu);
  ASSERT_TRUE(cpu->SetKernelMode(kLiteRtCpuKernelModeReference));
  auto model = CompiledModel::Create(
      *env, testing::GetTestFilePath("simple_model.tflite"), *options);
  ASSERT_FALSE(model);
  EXPECT_EQ(model.Error().Status(), kLiteRtStatusErrorUnsupported);
}

}  // namespace
}  // namespace litert
