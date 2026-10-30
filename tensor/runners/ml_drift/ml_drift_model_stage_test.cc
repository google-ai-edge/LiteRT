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

#include "tensor/runners/ml_drift/ml_drift_model_stage.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer_types.h"
#include "litert/test/common.h"
#include "litert/test/testdata/simple_model_test_vectors.h"
#include "tensor/arithmetic.h"
#include "tensor/backends/ml_drift/arithmetic_ml_drift.h"
#include "tensor/datatypes.h"
#include "tensor/runners/litert/litert_buffer.h"
#include "tensor/runners/model_chain.h"
#include "tensor/tensor.h"
#include "tensor/utils/matchers.h"

namespace litert::tensor {
namespace {

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::NotNull;

using TensorMLD = Tensor<MlDriftMixinTag>;
using Stage = MlDriftClModelStage;

TEST(MlDriftModelStageTest, DescriptorsDerivedCorrectly) {
  TensorMLD in1({.name = "in1", .type = Type::kFP32, .shape = {1, 4}});
  TensorMLD in2({.name = "in2", .type = Type::kFP32, .shape = {1, 4}});
  auto out = litert::tensor::Add(in1, in2);
  out.SetName("out");

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto stage, Stage::Create("AddStage", {in1, in2}, {out}));

  EXPECT_THAT(stage->Name(), Eq("AddStage"));
  EXPECT_THAT(stage->InputNames(), ElementsAre("in1", "in2"));
  EXPECT_THAT(stage->OutputNames(), ElementsAre("out"));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto in1_desc,
                                  stage->GetInputDescriptor("in1"));
  EXPECT_THAT(in1_desc.element_type, Eq(litert::ElementType::Float32));
  EXPECT_THAT(in1_desc.shape, ElementsAre(1, 4));
  EXPECT_THAT(in1_desc.size_bytes, Eq(4 * sizeof(float)));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto out_desc,
                                  stage->GetOutputDescriptor("out"));
  EXPECT_THAT(out_desc.element_type, Eq(litert::ElementType::Float32));
  EXPECT_THAT(out_desc.shape, ElementsAre(1, 4));
  EXPECT_THAT(out_desc.size_bytes, Eq(4 * sizeof(float)));
}

TEST(MlDriftModelStageTest, StandaloneExecution) {
  auto env_or = litert::Environment::Create({});
  ASSERT_TRUE(env_or.HasValue());
  auto env = std::make_shared<litert::Environment>(std::move(*env_or));

  TensorMLD x({.name = "x", .type = Type::kFP32, .shape = {1, 2}});
  auto y = litert::tensor::Add(x, x);
  y.SetName("y");

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto stage, Stage::Create("DoubleStage", {x}, {y}));
  stage->SetEnvironment(env);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto in_buf,
      LitertBuffer::CreateManagedHost(env, {1, 2}, litert::ElementType::Float32,
                                      2 * sizeof(float)));
  {
    auto span = in_buf->LockMutable();
    auto* d = reinterpret_cast<float*>(span.data());
    d[0] = 3.0f;
    d[1] = 5.0f;
  }
  ASSERT_TRUE(stage->SetInputBuffer("x", in_buf).ok());

  ASSERT_TRUE(stage->Run().ok());

  auto out_buf = stage->GetOutputBuffer("y");
  ASSERT_THAT(out_buf, NotNull());
  auto span = out_buf->Lock();
  const auto* d = reinterpret_cast<const float*>(span.data());
  EXPECT_FLOAT_EQ(d[0], 6.0f);
  EXPECT_FLOAT_EQ(d[1], 10.0f);
}

TEST(MlDriftModelStageTest, InteroperatesWithModelChain) {
  auto env_or = litert::Environment::Create({});
  ASSERT_TRUE(env_or.HasValue());
  auto env = std::make_shared<litert::Environment>(std::move(*env_or));

  // Stage 1: MlDriftModelStage doubles input: x + x -> y
  TensorMLD x({.name = "x", .type = Type::kFP32, .shape = {1, 2}});
  auto y = litert::tensor::Add(x, x);
  y.SetName("y");

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto stage1, Stage::Create("GpuDouble", {x}, {y}));

  // Stage 2: FunctionalModelStage adds 100 to input: y + 100 -> z
  HardwareBufferDescriptor desc{
      .shape = {1, 2},
      .element_type = litert::ElementType::Float32,
      .size_bytes = 2 * sizeof(float),
      .buffer_type = litert::TensorBufferType::kHostMemory,
  };
  auto stage2 = std::make_shared<FunctionalModelStage>(
      "CpuAdd100",
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>{{"in", desc}},
      absl::flat_hash_map<std::string, HardwareBufferDescriptor>{{"out", desc}},
      [](const auto& inputs, const auto& outputs) {
        auto in_span = inputs.at("in")->Lock();
        const auto* in_data = reinterpret_cast<const float*>(in_span.data());
        auto out_span = outputs.at("out")->LockMutable();
        auto* out_data = reinterpret_cast<float*>(out_span.data());
        out_data[0] = in_data[0] + 100.0f;
        out_data[1] = in_data[1] + 100.0f;
        return absl::OkStatus();
      });

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto chain, ModelChain::Builder()
                      .WithEnvironment(env)
                      .AddStage(stage1)
                      .AddStage(stage2)
                      .Connect("GpuDouble", "y", "CpuAdd100", "in")
                      .Build());

  // Set input: [1.5, 4.0]
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto in_buf,
      LitertBuffer::CreateManagedHost(env, {1, 2}, litert::ElementType::Float32,
                                      2 * sizeof(float)));
  {
    auto span = in_buf->LockMutable();
    auto* d = reinterpret_cast<float*>(span.data());
    d[0] = 1.5f;
    d[1] = 4.0f;
  }
  ASSERT_TRUE(chain.SetInputBuffer("GpuDouble", "x", in_buf).ok());

  ASSERT_TRUE(chain.Execute().ok());

  // Expected:
  // Stage 1 (x * 2): [3.0, 8.0]
  // Stage 2 (+ 100): [103.0, 108.0]
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto out_buf, chain.GetOutputBuffer("out"));
  auto span = out_buf->Lock();
  const auto* d = reinterpret_cast<const float*>(span.data());
  EXPECT_FLOAT_EQ(d[0], 103.0f);
  EXPECT_FLOAT_EQ(d[1], 108.0f);
}

TEST(MlDriftModelStageTest, ChainedGpuStages) {
  auto env_or = litert::Environment::Create({});
  ASSERT_TRUE(env_or.HasValue());
  auto env = std::make_shared<litert::Environment>(std::move(*env_or));

  // Stage 1: a + a -> b
  TensorMLD a({.name = "a", .type = Type::kFP32, .shape = {1, 2}});
  auto b = litert::tensor::Add(a, a);
  b.SetName("b");
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto stage1, Stage::Create("Stage1", {a}, {b}));

  // Stage 2: b + b -> c
  TensorMLD b_in({.name = "b_in", .type = Type::kFP32, .shape = {1, 2}});
  auto c = litert::tensor::Add(b_in, b_in);
  c.SetName("c");
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto stage2, Stage::Create("Stage2", {b_in}, {c}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto chain, ModelChain::Builder()
                      .WithEnvironment(env)
                      .AddStage(stage1)
                      .AddStage(stage2)
                      .Connect("Stage1", "b", "Stage2", "b_in")
                      .Build());

  // Set input: [2.0, 7.0]
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto in_buf,
      LitertBuffer::CreateManagedHost(env, {1, 2}, litert::ElementType::Float32,
                                      2 * sizeof(float)));
  {
    auto span = in_buf->LockMutable();
    auto* d = reinterpret_cast<float*>(span.data());
    d[0] = 2.0f;
    d[1] = 7.0f;
  }
  ASSERT_TRUE(chain.SetInputBuffer("Stage1", "a", in_buf).ok());

  ASSERT_TRUE(chain.Execute().ok());

  // Expected:
  // Stage 1: [4.0, 14.0]
  // Stage 2: [8.0, 28.0]
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto out_buf, chain.GetOutputBuffer("c"));
  auto span = out_buf->Lock();
  const auto* d = reinterpret_cast<const float*>(span.data());
  EXPECT_FLOAT_EQ(d[0], 8.0f);
  EXPECT_FLOAT_EQ(d[1], 28.0f);
}

TEST(MlDriftModelStageTest, MlDriftToCompiledModelZeroCopyAndComputation) {
  auto env_or = litert::Environment::Create({});
  ASSERT_TRUE(env_or.HasValue());
  auto env = std::make_shared<litert::Environment>(std::move(*env_or));

  // Stage 1: MlDriftClModelStage doubles input: x + x -> y (shape {1, 2})
  TensorMLD x({.name = "x", .type = Type::kFP32, .shape = {1, 2}});
  auto y = litert::tensor::Add(x, x);
  y.SetName("y");
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto stage1, Stage::Create("GpuDouble", {x}, {y}));

  // Stage 2: CompiledModelStage adds input_0 + input_1 -> output_0 (shape {2})
  auto options_or = litert::Options::Create();
  ASSERT_TRUE(options_or.HasValue());
  options_or->SetHardwareAccelerators(litert::HwAccelerators::kCpu);

  std::string model_path = testing::GetTestFilePath(kModelFileName);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto stage2, CompiledModelStage::Create(env, "CompiledAdd", model_path,
                                              std::move(*options_or)));

  auto input_names = stage2->InputNames();
  auto output_names = stage2->OutputNames();
  ASSERT_EQ(input_names.size(), 2);
  ASSERT_EQ(output_names.size(), 1);

  // Connect Stage 1 ("y", shape {1, 2}) to Stage 2 (input_names[0], shape {2})
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto chain, ModelChain::Builder()
                      .WithEnvironment(env)
                      .AddStage(stage1)
                      .AddStage(stage2)
                      .Connect("GpuDouble", "y", "CompiledAdd", input_names[0])
                      .Build());

  // Verify zero-copy shared boundary buffer reuse between MlDriftModelStage
  // output and CompiledModelStage input:
  auto stage1_out_buf = stage1->GetOutputBuffer("y");
  auto stage2_in0_buf = stage2->GetInputBuffer(input_names[0]);
  ASSERT_THAT(stage1_out_buf, NotNull());
  ASSERT_THAT(stage2_in0_buf, NotNull());
  EXPECT_EQ(stage1_out_buf.get(), stage2_in0_buf.get());
  EXPECT_EQ(stage1_out_buf->tensor_buffer().Get(),
            stage2_in0_buf->tensor_buffer().Get());

  // Bind external inputs:
  // x = [3.0, 5.0] -> Stage 1 computes y = [6.0, 10.0]
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto in_x,
      LitertBuffer::CreateManagedHost(env, {1, 2}, litert::ElementType::Float32,
                                      2 * sizeof(float)));
  {
    auto span = in_x->LockMutable();
    auto* d = reinterpret_cast<float*>(span.data());
    d[0] = 3.0f;
    d[1] = 5.0f;
  }
  ASSERT_TRUE(chain.SetInputBuffer("GpuDouble", "x", in_x).ok());

  // input_1 = [10.0, 20.0] -> Stage 2 computes [6.0, 10.0] + [10.0, 20.0] =
  // [16.0, 30.0]
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto in_1,
      LitertBuffer::CreateManagedHost(env, {2}, litert::ElementType::Float32,
                                      2 * sizeof(float)));
  {
    auto span = in_1->LockMutable();
    auto* d = reinterpret_cast<float*>(span.data());
    d[0] = 10.0f;
    d[1] = 20.0f;
  }
  ASSERT_TRUE(chain.SetInputBuffer("CompiledAdd", input_names[1], in_1).ok());

  ASSERT_TRUE(chain.Execute().ok());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto out_buf, chain.GetOutputBuffer("CompiledAdd", output_names[0]));
  auto out_span = out_buf->Lock();
  const auto* out_data = reinterpret_cast<const float*>(out_span.data());
  EXPECT_FLOAT_EQ(out_data[0], 16.0f);
  EXPECT_FLOAT_EQ(out_data[1], 30.0f);
}

TEST(MlDriftModelStageTest, CompiledModelToMlDriftZeroCopyAndComputation) {
  auto env_or = litert::Environment::Create({});
  ASSERT_TRUE(env_or.HasValue());
  auto env = std::make_shared<litert::Environment>(std::move(*env_or));

  // Stage 1: CompiledModelStage adds input_0 + input_1 -> output_0 (shape {2})
  auto options_or = litert::Options::Create();
  ASSERT_TRUE(options_or.HasValue());
  options_or->SetHardwareAccelerators(litert::HwAccelerators::kCpu);

  std::string model_path = testing::GetTestFilePath(kModelFileName);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto stage1, CompiledModelStage::Create(env, "CompiledAdd", model_path,
                                              std::move(*options_or)));

  auto input_names = stage1->InputNames();
  auto output_names = stage1->OutputNames();
  ASSERT_EQ(input_names.size(), 2);
  ASSERT_EQ(output_names.size(), 1);

  // Stage 2: MlDriftClModelStage doubles input: x + x -> y (shape {1, 2})
  TensorMLD x({.name = "x", .type = Type::kFP32, .shape = {1, 2}});
  auto y = litert::tensor::Add(x, x);
  y.SetName("y");
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto stage2, Stage::Create("GpuDouble", {x}, {y}));

  // Connect Stage 1 (output_names[0], shape {2}) to Stage 2 ("x", shape {1, 2})
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto chain, ModelChain::Builder()
                      .WithEnvironment(env)
                      .AddStage(stage1)
                      .AddStage(stage2)
                      .Connect("CompiledAdd", output_names[0], "GpuDouble", "x")
                      .Build());

  // Verify zero-copy shared boundary buffer reuse between CompiledModelStage
  // output and MlDriftModelStage input:
  auto stage1_out_buf = stage1->GetOutputBuffer(output_names[0]);
  auto stage2_in_buf = stage2->GetInputBuffer("x");
  ASSERT_THAT(stage1_out_buf, NotNull());
  ASSERT_THAT(stage2_in_buf, NotNull());
  EXPECT_EQ(stage1_out_buf.get(), stage2_in_buf.get());
  EXPECT_EQ(stage1_out_buf->tensor_buffer().Get(),
            stage2_in_buf->tensor_buffer().Get());

  // Bind external inputs:
  // input_0 = [2.0, 4.0], input_1 = [3.0, 5.0] -> Stage 1 computes [5.0, 9.0]
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto in_0,
      LitertBuffer::CreateManagedHost(env, {2}, litert::ElementType::Float32,
                                      2 * sizeof(float)));
  {
    auto span = in_0->LockMutable();
    auto* d = reinterpret_cast<float*>(span.data());
    d[0] = 2.0f;
    d[1] = 4.0f;
  }
  ASSERT_TRUE(chain.SetInputBuffer("CompiledAdd", input_names[0], in_0).ok());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto in_1,
      LitertBuffer::CreateManagedHost(env, {2}, litert::ElementType::Float32,
                                      2 * sizeof(float)));
  {
    auto span = in_1->LockMutable();
    auto* d = reinterpret_cast<float*>(span.data());
    d[0] = 3.0f;
    d[1] = 5.0f;
  }
  ASSERT_TRUE(chain.SetInputBuffer("CompiledAdd", input_names[1], in_1).ok());

  // Stage 2 computes [5.0, 9.0] * 2 = [10.0, 18.0]
  ASSERT_TRUE(chain.Execute().ok());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto out_buf, chain.GetOutputBuffer("GpuDouble", "y"));
  auto out_span = out_buf->Lock();
  const auto* out_data = reinterpret_cast<const float*>(out_span.data());
  EXPECT_FLOAT_EQ(out_data[0], 10.0f);
  EXPECT_FLOAT_EQ(out_data[1], 18.0f);
}

}  // namespace
}  // namespace litert::tensor
