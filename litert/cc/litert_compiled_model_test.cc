// Copyright 2024 Google LLC.
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

#include "litert/cc/litert_compiled_model.h"

#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_profiler.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_requirements.h"
#include "litert/cc/options/litert_runtime_options.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "litert/test/testdata/simple_model_test_vectors.h"

using ::testing::ElementsAre;
using ::testing::FloatNear;
using ::testing::Pointwise;
using ::testing::SizeIs;
using ::testing::litert::IsOkAndHolds;

namespace litert {
namespace {

TEST(CompiledModelTest, Basic) {
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, litert::Environment::Create({}));

  // Create Model.
  Model model = testing::LoadTestFileModel(kModelFileName);
  ASSERT_TRUE(model);

  // Create CompiledModel.
  LITERT_ASSERT_OK_AND_ASSIGN(
      CompiledModel compiled_model,
      CompiledModel::Create(env, model, kLiteRtHwAcceleratorCpu));

  // Check fully accelerated.
  LITERT_ASSERT_OK_AND_ASSIGN(auto fullyAccelerated,
                              compiled_model.IsFullyAccelerated());
  ASSERT_TRUE(fullyAccelerated);

  // Check CompiledModel buffer requirements.
  // input and output expect host memory.
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBufferRequirements input_buffer_requirements_arg0,
      compiled_model.GetInputBufferRequirements(/*input_name=*/"arg0"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<LiteRtTensorBufferType> input_buffer_types_arg0,
      input_buffer_requirements_arg0.SupportedTypes());
  EXPECT_THAT(input_buffer_types_arg0,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBufferRequirements input_buffer_requirements_arg1,
      compiled_model.GetInputBufferRequirements(/*input_name=*/"arg1"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<LiteRtTensorBufferType> input_buffer_types_arg1,
      input_buffer_requirements_arg1.SupportedTypes());
  EXPECT_THAT(input_buffer_types_arg1,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBufferRequirements output_buffer_requirements,
      compiled_model.GetOutputBufferRequirements(/*output_name=*/"tfl.add"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<LiteRtTensorBufferType> output_buffer_types,
      output_buffer_requirements.SupportedTypes());
  EXPECT_THAT(output_buffer_types,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  // Create and fill input and output buffers.
  LITERT_ASSERT_OK_AND_ASSIGN(std::vector<TensorBuffer> input_buffers,
                              compiled_model.CreateInputBuffers());

  LITERT_ASSERT_OK_AND_ASSIGN(std::vector<TensorBuffer> output_buffers,
                              compiled_model.CreateOutputBuffers());

  ASSERT_TRUE(input_buffers[0].Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size)));
  ASSERT_TRUE(input_buffers[1].Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size)));

  // Execute model with input and output buffers.
  compiled_model.Run(input_buffers, output_buffers);

  // Check model output.
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create<const float>(
            output_buffers[0], TensorBuffer::LockMode::kRead));
    auto output = absl::MakeSpan(lock_and_addr.second, kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));
  }
}

TEST(CompiledModelTest,
     ResizeInputTensorReflectsInCreatedInputBufferForSignature) {
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, litert::Environment::Create({}));

  Model model = testing::LoadTestFileModel(kDynamicModelFileName);
  ASSERT_TRUE(model);

  LITERT_ASSERT_OK_AND_ASSIGN(
      CompiledModel compiled_model,
      CompiledModel::Create(env, model, kLiteRtHwAcceleratorCpu));

  absl::string_view signature_key = model.DefaultSignatureKey();

  const std::vector<int> resized_dims = {4, 2, 3};
  LITERT_ASSERT_OK(compiled_model.ResizeInputTensor(
      signature_key, "arg0", absl::MakeConstSpan(resized_dims)));

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer input_buffer,
      compiled_model.CreateInputBuffer(signature_key, "arg0"));
  LITERT_ASSERT_OK_AND_ASSIGN(RankedTensorType buffer_type,
                              input_buffer.TensorType());
  EXPECT_THAT(buffer_type.Layout().Dimensions(),
              ElementsAre(resized_dims[0], resized_dims[1], resized_dims[2]));
}

TEST(CompiledModelTest, BasicSignatureIndex) {
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, litert::Environment::Create({}));

  // Create Model and check signatures.
  Model model = testing::LoadTestFileModel(kModelFileName);
  ASSERT_TRUE(model);

  EXPECT_EQ(model.GetNumSignatures(), 1);
  size_t signature_index = 0;

  LITERT_ASSERT_OK_AND_ASSIGN(auto input_names,
                              model.GetSignatureInputNames(signature_index));
  EXPECT_THAT(input_names, ElementsAre("arg0", "arg1"));

  LITERT_ASSERT_OK_AND_ASSIGN(auto output_names,
                              model.GetSignatureOutputNames(signature_index));
  EXPECT_THAT(output_names, ElementsAre("tfl.add"));

  // Create CompiledModel.
  LITERT_ASSERT_OK_AND_ASSIGN(
      CompiledModel compiled_model,
      CompiledModel::Create(env, model, kLiteRtHwAcceleratorCpu));

  // Check CompiledModel buffer requirements.
  // input and output expect host memory.
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBufferRequirements input_buffer_requirements_arg0,
      compiled_model.GetInputBufferRequirements(signature_index,
                                                /*input_name=*/"arg0"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<LiteRtTensorBufferType> input_buffer_types_arg0,
      input_buffer_requirements_arg0.SupportedTypes());
  EXPECT_THAT(input_buffer_types_arg0,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBufferRequirements input_buffer_requirements_arg1,
      compiled_model.GetInputBufferRequirements(signature_index,
                                                /*input_name=*/"arg1"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<LiteRtTensorBufferType> input_buffer_types_arg1,
      input_buffer_requirements_arg1.SupportedTypes());
  EXPECT_THAT(input_buffer_types_arg1,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBufferRequirements output_buffer_requirements,
      compiled_model.GetOutputBufferRequirements(signature_index,
                                                 /*output_name=*/"tfl.add"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<LiteRtTensorBufferType> output_buffer_types,
      output_buffer_requirements.SupportedTypes());
  EXPECT_THAT(output_buffer_types,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  // Create and fill input and output buffers.
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<TensorBuffer> input_buffers,
      compiled_model.CreateInputBuffers(signature_index));

  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<TensorBuffer> output_buffers,
      compiled_model.CreateOutputBuffers(signature_index));

  ASSERT_TRUE(input_buffers[0].Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size)));
  ASSERT_TRUE(input_buffers[1].Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size)));

  // Execute model with input and output buffers.
  compiled_model.Run(signature_index, input_buffers, output_buffers);

  // Check model output.
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create<const float>(
            output_buffers[0], TensorBuffer::LockMode::kRead));
    auto output = absl::MakeSpan(lock_and_addr.second, kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));
  }
}

TEST(CompiledModelTest, RunWithInputOutputMap) {
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, litert::Environment::Create({}));

  // Create Model and check signatures.
  Model model = testing::LoadTestFileModel(kModelFileName);
  ASSERT_TRUE(model);

  EXPECT_EQ(model.GetNumSignatures(), 1);

  LITERT_ASSERT_OK_AND_ASSIGN(auto input_names, model.GetSignatureInputNames());
  EXPECT_THAT(input_names, ElementsAre("arg0", "arg1"));

  LITERT_ASSERT_OK_AND_ASSIGN(auto output_names,
                              model.GetSignatureOutputNames());
  EXPECT_THAT(output_names, ElementsAre("tfl.add"));

  // Create CompiledModel.
  LITERT_ASSERT_OK_AND_ASSIGN(
      CompiledModel compiled_model,
      CompiledModel::Create(env, model, kLiteRtHwAcceleratorCpu));

  // Check CompiledModel buffer requirements.
  // input and output expect host memory.
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBufferRequirements input_buffer_requirements_arg0,
      compiled_model.GetInputBufferRequirements(
          /*input_name=*/"arg0"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<LiteRtTensorBufferType> input_buffer_types_arg0,
      input_buffer_requirements_arg0.SupportedTypes());
  EXPECT_THAT(input_buffer_types_arg0,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBufferRequirements input_buffer_requirements_arg1,
      compiled_model.GetInputBufferRequirements(
          /*input_name=*/"arg1"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<LiteRtTensorBufferType> input_buffer_types_arg1,
      input_buffer_requirements_arg1.SupportedTypes());
  EXPECT_THAT(input_buffer_types_arg1,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBufferRequirements output_buffer_requirements,
      compiled_model.GetOutputBufferRequirements(
          /*output_name=*/"tfl.add"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<LiteRtTensorBufferType> output_buffer_types,
      output_buffer_requirements.SupportedTypes());
  EXPECT_THAT(output_buffer_types,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  // Create and fill input and output buffers.
  LITERT_ASSERT_OK_AND_ASSIGN(TensorBuffer input_buffer0,
                              compiled_model.CreateInputBuffer("arg0"));
  LITERT_ASSERT_OK_AND_ASSIGN(TensorBuffer input_buffer1,
                              compiled_model.CreateInputBuffer("arg1"));
  LITERT_ASSERT_OK_AND_ASSIGN(TensorBuffer output_buffer0,
                              compiled_model.CreateOutputBuffer("tfl.add"));

  ASSERT_TRUE(input_buffer0.Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size)));
  ASSERT_TRUE(input_buffer1.Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size)));

  // Create input and output map.
  absl::flat_hash_map<absl::string_view, TensorBuffer> input_map;
  input_map["arg0"] = std::move(input_buffer0);
  input_map["arg1"] = std::move(input_buffer1);

  absl::flat_hash_map<absl::string_view, TensorBuffer> output_map;
  output_map["tfl.add"] = std::move(output_buffer0);

  // Execute model with input and output maps instead of buffers.
  compiled_model.Run(input_map, output_map);

  // Check model output.
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create<const float>(
            output_map["tfl.add"], TensorBuffer::LockMode::kRead));
    auto output = absl::MakeSpan(lock_and_addr.second, kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));
  }
}

// Tests Compiled Model async API on CPU. In the CPU case, the async API should
// always return false.
TEST(CompiledModelTest, RunAsyncReturnsFalse) {
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, litert::Environment::Create({}));

  // Create Model and check signatures.
  Model model = testing::LoadTestFileModel(kModelFileName);
  ASSERT_TRUE(model);

  // Create CompiledModel.
  LITERT_ASSERT_OK_AND_ASSIGN(
      CompiledModel compiled_model,
      CompiledModel::Create(env, model, kLiteRtHwAcceleratorCpu));

  // Create input and output buffers.
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<TensorBuffer> input_buffers,
      compiled_model.CreateInputBuffers(model.DefaultSignatureKey()));
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<TensorBuffer> output_buffers,
      compiled_model.CreateOutputBuffers(model.DefaultSignatureKey()));

  // Confirm input and output buffers are host memory.
  EXPECT_THAT(input_buffers[0].BufferType(),
              IsOkAndHolds(kLiteRtTensorBufferTypeHostMemory));
  EXPECT_THAT(input_buffers[1].BufferType(),
              IsOkAndHolds(kLiteRtTensorBufferTypeHostMemory));
  EXPECT_THAT(output_buffers[0].BufferType(),
              IsOkAndHolds(kLiteRtTensorBufferTypeHostMemory));

  ASSERT_THAT(input_buffers, SizeIs(2));
  ASSERT_THAT(output_buffers, SizeIs(1));

  ASSERT_TRUE(input_buffers[0].Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size)));
  ASSERT_TRUE(input_buffers[1].Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size)));

  // Execute model with input and output buffers.
  bool async;
  compiled_model.RunAsync(model.DefaultSignatureKey(), input_buffers,
                          output_buffers, async);
  // Since there are no events on the output buffers, async should be false.
  ASSERT_FALSE(async);

  // Check model output.
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create<const float>(
            output_buffers[0], TensorBuffer::LockMode::kRead));
    auto output = absl::MakeSpan(lock_and_addr.second, kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));
  }
}
TEST(CompiledModelTest, WithProfiler) {
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, litert::Environment::Create({}));

  // Create Model.
  Model model = testing::LoadTestFileModel(kModelFileName);
  ASSERT_TRUE(model);

  LITERT_ASSIGN_OR_ABORT(Options compilation_options,
                         litert::Options::Create());
  compilation_options.SetHardwareAccelerators(kLiteRtHwAcceleratorCpu);
  LITERT_ASSIGN_OR_ABORT(auto runtime_options, RuntimeOptions::Create());
  runtime_options.SetEnableProfiling(/*enabled=*/true);
  compilation_options.AddOpaqueOptions(std::move(runtime_options));

  // Create CompiledModel.
  LITERT_ASSERT_OK_AND_ASSIGN(
      CompiledModel compiled_model,
      CompiledModel::Create(env, model, compilation_options));

  // Check fully accelerated.
  LITERT_ASSERT_OK_AND_ASSIGN(auto fullyAccelerated,
                              compiled_model.IsFullyAccelerated());
  ASSERT_TRUE(fullyAccelerated);

  // Create profiler.
  LITERT_ASSERT_OK_AND_ASSIGN(auto profiler, compiled_model.GetProfiler());
  ASSERT_TRUE(profiler);
  ASSERT_TRUE(profiler.StartProfiling());

  // Check CompiledModel buffer requirements.
  // input and output expect host memory.
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBufferRequirements input_buffer_requirements_arg0,
      compiled_model.GetInputBufferRequirements(/*input_name=*/"arg0"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<LiteRtTensorBufferType> input_buffer_types_arg0,
      input_buffer_requirements_arg0.SupportedTypes());
  EXPECT_THAT(input_buffer_types_arg0,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBufferRequirements input_buffer_requirements_arg1,
      compiled_model.GetInputBufferRequirements(/*input_name=*/"arg1"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<LiteRtTensorBufferType> input_buffer_types_arg1,
      input_buffer_requirements_arg1.SupportedTypes());
  EXPECT_THAT(input_buffer_types_arg1,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBufferRequirements output_buffer_requirements,
      compiled_model.GetOutputBufferRequirements(/*output_name=*/"tfl.add"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<LiteRtTensorBufferType> output_buffer_types,
      output_buffer_requirements.SupportedTypes());
  EXPECT_THAT(output_buffer_types,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  // Create and fill input and output buffers.
  LITERT_ASSERT_OK_AND_ASSIGN(std::vector<TensorBuffer> input_buffers,
                              compiled_model.CreateInputBuffers());

  LITERT_ASSERT_OK_AND_ASSIGN(std::vector<TensorBuffer> output_buffers,
                              compiled_model.CreateOutputBuffers());

  ASSERT_TRUE(input_buffers[0].Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size)));
  ASSERT_TRUE(input_buffers[1].Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size)));

  // Execute model with input and output buffers.
  compiled_model.Run(input_buffers, output_buffers);

  // Check profiler events.
  LITERT_ASSERT_OK_AND_ASSIGN(auto events, profiler.GetEvents());
  EXPECT_GT(events.size(), 2);
  ASSERT_TRUE(profiler.Reset());
  LITERT_ASSERT_OK_AND_ASSIGN(events, profiler.GetEvents());
  EXPECT_EQ(events.size(), 0);

  // Check model output.
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create<const float>(
            output_buffers[0], TensorBuffer::LockMode::kRead));
    auto output = absl::MakeSpan(lock_and_addr.second, kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));
  }
}
TEST(CompiledModelTest, ResizeInputTensorWithDynamicModel) {
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, litert::Environment::Create({}));

  // Create Model with dynamic shapes.
  Model model = testing::LoadTestFileModel(kDynamicModelFileName);
  ASSERT_TRUE(model);

  // Create CompiledModel.
  LITERT_ASSERT_OK_AND_ASSIGN(
      CompiledModel compiled_model,
      CompiledModel::Create(env, model, kLiteRtHwAcceleratorCpu));

  // Test resizing input tensor by index - resize from (?, 2, 3) to (1, 2, 3)
  {
    const std::vector<int> new_dims = {1, 2, 3};
    LITERT_ASSERT_OK(compiled_model.ResizeInputTensor(
        /*input_index=*/size_t(0), absl::MakeConstSpan(new_dims)));

    // Verify buffer requirements after resize
    LITERT_ASSERT_OK_AND_ASSIGN(
        TensorBufferRequirements requirements,
        compiled_model.GetInputBufferRequirements(/*input_index=*/size_t(0)));
    LITERT_ASSERT_OK_AND_ASSIGN(size_t buffer_size, requirements.BufferSize());
    EXPECT_EQ(buffer_size, 1 * 2 * 3 * sizeof(float));
  }

  // Test resizing input tensor by name - resize to (2, 2, 3)
  {
    const std::vector<int> new_dims = {2, 2, 3};
    LITERT_ASSERT_OK(compiled_model.ResizeInputTensor(
        /*input_name=*/"arg0", absl::MakeConstSpan(new_dims)));

    // Verify buffer requirements after resize
    LITERT_ASSERT_OK_AND_ASSIGN(
        TensorBufferRequirements requirements,
        compiled_model.GetInputBufferRequirements(/*input_name=*/"arg0"));
    LITERT_ASSERT_OK_AND_ASSIGN(size_t buffer_size, requirements.BufferSize());
    EXPECT_EQ(buffer_size, 2 * 2 * 3 * sizeof(float));
  }

  // Test resizing with signature index
  {
    const std::vector<int> new_dims = {3, 2, 3};
    LITERT_ASSERT_OK(compiled_model.ResizeInputTensor(
        /*signature_index=*/size_t(0), /*input_index=*/size_t(0),
        absl::MakeConstSpan(new_dims)));

    LITERT_ASSERT_OK_AND_ASSIGN(
        TensorBufferRequirements requirements,
        compiled_model.GetInputBufferRequirements(/*signature_index=*/size_t(0),
                                                  /*input_index=*/size_t(0)));
    LITERT_ASSERT_OK_AND_ASSIGN(size_t buffer_size, requirements.BufferSize());
    EXPECT_EQ(buffer_size, 3 * 2 * 3 * sizeof(float));
  }

  // Test execution after resize, using tensor buffer APIs.
  {
    // Resize to specific shape for execution
    const std::vector<int> exec_dims = {1, 2, 3};
    LITERT_ASSERT_OK(compiled_model.ResizeInputTensor(
        /*input_index=*/size_t(0), absl::MakeConstSpan(exec_dims)));
    LITERT_ASSERT_OK(compiled_model.ResizeInputTensor(
        /*input_index=*/size_t(1), absl::MakeConstSpan(exec_dims)));

    // Create input and output buffers
    LITERT_ASSERT_OK_AND_ASSIGN(
        TensorBufferRequirements req0,
        compiled_model.GetInputBufferRequirements(size_t(0)));
    LITERT_ASSERT_OK_AND_ASSIGN(size_t size0, req0.BufferSize());
    LITERT_ASSERT_OK_AND_ASSIGN(
        TensorBufferRequirements req1,
        compiled_model.GetInputBufferRequirements(size_t(1)));
    LITERT_ASSERT_OK_AND_ASSIGN(size_t size1, req1.BufferSize());

    // Get the element type from the original model.
    LITERT_ASSERT_OK_AND_ASSIGN(const RankedTensorType& type0,
                                model.GetInputTensorType(0, "arg0"));
    LITERT_ASSERT_OK_AND_ASSIGN(const RankedTensorType& type1,
                                model.GetInputTensorType(0, "arg1"));

    // Manually create a new RankedTensorType with the new shape.
    auto new_type0 = RankedTensorType(
        type0.ElementType(),
        Layout(Dimensions(exec_dims.begin(), exec_dims.end())));
    auto new_type1 = RankedTensorType(
        type1.ElementType(),
        Layout(Dimensions(exec_dims.begin(), exec_dims.end())));

    LITERT_ASSERT_OK_AND_ASSIGN(
        TensorBuffer input_buffer0,
        TensorBuffer::CreateManaged(
            env.Get(), kLiteRtTensorBufferTypeHostMemory, new_type0, size0));
    LITERT_ASSERT_OK_AND_ASSIGN(
        TensorBuffer input_buffer1,
        TensorBuffer::CreateManaged(
            env.Get(), kLiteRtTensorBufferTypeHostMemory, new_type1, size1));
    std::vector<TensorBuffer> input_buffers;
    input_buffers.push_back(std::move(input_buffer0));
    input_buffers.push_back(std::move(input_buffer1));

    LITERT_ASSERT_OK_AND_ASSIGN(
        TensorBufferRequirements out_req,
        compiled_model.GetOutputBufferRequirements(size_t(0)));
    LITERT_ASSERT_OK_AND_ASSIGN(size_t out_size, out_req.BufferSize());
    LITERT_ASSERT_OK_AND_ASSIGN(const RankedTensorType& out_type,
                                model.GetOutputTensorType(0, "tfl.add"));

    // Get the output tensor shape from the compiled model.
    LITERT_ASSERT_OK_AND_ASSIGN(
        std::vector<Layout> output_tensor_layouts,
        compiled_model.GetOutputTensorLayouts(
            /*signature_index=*/size_t(0), /*update_allocation=*/true));

    ASSERT_EQ(output_tensor_layouts.size(), 1);

    auto new_out_type = RankedTensorType(out_type.ElementType(),
                                         std::move(output_tensor_layouts[0]));

    LITERT_ASSERT_OK_AND_ASSIGN(
        TensorBuffer output_buffer,
        TensorBuffer::CreateManaged(env.Get(),
                                    kLiteRtTensorBufferTypeHostMemory,
                                    new_out_type, out_size));

    std::vector<TensorBuffer> output_buffers;
    output_buffers.push_back(std::move(output_buffer));

    // Fill input buffers with test data
    const float test_input[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    ASSERT_TRUE(input_buffers[0].Write<float>(absl::MakeConstSpan(test_input)));
    ASSERT_TRUE(input_buffers[1].Write<float>(absl::MakeConstSpan(test_input)));

    // Execute model
    LITERT_ASSERT_OK(compiled_model.Run(input_buffers, output_buffers));
    // Verify output
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create<const float>(
            output_buffers[0], TensorBuffer::LockMode::kRead));
    auto output = absl::MakeSpan(lock_and_addr.second, 6);

    // For an add operation, expected output is input + input
    const float expected_output[] = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), expected_output));
  }
}
// Test error reporter with BufferErrorReporter mode
TEST(CompiledModelTest, ErrorReporterBufferMode) {
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, litert::Environment::Create({}));

  // Create Model.
  Model model = testing::LoadTestFileModel(kModelFileName);
  ASSERT_TRUE(model);

  // Create compilation options with BufferErrorReporter
  LITERT_ASSERT_OK_AND_ASSIGN(Options compilation_options, Options::Create());
  compilation_options.SetHardwareAccelerators(kLiteRtHwAcceleratorCpu);

  // Configure BufferErrorReporter mode
  LITERT_ASSERT_OK_AND_ASSIGN(auto runtime_options, RuntimeOptions::Create());
  LITERT_ASSERT_OK(
      runtime_options.SetErrorReporterMode(kLiteRtErrorReporterModeBuffer));
  LITERT_ASSERT_OK(
      compilation_options.AddOpaqueOptions(std::move(runtime_options)));

  // Create CompiledModel with BufferErrorReporter
  LITERT_ASSERT_OK_AND_ASSIGN(
      CompiledModel compiled_model,
      CompiledModel::Create(env, model, compilation_options));

  // Test 1: Basic ReportError functionality
  LITERT_ASSERT_OK(compiled_model.ReportError("Simple error message"));

  // Test 2: ReportError with formatting - integers
  LITERT_ASSERT_OK(compiled_model.ReportError("Error code: %d", 404));

  // Test 3: ReportError with formatting - strings
  LITERT_ASSERT_OK(
      compiled_model.ReportError("Failed operation: %s", "tensor_allocation"));

  // Test 4: ReportError with multiple format specifiers
  LITERT_ASSERT_OK(compiled_model.ReportError(
      "Complex error: %s at line %d with value %f", "overflow", 42, 3.14159));

  // Test 5: GetErrorMessages - verify all messages are captured
  LITERT_ASSERT_OK_AND_ASSIGN(std::string error_messages,
                              compiled_model.GetErrorMessages());
  EXPECT_THAT(error_messages, ::testing::HasSubstr("Simple error message"));
  EXPECT_THAT(error_messages, ::testing::HasSubstr("Error code: 404"));
  EXPECT_THAT(error_messages,
              ::testing::HasSubstr("Failed operation: tensor_allocation"));
  EXPECT_THAT(error_messages,
              ::testing::HasSubstr(
                  "Complex error: overflow at line 42 with value 3.14"));

  // Test 6: ClearErrors functionality
  LITERT_ASSERT_OK(compiled_model.ClearErrors());

  // Test 7: Verify errors are cleared
  LITERT_ASSERT_OK_AND_ASSIGN(error_messages,
                              compiled_model.GetErrorMessages());
  EXPECT_TRUE(error_messages.empty());

  // Test 8: Add errors after clearing
  LITERT_ASSERT_OK(compiled_model.ReportError("New error after clear"));
  LITERT_ASSERT_OK_AND_ASSIGN(error_messages,
                              compiled_model.GetErrorMessages());
  EXPECT_THAT(error_messages, ::testing::HasSubstr("New error after clear"));
  EXPECT_THAT(error_messages,
              ::testing::Not(::testing::HasSubstr("Simple error message")));
}

// Test error reporter with default StderrReporter mode
TEST(CompiledModelTest, ErrorReporterStderrMode) {
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, litert::Environment::Create({}));

  // Create Model.
  Model model = testing::LoadTestFileModel(kModelFileName);
  ASSERT_TRUE(model);

  // Create CompiledModel with default StderrReporter
  LITERT_ASSERT_OK_AND_ASSIGN(
      CompiledModel compiled_model,
      CompiledModel::Create(env, model, kLiteRtHwAcceleratorCpu));

  // ReportError should work with StderrReporter (prints to stderr)
  LITERT_ASSERT_OK(compiled_model.ReportError("This goes to stderr: %d", 123));

  // GetErrorMessages should fail with StderrReporter
  auto messages_result = compiled_model.GetErrorMessages();
  EXPECT_FALSE(messages_result.HasValue());

  // ClearErrors should fail with StderrReporter
  auto clear_result = compiled_model.ClearErrors();
  EXPECT_FALSE(clear_result.HasValue());
}

// Test error reporter with edge cases
TEST(CompiledModelTest, ErrorReporterEdgeCases) {
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, litert::Environment::Create({}));

  // Create Model.
  Model model = testing::LoadTestFileModel(kModelFileName);
  ASSERT_TRUE(model);

  // Create compilation options with BufferErrorReporter
  LITERT_ASSERT_OK_AND_ASSIGN(Options compilation_options, Options::Create());
  compilation_options.SetHardwareAccelerators(kLiteRtHwAcceleratorCpu);

  LITERT_ASSERT_OK_AND_ASSIGN(auto runtime_options, RuntimeOptions::Create());
  LITERT_ASSERT_OK(
      runtime_options.SetErrorReporterMode(kLiteRtErrorReporterModeBuffer));
  LITERT_ASSERT_OK(
      compilation_options.AddOpaqueOptions(std::move(runtime_options)));

  LITERT_ASSERT_OK_AND_ASSIGN(
      CompiledModel compiled_model,
      CompiledModel::Create(env, model, compilation_options));

  // Test 1: Empty error message
  LITERT_ASSERT_OK(compiled_model.ReportError(""));

  // Test 2: Very long error message
  std::string long_message(1000, 'A');
  LITERT_ASSERT_OK(compiled_model.ReportError("%s", long_message.c_str()));

  // Test 3: Special characters in error message
  LITERT_ASSERT_OK(
      compiled_model.ReportError("Special chars: \n\t\r %% %s", "test"));

  // Test 4: Multiple consecutive clears
  LITERT_ASSERT_OK(compiled_model.ClearErrors());
  LITERT_ASSERT_OK(compiled_model.ClearErrors());

  // Test 5: GetErrorMessages when no errors reported
  LITERT_ASSERT_OK_AND_ASSIGN(std::string error_messages,
                              compiled_model.GetErrorMessages());
  EXPECT_TRUE(error_messages.empty());

  // Test 6: Large number of error messages
  for (int i = 0; i < 100; ++i) {
    LITERT_ASSERT_OK(compiled_model.ReportError("Error number %d", i));
  }

  LITERT_ASSERT_OK_AND_ASSIGN(error_messages,
                              compiled_model.GetErrorMessages());
  // Verify we have all 100 error messages
  for (int i = 0; i < 100; ++i) {
    EXPECT_THAT(error_messages,
                ::testing::HasSubstr(absl::StrFormat("Error number %d", i)));
  }
}

// Test error reporter with None mode (no error reporting)
TEST(CompiledModelTest, ErrorReporterNoneMode) {
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, litert::Environment::Create({}));

  // Create Model.
  Model model = testing::LoadTestFileModel(kModelFileName);
  ASSERT_TRUE(model);

  // Create compilation options with no error reporter
  LITERT_ASSERT_OK_AND_ASSIGN(Options compilation_options, Options::Create());
  compilation_options.SetHardwareAccelerators(kLiteRtHwAcceleratorCpu);

  LITERT_ASSERT_OK_AND_ASSIGN(auto runtime_options, RuntimeOptions::Create());
  LITERT_ASSERT_OK(
      runtime_options.SetErrorReporterMode(kLiteRtErrorReporterModeNone));
  LITERT_ASSERT_OK(
      compilation_options.AddOpaqueOptions(std::move(runtime_options)));

  LITERT_ASSERT_OK_AND_ASSIGN(
      CompiledModel compiled_model,
      CompiledModel::Create(env, model, compilation_options));

  // ReportError should work but do nothing
  LITERT_ASSERT_OK(
      compiled_model.ReportError("This should be ignored: %d", 999));

  // GetErrorMessages should fail with None mode
  auto messages_result = compiled_model.GetErrorMessages();
  EXPECT_FALSE(messages_result.HasValue());

  // ClearErrors should fail with None mode
  auto clear_result = compiled_model.ClearErrors();
  EXPECT_FALSE(clear_result.HasValue());
}

TEST(CompiledModelTest, DispatchAnnotations) {
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, litert::Environment::Create({}));

  // Create Model.
  Model model = testing::LoadTestFileModel(kModelFileName);
  ASSERT_TRUE(model);

  // Create CompiledModel.
  LITERT_ASSERT_OK_AND_ASSIGN(
      CompiledModel compiled_model,
      CompiledModel::Create(env, model, kLiteRtHwAcceleratorCpu));

  // Test 1: Set and get annotation for signature index 0
  {
    LITERT_ASSERT_OK(
        compiled_model.SetDispatchAnnotation(0, "priority", "high"));

    LITERT_ASSERT_OK_AND_ASSIGN(
        auto value, compiled_model.GetDispatchAnnotation(0, "priority"));
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(value.value(), "high");
  }

  // Test 2: Set and get annotation for default signature (overload without
  // index)
  {
    LITERT_ASSERT_OK(
        compiled_model.SetDispatchAnnotation("memory_type", "shared"));

    LITERT_ASSERT_OK_AND_ASSIGN(
        auto value, compiled_model.GetDispatchAnnotation("memory_type"));
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(value.value(), "shared");
  }

  // Test 3: Get non-existent annotation
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto value, compiled_model.GetDispatchAnnotation(0, "nonexistent"));
    EXPECT_FALSE(value.has_value());
  }

  // Test 4: Remove annotation
  {
    // First set an annotation
    LITERT_ASSERT_OK(
        compiled_model.SetDispatchAnnotation(0, "to_remove", "value"));

    // Verify it exists
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto value_before,
        compiled_model.GetDispatchAnnotation(0, "to_remove"));
    ASSERT_TRUE(value_before.has_value());

    // Remove it
    LITERT_ASSERT_OK(compiled_model.RemoveDispatchAnnotation(0, "to_remove"));

    // Verify it's gone
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto value_after, compiled_model.GetDispatchAnnotation(0, "to_remove"));
    EXPECT_FALSE(value_after.has_value());
  }

  // Test 5: Remove non-existent annotation (should succeed)
  {
    LITERT_ASSERT_OK(
        compiled_model.RemoveDispatchAnnotation(0, "never_existed"));
  }

  // Test 6: Update existing annotation
  {
    LITERT_ASSERT_OK(
        compiled_model.SetDispatchAnnotation(0, "accelerator", "gpu"));

    LITERT_ASSERT_OK_AND_ASSIGN(
        auto value1, compiled_model.GetDispatchAnnotation(0, "accelerator"));
    ASSERT_TRUE(value1.has_value());
    EXPECT_EQ(value1.value(), "gpu");

    // Update to new value
    LITERT_ASSERT_OK(
        compiled_model.SetDispatchAnnotation(0, "accelerator", "npu"));

    LITERT_ASSERT_OK_AND_ASSIGN(
        auto value2, compiled_model.GetDispatchAnnotation(0, "accelerator"));
    ASSERT_TRUE(value2.has_value());
    EXPECT_EQ(value2.value(), "npu");
  }

  // Test 7: Multiple annotations on same signature
  {
    LITERT_ASSERT_OK(compiled_model.SetDispatchAnnotation(0, "key1", "value1"));
    LITERT_ASSERT_OK(compiled_model.SetDispatchAnnotation(0, "key2", "value2"));
    LITERT_ASSERT_OK(compiled_model.SetDispatchAnnotation(0, "key3", "value3"));

    LITERT_ASSERT_OK_AND_ASSIGN(
        auto val1, compiled_model.GetDispatchAnnotation(0, "key1"));
    ASSERT_TRUE(val1.has_value());
    EXPECT_EQ(val1.value(), "value1");

    LITERT_ASSERT_OK_AND_ASSIGN(
        auto val2, compiled_model.GetDispatchAnnotation(0, "key2"));
    ASSERT_TRUE(val2.has_value());
    EXPECT_EQ(val2.value(), "value2");

    LITERT_ASSERT_OK_AND_ASSIGN(
        auto val3, compiled_model.GetDispatchAnnotation(0, "key3"));
    ASSERT_TRUE(val3.has_value());
    EXPECT_EQ(val3.value(), "value3");
  }

  // Test 8: Empty string values
  {
    LITERT_ASSERT_OK(
        compiled_model.SetDispatchAnnotation(0, "empty_value", ""));

    LITERT_ASSERT_OK_AND_ASSIGN(
        auto value, compiled_model.GetDispatchAnnotation(0, "empty_value"));
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(value.value(), "");
  }

  // Test 9: Special characters in keys and values
  {
    LITERT_ASSERT_OK(compiled_model.SetDispatchAnnotation(
        0, "key-with-dashes", "value/with/slashes"));

    LITERT_ASSERT_OK_AND_ASSIGN(
        auto value, compiled_model.GetDispatchAnnotation(0, "key-with-dashes"));
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(value.value(), "value/with/slashes");
  }
}

TEST(CompiledModelTest, DispatchAnnotationsWithSignatureName) {
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, litert::Environment::Create({}));

  // Create Model.
  Model model = testing::LoadTestFileModel(kModelFileName);
  ASSERT_TRUE(model);

  EXPECT_EQ(model.GetNumSignatures(), 1);

  // Create CompiledModel.
  LITERT_ASSERT_OK_AND_ASSIGN(
      CompiledModel compiled_model,
      CompiledModel::Create(env, model, kLiteRtHwAcceleratorCpu));

  // Test 1: Set and get annotation using signature name
  {
    LITERT_ASSERT_OK(compiled_model.SetDispatchAnnotation("precision", "fp16"));

    LITERT_ASSERT_OK_AND_ASSIGN(
        auto value, compiled_model.GetDispatchAnnotation("precision"));
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(value.value(), "fp16");
  }

  // Test 2: Remove annotation using signature name
  {
    LITERT_ASSERT_OK(
        compiled_model.SetDispatchAnnotation("to_remove", "value"));

    LITERT_ASSERT_OK_AND_ASSIGN(
        auto value_before, compiled_model.GetDispatchAnnotation("to_remove"));
    ASSERT_TRUE(value_before.has_value());

    LITERT_ASSERT_OK(compiled_model.RemoveDispatchAnnotation("to_remove"));

    LITERT_ASSERT_OK_AND_ASSIGN(
        auto value_after, compiled_model.GetDispatchAnnotation("to_remove"));
    EXPECT_FALSE(value_after.has_value());
  }

  // Test 3: Annotations set by name should be retrievable by index
  {
    LITERT_ASSERT_OK(
        compiled_model.SetDispatchAnnotation("cross_check", "test_value"));

    // Since this is signature 0, we should be able to get it by index 0
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto value, compiled_model.GetDispatchAnnotation(0, "cross_check"));
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(value.value(), "test_value");
  }
}

TEST(CompiledModelTest, DispatchAnnotationsInvalidSignature) {
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, litert::Environment::Create({}));

  // Create Model.
  Model model = testing::LoadTestFileModel(kModelFileName);
  ASSERT_TRUE(model);

  // Create CompiledModel.
  LITERT_ASSERT_OK_AND_ASSIGN(
      CompiledModel compiled_model,
      CompiledModel::Create(env, model, kLiteRtHwAcceleratorCpu));

  // Test with invalid signature index (model only has 1 signature at index 0)
  {
    auto result = compiled_model.SetDispatchAnnotation(999, "key", "value");
    EXPECT_FALSE(result.HasValue());
  }

  {
    auto result = compiled_model.GetDispatchAnnotation(999, "key");
    EXPECT_FALSE(result.HasValue());
  }

  {
    auto result = compiled_model.RemoveDispatchAnnotation(999, "key");
    EXPECT_FALSE(result.HasValue());
  }

  // Test with invalid signature name
  {
    auto result = compiled_model.SetDispatchAnnotation("nonexistent_signature",
                                                       "key", "value");
    EXPECT_FALSE(result.HasValue());
  }

  {
    auto result =
        compiled_model.GetDispatchAnnotation("nonexistent_signature", "key");
    EXPECT_FALSE(result.HasValue());
  }

  {
    auto result =
        compiled_model.RemoveDispatchAnnotation("nonexistent_signature", "key");
    EXPECT_FALSE(result.HasValue());
  }
}

// Test for constant output tensor support
TEST(CompiledModelTest, ConstantOutputTensor) {
  // Create Model with constant output tensor.
  Model model = testing::LoadTestFileModel(kConstantOutputTensorModelFileName);
  ASSERT_TRUE(model);

  // Environment setup
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, litert::Environment::Create({}));

  // Create CompiledModel
  LITERT_ASSERT_OK_AND_ASSIGN(
      CompiledModel compiled_model,
      CompiledModel::Create(env, model, kLiteRtHwAcceleratorCpu));

  EXPECT_EQ(model.GetNumSignatures(), 1);
  size_t signature_index = 0;

  // Create input and output buffers
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<TensorBuffer> input_buffers,
      compiled_model.CreateInputBuffers(signature_index));
  ASSERT_EQ(input_buffers.size(), 1);

  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<TensorBuffer> output_buffers,
      compiled_model.CreateOutputBuffers(signature_index));
  ASSERT_EQ(output_buffers.size(), 2);  // normal_output and constant_output

  // Set input values
  const float input_data[] = {5.0f, 10.0f};
  ASSERT_TRUE(
      input_buffers[0].Write<float>(absl::MakeConstSpan(input_data, 2)));

  // Run the model
  LITERT_ASSERT_OK(
      compiled_model.Run(signature_index, input_buffers, output_buffers));

  // Note: TFLite might reorder outputs - check which is which by size
  // The constant output has 4 elements, the normal output has 2 elements
  int constant_output_idx = -1;
  int normal_output_idx = -1;

  // Determine which output is which based on size
  for (int i = 0; i < 2; i++) {
    LITERT_ASSERT_OK_AND_ASSIGN(auto size, output_buffers[i].Size());
    if (size == 4 * sizeof(float)) {
      constant_output_idx = i;
    } else if (size == 2 * sizeof(float)) {
      normal_output_idx = i;
    }
  }

  ASSERT_NE(constant_output_idx, -1) << "Could not find constant output";
  ASSERT_NE(normal_output_idx, -1) << "Could not find normal output";

  // Check normal output (should be [10.0, 20.0])
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create<const float>(
            output_buffers[normal_output_idx], TensorBuffer::LockMode::kRead));
    auto output = absl::MakeSpan(lock_and_addr.second, 2);
    EXPECT_THAT(output,
                ElementsAre(FloatNear(10.0f, 1e-5), FloatNear(20.0f, 1e-5)));
  }

  // Check constant output (should always be [1.0, 2.0, 3.0, 4.0])
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr, litert::TensorBufferScopedLock::Create<const float>(
                                output_buffers[constant_output_idx],
                                TensorBuffer::LockMode::kRead));
    auto output = absl::MakeSpan(lock_and_addr.second, 4);
    EXPECT_THAT(output,
                ElementsAre(FloatNear(1.0f, 1e-5), FloatNear(2.0f, 1e-5),
                            FloatNear(3.0f, 1e-5), FloatNear(4.0f, 1e-5)));
    ABSL_LOG(INFO) << "Constant output tensor test passed. Values: ["
                   << output[0] << ", " << output[1] << ", " << output[2]
                   << ", " << output[3] << "]";
  }

  // Run again with different input to verify constant output doesn't change
  const float input_data2[] = {100.0f, 200.0f};
  ASSERT_TRUE(
      input_buffers[0].Write<float>(absl::MakeConstSpan(input_data2, 2)));
  LITERT_ASSERT_OK(
      compiled_model.Run(signature_index, input_buffers, output_buffers));

  // Check normal output changed (should be [200.0, 400.0])
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create<const float>(
            output_buffers[normal_output_idx], TensorBuffer::LockMode::kRead));
    auto output = absl::MakeSpan(lock_and_addr.second, 2);
    EXPECT_THAT(output,
                ElementsAre(FloatNear(200.0f, 1e-5), FloatNear(400.0f, 1e-5)))
        << "Normal output should reflect new input values";
  }

  // Check that constant output is still [1.0, 2.0, 3.0, 4.0]
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr, litert::TensorBufferScopedLock::Create<const float>(
                                output_buffers[constant_output_idx],
                                TensorBuffer::LockMode::kRead));
    auto output = absl::MakeSpan(lock_and_addr.second, 4);
    EXPECT_THAT(output,
                ElementsAre(FloatNear(1.0f, 1e-5), FloatNear(2.0f, 1e-5),
                            FloatNear(3.0f, 1e-5), FloatNear(4.0f, 1e-5)))
        << "Constant output should not change with different inputs";
  }
}

TEST(CompiledModelTest, ExternalTensorBinding) {
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, litert::Environment::Create({}));

  // Create Model.
  Model model = testing::LoadTestFileModel(kModelFileName);
  ASSERT_TRUE(model);

  // Create weight tensor buffer.
  alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) float kWeightTensor[] = {1.0f,
                                                                        2.0f};
  constexpr int kWeightSize = sizeof(kWeightTensor);

  // Create Compilation options and bind weight tensor.
  LITERT_ASSERT_OK_AND_ASSIGN(Options compilation_options, Options::Create());
  compilation_options.SetHardwareAccelerators(kLiteRtHwAcceleratorCpu);
  LITERT_ASSERT_OK(compilation_options.AddExternalTensorBinding(
      /*signature_name=*/"", /*tensor_name=*/"arg1", kWeightTensor,
      kWeightSize));

  // Create CompiledModel.
  LITERT_ASSERT_OK_AND_ASSIGN(
      CompiledModel compiled_model,
      CompiledModel::Create(env, model, compilation_options));

  // Create and fill input and output buffers.
  LITERT_ASSERT_OK_AND_ASSIGN(std::vector<TensorBuffer> output_buffers,
                              compiled_model.CreateOutputBuffers());
  absl::flat_hash_map<absl::string_view, TensorBuffer> output_map;
  ASSERT_THAT(output_buffers, SizeIs(1));
  output_map["tfl.add"] = std::move(output_buffers[0]);

  absl::flat_hash_map<absl::string_view, TensorBuffer> input_map;
  float kInputTensor[] = {1.0f, 1.0f};
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer arg0_buffer,
      TensorBuffer::CreateManaged(
          env.Get(), kLiteRtTensorBufferTypeHostMemory,
          RankedTensorType(ElementType::Float32, Layout(Dimensions({2}))),
          sizeof(kInputTensor)));
  LITERT_ASSERT_OK(
      arg0_buffer.Write<float>(absl::MakeConstSpan(kInputTensor, 2)));
  input_map["arg0"] = std::move(arg0_buffer);

  // Execute model with input and output buffers.
  LITERT_ASSERT_OK(compiled_model.Run(input_map, output_map));

  // Check model output.
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto lock_and_addr,
      litert::TensorBufferScopedLock::Create<const float>(
          output_map["tfl.add"], TensorBuffer::LockMode::kRead));
  auto output = absl::MakeSpan(lock_and_addr.second, 2);
  constexpr float kExpectedOutput[] = {2.0f, 3.0f};
  EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kExpectedOutput));
}

}  // namespace
}  // namespace litert
