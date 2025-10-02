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

#include "litert/c/litert_compiled_model.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_logging.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/test/common.h"
#include "litert/test/testdata/simple_model_test_vectors.h"

using testing::FloatNear;
using testing::Pointwise;

namespace litert {
namespace {

TEST(CompiledModelTest, Basic) {
  auto path = testing::GetTestFilePath(kModelFileName);

  LiteRtModel model;
  ASSERT_EQ(LiteRtCreateModelFromFile(path.c_str(), &model), kLiteRtStatusOk);

  LiteRtOptions jit_compilation_options;
  ASSERT_EQ(LiteRtCreateOptions(&jit_compilation_options), kLiteRtStatusOk);
  ASSERT_EQ(LiteRtSetOptionsHardwareAccelerators(jit_compilation_options,
                                                 kLiteRtHwAcceleratorCpu),
            kLiteRtStatusOk);

  LiteRtEnvironment environment;
  LiteRtEnvOption options = {};
  ASSERT_EQ(LiteRtCreateEnvironment(/*num_options=*/0, &options, &environment),
            kLiteRtStatusOk);

  LiteRtCompiledModel compiled_model;
  ASSERT_EQ(LiteRtCreateCompiledModel(environment, model,
                                      jit_compilation_options, &compiled_model),
            kLiteRtStatusOk);

  LiteRtDestroyOptions(jit_compilation_options);

  LiteRtSubgraph subgraph;
  ASSERT_EQ(LiteRtGetModelSubgraph(model, 0, &subgraph), kLiteRtStatusOk);

  LiteRtParamIndex num_inputs;
  ASSERT_EQ(LiteRtGetNumSubgraphInputs(subgraph, &num_inputs), kLiteRtStatusOk);

  std::vector<LiteRtTensorBuffer> input_tensor_buffers;
  input_tensor_buffers.reserve(num_inputs);
  for (auto i = 0; i < num_inputs; ++i) {
    LiteRtTensorBufferRequirements tensor_buffer_requirements;
    ASSERT_EQ(LiteRtGetCompiledModelInputBufferRequirements(
                  compiled_model, /*signature_index=*/0, i,
                  &tensor_buffer_requirements),
              kLiteRtStatusOk);
    LiteRtTensorBufferType tensor_buffer_type;
    EXPECT_EQ(
        LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(
            tensor_buffer_requirements, /*type_index=*/0, &tensor_buffer_type),
        kLiteRtStatusOk);
    size_t tensor_buffer_size;
    EXPECT_EQ(LiteRtGetTensorBufferRequirementsBufferSize(
                  tensor_buffer_requirements, &tensor_buffer_size),
              kLiteRtStatusOk);
    LiteRtTensorBuffer tensor_buffer;
    EXPECT_EQ(LiteRtCreateManagedTensorBuffer(
                  environment, tensor_buffer_type, &kInput0TensorType,
                  tensor_buffer_size, &tensor_buffer),
              kLiteRtStatusOk);
    input_tensor_buffers.push_back(tensor_buffer);
  }

  LiteRtParamIndex num_outputs;
  ASSERT_EQ(LiteRtGetNumSubgraphOutputs(subgraph, &num_outputs),
            kLiteRtStatusOk);

  std::vector<LiteRtTensorBuffer> output_tensor_buffers;
  output_tensor_buffers.reserve(num_outputs);
  for (auto i = 0; i < num_outputs; ++i) {
    LiteRtTensorBufferRequirements tensor_buffer_requirements;
    ASSERT_EQ(LiteRtGetCompiledModelOutputBufferRequirements(
                  compiled_model, /*signature_index=*/0, i,
                  &tensor_buffer_requirements),
              kLiteRtStatusOk);
    LiteRtTensorBufferType tensor_buffer_type;
    EXPECT_EQ(
        LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(
            tensor_buffer_requirements, /*type_index=*/0, &tensor_buffer_type),
        kLiteRtStatusOk);
    size_t tensor_buffer_size;
    EXPECT_EQ(LiteRtGetTensorBufferRequirementsBufferSize(
                  tensor_buffer_requirements, &tensor_buffer_size),
              kLiteRtStatusOk);
    LiteRtTensorBuffer tensor_buffer;
    EXPECT_EQ(LiteRtCreateManagedTensorBuffer(
                  environment, tensor_buffer_type, &kInput0TensorType,
                  tensor_buffer_size, &tensor_buffer),
              kLiteRtStatusOk);
    output_tensor_buffers.push_back(tensor_buffer);
  }

  {
    ABSL_LOG(INFO) << "Filling inputs with data";
    void* host_mem_addr;

    ASSERT_EQ(LiteRtLockTensorBuffer(input_tensor_buffers[0], &host_mem_addr,
                                     kLiteRtTensorBufferLockModeWrite),
              kLiteRtStatusOk);
    std::memcpy(host_mem_addr, kTestInput0Tensor, sizeof(kTestInput0Tensor));
    ASSERT_EQ(LiteRtUnlockTensorBuffer(input_tensor_buffers[0]),
              kLiteRtStatusOk);

    ASSERT_EQ(LiteRtLockTensorBuffer(input_tensor_buffers[1], &host_mem_addr,
                                     kLiteRtTensorBufferLockModeWrite),
              kLiteRtStatusOk);
    std::memcpy(host_mem_addr, kTestInput1Tensor, sizeof(kTestInput1Tensor));
    ASSERT_EQ(LiteRtUnlockTensorBuffer(input_tensor_buffers[1]),
              kLiteRtStatusOk);
  }

  ASSERT_EQ(LiteRtRunCompiledModel(
                compiled_model, /*signature_index=*/0,
                input_tensor_buffers.size(), input_tensor_buffers.data(),
                output_tensor_buffers.size(), output_tensor_buffers.data()),
            kLiteRtStatusOk);

  {
    ABSL_LOG(INFO) << "Checking output...";
    void* host_mem_addr;
    ASSERT_EQ(LiteRtLockTensorBuffer(output_tensor_buffers[0], &host_mem_addr,
                                     kLiteRtTensorBufferLockModeRead),
              kLiteRtStatusOk);
    auto output = absl::MakeSpan(static_cast<const float*>(host_mem_addr),
                                 kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-3), kTestOutputTensor));
    ASSERT_EQ(LiteRtUnlockTensorBuffer(output_tensor_buffers[0]),
              kLiteRtStatusOk);
  }

  LiteRtDestroyCompiledModel(compiled_model);
  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(environment);

  for (auto tensor_buffer : input_tensor_buffers) {
    LiteRtDestroyTensorBuffer(tensor_buffer);
  }
  for (auto tensor_buffer : output_tensor_buffers) {
    LiteRtDestroyTensorBuffer(tensor_buffer);
  }
}

TEST(CompiledModelTest, ResizeInputTensorWithDynamicModel) {
  // Use the dynamic model for testing resize functionality
  auto path = testing::GetTestFilePath(kDynamicModelFileName);

  LiteRtModel model;
  ASSERT_EQ(LiteRtCreateModelFromFile(path.c_str(), &model), kLiteRtStatusOk);

  LiteRtOptions jit_compilation_options;
  ASSERT_EQ(LiteRtCreateOptions(&jit_compilation_options), kLiteRtStatusOk);
  ASSERT_EQ(LiteRtSetOptionsHardwareAccelerators(jit_compilation_options,
                                                 kLiteRtHwAcceleratorCpu),
            kLiteRtStatusOk);

  LiteRtEnvironment environment;
  LiteRtEnvOption options = {};
  ASSERT_EQ(LiteRtCreateEnvironment(/*num_options=*/0, &options, &environment),
            kLiteRtStatusOk);

  LiteRtCompiledModel compiled_model;
  ASSERT_EQ(LiteRtCreateCompiledModel(environment, model,
                                      jit_compilation_options, &compiled_model),
            kLiteRtStatusOk);

  LiteRtDestroyOptions(jit_compilation_options);

  // (?, 2, 3) => (1, 2, 3)
  const int new_dims[] = {1, 2, 3};
  ASSERT_EQ(
      LiteRtCompiledModelResizeInputTensor(compiled_model,
                                           /*signature_index=*/0,
                                           /*input_index=*/0, new_dims, 3),
      kLiteRtStatusOk);

  // Get new buffer requirements after resize
  LiteRtTensorBufferRequirements requirements;
  ASSERT_EQ(LiteRtGetCompiledModelInputBufferRequirements(
                compiled_model, /*signature_index=*/0, /*input_index=*/0,
                &requirements),
            kLiteRtStatusOk);

  size_t resized_input_tensor_size;
  ASSERT_EQ(LiteRtGetTensorBufferRequirementsBufferSize(
                requirements, &resized_input_tensor_size),
            kLiteRtStatusOk);

  // Verify that the size has doubled (batch size doubled)
  LITERT_LOG(LITERT_INFO, "New size: %zu", resized_input_tensor_size);
  EXPECT_EQ(resized_input_tensor_size, 1 * 2 * 3 * sizeof(float));

  // Test error cases
  // Invalid signature index
  EXPECT_NE(
      LiteRtCompiledModelResizeInputTensor(compiled_model,
                                           /*signature_index=*/999,
                                           /*input_index=*/0, new_dims, 3),
      kLiteRtStatusOk);

  // Invalid input index
  EXPECT_NE(
      LiteRtCompiledModelResizeInputTensor(compiled_model,
                                           /*signature_index=*/0,
                                           /*input_index=*/999, new_dims, 3),
      kLiteRtStatusOk);

  // Empty dims
  EXPECT_NE(LiteRtCompiledModelResizeInputTensor(
                compiled_model, /*signature_index=*/0, /*input_index=*/0,
                new_dims, 0),
            kLiteRtStatusOk);

  // Cleanup
  LiteRtDestroyCompiledModel(compiled_model);
  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(environment);
}

TEST(CompiledModelTest, ResizeInputTensorWithStaticModel) {
  // Use the simple model to ensure resize will error out.
  auto path = testing::GetTestFilePath(kModelFileName);

  LiteRtModel model;
  ASSERT_EQ(LiteRtCreateModelFromFile(path.c_str(), &model), kLiteRtStatusOk);

  LiteRtOptions jit_compilation_options;
  ASSERT_EQ(LiteRtCreateOptions(&jit_compilation_options), kLiteRtStatusOk);
  ASSERT_EQ(LiteRtSetOptionsHardwareAccelerators(jit_compilation_options,
                                                 kLiteRtHwAcceleratorCpu),
            kLiteRtStatusOk);

  LiteRtEnvironment environment;
  LiteRtEnvOption options = {};
  ASSERT_EQ(LiteRtCreateEnvironment(/*num_options=*/0, &options, &environment),
            kLiteRtStatusOk);

  LiteRtCompiledModel compiled_model;
  ASSERT_EQ(LiteRtCreateCompiledModel(environment, model,
                                      jit_compilation_options, &compiled_model),
            kLiteRtStatusOk);

  LiteRtDestroyOptions(jit_compilation_options);

  // (?, 2, 3) => (1, 2, 3)
  const int new_dims[] = {1, 2, 3};
  ASSERT_NE(
      LiteRtCompiledModelResizeInputTensor(compiled_model,
                                           /*signature_index=*/0,
                                           /*input_index=*/0, new_dims, 3),
      kLiteRtStatusOk);

  // Cleanup
  LiteRtDestroyCompiledModel(compiled_model);
  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(environment);
}

TEST(CompiledModelTest, GetOutputTensorShapesWithDynamicModel) {
  // Use the dynamic model for testing resize functionality
  auto path = testing::GetTestFilePath(kDynamicModelFileName);

  LiteRtModel model;
  ASSERT_EQ(LiteRtCreateModelFromFile(path.c_str(), &model), kLiteRtStatusOk);

  LiteRtOptions jit_compilation_options;
  ASSERT_EQ(LiteRtCreateOptions(&jit_compilation_options), kLiteRtStatusOk);
  ASSERT_EQ(LiteRtSetOptionsHardwareAccelerators(jit_compilation_options,
                                                 kLiteRtHwAcceleratorCpu),
            kLiteRtStatusOk);

  LiteRtEnvironment environment;
  LiteRtEnvOption options = {};
  ASSERT_EQ(LiteRtCreateEnvironment(/*num_options=*/0, &options, &environment),
            kLiteRtStatusOk);

  LiteRtCompiledModel compiled_model;
  ASSERT_EQ(LiteRtCreateCompiledModel(environment, model,
                                      jit_compilation_options, &compiled_model),
            kLiteRtStatusOk);

  LiteRtDestroyOptions(jit_compilation_options);

  // (?, 2, 3) => (1, 2, 3)
  const int new_dims[] = {1, 2, 3};
  ASSERT_EQ(
      LiteRtCompiledModelResizeInputTensor(compiled_model,
                                           /*signature_index=*/0,
                                           /*input_index=*/0, new_dims, 3),
      kLiteRtStatusOk);

  ASSERT_EQ(
      LiteRtCompiledModelAllocateTensors(compiled_model, /*signature_index=*/0),
      kLiteRtStatusOk);

  // (?, 2, 3) => (1, 2, 3)
  int expected_output_tensor_shapes[] = {1, 2, 3};
  int* output_tensor_shapes = nullptr;
  int rank = 0;
  ASSERT_EQ(LiteRtGetCompiledModelOutputTensorShapes(
                compiled_model, /*signature_index=*/0, /*output_index=*/0,
                &output_tensor_shapes, &rank),
            kLiteRtStatusOk);

  // Verify that the size has doubled (batch size doubled)
  EXPECT_EQ(rank, sizeof(expected_output_tensor_shapes) / sizeof(int));
  // litert check output_tensor_shapes equals expected_output_tensor_shapes
  for (int i = 0; i < rank; ++i) {
    EXPECT_EQ(output_tensor_shapes[i], expected_output_tensor_shapes[i]);
  }

  // Cleanup
  free(output_tensor_shapes);
  LiteRtDestroyCompiledModel(compiled_model);
  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(environment);
}

}  // namespace
}  // namespace litert
