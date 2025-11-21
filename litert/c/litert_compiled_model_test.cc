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
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "litert/test/testdata/simple_model_test_vectors.h"

using testing::ElementsAre;
using testing::FloatNear;
using testing::Pointwise;

namespace litert {
namespace {

TEST(CompiledModelTest, Basic) {
  auto path = testing::GetTestFilePath(kModelFileName);

  LiteRtModel model;
  LITERT_ASSERT_OK(LiteRtCreateModelFromFile(path.c_str(), &model));

  LiteRtOptions jit_compilation_options;
  LITERT_ASSERT_OK(LiteRtCreateOptions(&jit_compilation_options));
  LITERT_ASSERT_OK(LiteRtSetOptionsHardwareAccelerators(
      jit_compilation_options, kLiteRtHwAcceleratorCpu));

  LiteRtEnvironment environment;
  LiteRtEnvOption options = {};
  LITERT_ASSERT_OK(
      LiteRtCreateEnvironment(/*num_options=*/0, &options, &environment));

  LiteRtCompiledModel compiled_model;
  LITERT_ASSERT_OK(LiteRtCreateCompiledModel(
      environment, model, jit_compilation_options, &compiled_model));

  LiteRtDestroyOptions(jit_compilation_options);

  LiteRtSubgraph subgraph;
  LITERT_ASSERT_OK(LiteRtGetModelSubgraph(model, 0, &subgraph));

  LiteRtParamIndex num_inputs;
  LITERT_ASSERT_OK(LiteRtGetNumSubgraphInputs(subgraph, &num_inputs));

  std::vector<LiteRtTensorBuffer> input_tensor_buffers;
  input_tensor_buffers.reserve(num_inputs);
  for (auto i = 0; i < num_inputs; ++i) {
    LiteRtTensorBufferRequirements tensor_buffer_requirements;
    LITERT_ASSERT_OK(LiteRtGetCompiledModelInputBufferRequirements(
        compiled_model, /*signature_index=*/0, i, &tensor_buffer_requirements));
    LiteRtTensorBufferType tensor_buffer_type;
    LITERT_ASSERT_OK(LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(
        tensor_buffer_requirements, /*type_index=*/0, &tensor_buffer_type));
    size_t tensor_buffer_size;
    LITERT_ASSERT_OK(LiteRtGetTensorBufferRequirementsBufferSize(
        tensor_buffer_requirements, &tensor_buffer_size));
    LiteRtTensorBuffer tensor_buffer;
    LITERT_ASSERT_OK(LiteRtCreateManagedTensorBuffer(
        environment, tensor_buffer_type, &kInput0TensorType, tensor_buffer_size,
        &tensor_buffer));
    input_tensor_buffers.push_back(tensor_buffer);
  }

  LiteRtParamIndex num_outputs;
  LITERT_ASSERT_OK(LiteRtGetNumSubgraphOutputs(subgraph, &num_outputs));

  std::vector<LiteRtTensorBuffer> output_tensor_buffers;
  output_tensor_buffers.reserve(num_outputs);
  for (auto i = 0; i < num_outputs; ++i) {
    LiteRtTensorBufferRequirements tensor_buffer_requirements;
    LITERT_ASSERT_OK(LiteRtGetCompiledModelOutputBufferRequirements(
        compiled_model, /*signature_index=*/0, i, &tensor_buffer_requirements));
    LiteRtTensorBufferType tensor_buffer_type;
    LITERT_ASSERT_OK(LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(
        tensor_buffer_requirements, /*type_index=*/0, &tensor_buffer_type));
    size_t tensor_buffer_size;
    LITERT_ASSERT_OK(LiteRtGetTensorBufferRequirementsBufferSize(
        tensor_buffer_requirements, &tensor_buffer_size));
    LiteRtTensorBuffer tensor_buffer;
    LITERT_ASSERT_OK(LiteRtCreateManagedTensorBuffer(
        environment, tensor_buffer_type, &kInput0TensorType, tensor_buffer_size,
        &tensor_buffer));
    output_tensor_buffers.push_back(tensor_buffer);
  }

  {
    ABSL_LOG(INFO) << "Filling inputs with data";
    void* host_mem_addr;

    LITERT_ASSERT_OK(LiteRtLockTensorBuffer(input_tensor_buffers[0],
                                            &host_mem_addr,
                                            kLiteRtTensorBufferLockModeWrite));
    std::memcpy(host_mem_addr, kTestInput0Tensor, sizeof(kTestInput0Tensor));
    LITERT_ASSERT_OK(LiteRtUnlockTensorBuffer(input_tensor_buffers[0]));

    LITERT_ASSERT_OK(LiteRtLockTensorBuffer(input_tensor_buffers[1],
                                            &host_mem_addr,
                                            kLiteRtTensorBufferLockModeWrite));
    std::memcpy(host_mem_addr, kTestInput1Tensor, sizeof(kTestInput1Tensor));
    LITERT_ASSERT_OK(LiteRtUnlockTensorBuffer(input_tensor_buffers[1]));
  }

  LITERT_ASSERT_OK(LiteRtRunCompiledModel(
      compiled_model, /*signature_index=*/0, input_tensor_buffers.size(),
      input_tensor_buffers.data(), output_tensor_buffers.size(),
      output_tensor_buffers.data()));

  {
    ABSL_LOG(INFO) << "Checking output...";
    void* host_mem_addr;
    LITERT_ASSERT_OK(LiteRtLockTensorBuffer(output_tensor_buffers[0],
                                            &host_mem_addr,
                                            kLiteRtTensorBufferLockModeRead));
    auto output = absl::MakeSpan(static_cast<const float*>(host_mem_addr),
                                 kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-3), kTestOutputTensor));
    LITERT_ASSERT_OK(LiteRtUnlockTensorBuffer(output_tensor_buffers[0]));
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
  LITERT_ASSERT_OK(LiteRtCreateModelFromFile(path.c_str(), &model));

  LiteRtOptions jit_compilation_options;
  LITERT_ASSERT_OK(LiteRtCreateOptions(&jit_compilation_options));
  LITERT_ASSERT_OK(LiteRtSetOptionsHardwareAccelerators(
      jit_compilation_options, kLiteRtHwAcceleratorCpu));

  LiteRtEnvironment environment;
  LiteRtEnvOption options = {};
  LITERT_ASSERT_OK(
      LiteRtCreateEnvironment(/*num_options=*/0, &options, &environment));

  LiteRtCompiledModel compiled_model;
  LITERT_ASSERT_OK(LiteRtCreateCompiledModel(
      environment, model, jit_compilation_options, &compiled_model));

  LiteRtDestroyOptions(jit_compilation_options);

  // (?, 2, 3) => (1, 2, 3)
  const int new_dims[] = {1, 2, 3};
  LITERT_ASSERT_OK(LiteRtCompiledModelResizeInputTensor(
      compiled_model,
      /*signature_index=*/0,
      /*input_index=*/0, new_dims, /*dims_size=*/3));

  // Get new buffer requirements after resize
  LiteRtTensorBufferRequirements requirements;
  LITERT_ASSERT_OK(LiteRtGetCompiledModelInputBufferRequirements(
      compiled_model, /*signature_index=*/0, /*input_index=*/0, &requirements));

  size_t resized_input_tensor_size;
  LITERT_ASSERT_OK(LiteRtGetTensorBufferRequirementsBufferSize(
      requirements, &resized_input_tensor_size));

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
  LITERT_ASSERT_OK(LiteRtCreateModelFromFile(path.c_str(), &model));

  LiteRtOptions jit_compilation_options;
  LITERT_ASSERT_OK(LiteRtCreateOptions(&jit_compilation_options));
  LITERT_ASSERT_OK(LiteRtSetOptionsHardwareAccelerators(
      jit_compilation_options, kLiteRtHwAcceleratorCpu));

  LiteRtEnvironment environment;
  LiteRtEnvOption options = {};
  LITERT_ASSERT_OK(
      LiteRtCreateEnvironment(/*num_options=*/0, &options, &environment));

  LiteRtCompiledModel compiled_model;
  LITERT_ASSERT_OK(LiteRtCreateCompiledModel(
      environment, model, jit_compilation_options, &compiled_model));

  LiteRtDestroyOptions(jit_compilation_options);

  // (?, 2, 3) => (1, 2, 3)
  const int new_dims[] = {1, 2, 3};
  ASSERT_NE(
      LiteRtCompiledModelResizeInputTensor(compiled_model,
                                           /*signature_index=*/0,
                                           /*input_index=*/0, new_dims, 3),
      kLiteRtStatusOk);
  EXPECT_EQ(LiteRtCompiledModelResizeInputTensorNonStrict(
                compiled_model, /*signature_index=*/0,
                /*input_index=*/0, new_dims, 3),
            kLiteRtStatusOk);

  // Cleanup
  LiteRtDestroyCompiledModel(compiled_model);
  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(environment);
}

TEST(CompiledModelTest, GetOutputTensorLayoutsWithDynamicModel) {
  // Use the dynamic model for testing resize functionality
  auto path = testing::GetTestFilePath(kDynamicModelFileName);

  LiteRtModel model;
  LITERT_ASSERT_OK(LiteRtCreateModelFromFile(path.c_str(), &model));

  LiteRtOptions jit_compilation_options;
  LITERT_ASSERT_OK(LiteRtCreateOptions(&jit_compilation_options));
  LITERT_ASSERT_OK(LiteRtSetOptionsHardwareAccelerators(
      jit_compilation_options, kLiteRtHwAcceleratorCpu));

  LiteRtEnvironment environment;
  LiteRtEnvOption options = {};
  LITERT_ASSERT_OK(
      LiteRtCreateEnvironment(/*num_options=*/0, &options, &environment));

  LiteRtCompiledModel compiled_model;
  LITERT_ASSERT_OK(LiteRtCreateCompiledModel(
      environment, model, jit_compilation_options, &compiled_model));

  LiteRtDestroyOptions(jit_compilation_options);

  // Check basic subgraph information
  LiteRtSubgraph litert_subgraph;
  LITERT_ASSERT_OK(LiteRtGetModelSubgraph(model, 0, &litert_subgraph));
  LiteRtParamIndex num_inputs;
  LITERT_ASSERT_OK(LiteRtGetNumSubgraphInputs(litert_subgraph, &num_inputs));
  ASSERT_EQ(num_inputs, 2);
  LiteRtParamIndex num_outputs;
  LITERT_ASSERT_OK(LiteRtGetNumSubgraphOutputs(litert_subgraph, &num_outputs));
  ASSERT_EQ(num_outputs, 1);

  std::vector<LiteRtTensor> input_tensor = litert_subgraph->Inputs();
  for (int i = 0; i < (size_t)num_inputs; ++i) {
    LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_type, input_tensor[i]->Ranked());
    // Check if the original input tensor is (?, 2, 3)
    EXPECT_THAT(absl::MakeConstSpan(tensor_type.layout.dimensions,
                                    tensor_type.layout.rank),
                ElementsAre(-1, 2, 3));
  }

  // (?, 2, 3) => (1, 2, 3)
  const int new_dims[] = {1, 2, 3};
  for (int i = 0; i < (size_t)num_inputs; ++i) {
    LITERT_ASSERT_OK(LiteRtCompiledModelResizeInputTensor(compiled_model,
                                                          /*signature_index=*/0,
                                                          /*input_index=*/i,
                                                          new_dims, 3));
  }

  // (?, 2, 3) => (1, 2, 3)
  int expected_output_tensor_shapes[] = {1, 2, 3};
  std::vector<LiteRtLayout> output_tensor_layouts(num_outputs);
  LITERT_ASSERT_OK(LiteRtGetCompiledModelOutputTensorLayouts(
      compiled_model, /*signature_index=*/0, num_outputs,
      output_tensor_layouts.data(),
      /*update_allocation=*/true));

  // Verify that the size has doubled (batch size doubled)
  // Output tensor number is 1
  LiteRtLayout layout = output_tensor_layouts[0];
  ASSERT_EQ(layout.rank, sizeof(expected_output_tensor_shapes) / sizeof(int));
  // litert check output_tensor_shapes equals expected_output_tensor_shapes
  for (int i = 0; i < layout.rank; ++i) {
    EXPECT_EQ(layout.dimensions[i], expected_output_tensor_shapes[i]);
  }

  // Cleanup
  LiteRtDestroyCompiledModel(compiled_model);
  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(environment);
}

}  // namespace
}  // namespace litert
