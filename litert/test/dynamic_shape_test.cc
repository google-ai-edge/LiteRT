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

#include <cstddef>
#include <cstring>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
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

// Test dynamic shape support with variable batch sizes
TEST(DynamicShapeTest, VariableBatchSize) {
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

  // Test with different batch sizes
  const int batch_sizes[] = {1, 2, 4, 8};
  
  for (int batch_size : batch_sizes) {
    ABSL_LOG(INFO) << "Testing batch size: " << batch_size;
    
    // Resize input tensor for first input (shape will be [batch_size, 5, 5, 1])
    const int dims_input0[] = {batch_size, 5, 5, 1};
    ASSERT_EQ(LiteRtCompiledModelResizeInputTensor(
                  compiled_model, /*signature_index=*/0, /*input_index=*/0,
                  dims_input0, /*num_dims=*/4),
              kLiteRtStatusOk);

    // Resize second input (shape will be [batch_size, 1, 1, 1])
    const int dims_input1[] = {batch_size, 1, 1, 1};
    ASSERT_EQ(LiteRtCompiledModelResizeInputTensor(
                  compiled_model, /*signature_index=*/0, /*input_index=*/1,
                  dims_input1, /*num_dims=*/4),
              kLiteRtStatusOk);

    // Create input buffers with new sizes
    std::vector<LiteRtTensorBuffer> input_tensor_buffers;
    
    // First input buffer
    {
      LiteRtTensorBufferRequirements requirements;
      ASSERT_EQ(LiteRtGetCompiledModelInputBufferRequirements(
                    compiled_model, /*signature_index=*/0, /*input_index=*/0,
                    &requirements),
                kLiteRtStatusOk);
      
      size_t buffer_size;
      ASSERT_EQ(LiteRtGetTensorBufferRequirementsBufferSize(requirements,
                                                            &buffer_size),
                kLiteRtStatusOk);
      
      LiteRtTensorBuffer tensor_buffer;
      ASSERT_EQ(LiteRtCreateManagedTensorBuffer(
                    environment, kLiteRtTensorBufferTypeHostMemory,
                    &kInput0TensorType, buffer_size, &tensor_buffer),
                kLiteRtStatusOk);
      
      // Fill with test data (repeat pattern for larger batches)
      void* host_mem_addr;
      ASSERT_EQ(LiteRtLockTensorBuffer(tensor_buffer, &host_mem_addr,
                                       kLiteRtTensorBufferLockModeWrite),
                kLiteRtStatusOk);
      auto input_data = absl::MakeSpan(static_cast<float*>(host_mem_addr),
                                      batch_size * 25);
      for (int b = 0; b < batch_size; ++b) {
        std::memcpy(input_data.data() + b * 25, kTestInput0Tensor, 
                   25 * sizeof(float));
      }
      ASSERT_EQ(LiteRtUnlockTensorBuffer(tensor_buffer), kLiteRtStatusOk);
      
      input_tensor_buffers.push_back(tensor_buffer);
    }
    
    // Second input buffer
    {
      LiteRtTensorBufferRequirements requirements;
      ASSERT_EQ(LiteRtGetCompiledModelInputBufferRequirements(
                    compiled_model, /*signature_index=*/0, /*input_index=*/1,
                    &requirements),
                kLiteRtStatusOk);
      
      size_t buffer_size;
      ASSERT_EQ(LiteRtGetTensorBufferRequirementsBufferSize(requirements,
                                                            &buffer_size),
                kLiteRtStatusOk);
      
      LiteRtTensorBuffer tensor_buffer;
      ASSERT_EQ(LiteRtCreateManagedTensorBuffer(
                    environment, kLiteRtTensorBufferTypeHostMemory,
                    &kInput1TensorType, buffer_size, &tensor_buffer),
                kLiteRtStatusOk);
      
      // Fill with test data
      void* host_mem_addr;
      ASSERT_EQ(LiteRtLockTensorBuffer(tensor_buffer, &host_mem_addr,
                                       kLiteRtTensorBufferLockModeWrite),
                kLiteRtStatusOk);
      auto input_data = absl::MakeSpan(static_cast<float*>(host_mem_addr),
                                      batch_size);
      for (int b = 0; b < batch_size; ++b) {
        input_data[b] = kTestInput1Tensor[0];
      }
      ASSERT_EQ(LiteRtUnlockTensorBuffer(tensor_buffer), kLiteRtStatusOk);
      
      input_tensor_buffers.push_back(tensor_buffer);
    }

    // Create output buffer
    std::vector<LiteRtTensorBuffer> output_tensor_buffers;
    {
      LiteRtTensorBufferRequirements requirements;
      ASSERT_EQ(LiteRtGetCompiledModelOutputBufferRequirements(
                    compiled_model, /*signature_index=*/0, /*output_index=*/0,
                    &requirements),
                kLiteRtStatusOk);
      
      size_t buffer_size;
      ASSERT_EQ(LiteRtGetTensorBufferRequirementsBufferSize(requirements,
                                                            &buffer_size),
                kLiteRtStatusOk);
      
      // Expected output size should scale with batch size
      EXPECT_EQ(buffer_size, batch_size * 50 * sizeof(float));
      
      LiteRtTensorBuffer tensor_buffer;
      ASSERT_EQ(LiteRtCreateManagedTensorBuffer(
                    environment, kLiteRtTensorBufferTypeHostMemory,
                    &kOutputTensorType, buffer_size, &tensor_buffer),
                kLiteRtStatusOk);
      
      output_tensor_buffers.push_back(tensor_buffer);
    }

    // Run inference
    ASSERT_EQ(LiteRtRunCompiledModel(
                  compiled_model, /*signature_index=*/0,
                  input_tensor_buffers.size(), input_tensor_buffers.data(),
                  output_tensor_buffers.size(), output_tensor_buffers.data()),
              kLiteRtStatusOk);

    // Verify output (should be same pattern repeated for each batch)
    {
      void* host_mem_addr;
      ASSERT_EQ(LiteRtLockTensorBuffer(output_tensor_buffers[0], &host_mem_addr,
                                       kLiteRtTensorBufferLockModeRead),
                kLiteRtStatusOk);
      auto output = absl::MakeSpan(static_cast<const float*>(host_mem_addr),
                                  batch_size * 50);
      
      // Check each batch has the expected output pattern
      for (int b = 0; b < batch_size; ++b) {
        auto batch_output = output.subspan(b * 50, 50);
        EXPECT_THAT(batch_output, Pointwise(FloatNear(1e-3), kTestOutputTensor))
            << "Mismatch in batch " << b;
      }
      
      ASSERT_EQ(LiteRtUnlockTensorBuffer(output_tensor_buffers[0]),
                kLiteRtStatusOk);
    }

    // Cleanup buffers for this iteration
    for (auto tensor_buffer : input_tensor_buffers) {
      LiteRtDestroyTensorBuffer(tensor_buffer);
    }
    for (auto tensor_buffer : output_tensor_buffers) {
      LiteRtDestroyTensorBuffer(tensor_buffer);
    }
  }

  // Cleanup
  LiteRtDestroyCompiledModel(compiled_model);
  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(environment);
}

// Test dynamic shape support with changing input dimensions
TEST(DynamicShapeTest, ChangingInputDimensions) {
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

  // Test sequence: resize -> get requirements -> resize again -> get requirements
  // This tests that buffer requirements are properly updated after each resize
  
  // First resize
  const int dims1[] = {1, 5, 5, 1};
  ASSERT_EQ(LiteRtCompiledModelResizeInputTensor(
                compiled_model, /*signature_index=*/0, /*input_index=*/0,
                dims1, /*num_dims=*/4),
            kLiteRtStatusOk);

  LiteRtTensorBufferRequirements req1;
  ASSERT_EQ(LiteRtGetCompiledModelInputBufferRequirements(
                compiled_model, /*signature_index=*/0, /*input_index=*/0,
                &req1),
            kLiteRtStatusOk);

  size_t size1;
  ASSERT_EQ(LiteRtGetTensorBufferRequirementsBufferSize(req1, &size1),
            kLiteRtStatusOk);
  EXPECT_EQ(size1, 1 * 5 * 5 * 1 * sizeof(float));

  // Second resize with different dimensions
  const int dims2[] = {3, 5, 5, 1};
  ASSERT_EQ(LiteRtCompiledModelResizeInputTensor(
                compiled_model, /*signature_index=*/0, /*input_index=*/0,
                dims2, /*num_dims=*/4),
            kLiteRtStatusOk);

  LiteRtTensorBufferRequirements req2;
  ASSERT_EQ(LiteRtGetCompiledModelInputBufferRequirements(
                compiled_model, /*signature_index=*/0, /*input_index=*/0,
                &req2),
            kLiteRtStatusOk);

  size_t size2;
  ASSERT_EQ(LiteRtGetTensorBufferRequirementsBufferSize(req2, &size2),
            kLiteRtStatusOk);
  EXPECT_EQ(size2, 3 * 5 * 5 * 1 * sizeof(float));

  // Verify that buffer requirements are updated correctly
  EXPECT_NE(size1, size2);

  // Cleanup
  LiteRtDestroyCompiledModel(compiled_model);
  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(environment);
}

}  // namespace
}  // namespace litert