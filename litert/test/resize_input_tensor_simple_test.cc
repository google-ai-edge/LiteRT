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

#include <iostream>

#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"

int main() {
  std::cout << "Testing ResizeInputTensor API..." << std::endl;

  // Create environment
  LiteRtEnvironment environment;
  LiteRtEnvOption options = {};
  if (LiteRtCreateEnvironment(/*num_options=*/0, &options, &environment) !=
      kLiteRtStatusOk) {
    std::cerr << "Failed to create environment" << std::endl;
    return 1;
  }

  // Use the v_einsum model for testing
  const char* model_path = "litert/test/testdata/v_einsum.tflite";
  
  LiteRtModel model;
  if (LiteRtCreateModelFromFile(model_path, &model) != kLiteRtStatusOk) {
    std::cerr << "Failed to load model from: " << model_path << std::endl;
    LiteRtDestroyEnvironment(environment);
    return 1;
  }

  // Create compilation options
  LiteRtOptions compilation_options;
  if (LiteRtCreateOptions(&compilation_options) != kLiteRtStatusOk) {
    std::cerr << "Failed to create options" << std::endl;
    LiteRtDestroyModel(model);
    LiteRtDestroyEnvironment(environment);
    return 1;
  }

  // Set CPU acceleration (to avoid XNNPACK issues with dynamic shapes)
  if (LiteRtSetOptionsHardwareAccelerators(compilation_options,
                                          kLiteRtHwAcceleratorCpu) !=
      kLiteRtStatusOk) {
    std::cerr << "Failed to set hardware accelerator" << std::endl;
    LiteRtDestroyOptions(compilation_options);
    LiteRtDestroyModel(model);
    LiteRtDestroyEnvironment(environment);
    return 1;
  }

  // Create compiled model
  LiteRtCompiledModel compiled_model;
  if (LiteRtCreateCompiledModel(environment, model, compilation_options,
                               &compiled_model) != kLiteRtStatusOk) {
    std::cerr << "Failed to create compiled model" << std::endl;
    LiteRtDestroyOptions(compilation_options);
    LiteRtDestroyModel(model);
    LiteRtDestroyEnvironment(environment);
    return 1;
  }

  LiteRtDestroyOptions(compilation_options);

  // Get original buffer requirements
  LiteRtTensorBufferRequirements original_requirements;
  if (LiteRtGetCompiledModelInputBufferRequirements(
          compiled_model, /*signature_index=*/0, /*input_index=*/0,
          &original_requirements) != kLiteRtStatusOk) {
    std::cerr << "Failed to get original buffer requirements" << std::endl;
    LiteRtDestroyCompiledModel(compiled_model);
    LiteRtDestroyModel(model);
    LiteRtDestroyEnvironment(environment);
    return 1;
  }

  size_t original_size;
  if (LiteRtGetTensorBufferRequirementsBufferSize(original_requirements,
                                                  &original_size) !=
      kLiteRtStatusOk) {
    std::cerr << "Failed to get original buffer size" << std::endl;
    LiteRtDestroyCompiledModel(compiled_model);
    LiteRtDestroyModel(model);
    LiteRtDestroyEnvironment(environment);
    return 1;
  }

  std::cout << "Original buffer size: " << original_size << " bytes" << std::endl;

  // Try to resize input tensor (this might fail if the model doesn't support it)
  const int new_dims[] = {2, 64, 768};  // Different batch size
  auto resize_status = LiteRtCompiledModelResizeInputTensor(
      compiled_model, /*signature_index=*/0, /*input_index=*/0,
      new_dims, /*num_dims=*/3);

  if (resize_status == kLiteRtStatusOk) {
    std::cout << "✓ ResizeInputTensor succeeded!" << std::endl;

    // Get new buffer requirements
    LiteRtTensorBufferRequirements new_requirements;
    if (LiteRtGetCompiledModelInputBufferRequirements(
            compiled_model, /*signature_index=*/0, /*input_index=*/0,
            &new_requirements) == kLiteRtStatusOk) {
      size_t new_size;
      if (LiteRtGetTensorBufferRequirementsBufferSize(new_requirements,
                                                      &new_size) ==
          kLiteRtStatusOk) {
        std::cout << "New buffer size: " << new_size << " bytes" << std::endl;
        std::cout << "Size changed: " << (new_size != original_size ? "YES" : "NO") 
                  << std::endl;
      }
    }
  } else {
    std::cout << "✗ ResizeInputTensor failed (model may not support dynamic shapes)" 
              << std::endl;
  }

  // Test error cases
  std::cout << "\nTesting error cases..." << std::endl;

  // Invalid signature index
  auto error_status = LiteRtCompiledModelResizeInputTensor(
      compiled_model, /*signature_index=*/999, /*input_index=*/0,
      new_dims, /*num_dims=*/3);
  std::cout << "Invalid signature index: " 
            << (error_status != kLiteRtStatusOk ? "✓ Correctly failed" : "✗ Should have failed")
            << std::endl;

  // Invalid input index
  error_status = LiteRtCompiledModelResizeInputTensor(
      compiled_model, /*signature_index=*/0, /*input_index=*/999,
      new_dims, /*num_dims=*/3);
  std::cout << "Invalid input index: " 
            << (error_status != kLiteRtStatusOk ? "✓ Correctly failed" : "✗ Should have failed")
            << std::endl;

  // NULL dims with non-zero num_dims
  error_status = LiteRtCompiledModelResizeInputTensor(
      compiled_model, /*signature_index=*/0, /*input_index=*/0,
      nullptr, /*num_dims=*/3);
  std::cout << "NULL dims: " 
            << (error_status != kLiteRtStatusOk ? "✓ Correctly failed" : "✗ Should have failed")
            << std::endl;

  // Cleanup
  LiteRtDestroyCompiledModel(compiled_model);
  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(environment);

  std::cout << "\nResizeInputTensor API test completed!" << std::endl;
  return 0;
}