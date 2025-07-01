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

// This sample demonstrates the dynamic shape support in LiteRT.
// It shows how to use the ResizeInputTensor API to change input dimensions
// at runtime and process variable-sized inputs.

#include <iostream>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
    return 1;
  }

  const char* model_path = argv[1];

  // Create environment
  LiteRtEnvironment environment;
  LiteRtEnvOption options = {};
  if (LiteRtCreateEnvironment(/*num_options=*/0, &options, &environment) !=
      kLiteRtStatusOk) {
    std::cerr << "Failed to create environment" << std::endl;
    return 1;
  }

  // Load model
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

  // Set CPU acceleration
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

  std::cout << "Model loaded successfully!" << std::endl;

  // Demonstrate dynamic shape support with different batch sizes
  const int batch_sizes[] = {1, 2, 4};
  
  for (int batch_size : batch_sizes) {
    std::cout << "\n--- Testing with batch size: " << batch_size << " ---" 
              << std::endl;

    // Resize input tensor (assuming first dimension is batch size)
    // For example, change from [1, 224, 224, 3] to [batch_size, 224, 224, 3]
    const int new_dims[] = {batch_size, 224, 224, 3};
    
    if (LiteRtCompiledModelResizeInputTensor(
            compiled_model, /*signature_index=*/0, /*input_index=*/0,
            new_dims, /*num_dims=*/4) != kLiteRtStatusOk) {
      std::cerr << "Failed to resize input tensor" << std::endl;
      continue;
    }

    // Get updated buffer requirements
    LiteRtTensorBufferRequirements requirements;
    if (LiteRtGetCompiledModelInputBufferRequirements(
            compiled_model, /*signature_index=*/0, /*input_index=*/0,
            &requirements) != kLiteRtStatusOk) {
      std::cerr << "Failed to get buffer requirements" << std::endl;
      continue;
    }

    size_t buffer_size;
    if (LiteRtGetTensorBufferRequirementsBufferSize(requirements,
                                                    &buffer_size) !=
        kLiteRtStatusOk) {
      std::cerr << "Failed to get buffer size" << std::endl;
      continue;
    }

    std::cout << "Required buffer size for batch " << batch_size << ": " 
              << buffer_size << " bytes" << std::endl;

    // Calculate expected size (batch_size * 224 * 224 * 3 * sizeof(float))
    size_t expected_size = batch_size * 224 * 224 * 3 * sizeof(float);
    std::cout << "Expected size: " << expected_size << " bytes" << std::endl;

    if (buffer_size == expected_size) {
      std::cout << "✓ Buffer size matches expected size!" << std::endl;
    } else {
      std::cout << "✗ Buffer size mismatch!" << std::endl;
    }
  }

  // Cleanup
  LiteRtDestroyCompiledModel(compiled_model);
  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(environment);

  std::cout << "\nDynamic shape demo completed successfully!" << std::endl;
  return 0;
}