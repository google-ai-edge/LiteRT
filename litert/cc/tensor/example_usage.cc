// Copyright 2025 Google LLC.
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

#include "litert/cc/tensor/litert_tensor.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_macros.h"

#include <iostream>
#include <vector>

namespace litert {
namespace tensor {
namespace examples {

// Example 1: Basic tensor operations with three API styles
void BasicTensorOperations() {
  std::cout << "=== Example 1: Basic Tensor Operations ===" << std::endl;
  
  // Create tensors using factory functions
  auto a_result = zeros<float>({2, 3});
  auto b_result = ones<float>({2, 3});
  
  if (!a_result || !b_result) {
    std::cerr << "Failed to create tensors" << std::endl;
    return;
  }
  
  auto a = std::move(*a_result);
  auto b = std::move(*b_result);
  
  // Fill tensor 'a' with some values
  auto fill_result = a.fill(2.0f);
  if (!fill_result) {
    std::cerr << "Failed to fill tensor" << std::endl;
    return;
  }
  
  std::cout << "Created tensors a (2x3, filled with 2.0) and b (2x3, filled with 1.0)" << std::endl;
  
  // Three equivalent ways to add tensors:
  
  // 1. Operator Overloading (C++ idiomatic)
  auto c1 = a + b;
  std::cout << "c1 = a + b (operator overloading)" << std::endl;
  
  // 2. Fluent / Method Style (TensorFlow.js-like)
  auto c2 = a.add(b);
  std::cout << "c2 = a.add(b) (fluent style)" << std::endl;
  
  // 3. Functional Style (NumPy/TensorFlow-like)
  auto c3 = add(a, b);
  std::cout << "c3 = add(a, b) (functional style)" << std::endl;
  
  // Verify results are the same
  auto c1_sum = c1.sum();
  auto c2_sum = c2.sum();
  auto c3_sum = c3.sum();
  
  std::cout << "Results: c1.sum() = " << c1_sum 
            << ", c2.sum() = " << c2_sum 
            << ", c3.sum() = " << c3_sum << std::endl;
}

// Example 2: Chained operations (fluent API)
void ChainedOperations() {
  std::cout << "\n=== Example 2: Chained Operations ===" << std::endl;
  
  auto tensor_result = full<float>({3, 4}, 0.5f);
  if (!tensor_result) {
    std::cerr << "Failed to create tensor" << std::endl;
    return;
  }
  
  auto tensor = std::move(*tensor_result);
  
  // Chain multiple operations together
  auto result = tensor
      .add(1.0f)           // Add scalar
      .mul(2.0f)           // Multiply by scalar  
      .sqrt()              // Element-wise square root
      .reshape({2, 6})     // Reshape from 3x4 to 2x6
      .transpose();        // Transpose to 6x2
  
  std::cout << "Chained operations: tensor.add(1.0).mul(2.0).sqrt().reshape({2,6}).transpose()" << std::endl;
  std::cout << "Final shape: " << result.shape()[0] << "x" << result.shape()[1] << std::endl;
  std::cout << "Final sum: " << result.sum() << std::endl;
}

// Example 3: Mathematical operations
void MathematicalOperations() {
  std::cout << "\n=== Example 3: Mathematical Operations ===" << std::endl;
  
  auto x_result = full<float>({3, 3}, 0.5f);
  if (!x_result) {
    std::cerr << "Failed to create tensor" << std::endl;
    return;
  }
  
  auto x = std::move(*x_result);
  
  // Universal functions (element-wise)
  auto sin_x = x.sin();
  auto cos_x = x.cos();
  auto exp_x = x.exp();
  auto log_x = x.log();
  
  std::cout << "Created 3x3 tensor filled with 0.5" << std::endl;
  std::cout << "sin(x) sum: " << sin_x.sum() << std::endl;
  std::cout << "cos(x) sum: " << cos_x.sum() << std::endl;
  std::cout << "exp(x) sum: " << exp_x.sum() << std::endl;
  std::cout << "log(x) sum: " << log_x.sum() << std::endl;
  
  // Trigonometric identity: sin²(x) + cos²(x) = 1
  auto identity_check = sin_x.mul(sin_x).add(cos_x.mul(cos_x));
  std::cout << "sin²(x) + cos²(x) sum (should be ~9.0): " << identity_check.sum() << std::endl;
}

// Example 4: Reduction operations
void ReductionOperations() {
  std::cout << "\n=== Example 4: Reduction Operations ===" << std::endl;
  
  auto data_result = zeros<float>({2, 3});
  if (!data_result) {
    std::cerr << "Failed to create tensor" << std::endl;
    return;
  }
  
  auto data = std::move(*data_result);
  
  // Fill with some sample data: [1, 2, 3; 4, 5, 6]
  std::vector<float> sample_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto fill_result = data.from_vector(sample_data);
  if (!fill_result) {
    std::cerr << "Failed to fill tensor with data" << std::endl;
    return;
  }
  
  std::cout << "Created 2x3 tensor: [[1, 2, 3], [4, 5, 6]]" << std::endl;
  
  // Reduction operations
  std::cout << "Sum: " << data.sum() << std::endl;                    // 21.0
  std::cout << "Mean: " << data.mean() << std::endl;                  // 3.5
  std::cout << "Max: " << data.max() << std::endl;                    // 6.0
  std::cout << "Min: " << data.min() << std::endl;                    // 1.0
  std::cout << "ArgMax: " << data.argmax() << std::endl;              // 5 (index of max element)
  std::cout << "ArgMin: " << data.argmin() << std::endl;              // 0 (index of min element)
}

// Example 5: Integration with LiteRT workflow
void LiteRTIntegrationExample() {
  std::cout << "\n=== Example 5: LiteRT Integration ===" << std::endl;
  
  // This example shows how the tensor API integrates with LiteRT workflow
  // Note: This is conceptual - actual model loading would require real .tflite file
  
  std::cout << "Conceptual LiteRT integration workflow:" << std::endl;
  std::cout << "1. Load model and create environment" << std::endl;
  std::cout << "2. Create input/output buffers" << std::endl;
  std::cout << "3. Wrap buffers with tensor API" << std::endl;
  std::cout << "4. Perform preprocessing with tensor operations" << std::endl;
  std::cout << "5. Run inference" << std::endl;
  std::cout << "6. Perform postprocessing with tensor operations" << std::endl;
  
  // Simulate image preprocessing pipeline
  auto image_result = full<float>({1, 224, 224, 3}, 128.0f);  // Simulated RGB image
  if (!image_result) {
    std::cerr << "Failed to create image tensor" << std::endl;
    return;
  }
  
  auto image = std::move(*image_result);
  
  // Typical image preprocessing pipeline
  auto preprocessed = image
      .div(255.0f)              // Normalize to [0, 1]
      .sub(0.5f)                // Center around 0
      .mul(2.0f);               // Scale to [-1, 1]
  
  std::cout << "Image preprocessing: normalize -> center -> scale" << std::endl;
  std::cout << "Original image mean: " << image.mean() << std::endl;
  std::cout << "Preprocessed image mean: " << preprocessed.mean() << std::endl;
  
  // Simulate model output (e.g., classification logits)
  auto logits_result = full<float>({1, 1000}, 0.1f);
  if (!logits_result) {
    std::cerr << "Failed to create logits tensor" << std::endl;
    return;
  }
  
  auto logits = std::move(*logits_result);
  
  // Set one class to have higher score
  logits(0, 42) = 2.0f;  // Class 42 has highest score
  
  // Find predicted class
  auto predicted_class = logits.argmax();
  std::cout << "Predicted class: " << predicted_class << std::endl;
}

// Example 6: Element access and slicing
void ElementAccessExample() {
  std::cout << "\n=== Example 6: Element Access and Slicing ===" << std::endl;
  
  auto matrix_result = zeros<float>({3, 4});
  if (!matrix_result) {
    std::cerr << "Failed to create matrix" << std::endl;
    return;
  }
  
  auto matrix = std::move(*matrix_result);
  
  // Set individual elements
  matrix(0, 0) = 1.0f;
  matrix(0, 1) = 2.0f;
  matrix(1, 0) = 3.0f;
  matrix(1, 1) = 4.0f;
  matrix(2, 2) = 5.0f;
  
  std::cout << "Set individual elements in 3x4 matrix" << std::endl;
  std::cout << "matrix(0,0) = " << matrix(0, 0) << std::endl;
  std::cout << "matrix(1,1) = " << matrix(1, 1) << std::endl;
  std::cout << "matrix(2,2) = " << matrix(2, 2) << std::endl;
  
  // Demonstrate different indexing methods
  std::cout << "Different indexing styles:" << std::endl;
  std::cout << "matrix(0, 1) = " << matrix(0, 1) << std::endl;           // 2D indexing
  std::cout << "matrix({1, 0}) = " << matrix({1, 0}) << std::endl;       // Initializer list
}

// Example 7: Shape manipulation
void ShapeManipulationExample() {
  std::cout << "\n=== Example 7: Shape Manipulation ===" << std::endl;
  
  auto tensor_result = full<float>({2, 3, 4}, 1.0f);
  if (!tensor_result) {
    std::cerr << "Failed to create tensor" << std::endl;
    return;
  }
  
  auto tensor = std::move(*tensor_result);
  
  std::cout << "Original shape: ";
  for (size_t i = 0; i < tensor.shape().size(); ++i) {
    std::cout << tensor.shape()[i];
    if (i < tensor.shape().size() - 1) std::cout << "x";
  }
  std::cout << " (total size: " << tensor.size() << ")" << std::endl;
  
  // Reshape to different configurations
  auto reshaped1 = tensor.reshape({6, 4});
  std::cout << "Reshaped to: " << reshaped1.shape()[0] << "x" << reshaped1.shape()[1] << std::endl;
  
  auto reshaped2 = tensor.reshape({24});
  std::cout << "Reshaped to 1D: " << reshaped2.shape()[0] << " elements" << std::endl;
  
  // Add and remove dimensions
  auto expanded = tensor.expand_dims(0);  // Add dimension at axis 0
  std::cout << "Expanded dimensions: ";
  for (size_t i = 0; i < expanded.shape().size(); ++i) {
    std::cout << expanded.shape()[i];
    if (i < expanded.shape().size() - 1) std::cout << "x";
  }
  std::cout << std::endl;
  
  auto squeezed = expanded.squeeze();  // Remove dimensions of size 1
  std::cout << "After squeeze: ";
  for (size_t i = 0; i < squeezed.shape().size(); ++i) {
    std::cout << squeezed.shape()[i];
    if (i < squeezed.shape().size() - 1) std::cout << "x";
  }
  std::cout << std::endl;
}

}  // namespace examples

// Main function to run all examples
void RunExamples() {
  std::cout << "LiteRT Tensor API Examples" << std::endl;
  std::cout << "=========================" << std::endl;
  
  try {
    examples::BasicTensorOperations();
    examples::ChainedOperations();
    examples::MathematicalOperations();
    examples::ReductionOperations();
    examples::LiteRTIntegrationExample();
    examples::ElementAccessExample();
    examples::ShapeManipulationExample();
    
    std::cout << "\n✓ All examples completed successfully!" << std::endl;
    
  } catch (const std::exception& e) {
    std::cerr << "Error running examples: " << e.what() << std::endl;
  }
}

}  // namespace tensor
}  // namespace litert

// Example main function
int main() {
  litert::tensor::RunExamples();
  return 0;
}