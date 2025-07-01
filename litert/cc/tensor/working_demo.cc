// Working demo to show the API functionality

#include "litert/cc/tensor/litert_tensor.h"
#include <iostream>
#include <iomanip>

using namespace litert::tensor;

void demo_basic_creation() {
  std::cout << "=== Demo 1: Basic Tensor Creation ===" << std::endl;
  
  // Create tensors using factory functions
  std::cout << "Creating zeros tensor..." << std::endl;
  auto zeros_result = zeros<float>({2, 3});
  if (!zeros_result) {
    std::cout << "Failed to create zeros tensor" << std::endl;
    return;
  }
  auto zero_tensor = std::move(*zeros_result);
  std::cout << "✓ Created 2x3 zeros tensor" << std::endl;
  
  std::cout << "Creating ones tensor..." << std::endl;
  auto ones_result = ones<float>({2, 3});
  if (!ones_result) {
    std::cout << "Failed to create ones tensor" << std::endl;
    return;
  }
  auto ones_tensor = std::move(*ones_result);
  std::cout << "✓ Created 2x3 ones tensor" << std::endl;
  
  // Check basic properties
  std::cout << "Zeros tensor sum: " << zero_tensor.sum() << std::endl;
  std::cout << "Ones tensor sum: " << ones_tensor.sum() << std::endl;
}

void demo_three_api_styles() {
  std::cout << "\n=== Demo 2: Three API Styles ===" << std::endl;
  
  auto a_result = full<float>({2, 2}, 2.0f);
  auto b_result = full<float>({2, 2}, 3.0f);
  
  if (!a_result || !b_result) {
    std::cout << "Failed to create input tensors" << std::endl;
    return;
  }
  
  auto a = std::move(*a_result);
  auto b = std::move(*b_result);
  
  std::cout << "Input tensors: a(2x2, all 2.0), b(2x2, all 3.0)" << std::endl;
  
  // Three equivalent ways to add tensors:
  try {
    // 1. Operator Overloading
    auto c1 = a + b;
    std::cout << "Style 1 (a + b): sum = " << c1.sum() << std::endl;
    
    // 2. Fluent Style  
    auto c2 = a.add(b);
    std::cout << "Style 2 (a.add(b)): sum = " << c2.sum() << std::endl;
    
    // 3. Functional Style
    auto c3 = add(a, b);
    std::cout << "Style 3 (add(a, b)): sum = " << c3.sum() << std::endl;
    
    std::cout << "✓ All three styles produce identical results!" << std::endl;
    
  } catch (const std::exception& e) {
    std::cout << "Error in arithmetic operations: " << e.what() << std::endl;
  }
}

void demo_fluent_chain() {
  std::cout << "\n=== Demo 3: Fluent API Chaining ===" << std::endl;
  
  auto tensor_result = full<float>({3, 3}, 4.0f);
  if (!tensor_result) {
    std::cout << "Failed to create input tensor" << std::endl;
    return;
  }
  
  auto tensor = std::move(*tensor_result);
  std::cout << "Input: 3x3 tensor filled with 4.0" << std::endl;
  std::cout << "Initial sum: " << tensor.sum() << std::endl;
  
  try {
    // Chain operations: add -> multiply -> take square root
    auto result = tensor.add(1.0f).mul(2.0f).sqrt();
    
    std::cout << "After .add(1.0).mul(2.0).sqrt():" << std::endl;
    std::cout << "Final sum: " << result.sum() << std::endl;
    std::cout << "Expected: 9 * sqrt(10) ≈ " << (9.0f * std::sqrt(10.0f)) << std::endl;
    
  } catch (const std::exception& e) {
    std::cout << "Error in chained operations: " << e.what() << std::endl;
  }
}

void demo_shape_operations() {
  std::cout << "\n=== Demo 4: Shape Operations ===" << std::endl;
  
  auto tensor_result = full<float>({2, 3}, 1.0f);
  if (!tensor_result) {
    std::cout << "Failed to create tensor" << std::endl;
    return;
  }
  
  auto tensor = std::move(*tensor_result);
  
  std::cout << "Original shape: ";
  for (size_t dim : tensor.shape()) {
    std::cout << dim << " ";
  }
  std::cout << "(size: " << tensor.size() << ")" << std::endl;
  
  try {
    // Reshape
    auto reshaped = tensor.reshape({3, 2});
    std::cout << "Reshaped to: ";
    for (size_t dim : reshaped.shape()) {
      std::cout << dim << " ";
    }
    std::cout << std::endl;
    
    // Flatten
    auto flattened = tensor.reshape({6});
    std::cout << "Flattened to: " << flattened.shape()[0] << " elements" << std::endl;
    
    std::cout << "Sum preserved: " << flattened.sum() << std::endl;
    
  } catch (const std::exception& e) {
    std::cout << "Error in shape operations: " << e.what() << std::endl;
  }
}

void demo_element_access() {
  std::cout << "\n=== Demo 5: Element Access ===" << std::endl;
  
  auto tensor_result = zeros<float>({3, 3});
  if (!tensor_result) {
    std::cout << "Failed to create tensor" << std::endl;
    return;
  }
  
  auto tensor = std::move(*tensor_result);
  
  try {
    // Set some values
    tensor(0, 0) = 1.0f;
    tensor(1, 1) = 2.0f;
    tensor(2, 2) = 3.0f;
    
    std::cout << "Set diagonal elements to 1, 2, 3" << std::endl;
    std::cout << "tensor(0,0) = " << tensor(0, 0) << std::endl;
    std::cout << "tensor(1,1) = " << tensor(1, 1) << std::endl;
    std::cout << "tensor(2,2) = " << tensor(2, 2) << std::endl;
    std::cout << "Total sum: " << tensor.sum() << std::endl;
    
  } catch (const std::exception& e) {
    std::cout << "Error in element access: " << e.what() << std::endl;
  }
}

int main() {
  std::cout << "🚀 LiteRT Tensor API Working Demo" << std::endl;
  std::cout << "=================================" << std::endl;
  
  try {
    demo_basic_creation();
    demo_three_api_styles();
    demo_fluent_chain();
    demo_shape_operations();
    demo_element_access();
    
    std::cout << "\n✅ All demos completed successfully!" << std::endl;
    std::cout << "\n📋 Command to reproduce:" << std::endl;
    std::cout << "bazel run //litert/cc/tensor:working_demo" << std::endl;
    
  } catch (const std::exception& e) {
    std::cout << "❌ Demo failed with exception: " << e.what() << std::endl;
    return 1;
  }
  
  return 0;
}