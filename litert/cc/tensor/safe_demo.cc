// Safe demo that avoids potentially problematic reduction operations

#include "litert/cc/tensor/litert_tensor.h"
#include <iostream>

using namespace litert::tensor;

int main() {
  std::cout << "🔒 Safe LiteRT Tensor API Demo" << std::endl;
  std::cout << "==============================" << std::endl;
  
  try {
    // Demo 1: Basic Creation
    std::cout << "\n=== Demo 1: Tensor Creation ===\n" << std::endl;
    auto a_result = full<float>({2, 2}, 2.0f);
    auto b_result = full<float>({2, 2}, 3.0f);
    
    if (!a_result || !b_result) {
      std::cout << "❌ Failed to create tensors" << std::endl;
      return 1;
    }
    
    auto a = std::move(*a_result);
    auto b = std::move(*b_result);
    
    std::cout << "✅ Created tensor a: " << a.shape()[0] << "x" << a.shape()[1] << " (filled with 2.0)" << std::endl;
    std::cout << "✅ Created tensor b: " << b.shape()[0] << "x" << b.shape()[1] << " (filled with 3.0)" << std::endl;
    
    // Demo 2: Three API Styles
    std::cout << "\n=== Demo 2: Three API Styles ===\n" << std::endl;
    
    // Style 1: Operator overloading
    auto c1 = a + b;
    std::cout << "✅ Style 1 (a + b): Created result tensor" << std::endl;
    std::cout << "   Result size: " << c1.size() << ", element (0,0): " << c1(0,0) << std::endl;
    
    // Style 2: Fluent style
    auto c2 = a.add(b);
    std::cout << "✅ Style 2 (a.add(b)): Created result tensor" << std::endl;
    std::cout << "   Result size: " << c2.size() << ", element (0,0): " << c2(0,0) << std::endl;
    
    // Style 3: Functional style
    auto c3 = add(a, b);
    std::cout << "✅ Style 3 (add(a, b)): Created result tensor" << std::endl;
    std::cout << "   Result size: " << c3.size() << ", element (0,0): " << c3(0,0) << std::endl;
    
    // Verify all give same result
    if (c1(0,0) == c2(0,0) && c2(0,0) == c3(0,0)) {
      std::cout << "🎯 All three styles produce identical results!" << std::endl;
    }
    
    // Demo 3: Scalar Operations
    std::cout << "\n=== Demo 3: Scalar Operations ===\n" << std::endl;
    
    auto d = a + 1.0f;
    std::cout << "✅ a + 1.0: element (0,0) = " << d(0,0) << " (expected: 3.0)" << std::endl;
    
    auto e = a * 2.0f;
    std::cout << "✅ a * 2.0: element (0,0) = " << e(0,0) << " (expected: 4.0)" << std::endl;
    
    // Demo 4: Shape Operations
    std::cout << "\n=== Demo 4: Shape Operations ===\n" << std::endl;
    
    auto tensor_result = full<float>({2, 3}, 1.0f);
    if (!tensor_result) {
      std::cout << "❌ Failed to create tensor for reshape test" << std::endl;
      return 1;
    }
    
    auto tensor = std::move(*tensor_result);
    std::cout << "✅ Original shape: " << tensor.shape()[0] << "x" << tensor.shape()[1] << " (size: " << tensor.size() << ")" << std::endl;
    
    auto reshaped = tensor.reshape({3, 2});
    std::cout << "✅ Reshaped to: " << reshaped.shape()[0] << "x" << reshaped.shape()[1] << " (size: " << reshaped.size() << ")" << std::endl;
    
    auto flattened = tensor.reshape({6});
    std::cout << "✅ Flattened to: " << flattened.shape()[0] << " elements" << std::endl;
    
    // Demo 5: Element Access
    std::cout << "\n=== Demo 5: Element Access ===\n" << std::endl;
    
    auto matrix_result = zeros<float>({3, 3});
    if (!matrix_result) {
      std::cout << "❌ Failed to create matrix" << std::endl;
      return 1;
    }
    
    auto matrix = std::move(*matrix_result);
    
    // Set diagonal elements
    matrix(0, 0) = 1.0f;
    matrix(1, 1) = 2.0f;
    matrix(2, 2) = 3.0f;
    
    std::cout << "✅ Set diagonal: matrix(0,0)=" << matrix(0,0) 
              << ", matrix(1,1)=" << matrix(1,1) 
              << ", matrix(2,2)=" << matrix(2,2) << std::endl;
    
    std::cout << "\n🎉 All demos completed successfully!" << std::endl;
    std::cout << "\n📋 Three ways to build and run:" << std::endl;
    std::cout << "  1. Basic functionality: bazel run //litert/cc/tensor:simple_test" << std::endl;
    std::cout << "  2. Core operations:     bazel run //litert/cc/tensor:minimal_demo" << std::endl;
    std::cout << "  3. All API features:    bazel run //litert/cc/tensor:safe_demo" << std::endl;
    
  } catch (const std::exception& e) {
    std::cout << "❌ Error: " << e.what() << std::endl;
    return 1;
  }
  
  return 0;
}