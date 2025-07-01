// Minimal demo to test tensor creation without advanced operations

#include "litert/cc/tensor/litert_tensor.h"
#include <iostream>

using namespace litert::tensor;

int main() {
  std::cout << "🧪 Minimal LiteRT Tensor Demo" << std::endl;
  std::cout << "=============================" << std::endl;
  
  try {
    // Test basic tensor creation
    std::cout << "1. Creating zeros tensor..." << std::endl;
    auto zeros_result = zeros<float>({2, 3});
    if (!zeros_result) {
      std::cout << "❌ Failed to create zeros tensor" << std::endl;
      return 1;
    }
    auto zero_tensor = std::move(*zeros_result);
    std::cout << "✅ Created zeros tensor: " << zero_tensor.shape()[0] << "x" << zero_tensor.shape()[1] << std::endl;
    
    std::cout << "2. Creating ones tensor..." << std::endl;
    auto ones_result = ones<float>({2, 2});
    if (!ones_result) {
      std::cout << "❌ Failed to create ones tensor" << std::endl;
      return 1;
    }
    auto ones_tensor = std::move(*ones_result);
    std::cout << "✅ Created ones tensor: " << ones_tensor.shape()[0] << "x" << ones_tensor.shape()[1] << std::endl;
    
    std::cout << "3. Testing size calculation..." << std::endl;
    std::cout << "✅ Zeros tensor size: " << zero_tensor.size() << std::endl;
    std::cout << "✅ Ones tensor size: " << ones_tensor.size() << std::endl;
    
    std::cout << "4. Testing basic element access..." << std::endl;
    zero_tensor(0, 0) = 42.0f;
    std::cout << "✅ Set zero_tensor(0,0) = 42.0" << std::endl;
    std::cout << "✅ Read zero_tensor(0,0) = " << zero_tensor(0, 0) << std::endl;
    
    std::cout << "\n🎉 All basic operations successful!" << std::endl;
    std::cout << "\n📋 Reproduction command:" << std::endl;
    std::cout << "bazel run //litert/cc/tensor:minimal_demo" << std::endl;
    
  } catch (const std::exception& e) {
    std::cout << "❌ Error: " << e.what() << std::endl;
    return 1;
  }
  
  return 0;
}