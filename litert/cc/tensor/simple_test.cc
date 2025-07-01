// Simple test to debug tensor creation issues

#include "litert/cc/tensor/litert_tensor.h"
#include <iostream>

int main() {
  std::cout << "Starting simple tensor test..." << std::endl;
  
  try {
    std::cout << "Creating tensor..." << std::endl;
    
    auto tensor_result = litert::tensor::zeros<float>({2, 3});
    
    if (!tensor_result) {
      std::cout << "Failed to create tensor" << std::endl;
      return 1;
    }
    
    std::cout << "Tensor created successfully!" << std::endl;
    auto tensor = std::move(*tensor_result);
    
    std::cout << "Tensor size: " << tensor.size() << std::endl;
    std::cout << "Tensor shape: ";
    for (size_t dim : tensor.shape()) {
      std::cout << dim << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Test completed successfully!" << std::endl;
    
  } catch (const std::exception& e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
    return 1;
  }
  
  return 0;
}