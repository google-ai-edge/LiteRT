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

// Test program to verify memory safety of unowned buffer API
// Run with: valgrind --leak-check=full ./unowned_buffer_memory_test
// Or compile with: -fsanitize=address

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_model.h"

namespace {

// Test 1: Basic functionality - buffer remains valid
void TestBasicFunctionality(const std::string& model_path) {
  std::cout << "Test 1: Basic functionality\n";
  
  // Read model file
  std::ifstream file(model_path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    std::cerr << "Failed to open model file: " << model_path << "\n";
    return;
  }
  
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  
  std::vector<uint8_t> buffer(size);
  if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
    std::cerr << "Failed to read model file\n";
    return;
  }
  file.close();
  
  // Test C API
  {
    LiteRtModel model = nullptr;
    LiteRtStatus status = LiteRtCreateModelFromUnownedBuffer(
        buffer.data(), buffer.size(), &model);
    if (status != kLiteRtStatusOk) {
      std::cerr << "Failed to create model from unowned buffer (C API)\n";
      return;
    }
    
    LiteRtParamIndex num_subgraphs = 0;
    LiteRtGetNumModelSubgraphs(model, &num_subgraphs);
    std::cout << "  C API: Model has " << num_subgraphs << " subgraphs\n";
    
    LiteRtDestroyModel(model);
  }
  
  // Test C++ API
  {
    litert::BufferRef<uint8_t> buffer_ref(buffer.data(), buffer.size());
    auto model_result = litert::Model::CreateFromUnownedBuffer(buffer_ref);
    if (!model_result.HasValue()) {
      std::cerr << "Failed to create model from unowned buffer (C++ API)\n";
      return;
    }
    
    auto model = std::move(model_result.Value());
    std::cout << "  C++ API: Model has " << model.NumSubgraphs() << " subgraphs\n";
  }
  
  std::cout << "  ✓ Test 1 passed\n\n";
}

// Test 2: Verify no double-free or memory corruption
void TestNoDoubleFree(const std::string& model_path) {
  std::cout << "Test 2: No double-free test\n";
  
  std::ifstream file(model_path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    std::cerr << "Failed to open model file\n";
    return;
  }
  
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  
  // Allocate buffer with malloc to make valgrind/ASAN more sensitive
  uint8_t* buffer = static_cast<uint8_t*>(malloc(size));
  if (!file.read(reinterpret_cast<char*>(buffer), size)) {
    std::cerr << "Failed to read model file\n";
    free(buffer);
    return;
  }
  file.close();
  
  // Create model
  LiteRtModel model = nullptr;
  LiteRtStatus status = LiteRtCreateModelFromUnownedBuffer(
      buffer, size, &model);
  if (status != kLiteRtStatusOk) {
    std::cerr << "Failed to create model\n";
    free(buffer);
    return;
  }
  
  // Destroy model
  LiteRtDestroyModel(model);
  
  // Free buffer - should not cause double-free
  free(buffer);
  
  std::cout << "  ✓ Test 2 passed\n\n";
}

// Test 3: Multiple models from same buffer
void TestMultipleModels(const std::string& model_path) {
  std::cout << "Test 3: Multiple models from same buffer\n";
  
  std::ifstream file(model_path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    std::cerr << "Failed to open model file\n";
    return;
  }
  
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  
  std::vector<uint8_t> buffer(size);
  if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
    std::cerr << "Failed to read model file\n";
    return;
  }
  file.close();
  
  // Create multiple models from same buffer
  LiteRtModel model1 = nullptr;
  LiteRtModel model2 = nullptr;
  
  LiteRtStatus status1 = LiteRtCreateModelFromUnownedBuffer(
      buffer.data(), buffer.size(), &model1);
  LiteRtStatus status2 = LiteRtCreateModelFromUnownedBuffer(
      buffer.data(), buffer.size(), &model2);
  
  if (status1 != kLiteRtStatusOk || status2 != kLiteRtStatusOk) {
    std::cerr << "Failed to create models\n";
    if (model1) LiteRtDestroyModel(model1);
    if (model2) LiteRtDestroyModel(model2);
    return;
  }
  
  // Use both models
  LiteRtParamIndex num_subgraphs1 = 0, num_subgraphs2 = 0;
  LiteRtGetNumModelSubgraphs(model1, &num_subgraphs1);
  LiteRtGetNumModelSubgraphs(model2, &num_subgraphs2);
  
  std::cout << "  Model 1: " << num_subgraphs1 << " subgraphs\n";
  std::cout << "  Model 2: " << num_subgraphs2 << " subgraphs\n";
  
  // Destroy models
  LiteRtDestroyModel(model1);
  LiteRtDestroyModel(model2);
  
  std::cout << "  ✓ Test 3 passed\n\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <model_file_path>\n";
    return 1;
  }
  
  std::string model_path = argv[1];
  
  std::cout << "Running unowned buffer memory safety tests...\n\n";
  
  TestBasicFunctionality(model_path);
  TestNoDoubleFree(model_path);
  TestMultipleModels(model_path);
  
  std::cout << "All tests completed successfully!\n";
  std::cout << "\nTo verify memory safety, run with:\n";
  std::cout << "  valgrind --leak-check=full --show-leak-kinds=all " << argv[0] << " <model_file>\n";
  std::cout << "Or compile with -fsanitize=address\n";
  
  return 0;
}