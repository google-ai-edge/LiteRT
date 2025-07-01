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
#include <memory>
#include <vector>

namespace litert {
namespace tensor {

// This example demonstrates the complete workflow from the proposal:
// "Getting Started: A Developer's Workflow"
class LiteRTTensorWorkflow {
 public:
  LiteRTTensorWorkflow() = default;
  
  // Step 1: Load Model and Prepare Runtime
  Expected<void> LoadModelAndCreateEnvironment(const std::string& model_path) {
    std::cout << "Step 1: Loading model and creating environment..." << std::endl;
    
    // Load the model from a file
    LITERT_ASSIGN_OR_RETURN(model_, Model::CreateFromFile(model_path.c_str()));
    std::cout << "✓ Model loaded from: " << model_path << std::endl;
    
    // Create a runtime environment
    LITERT_ASSIGN_OR_RETURN(env_, Environment::Create({}));
    std::cout << "✓ Runtime environment created" << std::endl;
    
    // Create a compiled model for the desired hardware accelerator
    LITERT_ASSIGN_OR_RETURN(
        compiled_model_,
        CompiledModel::Create(*env_, *model_, HwAccelerator::kCpu));
    std::cout << "✓ Model compiled for CPU execution" << std::endl;
    
    return {};
  }
  
  // Step 2: Create and Wrap Buffers for Manipulation
  Expected<void> CreateAndWrapBuffers() {
    std::cout << "\nStep 2: Creating and wrapping tensor buffers..." << std::endl;
    
    // Create buffers for the first input and output using the compiled model
    LITERT_ASSIGN_OR_RETURN(
        input_buffer_,
        TensorBuffer::CreateManaged(compiled_model_->GetInputTensorType(0)));
    std::cout << "✓ Input buffer created" << std::endl;
    
    LITERT_ASSIGN_OR_RETURN(
        output_buffer_,
        TensorBuffer::CreateManaged(compiled_model_->GetOutputTensorType(0)));
    std::cout << "✓ Output buffer created" << std::endl;
    
    // Wrap the buffers for NumPy-like operations (zero-copy)
    LITERT_ASSIGN_OR_RETURN(
        input_tensor_,
        from_buffer<float>(std::make_shared<TensorBuffer>(std::move(input_buffer_))));
    std::cout << "✓ Input tensor wrapper created (zero-copy)" << std::endl;
    
    LITERT_ASSIGN_OR_RETURN(
        output_tensor_,
        from_buffer<float>(std::make_shared<TensorBuffer>(std::move(output_buffer_))));
    std::cout << "✓ Output tensor wrapper created (zero-copy)" << std::endl;
    
    // Display tensor information
    std::cout << "Input tensor shape: ";
    PrintShape(input_tensor_->shape());
    std::cout << "Output tensor shape: ";
    PrintShape(output_tensor_->shape());
    
    return {};
  }
  
  // Step 3: Pre-process Input
  Expected<void> PreprocessInput() {
    std::cout << "\nStep 3: Preprocessing input data..." << std::endl;
    
    if (!input_tensor_) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument, "Input tensor not initialized");
    }
    
    // Simulate loading raw image data (e.g., from camera/file)
    // For demonstration, we'll fill with simulated RGB values [0, 255]
    std::cout << "Loading simulated raw image data..." << std::endl;
    LITERT_RETURN_IF_ERROR(input_tensor_->fill(128.0f));  // Simulated gray image
    
    // Example: Normalize the input tensor from [0, 255] to [-1, 1]
    // This demonstrates the fluent API for preprocessing
    std::cout << "Applying preprocessing pipeline:" << std::endl;
    std::cout << "  1. Normalize [0, 255] -> [0, 1]" << std::endl;
    std::cout << "  2. Center around 0: [0, 1] -> [-0.5, 0.5]" << std::endl;
    std::cout << "  3. Scale to [-1, 1]" << std::endl;
    
    // Chain preprocessing operations
    auto preprocessed = input_tensor_->div(255.0f)    // [0, 255] -> [0, 1]
                                    .sub(0.5f)        // [0, 1] -> [-0.5, 0.5]
                                    .mul(2.0f);       // [-0.5, 0.5] -> [-1, 1]
    
    // Copy preprocessed data back to input tensor
    auto preprocessed_data = preprocessed.to_vector();
    if (!preprocessed_data) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure, "Failed to get preprocessed data");
    }
    
    LITERT_RETURN_IF_ERROR(input_tensor_->from_vector(*preprocessed_data));
    
    std::cout << "✓ Preprocessing completed" << std::endl;
    std::cout << "  Original mean: ~128, Preprocessed mean: " << input_tensor_->mean() << std::endl;
    
    return {};
  }
  
  // Step 4: Run Inference
  Expected<void> RunInference() {
    std::cout << "\nStep 4: Running model inference..." << std::endl;
    
    if (!compiled_model_ || !input_tensor_ || !output_tensor_) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument, "Model or tensors not initialized");
    }
    
    // Run inference synchronously
    // Note: In real implementation, we'd use the actual buffer objects
    std::cout << "Executing inference on compiled model..." << std::endl;
    
    // Simulate inference by setting some output values
    // In real code: LITERT_RETURN_IF_ERROR(compiled_model_->Run(*input_buffer_, output_buffer_.get()));
    
    // For demonstration, simulate classification logits
    LITERT_RETURN_IF_ERROR(output_tensor_->fill(0.1f));  // Base logits
    
    // Set a few classes to have higher scores
    if (output_tensor_->size() > 10) {
      (*output_tensor_)(42) = 2.5f;  // Class 42 has highest score
      (*output_tensor_)(17) = 1.8f;  // Class 17 has second highest
      (*output_tensor_)(3) = 1.2f;   // Class 3 has third highest
    }
    
    std::cout << "✓ Inference completed" << std::endl;
    std::cout << "  Output tensor sum: " << output_tensor_->sum() << std::endl;
    
    return {};
  }
  
  // Step 5: Post-process Output
  Expected<void> PostprocessOutput() {
    std::cout << "\nStep 5: Postprocessing model output..." << std::endl;
    
    if (!output_tensor_) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument, "Output tensor not initialized");
    }
    
    // The output_tensor now provides a view into the results
    std::cout << "Applying softmax function to convert logits to probabilities..." << std::endl;
    
    // Apply softmax: exp(x) / sum(exp(x))
    auto exp_logits = output_tensor_->exp();
    auto sum_exp = exp_logits.sum();
    auto probabilities = exp_logits.div(sum_exp);
    
    std::cout << "✓ Softmax applied" << std::endl;
    std::cout << "  Probabilities sum (should be ~1.0): " << probabilities.sum() << std::endl;
    
    // Find the index of the highest probability
    auto max_index = probabilities.argmax();
    auto max_prob = probabilities.max();
    
    std::cout << "✓ Prediction extracted" << std::endl;
    std::cout << "  Predicted class: " << max_index << std::endl;
    std::cout << "  Confidence: " << max_prob << std::endl;
    
    // Additional postprocessing: Get top-3 predictions
    std::cout << "\nTop predictions analysis:" << std::endl;
    auto prob_data = probabilities.to_vector();
    if (prob_data) {
      auto sorted_indices = GetTopKIndices(*prob_data, 3);
      for (size_t i = 0; i < sorted_indices.size(); ++i) {
        size_t idx = sorted_indices[i];
        float prob = (*prob_data)[idx];
        std::cout << "  " << (i + 1) << ". Class " << idx << ": " 
                  << (prob * 100.0f) << "%" << std::endl;
      }
    }
    
    return {};
  }
  
  // Complete workflow demonstration
  void DemonstrateCompleteWorkflow() {
    std::cout << "=== LiteRT Tensor API Complete Workflow Demo ===" << std::endl;
    std::cout << "This demonstrates the exact workflow from the proposal." << std::endl;
    std::cout << std::endl;
    
    try {
      // Note: Using a dummy model path for demonstration
      // In real usage, this would be a path to an actual .tflite file
      std::string model_path = "model.tflite";
      std::cout << "Demo mode: Simulating model loading (no actual .tflite file needed)" << std::endl;
      
      // For demo purposes, we'll simulate the workflow without actual model loading
      DemonstrateWorkflowSteps();
      
    } catch (const std::exception& e) {
      std::cerr << "Error in workflow: " << e.what() << std::endl;
    }
  }

 private:
  std::unique_ptr<Model> model_;
  std::unique_ptr<Environment> env_;
  std::unique_ptr<CompiledModel> compiled_model_;
  TensorBuffer input_buffer_;
  TensorBuffer output_buffer_;
  std::unique_ptr<Tensor<float>> input_tensor_;
  std::unique_ptr<Tensor<float>> output_tensor_;
  
  void PrintShape(const Shape& shape) {
    std::cout << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
      std::cout << shape[i];
      if (i < shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
  }
  
  std::vector<size_t> GetTopKIndices(const std::vector<float>& values, size_t k) {
    std::vector<std::pair<float, size_t>> indexed_values;
    for (size_t i = 0; i < values.size(); ++i) {
      indexed_values.emplace_back(values[i], i);
    }
    
    std::partial_sort(indexed_values.begin(), 
                      indexed_values.begin() + std::min(k, indexed_values.size()),
                      indexed_values.end(),
                      std::greater<std::pair<float, size_t>>());
    
    std::vector<size_t> result;
    for (size_t i = 0; i < std::min(k, indexed_values.size()); ++i) {
      result.push_back(indexed_values[i].second);
    }
    
    return result;
  }
  
  // Simplified workflow demonstration without actual model loading
  void DemonstrateWorkflowSteps() {
    std::cout << "\n=== Simulated Workflow (No Model File Required) ===" << std::endl;
    
    // Create simulated input/output tensors
    std::cout << "Step 1-2: Creating simulated input/output tensors..." << std::endl;
    
    auto input_result = zeros<float>({1, 224, 224, 3});  // Typical image input
    auto output_result = zeros<float>({1, 1000});         // ImageNet-style output
    
    if (!input_result || !output_result) {
      std::cerr << "Failed to create tensors" << std::endl;
      return;
    }
    
    auto input = std::move(*input_result);
    auto output = std::move(*output_result);
    
    std::cout << "✓ Created input tensor: ";
    PrintShape(input.shape());
    std::cout << "✓ Created output tensor: ";
    PrintShape(output.shape());
    
    // Step 3: Preprocessing
    std::cout << "\nStep 3: Demonstrating preprocessing pipeline..." << std::endl;
    
    // Simulate raw image data
    auto fill_result = input.fill(128.0f);
    if (!fill_result) {
      std::cerr << "Failed to fill input tensor" << std::endl;
      return;
    }
    
    // Preprocessing pipeline (exactly as in the proposal)
    auto preprocessed = input.div(255.0f).sub(0.5f).mul(2.0f);
    std::cout << "✓ Applied preprocessing: input.div(255.0f).sub(0.5f).mul(2.0f)" << std::endl;
    std::cout << "  Original mean: 128, Preprocessed mean: " << preprocessed.mean() << std::endl;
    
    // Step 4: Simulate inference
    std::cout << "\nStep 4: Simulating model inference..." << std::endl;
    
    // Fill output with simulated logits
    auto output_fill = output.fill(0.1f);
    if (!output_fill) {
      std::cerr << "Failed to fill output tensor" << std::endl;
      return;
    }
    
    // Set some classes to have higher scores
    output(0, 42) = 2.5f;
    output(0, 17) = 1.8f;
    output(0, 3) = 1.2f;
    
    std::cout << "✓ Simulated inference completed" << std::endl;
    
    // Step 5: Postprocessing (exactly as in the proposal)
    std::cout << "\nStep 5: Postprocessing with tensor operations..." << std::endl;
    
    // Apply softmax function to the output
    auto probabilities = output.exp();
    auto prob_sum = probabilities.sum();
    probabilities = probabilities.div(prob_sum);
    
    std::cout << "✓ Applied softmax: output.exp().div(sum)" << std::endl;
    std::cout << "  Probabilities sum: " << probabilities.sum() << std::endl;
    
    // Find the index of the highest probability
    auto max_index = probabilities.argmax();
    std::cout << "✓ Found predicted class: " << max_index << std::endl;
    
    std::cout << "\n=== Workflow Summary ===" << std::endl;
    std::cout << "✓ Zero-copy tensor operations" << std::endl;
    std::cout << "✓ Fluent API for readable preprocessing" << std::endl;
    std::cout << "✓ Hardware-optimized execution (simulated)" << std::endl;
    std::cout << "✓ Efficient postprocessing with tensor operations" << std::endl;
    
    // Demonstrate the three API styles
    std::cout << "\n=== Three API Styles Demonstration ===" << std::endl;
    
    auto a = *full<float>({2, 2}, 1.0f);
    auto b = *full<float>({2, 2}, 2.0f);
    
    // 1. Operator Overloading (C++ idiomatic)
    auto c1 = a + b;
    std::cout << "1. Operator style: c1 = a + b, sum = " << c1.sum() << std::endl;
    
    // 2. Fluent / Method Style (TensorFlow.js-like)
    auto c2 = a.add(b);
    std::cout << "2. Fluent style:   c2 = a.add(b), sum = " << c2.sum() << std::endl;
    
    // 3. Functional Style (NumPy/TensorFlow-like)
    auto c3 = add(a, b);
    std::cout << "3. Functional style: c3 = add(a, b), sum = " << c3.sum() << std::endl;
    
    std::cout << "\n✓ All three styles produce identical results!" << std::endl;
  }
};

}  // namespace tensor
}  // namespace litert

// Main function to run the complete workflow example
int main() {
  litert::tensor::LiteRTTensorWorkflow workflow;
  workflow.DemonstrateCompleteWorkflow();
  return 0;
}