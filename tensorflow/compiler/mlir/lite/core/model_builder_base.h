#ifndef TENSORFLOW_COMPILER_MLIR_LITE_CORE_MODEL_BUILDER_BASE_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_CORE_MODEL_BUILDER_BASE_H_

#include <memory>
#include <string>
#include "tensorflow/compiler/mlir/lite/allocation.h"

namespace tflite {

// Base class for model builders in TensorFlow Lite
class ModelBuilderBase {
 public:
  virtual ~ModelBuilderBase() = default;
  
  // Build a model from an allocation
  virtual bool BuildFromAllocation(std::unique_ptr<tensorflow::Allocation> allocation) = 0;
  
  // Build a model from a buffer
  virtual bool BuildFromBuffer(const char* buffer, size_t buffer_size) = 0;
  
  // Get error string if build failed
  virtual std::string GetErrorString() const = 0;
  
  // Check if the model is valid
  virtual bool IsValid() const = 0;

 protected:
  ModelBuilderBase() = default;
};

}  // namespace tflite

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_CORE_MODEL_BUILDER_BASE_H_