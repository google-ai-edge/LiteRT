#ifndef TENSORFLOW_LITE_CORE_FLATBUFFER_MODEL_BASE_H_
#define TENSORFLOW_LITE_CORE_FLATBUFFER_MODEL_BASE_H_

#include <memory>
#include "tensorflow/compiler/mlir/lite/allocation.h"

// Forward declarations
namespace tflite {
class Model;
}

namespace tflite {
namespace impl {

// Base template class for FlatBuffer models
template<typename Derived>
class FlatBufferModelBase {
 public:
  FlatBufferModelBase() = default;
  virtual ~FlatBufferModelBase() = default;
  
  // Get the allocation object
  virtual const tensorflow::Allocation* allocation() const {
    return allocation_.get();
  }
  
  // Get the model (stub implementation)
  virtual const tflite::Model* GetModel() const {
    return nullptr;  // Stub implementation
  }
  
 protected:
  std::unique_ptr<tensorflow::Allocation> allocation_;
};

}  // namespace impl
}  // namespace tflite

#endif  // TENSORFLOW_LITE_CORE_FLATBUFFER_MODEL_BASE_H_