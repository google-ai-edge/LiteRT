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

#ifndef ODML_LITERT_CC_TENSOR_LITERT_TENSOR_H_
#define ODML_LITERT_CC_TENSOR_LITERT_TENSOR_H_

#include <cstddef>
#include <cstring>
#include <memory>
#include <vector>
#include <initializer_list>
#include <functional>

#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_element_type.h"

namespace litert {
namespace tensor {

// Forward declarations
template<typename T>
class Tensor;

template<typename T>
class TensorView;

// Type aliases for common tensor types
using FloatTensor = Tensor<float>;
using Int32Tensor = Tensor<int32_t>;
using Uint8Tensor = Tensor<uint8_t>;

// Shape type for tensor dimensions
using Shape = std::vector<size_t>;

// Core Tensor class that provides NumPy-like interface
template<typename T>
class Tensor {
 public:
  // Default constructor
  Tensor() = default;

  // Move constructor and assignment
  Tensor(Tensor&& other) noexcept 
      : buffer_(std::move(other.buffer_)),
        shape_(std::move(other.shape_)),
        strides_(std::move(other.strides_)),
        offset_(other.offset_) {
    other.offset_ = 0;
  }

  Tensor& operator=(Tensor&& other) noexcept {
    if (this != &other) {
      buffer_ = std::move(other.buffer_);
      shape_ = std::move(other.shape_);
      strides_ = std::move(other.strides_);
      offset_ = other.offset_;
      other.offset_ = 0;
    }
    return *this;
  }

  // Delete copy constructor and assignment to enforce move semantics
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;

  // Constructor from TensorBuffer (zero-copy)
  explicit Tensor(std::shared_ptr<TensorBuffer> buffer) 
      : buffer_(std::move(buffer)), offset_(0) {
    if (buffer_) {
      auto tensor_type = buffer_->TensorType();
      if (tensor_type) {
        auto dimensions = tensor_type->Layout().Dimensions();
        shape_.clear();
        for (int32_t dim : dimensions) {
          shape_.push_back(static_cast<size_t>(dim));
        }
        ComputeStrides();
      }
    }
  }

  // Element access operator
  T& operator()(std::initializer_list<size_t> indices) {
    return const_cast<T&>(static_cast<const Tensor*>(this)->operator()(indices));
  }

  const T& operator()(std::initializer_list<size_t> indices) const {
    size_t flat_index = ComputeFlatIndex(indices);
    auto lock_result = TensorBufferScopedLock::Create<T>(*buffer_);
    if (!lock_result) {
      throw std::runtime_error("Failed to lock tensor buffer");
    }
    auto [lock, data] = std::move(*lock_result);
    return data[flat_index + offset_];
  }

  // Convenience operators for common dimensions
  T& operator()(size_t i) { return operator()({i}); }
  const T& operator()(size_t i) const { return operator()({i}); }
  
  T& operator()(size_t i, size_t j) { return operator()({i, j}); }
  const T& operator()(size_t i, size_t j) const { return operator()({i, j}); }

  T& operator()(size_t i, size_t j, size_t k) { return operator()({i, j, k}); }
  const T& operator()(size_t i, size_t j, size_t k) const { return operator()({i, j, k}); }

  // Shape and properties
  const Shape& shape() const { return shape_; }
  size_t ndim() const { return shape_.size(); }
  size_t size() const {
    size_t total = 1;
    for (size_t dim : shape_) total *= dim;
    return total;
  }

  // Arithmetic operations (fluent style)
  Tensor add(const Tensor& other) const;
  Tensor add(T scalar) const;
  Tensor sub(const Tensor& other) const;
  Tensor sub(T scalar) const;
  Tensor mul(const Tensor& other) const;
  Tensor mul(T scalar) const;
  Tensor div(const Tensor& other) const;
  Tensor div(T scalar) const;

  // Operator overloading (C++ idiomatic style)
  Tensor operator+(const Tensor& other) const { return add(other); }
  Tensor operator+(T scalar) const { return add(scalar); }
  Tensor operator-(const Tensor& other) const { return sub(other); }
  Tensor operator-(T scalar) const { return sub(scalar); }
  Tensor operator*(const Tensor& other) const { return mul(other); }
  Tensor operator*(T scalar) const { return mul(scalar); }
  Tensor operator/(const Tensor& other) const { return div(other); }
  Tensor operator/(T scalar) const { return div(scalar); }

  // Universal functions (element-wise)
  Tensor sin() const;
  Tensor cos() const;
  Tensor exp() const;
  Tensor log() const;
  Tensor sqrt() const;
  Tensor abs() const;

  // Shape manipulation
  Tensor reshape(const Shape& new_shape) const;
  Tensor squeeze() const;
  Tensor expand_dims(size_t axis) const;
  Tensor transpose() const;  // 2D transpose
  Tensor transpose(const std::vector<size_t>& axes) const;  // N-D transpose

  // Slicing and indexing
  Tensor slice(const std::vector<std::pair<size_t, size_t>>& ranges) const;
  
  // Reduction operations
  T sum() const;
  T mean() const;
  T max() const;
  T min() const;
  size_t argmax() const;
  size_t argmin() const;
  
  // Axis-aware reductions
  Tensor sum(size_t axis) const;
  Tensor mean(size_t axis) const;
  Tensor max(size_t axis) const;
  Tensor min(size_t axis) const;

  // Linear algebra
  Tensor matmul(const Tensor& other) const;
  Tensor dot(const Tensor& other) const;  // 1D dot product

  // Data access
  Expected<void> fill(T value);
  Expected<std::vector<T>> to_vector() const;
  Expected<void> from_vector(const std::vector<T>& data);

  // TensorBuffer access
  std::shared_ptr<TensorBuffer> buffer() const { return buffer_; }

 private:
  std::shared_ptr<TensorBuffer> buffer_;
  Shape shape_;
  std::vector<size_t> strides_;
  size_t offset_ = 0;

  void ComputeStrides() {
    strides_.resize(shape_.size());
    if (!shape_.empty()) {
      strides_.back() = 1;
      for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i) {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
      }
    }
  }

  size_t ComputeFlatIndex(std::initializer_list<size_t> indices) const {
    if (indices.size() != shape_.size()) {
      throw std::invalid_argument("Index dimension mismatch");
    }
    size_t flat_index = 0;
    auto it = indices.begin();
    for (size_t i = 0; i < shape_.size(); ++i, ++it) {
      if (*it >= shape_[i]) {
        throw std::out_of_range("Index out of bounds");
      }
      flat_index += (*it) * strides_[i];
    }
    return flat_index;
  }

  // Helper for creating result tensors
  Tensor CreateResultTensor(const Shape& result_shape) const;
  
  // Element-wise operation helper
  template<typename Op>
  Tensor ElementWiseOp(const Tensor& other, Op op) const;
  
  template<typename Op>
  Tensor ElementWiseOp(T scalar, Op op) const;

  // Constructor for internal use (views)
  Tensor(std::shared_ptr<TensorBuffer> buffer, const Shape& shape, 
         const std::vector<size_t>& strides, size_t offset)
      : buffer_(std::move(buffer)), shape_(shape), strides_(strides), offset_(offset) {}

  friend class TensorView<T>;
};

// Factory functions for tensor creation
template<typename T>
Expected<Tensor<T>> zeros(const Shape& shape);

template<typename T>
Expected<Tensor<T>> ones(const Shape& shape);

template<typename T>
Expected<Tensor<T>> full(const Shape& shape, T value);

template<typename T>
Expected<Tensor<T>> from_data(const Shape& shape, const std::vector<T>& data);

// Create tensor from existing TensorBuffer (zero-copy)
template<typename T>
Expected<Tensor<T>> from_buffer(std::shared_ptr<TensorBuffer> buffer);

// Functional style operations (NumPy/TensorFlow-like)
template<typename T>
Tensor<T> add(const Tensor<T>& a, const Tensor<T>& b);

template<typename T>
Tensor<T> add(const Tensor<T>& a, T scalar);

template<typename T>
Tensor<T> sub(const Tensor<T>& a, const Tensor<T>& b);

template<typename T>
Tensor<T> mul(const Tensor<T>& a, const Tensor<T>& b);

template<typename T>
Tensor<T> div(const Tensor<T>& a, const Tensor<T>& b);

template<typename T>
Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b);

template<typename T>
Tensor<T> sin(const Tensor<T>& tensor);

template<typename T>
Tensor<T> cos(const Tensor<T>& tensor);

template<typename T>
Tensor<T> exp(const Tensor<T>& tensor);

template<typename T>
Tensor<T> log(const Tensor<T>& tensor);

template<typename T>
Tensor<T> sqrt(const Tensor<T>& tensor);

// Reduction operations
template<typename T>
T sum(const Tensor<T>& tensor);

template<typename T>
T mean(const Tensor<T>& tensor);

template<typename T>
T max(const Tensor<T>& tensor);

template<typename T>
T min(const Tensor<T>& tensor);

template<typename T>
size_t argmax(const Tensor<T>& tensor);

template<typename T>
size_t argmin(const Tensor<T>& tensor);

// Shape operations
template<typename T>
Tensor<T> reshape(const Tensor<T>& tensor, const Shape& new_shape);

template<typename T>
Tensor<T> transpose(const Tensor<T>& tensor);

template<typename T>
Tensor<T> concatenate(const std::vector<Tensor<T>>& tensors, size_t axis);

}  // namespace tensor
}  // namespace litert

#endif  // ODML_LITERT_CC_TENSOR_LITERT_TENSOR_H_