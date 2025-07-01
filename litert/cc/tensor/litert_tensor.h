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
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_tensor_buffer.h"

namespace litert {
namespace tensor {

template <typename T>
class Tensor;

template <typename T>
class TensorView;

// Type aliases for mostly used tensor types
using FloatTensor = Tensor<float>;
using Int32Tensor = Tensor<int32_t>;
using Uint8Tensor = Tensor<uint8_t>;

// Shape type for tensor dimensions
using Shape = std::vector<size_t>;

// Core Tensor template.
template <typename T>
class Tensor {
 public:
  Tensor() = default;

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

  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;

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
      }
    }
  }

  // Element access operators - these require successful operations or program
  // termination. For error-safe access, use at() methods instead.
  T& operator()(std::initializer_list<size_t> indices) {
    auto result = at(indices);
    if (!result) {
      // This should never happen in correct usage - indicates programming error
      std::abort();
    }
    return **result;
  }

  const T& operator()(std::initializer_list<size_t> indices) const {
    auto result = at(indices);
    if (!result) {
      // This should never happen in correct usage - indicates programming error
      std::abort();
    }
    return **result;
  }

  // Safe element access with error handling.
  Expected<T*> at(std::initializer_list<size_t> indices);
  Expected<const T*> at(std::initializer_list<size_t> indices) const;

  // Convenience operators for common dimensions.
  T& operator()(size_t i) { return operator()({i}); }
  const T& operator()(size_t i) const { return operator()({i}); }

  T& operator()(size_t i, size_t j) { return operator()({i, j}); }
  const T& operator()(size_t i, size_t j) const { return operator()({i, j}); }

  T& operator()(size_t i, size_t j, size_t k) { return operator()({i, j, k}); }
  const T& operator()(size_t i, size_t j, size_t k) const {
    return operator()({i, j, k});
  }

  // Safe convenience access methods.
  Expected<T*> at(size_t i) { return at({i}); }
  Expected<const T*> at(size_t i) const { return at({i}); }
  Expected<T*> at(size_t i, size_t j) { return at({i, j}); }
  Expected<const T*> at(size_t i, size_t j) const { return at({i, j}); }
  Expected<T*> at(size_t i, size_t j, size_t k) { return at({i, j, k}); }
  Expected<const T*> at(size_t i, size_t j, size_t k) const {
    return at({i, j, k});
  }

  // Shape and properties.
  const Shape& shape() const { return shape_; }
  size_t ndim() const { return shape_.size(); }
  size_t size() const {
    size_t total = 1;
    for (size_t dim : shape_) total *= dim;
    return total;
  }

  // Arithmetic operations (fluent style).
  Expected<Tensor> add(const Tensor& other) const;
  Expected<Tensor> add(T scalar) const;
  Expected<Tensor> sub(const Tensor& other) const;
  Expected<Tensor> sub(T scalar) const;
  Expected<Tensor> mul(const Tensor& other) const;
  Expected<Tensor> mul(T scalar) const;
  Expected<Tensor> div(const Tensor& other) const;
  Expected<Tensor> div(T scalar) const;

  // Operator overloading (C++ idiomatic style) - these abort on error.
  // For error-safe operations, use the named methods (add, sub, etc.) instead.
  Tensor operator+(const Tensor& other) const {
    auto result = add(other);
    if (!result) {
      std::abort();
    }
    return std::move(*result);
  }
  Tensor operator+(T scalar) const {
    auto result = add(scalar);
    if (!result) {
      std::abort();
    }
    return std::move(*result);
  }
  Tensor operator-(const Tensor& other) const {
    auto result = sub(other);
    if (!result) {
      std::abort();
    }
    return std::move(*result);
  }
  Tensor operator-(T scalar) const {
    auto result = sub(scalar);
    if (!result) {
      std::abort();
    }
    return std::move(*result);
  }
  Tensor operator*(const Tensor& other) const {
    auto result = mul(other);
    if (!result) {
      std::abort();
    }
    return std::move(*result);
  }
  Tensor operator*(T scalar) const {
    auto result = mul(scalar);
    if (!result) {
      std::abort();
    }
    return std::move(*result);
  }
  Tensor operator/(const Tensor& other) const {
    auto result = div(other);
    if (!result) {
      std::abort();
    }
    return std::move(*result);
  }
  Tensor operator/(T scalar) const {
    auto result = div(scalar);
    if (!result) {
      std::abort();
    }
    return std::move(*result);
  }

  // Common element wise mathematical operations.
  Expected<Tensor> sin() const;
  Expected<Tensor> cos() const;
  Expected<Tensor> exp() const;
  Expected<Tensor> log() const;
  Expected<Tensor> sqrt() const;
  Expected<Tensor> abs() const;

  // Reduction operations.
  Expected<T> sum() const;
  Expected<T> mean() const;
  Expected<T> max() const;
  Expected<T> min() const;
  Expected<size_t> argmax() const;
  Expected<size_t> argmin() const;

  // Data access
  Expected<void> fill(T value);

  // TensorBuffer access
  std::shared_ptr<TensorBuffer> buffer() const { return buffer_; }

 private:
  std::shared_ptr<TensorBuffer> buffer_;
  Shape shape_;
  std::vector<size_t> strides_;
  size_t offset_ = 0;

  Expected<size_t> ComputeFlatIndex(
      std::initializer_list<size_t> indices) const {
    if (indices.size() != shape_.size()) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "Index dimension mismatch");
    }
    size_t flat_index = 0;
    auto it = indices.begin();
    for (size_t i = 0; i < shape_.size(); ++i, ++it) {
      if (*it >= shape_[i]) {
        return Unexpected(kLiteRtStatusErrorInvalidArgument,
                          "Index out of bounds");
      }
      flat_index += (*it) * strides_[i];
    }
    return flat_index;
  }

  // Helper for creating result tensors
  Expected<Tensor> CreateResultTensor(const Shape& result_shape) const;

  // Element-wise operation helper
  template <typename Op>
  Expected<Tensor> ElementWiseOp(const Tensor& other, Op op) const;

  template <typename Op>
  Expected<Tensor> ElementWiseOp(T scalar, Op op) const;

  // Constructor for internal use (views)
  Tensor(std::shared_ptr<TensorBuffer> buffer, const Shape& shape,
         const std::vector<size_t>& strides, size_t offset)
      : buffer_(std::move(buffer)),
        shape_(shape),
        strides_(strides),
        offset_(offset) {}

  friend class TensorView<T>;
};

// Factory functions for tensor creation
template <typename T>
Expected<Tensor<T>> zeros(const Shape& shape);

template <typename T>
Expected<Tensor<T>> ones(const Shape& shape);

template <typename T>
Expected<Tensor<T>> full(const Shape& shape, T value);

// Common element-wise mathematical operations.
template <typename T>
Expected<Tensor<T>> add(const Tensor<T>& a, const Tensor<T>& b);

template <typename T>
Expected<Tensor<T>> add(const Tensor<T>& a, T scalar);

template <typename T>
Expected<Tensor<T>> sub(const Tensor<T>& a, const Tensor<T>& b);

template <typename T>
Expected<Tensor<T>> mul(const Tensor<T>& a, const Tensor<T>& b);

template <typename T>
Expected<Tensor<T>> div(const Tensor<T>& a, const Tensor<T>& b);

template <typename T>
Expected<Tensor<T>> sin(const Tensor<T>& tensor);

template <typename T>
Expected<Tensor<T>> cos(const Tensor<T>& tensor);

template <typename T>
Expected<Tensor<T>> exp(const Tensor<T>& tensor);

template <typename T>
Expected<Tensor<T>> log(const Tensor<T>& tensor);

template <typename T>
Expected<Tensor<T>> sqrt(const Tensor<T>& tensor);

// Reduction operations.
template <typename T>
Expected<T> sum(const Tensor<T>& tensor);

template <typename T>
Expected<T> mean(const Tensor<T>& tensor);

template <typename T>
Expected<T> max(const Tensor<T>& tensor);

template <typename T>
Expected<T> min(const Tensor<T>& tensor);

template <typename T>
Expected<size_t> argmax(const Tensor<T>& tensor);

template <typename T>
Expected<size_t> argmin(const Tensor<T>& tensor);
}  // namespace tensor
}  // namespace litert

#endif  // ODML_LITERT_CC_TENSOR_LITERT_TENSOR_H_
