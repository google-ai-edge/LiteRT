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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <type_traits>
#include <utility>

#include "litert/c/litert_common.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_tensor_buffer.h"

namespace litert {
namespace tensor {

// Explicit template instantiations for commonly used.
template class Tensor<float>;
template class Tensor<int32_t>;
template class Tensor<uint8_t>;

// Safe element access implementation.
template <typename T>
Expected<T*> Tensor<T>::at(std::initializer_list<size_t> indices) {
  LITERT_ASSIGN_OR_RETURN(size_t flat_index, ComputeFlatIndex(indices));
  auto lock_result = TensorBufferScopedLock::Create<T>(*buffer_);
  if (!lock_result) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to lock tensor buffer");
  }
  auto [lock, data] = std::move(*lock_result);
  return &data[flat_index + offset_];
}

template <typename T>
Expected<const T*> Tensor<T>::at(std::initializer_list<size_t> indices) const {
  LITERT_ASSIGN_OR_RETURN(size_t flat_index, ComputeFlatIndex(indices));
  auto lock_result = TensorBufferScopedLock::Create<T>(*buffer_);
  if (!lock_result) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to lock tensor buffer");
  }
  auto [lock, data] = std::move(*lock_result);
  return &data[flat_index + offset_];
}

// Implementation of arithmetic operations (fluent style)
template <typename T>
Expected<Tensor<T>> Tensor<T>::add(const Tensor& other) const {
  return ElementWiseOp(other, std::plus<T>{});
}

template <typename T>
Expected<Tensor<T>> Tensor<T>::add(T scalar) const {
  return ElementWiseOp(scalar, std::plus<T>{});
}

template <typename T>
Expected<Tensor<T>> Tensor<T>::sub(const Tensor& other) const {
  return ElementWiseOp(other, std::minus<T>{});
}

template <typename T>
Expected<Tensor<T>> Tensor<T>::sub(T scalar) const {
  return ElementWiseOp(scalar, std::minus<T>{});
}

template <typename T>
Expected<Tensor<T>> Tensor<T>::mul(const Tensor& other) const {
  return ElementWiseOp(other, std::multiplies<T>{});
}

template <typename T>
Expected<Tensor<T>> Tensor<T>::mul(T scalar) const {
  return ElementWiseOp(scalar, std::multiplies<T>{});
}

template <typename T>
Expected<Tensor<T>> Tensor<T>::div(const Tensor& other) const {
  return ElementWiseOp(other, std::divides<T>{});
}

template <typename T>
Expected<Tensor<T>> Tensor<T>::div(T scalar) const {
  return ElementWiseOp(scalar, std::divides<T>{});
}

// Universal functions
template <typename T>
Expected<Tensor<T>> Tensor<T>::sin() const {
  return ElementWiseOp(T{0}, [](T a, T) { return std::sin(a); });
}

template <typename T>
Expected<Tensor<T>> Tensor<T>::cos() const {
  return ElementWiseOp(T{0}, [](T a, T) { return std::cos(a); });
}

template <typename T>
Expected<Tensor<T>> Tensor<T>::exp() const {
  return ElementWiseOp(T{0}, [](T a, T) { return std::exp(a); });
}

template <typename T>
Expected<Tensor<T>> Tensor<T>::log() const {
  return ElementWiseOp(T{0}, [](T a, T) { return std::log(a); });
}

template <typename T>
Expected<Tensor<T>> Tensor<T>::sqrt() const {
  return ElementWiseOp(T{0}, [](T a, T) { return std::sqrt(a); });
}

template <typename T>
Expected<Tensor<T>> Tensor<T>::abs() const {
  return ElementWiseOp(T{0}, [](T a, T) { return std::abs(a); });
}

// Reduction operations.
template <typename T>
Expected<T> Tensor<T>::sum() const {
  auto lock_result = TensorBufferScopedLock::Create<T>(*buffer_);
  if (!lock_result) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to lock tensor buffer");
  }
  auto [lock, data] = std::move(*lock_result);

  T result = T{0};
  size_t total_size = size();

  for (size_t i = 0; i < total_size; ++i) {
    result += data[i];
  }

  return result;
}

template <typename T>
Expected<T> Tensor<T>::mean() const {
  LITERT_ASSIGN_OR_RETURN(T total, sum());
  return total / static_cast<T>(size());
}

template <typename T>
Expected<T> Tensor<T>::max() const {
  auto lock_result = TensorBufferScopedLock::Create<T>(*buffer_);
  if (!lock_result) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to lock tensor buffer");
  }
  auto [lock, data] = std::move(*lock_result);

  if (size() == 0) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Cannot find max of empty tensor");
  }

  T result = data[0];
  size_t total_size = size();

  for (size_t i = 1; i < total_size; ++i) {
    result = std::max(result, data[i]);
  }

  return result;
}

template <typename T>
Expected<T> Tensor<T>::min() const {
  auto lock_result = TensorBufferScopedLock::Create<T>(*buffer_);
  if (!lock_result) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to lock tensor buffer");
  }
  auto [lock, data] = std::move(*lock_result);

  if (size() == 0) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Cannot find min of empty tensor");
  }

  T result = data[0];
  size_t total_size = size();

  for (size_t i = 1; i < total_size; ++i) {
    result = std::min(result, data[i]);
  }

  return result;
}

template <typename T>
Expected<size_t> Tensor<T>::argmax() const {
  auto lock_result = TensorBufferScopedLock::Create<T>(*buffer_);
  if (!lock_result) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to lock tensor buffer");
  }
  auto [lock, data] = std::move(*lock_result);

  if (size() == 0) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Cannot find argmax of empty tensor");
  }

  size_t max_idx = 0;
  T max_val = data[0];
  size_t total_size = size();

  for (size_t i = 1; i < total_size; ++i) {
    if (data[i] > max_val) {
      max_val = data[i];
      max_idx = i;
    }
  }

  return max_idx;
}

template <typename T>
Expected<size_t> Tensor<T>::argmin() const {
  auto lock_result = TensorBufferScopedLock::Create<T>(*buffer_);
  if (!lock_result) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to lock tensor buffer");
  }
  auto [lock, data] = std::move(*lock_result);

  if (size() == 0) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Cannot find argmin of empty tensor");
  }

  size_t min_idx = 0;
  T min_val = data[0];
  size_t total_size = size();

  for (size_t i = 1; i < total_size; ++i) {
    if (data[i] < min_val) {
      min_val = data[i];
      min_idx = i;
    }
  }

  return min_idx;
}

template <typename T>
Expected<void> Tensor<T>::fill(T value) {
  auto lock_result = TensorBufferScopedLock::Create<T>(*buffer_);
  if (!lock_result) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to lock tensor buffer");
  }
  auto [lock, data] = std::move(*lock_result);

  size_t total_size = size();
  for (size_t i = 0; i < total_size; ++i) {
    data[i] = value;
  }

  return {};
}

// Helper functions.
template <typename T>
Expected<Tensor<T>> Tensor<T>::CreateResultTensor(
    const Shape& result_shape) const {
  // Create a new TensorBuffer for the result
  size_t total_size = 1;
  for (size_t dim : result_shape) total_size *= dim;

  // Create tensor type based on T and result_shape
  Dimensions dims;
  for (size_t dim : result_shape) dims.push_back(static_cast<int32_t>(dim));

  ElementType element_type;
  if constexpr (std::is_same_v<T, float>) {
    element_type = ElementType::Float32;
  } else if constexpr (std::is_same_v<T, int32_t>) {
    element_type = ElementType::Int32;
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    element_type = ElementType::UInt8;
  } else {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Unsupported tensor type");
  }

  Layout layout(dims);
  RankedTensorType tensor_type(element_type, std::move(layout));

  LITERT_ASSIGN_OR_RETURN(
      auto buffer,
      TensorBuffer::CreateManaged(kLiteRtTensorBufferTypeHostMemory,
                                  tensor_type, total_size * sizeof(T)));

  return Tensor(std::make_shared<TensorBuffer>(std::move(buffer)));
}

template <typename T>
template <typename Op>
Expected<Tensor<T>> Tensor<T>::ElementWiseOp(const Tensor& other, Op op) const {
  if (shape_ != other.shape_) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Tensor shapes must match for element-wise operations");
  }

  LITERT_ASSIGN_OR_RETURN(Tensor result, CreateResultTensor(shape_));

  auto this_lock = TensorBufferScopedLock::Create<T>(*buffer_);
  auto other_lock = TensorBufferScopedLock::Create<T>(*other.buffer_);
  auto result_lock = TensorBufferScopedLock::Create<T>(*result.buffer_);

  if (!this_lock || !other_lock || !result_lock) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to lock tensor buffers");
  }

  auto [this_guard, this_data] = std::move(*this_lock);
  auto [other_guard, other_data] = std::move(*other_lock);
  auto [result_guard, result_data] = std::move(*result_lock);

  size_t total_size = size();
  for (size_t i = 0; i < total_size; ++i) {
    result_data[i] = op(this_data[i], other_data[i]);
  }

  return std::move(result);
}

template <typename T>
template <typename Op>
Expected<Tensor<T>> Tensor<T>::ElementWiseOp(T scalar, Op op) const {
  LITERT_ASSIGN_OR_RETURN(Tensor result, CreateResultTensor(shape_));

  auto this_lock = TensorBufferScopedLock::Create<T>(*buffer_);
  auto result_lock = TensorBufferScopedLock::Create<T>(*result.buffer_);

  if (!this_lock || !result_lock) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to lock tensor buffers");
  }

  auto [this_guard, this_data] = std::move(*this_lock);
  auto [result_guard, result_data] = std::move(*result_lock);

  size_t total_size = size();
  for (size_t i = 0; i < total_size; ++i) {
    result_data[i] = op(this_data[i], scalar);
  }

  return std::move(result);
}

// Creation functions.
template <typename T>
Expected<Tensor<T>> zeros(const Shape& shape) {
  size_t total_size = 1;
  for (size_t dim : shape) total_size *= dim;

  // Create RankedTensorType based on shape and T.
  Dimensions dims;
  for (size_t dim : shape) dims.push_back(static_cast<int32_t>(dim));

  ElementType element_type;
  if constexpr (std::is_same_v<T, float>) {
    element_type = ElementType::Float32;
  } else if constexpr (std::is_same_v<T, int32_t>) {
    element_type = ElementType::Int32;
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    element_type = ElementType::UInt8;
  } else {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Unsupported tensor type");
  }

  Layout layout(dims);
  RankedTensorType tensor_type(element_type, std::move(layout));

  LITERT_ASSIGN_OR_RETURN(
      auto buffer,
      TensorBuffer::CreateManaged(kLiteRtTensorBufferTypeHostMemory,
                                  tensor_type, total_size * sizeof(T)));

  auto tensor = Tensor<T>(std::make_shared<TensorBuffer>(std::move(buffer)));
  LITERT_RETURN_IF_ERROR(tensor.fill(T{0}));

  return std::move(tensor);
}

template <typename T>
Expected<Tensor<T>> ones(const Shape& shape) {
  LITERT_ASSIGN_OR_RETURN(auto tensor, zeros<T>(shape));
  LITERT_RETURN_IF_ERROR(tensor.fill(T{1}));
  return std::move(tensor);
}

template <typename T>
Expected<Tensor<T>> full(const Shape& shape, T value) {
  LITERT_ASSIGN_OR_RETURN(auto tensor, zeros<T>(shape));
  LITERT_RETURN_IF_ERROR(tensor.fill(value));
  return std::move(tensor);
}

template <typename T>
Expected<Tensor<T>> from_buffer(std::shared_ptr<TensorBuffer> buffer) {
  if (!buffer) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Buffer cannot be null");
  }

  return Tensor<T>(std::move(buffer));
}

// Functional style operations.
template <typename T>
Expected<Tensor<T>> add(const Tensor<T>& a, const Tensor<T>& b) {
  return a.add(b);
}

template <typename T>
Expected<Tensor<T>> add(const Tensor<T>& a, T scalar) {
  return a.add(scalar);
}

template <typename T>
Expected<Tensor<T>> sub(const Tensor<T>& a, const Tensor<T>& b) {
  return a.sub(b);
}

template <typename T>
Expected<Tensor<T>> mul(const Tensor<T>& a, const Tensor<T>& b) {
  return a.mul(b);
}

template <typename T>
Expected<Tensor<T>> div(const Tensor<T>& a, const Tensor<T>& b) {
  return a.div(b);
}

template <typename T>
Expected<Tensor<T>> sin(const Tensor<T>& tensor) {
  return tensor.sin();
}

template <typename T>
Expected<Tensor<T>> cos(const Tensor<T>& tensor) {
  return tensor.cos();
}

template <typename T>
Expected<Tensor<T>> exp(const Tensor<T>& tensor) {
  return tensor.exp();
}

template <typename T>
Expected<Tensor<T>> sqrt(const Tensor<T>& tensor) {
  return tensor.sqrt();
}

template <typename T>
Expected<T> sum(const Tensor<T>& tensor) {
  return tensor.sum();
}

template <typename T>
Expected<T> mean(const Tensor<T>& tensor) {
  return tensor.mean();
}

template <typename T>
Expected<size_t> argmax(const Tensor<T>& tensor) {
  return tensor.argmax();
}

// Explicit template instantiations for commonly used.
template Expected<Tensor<float>> zeros(const Shape& shape);
template Expected<Tensor<int32_t>> zeros(const Shape& shape);
template Expected<Tensor<uint8_t>> zeros(const Shape& shape);

template Expected<Tensor<float>> ones(const Shape& shape);
template Expected<Tensor<int32_t>> ones(const Shape& shape);
template Expected<Tensor<uint8_t>> ones(const Shape& shape);

}  // namespace tensor
}  // namespace litert
