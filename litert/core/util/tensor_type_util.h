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

#ifndef ODML_LITERT_LITERT_CORE_UTIL_TENSOR_TYPE_UTIL_H_
#define ODML_LITERT_LITERT_CORE_UTIL_TENSOR_TYPE_UTIL_H_

#include <cstddef>
#include <limits>
#include <string>
#include <type_traits>

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_model.h"
#include "litert/cc/litert_expected.h"

namespace litert::internal {

struct Ratio {
  using Type = int;
  Type num;
  Type denom;
  std::string ToString() const { return absl::StrCat(num, "/", denom); }
};

Expected<Ratio> GetElementSize(LiteRtElementType element_type);

template <typename T>
bool IsNegative(T value) {
  if constexpr (std::numeric_limits<std::remove_cv_t<T>>::is_signed) {
    return value < 0;
  }
  return false;
}

inline Expected<size_t> GetNumBytesFromElements(size_t num_elements,
                                                Ratio element_size) {
  if (element_size.num <= 0 || element_size.denom <= 0) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Unexpected element size");
  }
  const auto numerator = static_cast<size_t>(element_size.num);
  const auto denominator = static_cast<size_t>(element_size.denom);
  if (num_elements >
      (std::numeric_limits<size_t>::max() - (denominator - 1)) / numerator) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Tensor byte size overflows size_t");
  }
  return ((num_elements * numerator) + (denominator - 1)) / denominator;
}

// Get the number of elements in a tensor with given dimensions.
template <typename T>
Expected<size_t> GetNumElements(absl::Span<T> dimensions) {
  size_t num_elements = 1;
  for (auto i = 0; i < dimensions.size(); ++i) {
    auto dim = dimensions[i];
    if (IsNegative(dim)) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "Unexpected negative dimension");
    } else if (dim == 0) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "Unexpected 0 dimension");
    }
    const size_t dimension = static_cast<size_t>(dim);
    if (dimension > std::numeric_limits<size_t>::max() / num_elements) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "Tensor element count overflows size_t");
    }
    num_elements *= dimension;
  }
  return num_elements;
}

inline Expected<size_t> GetNumElements(
    const LiteRtRankedTensorType& tensor_type) {
  return GetNumElements(
      absl::MakeSpan(tensor_type.layout.dimensions, tensor_type.layout.rank));
}

// Get the minimum number of bytes necessary to represent a packed tensor with a
// given element type and dimensions.
template <typename T>
Expected<size_t> GetNumPackedBytes(LiteRtElementType element_type,
                                   absl::Span<T> dimensions) {
  auto element_size = GetElementSize(element_type);
  if (!element_size) {
    return element_size.Error();
  }
  auto num_elements = GetNumElements(dimensions);
  if (!num_elements) {
    return num_elements.Error();
  }
  return GetNumBytesFromElements(*num_elements, *element_size);
}

// Get the number of bytes necessary to represent a packed tensor type, ignoring
// any stride information.
inline Expected<size_t> GetNumPackedBytes(
    const LiteRtRankedTensorType& tensor_type) {
  return GetNumPackedBytes(
      tensor_type.element_type,
      absl::MakeSpan(tensor_type.layout.dimensions, tensor_type.layout.rank));
}

// Get the minimum number of bytes necessary to represent a possibly unpacked
// tensor with a given element type, dimensions, and strides.
template <typename T, typename U>
Expected<size_t> GetNumBytes(LiteRtElementType element_type,
                             absl::Span<T> dimensions, absl::Span<U> strides) {
  if (dimensions.size() != strides.size()) {
    return Unexpected(
        kLiteRtStatusErrorInvalidArgument,
        "Dimensions and strides have different number of elements");
  }
  auto element_size = GetElementSize(element_type);
  if (!element_size) {
    return element_size.Error();
  }
  auto rank = dimensions.size();
  size_t num_elements = 1;
  for (auto i = 0; i < rank; ++i) {
    const auto dimension = dimensions[i];
    const auto stride = strides[i];
    if (IsNegative(dimension)) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "Unexpected negative dimension");
    } else if (dimension == 0) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "Unexpected 0 dimension");
    }
    if (IsNegative(stride)) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "Unexpected negative stride");
    }
    const size_t dim_less_one = static_cast<size_t>(dimension) - 1;
    const size_t stride_size = static_cast<size_t>(stride);
    if (dim_less_one != 0 &&
        stride_size > std::numeric_limits<size_t>::max() / dim_less_one) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "Tensor stride span overflows size_t");
    }
    const size_t span = dim_less_one * stride_size;
    if (span > std::numeric_limits<size_t>::max() - num_elements) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "Tensor element count overflows size_t");
    }
    num_elements += span;
  }
  return GetNumBytesFromElements(num_elements, *element_size);
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_UTIL_TENSOR_TYPE_UTIL_H_
