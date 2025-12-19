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

#ifndef ODML_LITERT_LITERT_CC_LITERT_RANKED_TENSOR_TYPE_H_
#define ODML_LITERT_LITERT_CC_LITERT_RANKED_TENSOR_TYPE_H_

#include <cstddef>
#include <initializer_list>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model_types.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"

/// @file
/// @brief Defines the C++ wrapper for a ranked tensor type.

namespace litert {

/// @brief A C++ wrapper for `LiteRtRankedTensorType`, representing a type for
/// tensors with known dimensions.
class RankedTensorType {
 public:
  RankedTensorType(ElementType element_type, Layout&& layout)
      : element_type_(element_type), layout_(std::move(layout)) {}
  explicit RankedTensorType(const LiteRtRankedTensorType& type)
      : element_type_(static_cast<enum ElementType>(type.element_type)),
        layout_(type.layout) {}

  explicit operator LiteRtRankedTensorType() const {
    return LiteRtRankedTensorType{
        /*.element_type=*/static_cast<LiteRtElementType>(element_type_),
        /*layout=*/static_cast<LiteRtLayout>(layout_),
    };
  }

  bool operator==(const RankedTensorType& other) const {
    return ElementType() == other.ElementType() && Layout() == other.Layout();
  }

  bool operator!=(const RankedTensorType& other) const {
    return !(*this == other);
  }

  ElementType ElementType() const { return element_type_; }

  const Layout& Layout() const { return layout_; }

  Expected<size_t> Bytes() const {
    LITERT_ASSIGN_OR_RETURN(const size_t num_elements, layout_.NumElements());
    auto byte_width = GetByteWidth(element_type_);
    if (!byte_width) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument);
    }
    return *byte_width * num_elements;
  }

 private:
  enum ElementType element_type_;
  class Layout layout_;
};

/// @brief Constructs a `RankedTensorType` from a C++ type and shape.
template <typename T>
RankedTensorType MakeRankedTensorType(
    std::initializer_list<Layout::Dim> shape) {
  return RankedTensorType(GetElementType<T>(), Layout(std::move(shape)));
}
template <typename T, typename Shape>
RankedTensorType MakeRankedTensorType(const Shape& shape) {
  return RankedTensorType(
      GetElementType<T>(),
      Layout(Dimensions(std::cbegin(shape), std::cend(shape))));
}

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_LITERT_RANKED_TENSOR_TYPE_H_
