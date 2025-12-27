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

#ifndef ODML_LITERT_LITERT_CC_LITERT_LAYOUT_H_
#define ODML_LITERT_LITERT_CC_LITERT_LAYOUT_H_

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <optional>
#include <utility>

#include "absl/container/inlined_vector.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_layout.h"
#include "litert/cc/internal/litert_consts.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

/// @file
/// @brief Defines the C++ wrapper for a tensor layout and related utilities.

namespace litert {

using Dimensions = absl::InlinedVector<int32_t, kExpectedMaxTensorRank>;
using Strides = absl::InlinedVector<uint32_t, kExpectedMaxTensorRank>;

/// @brief Small standalone helper functions for working with the C layout API.

/// @brief Builds a layout from a given iterator of dimensions.
template <class Begin, class End>
inline constexpr LiteRtLayout BuildLayout(Begin begin, End end,
                                          const uint32_t* strides = nullptr) {
  auto rank = static_cast<uint32_t>(end - begin);
  bool has_strides = (strides != nullptr);
  LiteRtLayout layout{
      /*.rank=*/rank,
      /*.has_strides=*/has_strides,
      /*.dimensions=*/{},
      /*.strides=*/{},
  };
  size_t i = 0;
  for (auto it = begin; it != end; ++it, ++i) {
    layout.dimensions[i] = *it;
  }
  if (has_strides) {
    for (size_t i = 0; i < rank; ++i) {
      layout.strides[i] = strides[i];
    }
  }
  return layout;
}

/// @brief Builds a layout from a given iterable of dimensions.
template <class Dims>
inline constexpr LiteRtLayout BuildLayout(const Dims& dims,
                                          const uint32_t* strides = nullptr) {
  return BuildLayout(std::cbegin(dims), std::cend(dims), strides);
}

/// @brief Builds a layout from literal dimensions.
inline constexpr LiteRtLayout BuildLayout(std::initializer_list<int32_t> dims,
                                          const uint32_t* strides = nullptr) {
  return BuildLayout(dims.begin(), dims.end(), strides);
}

/// @brief Gets the dimensions as a span.
inline constexpr absl::Span<const int32_t> DimsSpan(
    const LiteRtLayout& layout) {
  return absl::MakeConstSpan(layout.dimensions, layout.rank);
}

/// @brief Gets the strides as a span if they exist.
inline constexpr std::optional<absl::Span<const uint32_t>> StridesSpan(
    const LiteRtLayout& layout) {
  if (layout.has_strides) {
    return absl::MakeConstSpan(layout.strides, layout.rank);
  }
  return {};
}

/// @brief A C++ wrapper for `LiteRtLayout`, representing a tensor layout.
class Layout {
 public:
  using Dim = int32_t;

  constexpr Layout()
      : lrt_layout_{/*.rank=*/0, /*.has_strides=*/false, /*.dimensions=*/{},
                    /*.strides=*/{}} {}

  constexpr explicit Layout(const litert::Dimensions& dimensions)
      : lrt_layout_(BuildLayout(dimensions)) {}

  Layout(const litert::Dimensions& dimensions, const litert::Strides& strides)
      : lrt_layout_(BuildLayout(dimensions,
                                strides.empty() ? nullptr : strides.data())) {}

  constexpr explicit Layout(const LiteRtLayout& layout) : lrt_layout_(layout) {}

  explicit operator const LiteRtLayout&() const { return lrt_layout_; }

  bool operator==(const Layout& other) const {
    bool is_same;
    (void)LiteRtIsSameLayout(&lrt_layout_, &other.lrt_layout_, &is_same);
    return is_same;
  }

  uint32_t Rank() const { return lrt_layout_.rank; }

  absl::Span<const Dim> Dimensions() const {
    return absl::MakeSpan(lrt_layout_.dimensions, Rank());
  }

  bool HasStrides() const { return lrt_layout_.has_strides; }

  absl::Span<const uint32_t> Strides() const {
    if (HasStrides()) {
      return {lrt_layout_.strides, Rank()};
    } else {
      return {};
    }
  }

  /// @brief Returns the number of scalar elements in the tensor layout.
  ///
  /// Returns an error if the layout includes dynamic dimensions.
  Expected<size_t> NumElements() const {
    size_t num_elements;
    LITERT_RETURN_IF_ERROR(
        LiteRtGetNumLayoutElements(&lrt_layout_, &num_elements));
    return num_elements;
  }

 private:
  LiteRtLayout lrt_layout_;
};

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_LITERT_LAYOUT_H_
