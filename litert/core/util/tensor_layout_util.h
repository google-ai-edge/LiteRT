// Copyright 2026 Google LLC.
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

#ifndef ODML_LITERT_LITERT_CORE_UTIL_TENSOR_LAYOUT_UTIL_H_
#define ODML_LITERT_LITERT_CORE_UTIL_TENSOR_LAYOUT_UTIL_H_

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "absl/base/attributes.h"  // from @com_google_absl
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"

namespace litert::internal {

// Lightweight non-owning core tensor view combining a typed data span with
// concrete dimensions. Backing storage for both `data` and `dims` must outlive
// the view. Core reference kernels validate buffer capacities against checked
// layout element counts without depending on `litert/test`.
template <typename T>
struct TensorView {
  absl::Span<T> data;
  absl::Span<const int32_t> dims;

  constexpr TensorView() = default;
  constexpr TensorView(absl::Span<T> data, absl::Span<const int32_t> dims)
      : data(data), dims(dims) {}

  // Allows implicit conversion from `TensorView<U>` to `TensorView<const U>`
  // (`ConstTensorView<U>`), matching `absl::Span<U>` -> `absl::Span<const U>`.
  template <typename U, typename = std::enable_if_t<!std::is_const_v<U> &&
                                                    std::is_same_v<T, const U>>>
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr TensorView(const TensorView<U>& other)
      : data(other.data), dims(other.dims) {}
};

template <typename T>
using ConstTensorView = TensorView<const T>;

// Validates a `(dims, rank)` pair and returns it as a non-owning span.
// Backing storage for `dims` must outlive the returned span. `dims` may be
// null only when `rank == 0`; negative ranks or null `dims` with `rank > 0`
// return `kLiteRtStatusErrorInvalidArgument`.
Expected<absl::Span<const int32_t>> MakeDimsSpan(
    const int32_t* absl_nullable dims ABSL_ATTRIBUTE_LIFETIME_BOUND, int rank);

// Validated contiguous row-major tensor layout for concrete nonnegative shapes.
// Valid empty tensors (`num_elements == 0`) are represented with zeroed strides
// so consumers distinguish valid empty layouts from errors returned via
// `Expected`.
struct ContiguousLayout {
  std::vector<size_t> strides;
  size_t num_elements = 0;
  size_t num_bytes = 0;

  bool IsEmpty() const { return num_elements == 0; }
};

// Validated broadcast layout for an input tensor aligned to an output shape.
struct BroadcastLayout {
  std::vector<size_t> strides;
  size_t num_elements = 0;
  size_t num_bytes = 0;

  bool IsEmpty() const { return num_elements == 0; }
};

// Validated outer/axis/inner decomposition around an existing tensor axis.
// `outer_size` and `inner_size` hold the exact checked element products of the
// dimensions before and after `axis` respectively, even when `IsEmpty()` is
// true (e.g., `{2, 0, 4}` around `axis = 1` yields `outer_size = 2`,
// `axis_size = 0`, `inner_size = 4`, `num_elements = 0`).
struct AxisLayout {
  size_t outer_size = 0;
  size_t axis_size = 0;
  size_t inner_size = 0;
  size_t num_elements = 0;
  size_t num_bytes = 0;

  bool IsEmpty() const { return num_elements == 0; }
};

// Validated outer/inner decomposition when inserting a new axis (e.g., Pack).
// `outer_size` and `inner_size` hold the exact checked element products before
// and from the insertion position respectively, even when `IsEmpty()` is true.
struct InsertAxisLayout {
  size_t outer_size = 0;
  size_t inner_size = 0;
  size_t num_elements = 0;
  size_t num_bytes = 0;

  bool IsEmpty() const { return num_elements == 0; }
};

// Returns checked byte count for `num_elements` elements of size `elem_size`.
Expected<size_t> CheckedByteCount(size_t num_elements, size_t elem_size);

// Validates concrete nonnegative `dims`, element count, and byte count, and
// computes row-major contiguous strides. Valid empty tensors return
// `num_elements == 0` with zeroed strides.
Expected<ContiguousLayout> PrepareContiguousLayout(
    absl::Span<const int32_t> dims, size_t elem_size = 1);

// Validates broadcast compatibility of `input_dims` into `output_dims`,
// checks element and byte counts for both shapes, and computes aligned
// broadcast strides. Valid empty outputs return `num_elements == 0` with zeroed
// strides after full shape validation.
Expected<BroadcastLayout> PrepareBroadcastLayout(
    absl::Span<const int32_t> input_dims, absl::Span<const int32_t> output_dims,
    size_t elem_size = 1);

// Computes the broadcasted output shape of any number of concrete nonnegative
// shapes under trailing-dimension NumPy broadcasting. Handles zero dimensions
// explicitly: broadcasting `0` with `1` yields `0`, while `0` with `>= 2` is
// rejected even if another dimension is `0`.
Expected<std::vector<int32_t>> ComputeConcreteBroadcastShape(
    absl::Span<const absl::Span<const int32_t>> shapes);

// Convenience overload for two concrete nonnegative shapes.
Expected<std::vector<int32_t>> ComputeConcreteBroadcastShape(
    absl::Span<const int32_t> lhs_dims, absl::Span<const int32_t> rhs_dims);

// Validates that an input or output data span has at least `required_elements`
// capacity and non-null data when `required_elements > 0`.
template <typename T>
inline Expected<void> ValidateBufferCapacity(absl::Span<T> buffer,
                                             size_t required_elements) {
  if (buffer.size() < required_elements) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Tensor buffer capacity is smaller than shape elements");
  }
  if (required_elements > 0 && buffer.data() == nullptr) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Tensor buffer data pointer is null for non-empty shape");
  }
  return {};
}

// Computes checked outer/axis/inner element counts and byte count around `axis`
// in `[-rank, rank - 1]` for a concrete nonnegative tensor shape `dims`.
// `outer_size` and `inner_size` are always exact checked sub-shape products on
// either side of `axis`, even when `IsEmpty()` is true.
Expected<AxisLayout> PrepareAxisLayout(absl::Span<const int32_t> dims,
                                       int32_t axis, size_t elem_size = 1);

// Computes checked outer/inner element counts and byte count when inserting a
// new axis at `axis` in `[-(rank + 1), rank]` (e.g., Pack). `outer_size` and
// `inner_size` are always exact checked sub-shape products before and from the
// insertion position, even when `IsEmpty()` is true.
Expected<InsertAxisLayout> PrepareInsertAxisLayout(
    absl::Span<const int32_t> dims, int32_t axis, size_t elem_size = 1);

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_UTIL_TENSOR_LAYOUT_UTIL_H_
