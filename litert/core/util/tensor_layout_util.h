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

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"  // from @com_google_absl
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/core/util/tensor_type_util.h"

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

// Controls how `PrepareBroadcastPlan` validates `output_dims` against the
// inputs.
enum class BroadcastOutputMode {
  // `output_dims` must equal the exact NumPy broadcast shape of all inputs
  // (e.g., binary ops, SelectV2, PReLU).
  kInferredExact,
  // `output_dims` is a caller-specified destination shape that every input must
  // broadcast into (e.g., BroadcastTo).
  kTargetShape,
};

// One contiguous output span emitted by `BroadcastCursor<N>`.
// Within a span of `count` elements, each input `k` either advances by 1
// element (`input_steps[k] == 1`) or stays constant (`input_steps[k] == 0`).
template <size_t N>
struct BroadcastSpan {
  size_t output_offset = 0;
  std::array<size_t, N> input_offsets{};
  size_t count = 0;
  std::array<size_t, N> input_steps{};
};

template <size_t N>
struct BroadcastPlan;

// Mutable odometer cursor iterating over `BroadcastSpan<N>` entries of an
// immutable `BroadcastPlan<N>`. The referenced `BroadcastPlan<N>` must outlive
// the cursor. Uses pure unsigned `size_t` arithmetic whose intermediate
// sums/differences are bounded by the validated tensor element counts.
template <size_t N>
class BroadcastCursor {
 public:
  using iterator_category = std::input_iterator_tag;
  using value_type = BroadcastSpan<N>;
  using difference_type = std::ptrdiff_t;
  using pointer = void;
  using reference = BroadcastSpan<N>;

  BroadcastCursor() = default;

  explicit BroadcastCursor(
      const BroadcastPlan<N>& plan ABSL_ATTRIBUTE_LIFETIME_BOUND)
      : plan_(&plan),
        coords_(plan.outer_extents.size(), 0),
        done_(plan.IsEmpty() || plan.span_count == 0) {}

  BroadcastCursor(const BroadcastPlan<N>&&) = delete;

  bool Done() const { return done_; }

  // Returns the current span, or an empty `BroadcastSpan<N>{}` if the cursor
  // is default-constructed or exhausted (`Done()` is true).
  BroadcastSpan<N> Current() const {
    if (done_ || plan_ == nullptr) {
      return BroadcastSpan<N>{};
    }
    return BroadcastSpan<N>{output_offset_, input_offsets_, plan_->span_count,
                            plan_->span_steps};
  }

  void Advance() {
    if (done_ || plan_ == nullptr) {
      return;
    }
    output_offset_ += plan_->span_count;
    for (size_t i = plan_->outer_extents.size(); i > 0; --i) {
      const size_t d = i - 1;
      ++coords_[d];
      if (coords_[d] < plan_->outer_extents[d]) {
        for (size_t k = 0; k < N; ++k) {
          input_offsets_[k] += plan_->outer_strides[k][d];
        }
        return;
      }
      // Every prior increment of `coords_[d]` added `outer_strides[k][d]` to
      // `input_offsets_[k]` without exceeding `input_num_elements[k]`, so both
      // the product and subtraction below are exact and overflow-free.
      const size_t reset_steps = plan_->outer_extents[d] - 1;
      coords_[d] = 0;
      for (size_t k = 0; k < N; ++k) {
        input_offsets_[k] -= reset_steps * plan_->outer_strides[k][d];
      }
    }
    done_ = true;
  }

  BroadcastSpan<N> operator*() const { return Current(); }

  BroadcastCursor& operator++() {
    Advance();
    return *this;
  }

  BroadcastCursor operator++(int) {
    BroadcastCursor copy = *this;
    Advance();
    return copy;
  }

  bool operator==(const BroadcastCursor& other) const {
    if (done_ && other.done_) {
      return true;
    }
    return done_ == other.done_ && plan_ == other.plan_ &&
           output_offset_ == other.output_offset_;
  }

  bool operator!=(const BroadcastCursor& other) const {
    return !(*this == other);
  }

 private:
  const BroadcastPlan<N>* plan_ = nullptr;
  size_t output_offset_ = 0;
  std::array<size_t, N> input_offsets_{};
  std::vector<size_t> coords_;
  bool done_ = true;
};

// Dtype-independent validated broadcast plan for `N` inputs (typically 1..3)
// into one contiguous output tensor. Adjacent dimensions with compatible
// addressing across all inputs are coalesced during preparation; empty outputs
// produce zero spans.
template <size_t N>
struct BroadcastPlan {
  static_assert(N >= 1, "BroadcastPlan requires at least one input.");

  struct SpanRange {
    const BroadcastPlan* plan = nullptr;
    BroadcastCursor<N> begin() const {
      return plan == nullptr ? BroadcastCursor<N>() : BroadcastCursor<N>(*plan);
    }
    BroadcastCursor<N> end() const { return BroadcastCursor<N>(); }
  };

  std::vector<int32_t> output_shape;
  size_t output_num_elements = 0;
  size_t output_num_bytes = 0;
  std::array<size_t, N> input_num_elements{};
  std::array<size_t, N> input_num_bytes{};

  size_t span_count = 0;
  std::array<size_t, N> span_steps{};
  std::vector<size_t> outer_extents;
  std::array<std::vector<size_t>, N> outer_strides;

  bool IsEmpty() const { return output_num_elements == 0; }

  size_t NumSpans() const {
    return (IsEmpty() || span_count == 0) ? 0
                                          : (output_num_elements / span_count);
  }

  // Returns a non-owning range over the plan's spans. `*this` must outlive the
  // returned `SpanRange` and any `BroadcastCursor` obtained from it.
  SpanRange Spans() const& ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return SpanRange{this};
  }
  SpanRange Spans() const&& = delete;
};

namespace detail {

// Drops singleton output dimensions and coalesces adjacent dimensions when
// every input's addressing satisfies `outer_stride == inner_stride * extent`.
template <size_t N>
inline void CoalesceBroadcastDimensions(
    absl::Span<const int32_t> output_dims,
    const std::array<std::vector<size_t>, N>& raw_strides,
    BroadcastPlan<N>& plan) {
  std::vector<size_t> merged_extents;
  std::array<std::vector<size_t>, N> merged_strides;
  for (size_t d = 0; d < output_dims.size(); ++d) {
    const size_t extent = static_cast<size_t>(output_dims[d]);
    if (extent == 1) {
      continue;
    }
    if (!merged_extents.empty()) {
      bool can_merge = true;
      for (size_t k = 0; k < N; ++k) {
        const size_t s_outer = merged_strides[k].back();
        const size_t s_inner = raw_strides[k][d];
        if (s_inner == 0) {
          if (s_outer != 0) {
            can_merge = false;
            break;
          }
        } else if (s_outer % extent != 0 || s_outer / extent != s_inner) {
          can_merge = false;
          break;
        }
      }
      if (can_merge) {
        merged_extents.back() *= extent;
        for (size_t k = 0; k < N; ++k) {
          merged_strides[k].back() = raw_strides[k][d];
        }
        continue;
      }
    }
    merged_extents.push_back(extent);
    for (size_t k = 0; k < N; ++k) {
      merged_strides[k].push_back(raw_strides[k][d]);
    }
  }

  if (merged_extents.empty()) {
    // Rank-0 scalar or all-ones shape: a single span of length 1.
    plan.span_count = 1;
    plan.span_steps.fill(0);
    return;
  }

  plan.span_count = merged_extents.back();
  merged_extents.pop_back();
  for (size_t k = 0; k < N; ++k) {
    plan.span_steps[k] = merged_strides[k].back();
    merged_strides[k].pop_back();
  }
  plan.outer_extents = std::move(merged_extents);
  plan.outer_strides = std::move(merged_strides);
}

}  // namespace detail

// Validates concrete nonnegative shapes, checked element/byte counts, and
// broadcast compatibility for `N` inputs and one output, then builds a
// coalesced `BroadcastPlan<N>`.
template <size_t N>
inline Expected<BroadcastPlan<N>> PrepareBroadcastPlan(
    const std::array<absl::Span<const int32_t>, N>& input_shapes,
    const std::array<size_t, N>& input_elem_sizes,
    absl::Span<const int32_t> output_dims, size_t output_elem_size,
    BroadcastOutputMode output_mode = BroadcastOutputMode::kInferredExact) {
  Expected<size_t> out_elems = GetNumElements(output_dims);
  if (!out_elems) {
    return out_elems.Error();
  }
  Expected<size_t> out_bytes = CheckedByteCount(*out_elems, output_elem_size);
  if (!out_bytes) {
    return out_bytes.Error();
  }

  std::array<size_t, N> in_elems{};
  std::array<size_t, N> in_bytes{};
  for (size_t k = 0; k < N; ++k) {
    Expected<size_t> elems = GetNumElements(input_shapes[k]);
    if (!elems) {
      return elems.Error();
    }
    Expected<size_t> bytes = CheckedByteCount(*elems, input_elem_sizes[k]);
    if (!bytes) {
      return bytes.Error();
    }
    in_elems[k] = *elems;
    in_bytes[k] = *bytes;
  }

  switch (output_mode) {
    case BroadcastOutputMode::kInferredExact: {
      // Validate mutual input compatibility across all dimensions (including
      // when another dimension is zero).
      Expected<std::vector<int32_t>> inferred_out =
          ComputeConcreteBroadcastShape(input_shapes);
      if (!inferred_out) {
        return inferred_out.Error();
      }
      if (output_dims.size() != inferred_out->size() ||
          !std::equal(output_dims.begin(), output_dims.end(),
                      inferred_out->begin())) {
        return Unexpected(
            kLiteRtStatusErrorInvalidArgument,
            "Output shape does not match inferred broadcast shape");
      }
      break;
    }
    case BroadcastOutputMode::kTargetShape: {
      // Every input must broadcast into `output_dims` (which also guarantees
      // mutual compatibility since every non-1 dim equals the target dim).
      for (size_t k = 0; k < N; ++k) {
        const absl::Span<const int32_t> in_dims = input_shapes[k];
        if (in_dims.size() > output_dims.size()) {
          return Unexpected(kLiteRtStatusErrorInvalidArgument,
                            "Input rank exceeds target broadcast output rank");
        }
        const size_t rank_diff = output_dims.size() - in_dims.size();
        for (size_t d = 0; d < in_dims.size(); ++d) {
          if (in_dims[d] != 1 && in_dims[d] != output_dims[d + rank_diff]) {
            return Unexpected(kLiteRtStatusErrorInvalidArgument,
                              "Input shape cannot broadcast to target shape");
          }
        }
      }
      break;
    }
  }

  BroadcastPlan<N> plan;
  plan.output_shape.assign(output_dims.begin(), output_dims.end());
  plan.output_num_elements = *out_elems;
  plan.output_num_bytes = *out_bytes;
  plan.input_num_elements = in_elems;
  plan.input_num_bytes = in_bytes;

  if (plan.IsEmpty()) {
    return plan;
  }

  // Compute raw aligned strides for every input. Because `output_num_elements`
  // is positive and every input broadcasts into `output_dims`, every input
  // dimension is positive and `in_elems[k]` bounds every suffix product.
  const size_t out_rank = output_dims.size();
  std::array<std::vector<size_t>, N> raw_strides;
  for (size_t k = 0; k < N; ++k) {
    raw_strides[k].assign(out_rank, 0);
    const absl::Span<const int32_t> in_dims = input_shapes[k];
    const size_t rank_diff = out_rank - in_dims.size();
    size_t current_stride = 1;
    for (size_t i = in_dims.size(); i > 0; --i) {
      const size_t d = i - 1;
      raw_strides[k][d + rank_diff] = (in_dims[d] == 1) ? 0 : current_stride;
      current_stride *= static_cast<size_t>(in_dims[d]);
    }
  }

  detail::CoalesceBroadcastDimensions<N>(output_dims, raw_strides, plan);
  return plan;
}

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
