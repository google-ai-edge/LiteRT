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
#include <limits>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/core/util/tensor_type_util.h"

namespace litert::internal {

// Lightweight core tensor view combining a typed data span with concrete
// dimensions. Core reference kernels validate buffer capacities against
// checked layout element counts without depending on `litert/test`.
template <typename T>
struct TensorView {
  absl::Span<T> data;
  absl::Span<const int32_t> dims;
};

template <typename T>
using ConstTensorView = TensorView<const T>;

// Validates a `(dims, rank)` pair and returns it as a span.
inline Expected<absl::Span<const int32_t>> MakeDimsSpan(const int32_t* dims,
                                                        int rank) {
  if (rank < 0 || (rank > 0 && dims == nullptr)) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Invalid tensor rank or null dimensions pointer");
  }
  return rank == 0 ? absl::Span<const int32_t>{}
                   : absl::MakeConstSpan(dims, static_cast<size_t>(rank));
}

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
struct AxisLayout {
  size_t outer_size = 1;
  size_t axis_size = 1;
  size_t inner_size = 1;
  size_t num_elements = 0;
  size_t num_bytes = 0;

  bool IsEmpty() const { return num_elements == 0; }
};

// Validated outer/inner decomposition when inserting a new axis (e.g. Pack).
struct InsertAxisLayout {
  size_t outer_size = 1;
  size_t inner_size = 1;
  size_t num_elements = 0;
  size_t num_bytes = 0;

  bool IsEmpty() const { return num_elements == 0; }
};

// Returns checked byte count for `num_elements` elements of size `elem_size`.
inline Expected<size_t> CheckedByteCount(size_t num_elements,
                                         size_t elem_size) {
  if (elem_size == 0 ||
      elem_size > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Invalid element byte size");
  }
  return GetNumBytesFromElements(num_elements,
                                 Ratio{static_cast<int>(elem_size), 1});
}

// Validates concrete nonnegative `dims`, element count, and byte count, and
// computes row-major contiguous strides. Valid empty tensors return
// `num_elements == 0` with zeroed strides.
inline Expected<ContiguousLayout> PrepareContiguousLayout(
    absl::Span<const int32_t> dims, size_t elem_size = 1) {
  auto total = GetNumElements(dims);
  if (!total) {
    return total.Error();
  }
  auto bytes = CheckedByteCount(*total, elem_size);
  if (!bytes) {
    return bytes.Error();
  }
  ContiguousLayout layout;
  layout.num_elements = *total;
  layout.num_bytes = *bytes;
  layout.strides.assign(dims.size(), 0);
  if (layout.IsEmpty() || dims.empty()) {
    return layout;
  }
  // Because `num_elements > 0`, every dimension is strictly positive and
  // `num_elements` bounds every suffix product computed below.
  size_t current_stride = 1;
  for (size_t i = dims.size(); i > 0; --i) {
    const size_t d = i - 1;
    layout.strides[d] = current_stride;
    current_stride *= static_cast<size_t>(dims[d]);
  }
  return layout;
}

// Validates broadcast compatibility of `input_dims` into `output_dims`,
// checks element and byte counts for both shapes, and computes aligned
// broadcast strides. Valid empty outputs return `num_elements == 0` with zeroed
// strides after full shape validation.
inline Expected<BroadcastLayout> PrepareBroadcastLayout(
    absl::Span<const int32_t> input_dims, absl::Span<const int32_t> output_dims,
    size_t elem_size = 1) {
  if (input_dims.size() > output_dims.size()) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Input rank exceeds output rank in broadcast");
  }
  auto in_elems = GetNumElements(input_dims);
  if (!in_elems) {
    return in_elems.Error();
  }
  auto in_bytes = CheckedByteCount(*in_elems, elem_size);
  if (!in_bytes) {
    return in_bytes.Error();
  }
  auto out_elems = GetNumElements(output_dims);
  if (!out_elems) {
    return out_elems.Error();
  }
  auto out_bytes = CheckedByteCount(*out_elems, elem_size);
  if (!out_bytes) {
    return out_bytes.Error();
  }
  const size_t rank_diff = output_dims.size() - input_dims.size();
  for (size_t d = 0; d < input_dims.size(); ++d) {
    const int32_t in_dim = input_dims[d];
    const int32_t out_dim = output_dims[d + rank_diff];
    if (in_dim != 1 && in_dim != out_dim) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "Incompatible dimensions in broadcast");
    }
  }
  BroadcastLayout layout;
  layout.num_elements = *out_elems;
  layout.num_bytes = *out_bytes;
  layout.strides.assign(output_dims.size(), 0);
  if (layout.IsEmpty() || output_dims.empty()) {
    return layout;
  }
  // Because `num_elements > 0` and every aligned input dimension is either 1 or
  // `out_dim`, every input dimension is strictly positive and `*in_elems`
  // bounds every suffix product computed below.
  size_t current_stride = 1;
  for (size_t i = input_dims.size(); i > 0; --i) {
    const size_t d = i - 1;
    const size_t out_d = d + rank_diff;
    layout.strides[out_d] = (input_dims[d] == 1) ? 0 : current_stride;
    current_stride *= static_cast<size_t>(input_dims[d]);
  }
  return layout;
}

// Computes the broadcasted output shape of any number of concrete nonnegative
// shapes under trailing-dimension NumPy broadcasting. Handles zero dimensions
// explicitly: broadcasting `0` with `1` yields `0`, while `0` with `>= 2` is
// rejected even if another dimension is `0`.
inline Expected<std::vector<int32_t>> ComputeConcreteBroadcastShape(
    absl::Span<const absl::Span<const int32_t>> shapes) {
  size_t out_rank = 0;
  for (const auto& shape : shapes) {
    out_rank = std::max(out_rank, shape.size());
  }
  std::vector<int32_t> out_dims(out_rank, 1);
  for (const auto& shape : shapes) {
    const size_t offset = out_rank - shape.size();
    for (size_t i = 0; i < shape.size(); ++i) {
      const int32_t dim = shape[i];
      if (dim < 0) {
        return Unexpected(kLiteRtStatusErrorInvalidArgument,
                          "Negative dimension in concrete broadcast shape");
      }
      int32_t& out_dim = out_dims[offset + i];
      if (dim == out_dim) {
        continue;
      } else if (out_dim == 1) {
        out_dim = dim;
      } else if (dim == 1) {
        continue;
      } else {
        return Unexpected(
            kLiteRtStatusErrorInvalidArgument,
            "Incompatible dimensions in concrete broadcast shape");
      }
    }
  }
  return out_dims;
}

// Convenience overload for two concrete nonnegative shapes.
inline Expected<std::vector<int32_t>> ComputeConcreteBroadcastShape(
    absl::Span<const int32_t> lhs_dims, absl::Span<const int32_t> rhs_dims) {
  const std::array<absl::Span<const int32_t>, 2> shapes = {lhs_dims, rhs_dims};
  return ComputeConcreteBroadcastShape(shapes);
}

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
// immutable `BroadcastPlan<N>`. Uses pure unsigned `size_t` arithmetic whose
// intermediate sums/differences are bounded by the validated tensor element
// counts.
template <size_t N>
class BroadcastCursor {
 public:
  using value_type = BroadcastSpan<N>;
  using difference_type = std::ptrdiff_t;

  BroadcastCursor() = default;

  explicit BroadcastCursor(const BroadcastPlan<N>& plan)
      : plan_(&plan),
        coords_(plan.outer_extents.size(), 0),
        done_(plan.IsEmpty()) {}

  bool Done() const { return done_; }

  BroadcastSpan<N> Current() const {
    return BroadcastSpan<N>{
        .output_offset = output_offset_,
        .input_offsets = input_offsets_,
        .count = plan_->span_count,
        .input_steps = plan_->span_steps,
    };
  }

  void Advance() {
    if (done_) {
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
    const BroadcastPlan* plan;
    BroadcastCursor<N> begin() const { return BroadcastCursor<N>(*plan); }
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
    return IsEmpty() ? 0 : (output_num_elements / span_count);
  }

  SpanRange Spans() const { return SpanRange{this}; }
};

// Validates concrete nonnegative shapes, checked element/byte counts, and
// broadcast compatibility for `N` inputs and one output, then builds a
// coalesced `BroadcastPlan<N>`.
template <size_t N>
inline Expected<BroadcastPlan<N>> PrepareBroadcastPlan(
    const std::array<absl::Span<const int32_t>, N>& input_shapes,
    const std::array<size_t, N>& input_elem_sizes,
    absl::Span<const int32_t> output_dims, size_t output_elem_size,
    BroadcastOutputMode output_mode = BroadcastOutputMode::kInferredExact) {
  auto out_elems = GetNumElements(output_dims);
  if (!out_elems) {
    return out_elems.Error();
  }
  auto out_bytes = CheckedByteCount(*out_elems, output_elem_size);
  if (!out_bytes) {
    return out_bytes.Error();
  }

  std::array<size_t, N> in_elems{};
  std::array<size_t, N> in_bytes{};
  for (size_t k = 0; k < N; ++k) {
    auto elems = GetNumElements(input_shapes[k]);
    if (!elems) {
      return elems.Error();
    }
    auto bytes = CheckedByteCount(*elems, input_elem_sizes[k]);
    if (!bytes) {
      return bytes.Error();
    }
    in_elems[k] = *elems;
    in_bytes[k] = *bytes;
  }

  // Validate mutual input compatibility across all dimensions (including when
  // another dimension is zero).
  auto inferred_out = ComputeConcreteBroadcastShape(input_shapes);
  if (!inferred_out) {
    return inferred_out.Error();
  }

  if (output_mode == BroadcastOutputMode::kInferredExact) {
    if (output_dims.size() != inferred_out->size() ||
        !std::equal(output_dims.begin(), output_dims.end(),
                    inferred_out->begin())) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "Output shape does not match inferred broadcast shape");
    }
  } else {
    // `kTargetShape`: every input must broadcast into `output_dims`.
    for (size_t k = 0; k < N; ++k) {
      const auto& in_dims = input_shapes[k];
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
    const auto& in_dims = input_shapes[k];
    const size_t rank_diff = out_rank - in_dims.size();
    size_t current_stride = 1;
    for (size_t i = in_dims.size(); i > 0; --i) {
      const size_t d = i - 1;
      raw_strides[k][d + rank_diff] = (in_dims[d] == 1) ? 0 : current_stride;
      current_stride *= static_cast<size_t>(in_dims[d]);
    }
  }

  // Drop singleton output dimensions (their coordinate is always 0) and
  // coalesce adjacent dimensions when every input's addressing satisfies
  // `outer_stride == inner_stride * inner_extent`.
  std::vector<size_t> merged_extents;
  std::array<std::vector<size_t>, N> merged_strides;
  for (size_t d = 0; d < out_rank; ++d) {
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
    return plan;
  }

  plan.span_count = merged_extents.back();
  merged_extents.pop_back();
  for (size_t k = 0; k < N; ++k) {
    plan.span_steps[k] = merged_strides[k].back();
    merged_strides[k].pop_back();
  }
  plan.outer_extents = std::move(merged_extents);
  plan.outer_strides = std::move(merged_strides);
  return plan;
}

// Validates that an input or output data span has at least `required_elements`
// capacity (and non-null data when `required_elements > 0`).
template <typename T>
inline Expected<void> ValidateBufferCapacity(absl::Span<T> buffer,
                                             size_t required_elements) {
  if (buffer.size() < required_elements ||
      (required_elements > 0 && buffer.data() == nullptr)) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Tensor buffer capacity is smaller than shape elements");
  }
  return {};
}

// Computes checked outer/axis/inner element counts and byte count around `axis`
// in [-rank, rank - 1] for a concrete nonnegative tensor shape `dims`.
inline Expected<AxisLayout> PrepareAxisLayout(absl::Span<const int32_t> dims,
                                              int32_t axis,
                                              size_t elem_size = 1) {
  if (dims.empty() ||
      dims.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Invalid rank for axis layout");
  }
  const int32_t rank = static_cast<int32_t>(dims.size());
  if (axis < 0) {
    axis += rank;
  }
  if (axis < 0 || axis >= rank) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Axis out of range for tensor rank");
  }
  auto total_elems = GetNumElements(dims);
  if (!total_elems) {
    return total_elems.Error();
  }
  auto total_bytes = CheckedByteCount(*total_elems, elem_size);
  if (!total_bytes) {
    return total_bytes.Error();
  }
  if (*total_elems == 0) {
    return AxisLayout{0, static_cast<size_t>(dims[axis]), 0, 0, 0};
  }
  size_t outer = 1;
  for (int32_t i = 0; i < axis; ++i) {
    outer *= static_cast<size_t>(dims[i]);
  }
  size_t inner = 1;
  for (int32_t i = axis + 1; i < rank; ++i) {
    inner *= static_cast<size_t>(dims[i]);
  }
  return AxisLayout{outer, static_cast<size_t>(dims[axis]), inner, *total_elems,
                    *total_bytes};
}

// Computes checked outer/inner element counts and byte count when inserting a
// new axis at `axis` in [-(rank + 1), rank] (e.g., Pack).
inline Expected<InsertAxisLayout> PrepareInsertAxisLayout(
    absl::Span<const int32_t> dims, int32_t axis, size_t elem_size = 1) {
  if (dims.size() >= static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
      axis > static_cast<int64_t>(dims.size()) ||
      axis < -static_cast<int64_t>(dims.size()) - 1) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Invalid rank or axis for insert-axis layout");
  }
  const int32_t rank = static_cast<int32_t>(dims.size());
  if (axis < 0) {
    axis += rank + 1;
  }
  auto total_elems = GetNumElements(dims);
  if (!total_elems) {
    return total_elems.Error();
  }
  auto total_bytes = CheckedByteCount(*total_elems, elem_size);
  if (!total_bytes) {
    return total_bytes.Error();
  }
  if (*total_elems == 0) {
    return InsertAxisLayout{0, 0, 0, 0};
  }
  size_t outer = 1;
  for (int32_t i = 0; i < axis; ++i) {
    outer *= static_cast<size_t>(dims[i]);
  }
  size_t inner = 1;
  for (int32_t i = axis; i < rank; ++i) {
    inner *= static_cast<size_t>(dims[i]);
  }
  return InsertAxisLayout{outer, inner, *total_elems, *total_bytes};
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_UTIL_TENSOR_LAYOUT_UTIL_H_
