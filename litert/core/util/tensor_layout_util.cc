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

#include "litert/core/util/tensor_layout_util.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "absl/base/attributes.h"  // from @com_google_absl
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/core/util/tensor_type_util.h"

namespace litert::internal {

Expected<absl::Span<const int32_t>> MakeDimsSpan(
    const int32_t* absl_nullable dims ABSL_ATTRIBUTE_LIFETIME_BOUND, int rank) {
  if (rank < 0 || (rank > 0 && dims == nullptr)) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Invalid tensor rank or null dimensions pointer");
  }
  return absl::MakeConstSpan(dims, static_cast<size_t>(rank));
}

Expected<size_t> CheckedByteCount(size_t num_elements, size_t elem_size) {
  if (elem_size == 0 ||
      elem_size > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Invalid element byte size");
  }
  return GetNumBytesFromElements(num_elements,
                                 Ratio{static_cast<int>(elem_size), 1});
}

Expected<ContiguousLayout> PrepareContiguousLayout(
    absl::Span<const int32_t> dims, size_t elem_size) {
  Expected<size_t> total = GetNumElements(dims);
  if (!total) {
    return total.Error();
  }
  Expected<size_t> bytes = CheckedByteCount(*total, elem_size);
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

Expected<BroadcastLayout> PrepareBroadcastLayout(
    absl::Span<const int32_t> input_dims, absl::Span<const int32_t> output_dims,
    size_t elem_size) {
  if (input_dims.size() > output_dims.size()) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Input rank exceeds output rank in broadcast");
  }
  Expected<size_t> in_elems = GetNumElements(input_dims);
  if (!in_elems) {
    return in_elems.Error();
  }
  Expected<size_t> in_bytes = CheckedByteCount(*in_elems, elem_size);
  if (!in_bytes) {
    return in_bytes.Error();
  }
  Expected<size_t> out_elems = GetNumElements(output_dims);
  if (!out_elems) {
    return out_elems.Error();
  }
  Expected<size_t> out_bytes = CheckedByteCount(*out_elems, elem_size);
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

Expected<std::vector<int32_t>> ComputeConcreteBroadcastShape(
    absl::Span<const absl::Span<const int32_t>> shapes) {
  size_t out_rank = 0;
  for (absl::Span<const int32_t> shape : shapes) {
    out_rank = std::max(out_rank, shape.size());
  }
  std::vector<int32_t> out_dims(out_rank, 1);
  for (absl::Span<const int32_t> shape : shapes) {
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

Expected<std::vector<int32_t>> ComputeConcreteBroadcastShape(
    absl::Span<const int32_t> lhs_dims, absl::Span<const int32_t> rhs_dims) {
  const std::array<absl::Span<const int32_t>, 2> shapes = {lhs_dims, rhs_dims};
  return ComputeConcreteBroadcastShape(shapes);
}

Expected<AxisLayout> PrepareAxisLayout(absl::Span<const int32_t> dims,
                                       int32_t axis, size_t elem_size) {
  if (dims.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
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
  Expected<size_t> total_elems = GetNumElements(dims);
  if (!total_elems) {
    return total_elems.Error();
  }
  Expected<size_t> total_bytes = CheckedByteCount(*total_elems, elem_size);
  if (!total_bytes) {
    return total_bytes.Error();
  }
  Expected<size_t> outer =
      GetNumElements(dims.subspan(0, static_cast<size_t>(axis)));
  if (!outer) {
    return outer.Error();
  }
  Expected<size_t> inner =
      GetNumElements(dims.subspan(static_cast<size_t>(axis) + 1));
  if (!inner) {
    return inner.Error();
  }
  return AxisLayout{*outer, static_cast<size_t>(dims[axis]), *inner,
                    *total_elems, *total_bytes};
}

Expected<InsertAxisLayout> PrepareInsertAxisLayout(
    absl::Span<const int32_t> dims, int32_t axis, size_t elem_size) {
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
  Expected<size_t> total_elems = GetNumElements(dims);
  if (!total_elems) {
    return total_elems.Error();
  }
  Expected<size_t> total_bytes = CheckedByteCount(*total_elems, elem_size);
  if (!total_bytes) {
    return total_bytes.Error();
  }
  Expected<size_t> outer =
      GetNumElements(dims.subspan(0, static_cast<size_t>(axis)));
  if (!outer) {
    return outer.Error();
  }
  Expected<size_t> inner =
      GetNumElements(dims.subspan(static_cast<size_t>(axis)));
  if (!inner) {
    return inner.Error();
  }
  return InsertAxisLayout{*outer, *inner, *total_elems, *total_bytes};
}

}  // namespace litert::internal
