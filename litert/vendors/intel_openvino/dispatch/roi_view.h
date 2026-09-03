// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
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

#ifndef ODML_LITERT_LITERT_VENDORS_INTEL_OPENVINO_DISPATCH_ROI_VIEW_H_
#define ODML_LITERT_LITERT_VENDORS_INTEL_OPENVINO_DISPATCH_ROI_VIEW_H_

#include <cstddef>
#include <exception>

#include "openvino/core/coordinate.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/strides.hpp"
#include "openvino/runtime/tensor.hpp"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"

namespace litert::openvino {

// Binds a graph port to a possibly-larger backing OpenVINO tensor by returning a
// zero-copy view sized to the port.
//
// This is the split-context KV-cache primitive: one canonical max-length KV
// buffer (e.g. decode seq 16383) is shared by signatures that declare a shorter
// port (e.g. prefill seq 16256). When the buffer is larger than the port, a
// region-of-interest sub-view is created that starts at offset 0 on every
// dimension and extends to the port's extent. The ROI inherits the parent's
// strides, which is exactly what the NPU/GPU plugin needs for strided access,
// and it is layout-agnostic: it handles both the K layout ([1,H,S,D], seq at
// dim 2) and the transposed-V layout ([1,H,D,S], seq at dim 3) uniformly, since
// only the differing dimension is trimmed.
//
// Returns the buffer unchanged when the shapes are equal. Returns an error (not
// a crash) on rank mismatch, an under-sized buffer, or a computed extent that
// would fall outside the parent allocation. Never throws: any OpenVINO exception
// is converted to a litert::Error.
inline litert::Expected<ov::Tensor> MakePortView(const ov::Tensor& buffer_tensor,
                                                 const ov::Shape& port_shape) {
  try {
    const ov::Shape& buffer_shape = buffer_tensor.get_shape();
    if (buffer_shape == port_shape) {
      return buffer_tensor;
    }
    if (buffer_shape.size() != port_shape.size()) {
      return litert::Error(
          kLiteRtStatusErrorRuntimeFailure,
          "KV buffer rank does not match graph port rank; cannot create view");
    }
    // The buffer must be at least as large as the port on every dimension. The
    // split-context design only ever trims a single (cache-length) dimension, so
    // more than one differing dim signals an unexpected shape pairing; warn but
    // still build the view (the bounds guard below remains authoritative).
    size_t differing_dims = 0;
    for (size_t i = 0; i < buffer_shape.size(); ++i) {
      if (buffer_shape[i] < port_shape[i]) {
        return litert::Error(
            kLiteRtStatusErrorRuntimeFailure,
            "KV buffer smaller than graph port; cannot create view");
      }
      if (buffer_shape[i] != port_shape[i]) {
        ++differing_dims;
      }
    }
    if (differing_dims > 1) {
      LITERT_LOG(LITERT_WARNING,
                 "Strided KV sub-view trims %zu dimensions; the split-context "
                 "design expects exactly one",
                 differing_dims);
    }
    ov::Coordinate begin(buffer_shape.size(), 0);
    ov::Coordinate end(port_shape.begin(), port_shape.end());
    // ov::Tensor(other, begin, end) is the ROI constructor: a zero-copy sub-view
    // sharing the parent's memory and strides.
    ov::Tensor view(buffer_tensor, begin, end);

    // Bounds guard for the strided view. The ROI inherits the PARENT's byte
    // strides, so the last element the driver may touch sits at
    //   sum_i (port_shape[i] - 1) * parent_byte_stride[i]
    // (the +1 element itself spans elem_size more bytes). If the parent buffer
    // was under-allocated relative to its declared shape, that tail lands
    // outside the allocation and the NPU driver walks off the end during graph
    // execution (observed as a SIGSEGV inside libze_intel_npu after all binds
    // succeed, never at bind time). Compute the extent explicitly and fail
    // loudly here rather than letting the driver fault on stripped frames. Since
    // every ROI starts at offset 0 and only trims dims, a full-size parent
    // always covers the view; a triggered check means the parent allocation is
    // short.
    const ov::Strides& parent_strides = buffer_tensor.get_strides();  // bytes
    const size_t elem_size = buffer_tensor.get_element_type().size();
    size_t max_byte_offset = 0;
    for (size_t i = 0; i < port_shape.size(); ++i) {
      if (port_shape[i] == 0) continue;
      max_byte_offset += (port_shape[i] - 1) * parent_strides[i];
    }
    const size_t roi_end_byte = max_byte_offset + elem_size;
    const size_t parent_capacity = buffer_tensor.get_byte_size();
    LITERT_LOG(LITERT_VERBOSE,
               "Strided KV sub-view bind: parent data=%p parent_capacity=%zu "
               "roi_end_byte=%zu elem_size=%zu",
               buffer_tensor.data(), parent_capacity, roi_end_byte, elem_size);
    if (roi_end_byte > parent_capacity) {
      return litert::Error(
          kLiteRtStatusErrorRuntimeFailure,
          "Strided KV sub-view extends past the parent allocation; the parent "
          "buffer was allocated smaller than its declared (max-length) shape");
    }
    return view;
  } catch (const std::exception& e) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure, e.what());
  }
}

}  // namespace litert::openvino

#endif  // ODML_LITERT_LITERT_VENDORS_INTEL_OPENVINO_DISPATCH_ROI_VIEW_H_
