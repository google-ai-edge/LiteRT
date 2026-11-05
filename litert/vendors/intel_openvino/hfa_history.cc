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

#include "litert/vendors/intel_openvino/hfa_history.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <string>

#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"

namespace litert {
namespace openvino {
namespace {

// History seen by one mask row of |len| values, the first |past| of which are
// the KV cache.
template <typename T>
int64_t RowHistory(const T* row, size_t past, size_t len) {
  float lo = static_cast<float>(row[0]);
  float hi = lo;
  for (size_t i = 1; i < len; ++i) {
    const float v = static_cast<float>(row[i]);
    lo = std::min(lo, v);
    hi = std::max(hi, v);
  }
  if (lo == hi) {
    return lo < 0.0f ? 0 : static_cast<int64_t>(past);
  }
  for (size_t i = past; i > 0; --i) {
    if (static_cast<float>(row[i - 1]) > lo) return static_cast<int64_t>(i);
  }
  return 0;
}

template <typename T>
int64_t MaskHistory(const T* data, size_t rows, size_t past, size_t len) {
  int64_t history = 0;
  const T* previous = nullptr;
  for (size_t r = 0; r < rows && history < static_cast<int64_t>(past); ++r) {
    const T* row = data + r * len;
    // Rows of a causal chunk differ only in the chunk's own columns. Two rows
    // with the same cache columns see the same history.
    if (previous != nullptr &&
        std::memcmp(row, previous, past * sizeof(T)) == 0) {
      continue;
    }
    history = std::max(history, RowHistory(row, past, len));
    previous = row;
  }
  return history;
}

}  // namespace

Expected<void> CheckHistoryMask(const ov::element::Type& type,
                                const ov::Shape& shape) {
  if (type != ov::element::f32 && type != ov::element::f16 &&
      type != ov::element::bf16) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "HFA history mask must be f32, f16 or bf16, got " +
                          type.get_type_name());
  }
  if (shape.size() != 4 || shape[0] != 1 || shape[1] != 1 || shape[2] == 0 ||
      shape[3] <= shape[2]) {
    return Unexpected(
        kLiteRtStatusErrorInvalidArgument,
        "HFA history mask must be [1, 1, Q, past + Q] with Q > 0 and "
        "past > 0, got " +
            shape.to_string());
  }
  return {};
}

Expected<int64_t> HistorySizeFromMask(const ov::Tensor& mask) {
  try {
    const auto type = mask.get_element_type();
    const auto shape = mask.get_shape();
    if (auto valid = CheckHistoryMask(type, shape); !valid.HasValue()) {
      return Unexpected(valid.Error());
    }
    if (!mask.is_continuous()) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "HFA history mask must be contiguous");
    }
    const size_t rows = shape[2];
    const size_t len = shape[3];
    const size_t past = len - rows;
    if (type == ov::element::f32) {
      return MaskHistory(mask.data<const float>(), rows, past, len);
    }
    if (type == ov::element::f16) {
      return MaskHistory(mask.data<const ov::float16>(), rows, past, len);
    }
    return MaskHistory(mask.data<const ov::bfloat16>(), rows, past, len);
  } catch (const std::exception& e) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      std::string("HFA history mask is not readable: ") +
                          e.what());
  }
}

}  // namespace openvino
}  // namespace litert
