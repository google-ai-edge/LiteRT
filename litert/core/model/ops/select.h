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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_OPS_SELECT_H_
#define ODML_LITERT_LITERT_CORE_MODEL_OPS_SELECT_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/core/util/tensor_layout_util.h"

namespace litert::internal {

inline LiteRtStatus InferSelect(const LiteRtOpT& op,
                                absl::Span<const Dims> input_shapes,
                                std::vector<Dims>& output_shapes) {
  constexpr int kSelectNumArgs = 3;
  constexpr int kConditionArgIndex = 0;
  constexpr int kTrueArgIndex = 1;
  constexpr int kFalseArgIndex = 2;

  if (input_shapes.size() != kSelectNumArgs || output_shapes.size() != 1) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  const auto& s1 = input_shapes[kConditionArgIndex];
  const auto& s2 = input_shapes[kTrueArgIndex];
  const auto& s3 = input_shapes[kFalseArgIndex];

  size_t rank = std::max({s1.size(), s2.size(), s3.size()});
  Dims out_shape(rank);

  for (int i = 0; i < rank; ++i) {
    int idx1 = s1.size() - 1 - i;
    int idx2 = s2.size() - 1 - i;
    int idx3 = s3.size() - 1 - i;

    int32_t d1 = (idx1 >= 0) ? s1[idx1] : 1;
    int32_t d2 = (idx2 >= 0) ? s2[idx2] : 1;
    int32_t d3 = (idx3 >= 0) ? s3[idx3] : 1;

    if (d1 == -1 || d2 == -1 || d3 == -1) {
      out_shape[rank - 1 - i] = -1;
      continue;
    }

    int32_t out_dim = 1;
    for (int32_t dim : {d1, d2, d3}) {
      if (dim != 1 && out_dim != 1 && dim != out_dim) {
        return kLiteRtStatusErrorInvalidArgument;
      }
      if (dim > out_dim) {
        out_dim = dim;
      }
    }
    out_shape[rank - 1 - i] = out_dim;
  }

  output_shapes[0] = std::move(out_shape);
  return kLiteRtStatusOk;
}

namespace detail {

template <typename T>
inline void ExecuteSelectWithPlan(const BroadcastPlan<3>& plan,
                                  const bool* cond_data, const T* a_data,
                                  const T* b_data, T* output_data) {
  for (const auto& span : plan.Spans()) {
    const size_t c_base = span.input_offsets[0];
    const size_t a_base = span.input_offsets[1];
    const size_t b_base = span.input_offsets[2];
    const size_t c_step = span.input_steps[0];
    const size_t a_step = span.input_steps[1];
    const size_t b_step = span.input_steps[2];
    T* out_ptr = output_data + span.output_offset;
    for (size_t i = 0; i < span.count; ++i) {
      out_ptr[i] = cond_data[c_base + i * c_step] ? a_data[a_base + i * a_step]
                                                  : b_data[b_base + i * b_step];
    }
  }
}

}  // namespace detail

// Evaluates SelectV2 (`cond ? a : b`) in one pass over three concrete tensors
// under trailing-dimension NumPy broadcasting (`output.dims ==
// broadcast(cond.dims, a.dims, b.dims)`).
//
// Validates concrete shapes, checked element/byte counts, and span capacities
// before accessing elements; valid empty outputs return `kLiteRtStatusOk`
// without dereferencing any data pointer. `output.data` must not alias `a.data`
// or `b.data` when broadcasting.
template <typename T>
inline LiteRtStatus ReferenceSelect(ConstTensorView<bool> cond,
                                    ConstTensorView<T> a, ConstTensorView<T> b,
                                    TensorView<T> output) {
  auto plan = PrepareBroadcastPlan<3>(
      {cond.dims, a.dims, b.dims}, {sizeof(bool), sizeof(T), sizeof(T)},
      output.dims, sizeof(T), BroadcastOutputMode::kInferredExact);
  if (!plan) {
    return plan.Error().Status();
  }
  if (!ValidateBufferCapacity(cond.data, plan->input_num_elements[0]) ||
      !ValidateBufferCapacity(a.data, plan->input_num_elements[1]) ||
      !ValidateBufferCapacity(b.data, plan->input_num_elements[2]) ||
      !ValidateBufferCapacity(output.data, plan->output_num_elements)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  detail::ExecuteSelectWithPlan(*plan, cond.data.data(), a.data.data(),
                                b.data.data(), output.data.data());
  return kLiteRtStatusOk;
}

// Evaluates legacy TFLite `SELECT` semantics: `a`, `b`, and `output` must share
// the exact same shape, and `cond` must be a scalar (rank 0), match that shape,
// or be a 1D tensor whose length matches the leading dimension of `a` (for
// rank >= 1).
template <typename T>
inline LiteRtStatus ReferenceSelectLegacy(ConstTensorView<bool> cond,
                                          ConstTensorView<T> a,
                                          ConstTensorView<T> b,
                                          TensorView<T> output) {
  if (a.dims.size() != b.dims.size() || a.dims.size() != output.dims.size() ||
      !std::equal(a.dims.begin(), a.dims.end(), b.dims.begin()) ||
      !std::equal(a.dims.begin(), a.dims.end(), output.dims.begin())) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (cond.dims.empty() ||
      (cond.dims.size() == a.dims.size() &&
       std::equal(cond.dims.begin(), cond.dims.end(), a.dims.begin()))) {
    return ReferenceSelect<T>(cond, a, b, output);
  }
  if (cond.dims.size() == 1 && !a.dims.empty() && cond.dims[0] == a.dims[0]) {
    std::vector<int32_t> expanded_cond_dims(a.dims.size(), 1);
    expanded_cond_dims[0] = cond.dims[0];
    return ReferenceSelect<T>(
        ConstTensorView<bool>{cond.data, expanded_cond_dims}, a, b, output);
  }
  return kLiteRtStatusErrorInvalidArgument;
}

template <typename T>
inline LiteRtStatus ReferenceSelect(const bool* cond_data,
                                    const int32_t* cond_dims, int cond_rank,
                                    const T* a_data, const int32_t* a_dims,
                                    int a_rank, const T* b_data,
                                    const int32_t* b_dims, int b_rank,
                                    T* output_data, const int32_t* output_dims,
                                    int rank) {
  auto cond_span = MakeDimsSpan(cond_dims, cond_rank);
  auto a_span = MakeDimsSpan(a_dims, a_rank);
  auto b_span = MakeDimsSpan(b_dims, b_rank);
  auto out_span = MakeDimsSpan(output_dims, rank);
  if (!cond_span || !a_span || !b_span || !out_span) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto plan = PrepareBroadcastPlan<3>(
      {*cond_span, *a_span, *b_span}, {sizeof(bool), sizeof(T), sizeof(T)},
      *out_span, sizeof(T), BroadcastOutputMode::kInferredExact);
  if (!plan) {
    return plan.Error().Status();
  }
  if (plan->IsEmpty()) {
    return kLiteRtStatusOk;
  }
  if (cond_data == nullptr || a_data == nullptr || b_data == nullptr ||
      output_data == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  detail::ExecuteSelectWithPlan(*plan, cond_data, a_data, b_data, output_data);
  return kLiteRtStatusOk;
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_SELECT_H_
