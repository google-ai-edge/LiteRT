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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_OPS_SLICE_H_
#define ODML_LITERT_LITERT_CORE_MODEL_OPS_SLICE_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"

namespace litert::internal {

inline LiteRtStatus InferSlice(const LiteRtOpT& op,
                               absl::Span<const Dims> input_shapes,
                               std::vector<Dims>& output_shapes) {
  constexpr int kSliceNumArgs = 3;
  constexpr int kInputArgIndex = 0;
  constexpr int kBeginArgIndex = 1;
  constexpr int kSizeArgIndex = 2;

  // Inputs: Input, Begin, Size.
  if (input_shapes.size() < kSliceNumArgs || output_shapes.empty() ||
      op.Inputs().size() < kSliceNumArgs ||
      op.Inputs()[kSizeArgIndex] == nullptr) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  auto get_element_type = [](const LiteRtTensorT* t) -> LiteRtElementType {
    if (!t) return kLiteRtElementTypeNone;
    if (t->Type().first == kLiteRtRankedTensorType) {
      return t->Type().second.ranked_tensor_type.element_type;
    }
    if (t->Type().first == kLiteRtUnrankedTensorType) {
      return t->Type().second.unranked_tensor_type.element_type;
    }
    return kLiteRtElementTypeNone;
  };

  const LiteRtTensorT* input_tensor = op.Inputs()[kInputArgIndex];
  const LiteRtTensorT* begin_tensor = op.Inputs()[kBeginArgIndex];
  const LiteRtTensorT* size_tensor = op.Inputs()[kSizeArgIndex];
  const LiteRtTensorT* output_tensor =
      op.Outputs().empty() ? nullptr : op.Outputs()[0];

  const LiteRtElementType input_type = get_element_type(input_tensor);
  const LiteRtElementType begin_type = get_element_type(begin_tensor);
  LiteRtElementType size_type = get_element_type(size_tensor);
  const LiteRtElementType output_type = get_element_type(output_tensor);

  if (begin_type != kLiteRtElementTypeNone &&
      begin_type != kLiteRtElementTypeInt32 &&
      begin_type != kLiteRtElementTypeInt64) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }
  if (size_type != kLiteRtElementTypeNone &&
      size_type != kLiteRtElementTypeInt32 &&
      size_type != kLiteRtElementTypeInt64) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }
  // Begin and Size must have the same data type.
  if (begin_type != kLiteRtElementTypeNone &&
      size_type != kLiteRtElementTypeNone && begin_type != size_type) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  // Output element type must match input element type.
  if (input_type != kLiteRtElementTypeNone &&
      output_type != kLiteRtElementTypeNone && input_type != output_type) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  const auto& input_shape = input_shapes[kInputArgIndex];
  const auto& begin_shape = input_shapes[kBeginArgIndex];
  const auto& size_shape = input_shapes[kSizeArgIndex];

  // Begin and Size must be 1D tensors with length equal to input rank.
  if (begin_shape.size() != 1 || size_shape.size() != 1) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }
  if ((begin_shape[0] != -1 &&
       begin_shape[0] != static_cast<int32_t>(input_shape.size())) ||
      (size_shape[0] != -1 &&
       size_shape[0] != static_cast<int32_t>(input_shape.size()))) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  auto read_index_vec = [](const LiteRtTensorT* tensor,
                           LiteRtElementType elem_type,
                           std::vector<int64_t>& out) -> bool {
    if (!tensor) return false;
    auto buf = tensor->Weights().Buffer();
    if (buf.Size() == 0) return false;
    if (elem_type == kLiteRtElementTypeInt64) {
      if (buf.Size() % sizeof(int64_t) != 0) return false;
      const int64_t* data = reinterpret_cast<const int64_t*>(buf.Data());
      size_t count = buf.Size() / sizeof(int64_t);
      out.assign(data, data + count);
      return true;
    }
    if (elem_type == kLiteRtElementTypeInt32 ||
        elem_type == kLiteRtElementTypeNone) {
      if (buf.Size() % sizeof(int32_t) != 0) return false;
      const int32_t* data = reinterpret_cast<const int32_t*>(buf.Data());
      size_t count = buf.Size() / sizeof(int32_t);
      out.resize(count);
      for (size_t i = 0; i < count; ++i) {
        out[i] = data[i];
      }
      return true;
    }
    return false;
  };

  std::vector<int64_t> begin_vals;
  std::vector<int64_t> size_vals;
  const bool has_begin = read_index_vec(begin_tensor, begin_type, begin_vals);
  const bool has_size = read_index_vec(size_tensor, size_type, size_vals);

  if (has_begin) {
    if (begin_vals.size() != input_shape.size()) {
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
    for (size_t i = 0; i < input_shape.size(); ++i) {
      if (begin_vals[i] < 0 ||
          (input_shape[i] != -1 && begin_vals[i] > input_shape[i])) {
        return kLiteRtStatusErrorShapeInferenceFailed;
      }
    }
  }

  // If size tensor is constant, validate and compute output shape.
  if (has_size) {
    if (size_vals.size() != input_shape.size()) {
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
    Dims out_shape;
    out_shape.reserve(input_shape.size());
    for (size_t i = 0; i < input_shape.size(); ++i) {
      int64_t s = size_vals[i];
      if (s < -1) {
        return kLiteRtStatusErrorShapeInferenceFailed;
      }
      if (s == -1) {
        if (input_shape[i] != -1 && has_begin) {
          s = input_shape[i] - begin_vals[i];
        } else {
          s = -1;
        }
      } else if (input_shape[i] != -1 && has_begin) {
        if (begin_vals[i] + s > input_shape[i]) {
          return kLiteRtStatusErrorShapeInferenceFailed;
        }
      } else if (input_shape[i] != -1 && s > input_shape[i]) {
        return kLiteRtStatusErrorShapeInferenceFailed;
      }
      out_shape.push_back(static_cast<int32_t>(s));
    }
    output_shapes[0] = std::move(out_shape);
    return kLiteRtStatusOk;
  }

  // If size is dynamic, output is dynamic with matching rank.
  output_shapes[0] = Dims(input_shape.size(), -1);
  return kLiteRtStatusOk;
}

inline LiteRtStatus InferStridedSlice(const ShapeInferenceContext& ctx,
                                      InferenceResult& result) {
  if (ctx.GetNumInputs() != 4) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  const auto& opts = ctx.GetOptions();
  const auto* ss_opts = opts.AsStridedSliceOptions();
  if (!ss_opts) return kLiteRtStatusErrorShapeInferenceFailed;

  const int32_t begin_mask = ss_opts->begin_mask;
  const int32_t end_mask = ss_opts->end_mask;
  const int32_t shrink_axis_mask = ss_opts->shrink_axis_mask;
  const int32_t new_axis_mask = ss_opts->new_axis_mask;

  const auto& input_shape = ctx.GetInputShape(0);
  const auto& begin_shape = ctx.GetInputShape(1);
  if (begin_shape.empty()) return kLiteRtStatusErrorShapeInferenceFailed;
  const int processing_dims = begin_shape[0];

  auto begin_data = ctx.GetInputData(1);
  auto end_data = ctx.GetInputData(2);
  auto strides_data = ctx.GetInputData(3);

  auto parse_index_vec = [&ctx](size_t arg_index,
                                absl::Span<const uint8_t> data,
                                std::vector<int64_t>& vals) -> bool {
    if (data.empty()) return false;
    LiteRtElementType elem_type = ctx.GetInputElementType(arg_index);
    Dims shape = ctx.GetInputShape(arg_index);
    bool is_int64 = (elem_type == kLiteRtElementTypeInt64) ||
                    (!shape.empty() && shape[0] > 0 &&
                     data.size() == shape[0] * sizeof(int64_t));
    if (is_int64) {
      if (data.size() % sizeof(int64_t) != 0) return false;
      const int64_t* ptr = reinterpret_cast<const int64_t*>(data.data());
      size_t count = data.size() / sizeof(int64_t);
      for (size_t i = 0; i < count; ++i) vals.push_back(ptr[i]);
    } else {
      if (data.size() % sizeof(int32_t) != 0) return false;
      const int32_t* ptr = reinterpret_cast<const int32_t*>(data.data());
      size_t count = data.size() / sizeof(int32_t);
      for (size_t i = 0; i < count; ++i) vals.push_back(ptr[i]);
    }
    return true;
  };

  std::vector<int64_t> begin_vec, end_vec, strides_vec;
  bool indices_known = parse_index_vec(1, begin_data, begin_vec) &&
                       parse_index_vec(2, end_data, end_vec) &&
                       parse_index_vec(3, strides_data, strides_vec);

  if (indices_known &&
      (begin_vec.size() < static_cast<size_t>(processing_dims) ||
       end_vec.size() < static_cast<size_t>(processing_dims) ||
       strides_vec.size() < static_cast<size_t>(processing_dims))) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  // Validate non-zero strides whenever strides are known.
  if (indices_known) {
    for (int i = 0; i < processing_dims; ++i) {
      if (strides_vec[i] == 0) return kLiteRtStatusErrorInvalidArgument;
    }
  }

  Dims out_shape;
  int i_in = 0;
  for (int i = 0; i < processing_dims; ++i) {
    if (new_axis_mask & (1 << i)) {
      out_shape.push_back(1);
    } else if (shrink_axis_mask & (1 << i)) {
      i_in++;
    } else {
      if (indices_known && i_in < static_cast<int>(input_shape.size()) &&
          input_shape[i_in] != -1) {
        int64_t dim_size = input_shape[i_in];
        int64_t stride = strides_vec[i];
        int64_t start = begin_vec[i];
        int64_t stop = end_vec[i];

        if (start < 0) start += dim_size;
        if (stride > 0) {
          start = std::clamp(start, int64_t{0}, dim_size);
        } else {
          start = std::clamp(start, int64_t{-1}, dim_size - 1);
        }
        if (begin_mask & (1 << i)) {
          start = (stride > 0) ? 0 : dim_size - 1;
        }

        if (stop < 0) stop += dim_size;
        if (stride > 0) {
          stop = std::clamp(stop, int64_t{0}, dim_size);
        } else {
          stop = std::clamp(stop, int64_t{-1}, dim_size - 1);
        }
        if (end_mask & (1 << i)) {
          stop = (stride > 0) ? dim_size : -1;
        }

        int64_t dim_shape = stop - start;
        if ((dim_shape < 0) != (stride < 0)) {
          dim_shape = 0;
        } else {
          if (stride < 0) {
            dim_shape = (dim_shape + 1) / stride + 1;
          } else {
            dim_shape = (dim_shape == 0) ? 0 : (dim_shape - 1) / stride + 1;
          }
        }
        out_shape.push_back(static_cast<int32_t>(dim_shape));
      } else {
        out_shape.push_back(-1);
      }
      i_in++;
    }
  }

  for (; i_in < static_cast<int>(input_shape.size()); ++i_in) {
    out_shape.push_back(input_shape[i_in]);
  }

  if (result.output_shapes.empty()) {
    result.output_shapes.resize(1);
  }
  result.output_shapes[0] = std::move(out_shape);

  // 1D / scalar data propagation for constant/transient input buffers.
  auto in_data = ctx.GetInputData(0);
  if (!in_data.empty() && indices_known && input_shape.size() == 1 &&
      processing_dims == 1) {
    LiteRtElementType in_elem_type = ctx.GetInputElementType(0);
    bool in_is_int64 = (in_elem_type == kLiteRtElementTypeInt64) ||
                       (in_data.size() == input_shape[0] * sizeof(int64_t));

    int64_t dim_size = input_shape[0];
    int64_t stride = strides_vec[0];
    int64_t start = begin_vec[0];
    int64_t stop = end_vec[0];

    if (start < 0) start += dim_size;
    if (stride > 0) {
      start = std::clamp(start, int64_t{0}, dim_size);
    } else {
      start = std::clamp(start, int64_t{-1}, dim_size - 1);
    }
    if (begin_mask & 1) start = (stride > 0) ? 0 : dim_size - 1;

    if (stop < 0) stop += dim_size;
    if (stride > 0) {
      stop = std::clamp(stop, int64_t{0}, dim_size);
    } else {
      stop = std::clamp(stop, int64_t{-1}, dim_size - 1);
    }
    if (end_mask & 1) stop = (stride > 0) ? dim_size : -1;

    if (shrink_axis_mask & 1) {
      if (start >= 0 && start < dim_size) {
        if (in_is_int64 && in_data.size() % sizeof(int64_t) == 0) {
          const int64_t* in_ptr =
              reinterpret_cast<const int64_t*>(in_data.data());
          std::vector<uint8_t> out_bytes(sizeof(int64_t));
          std::memcpy(out_bytes.data(), &in_ptr[start], sizeof(int64_t));
          result.propagated_data[0] = std::move(out_bytes);
        } else if (!in_is_int64 && in_data.size() % sizeof(int32_t) == 0) {
          const int32_t* in_ptr =
              reinterpret_cast<const int32_t*>(in_data.data());
          std::vector<uint8_t> out_bytes(sizeof(int32_t));
          std::memcpy(out_bytes.data(), &in_ptr[start], sizeof(int32_t));
          result.propagated_data[0] = std::move(out_bytes);
        }
      }
    } else if (stride != 0) {
      if (in_is_int64 && in_data.size() % sizeof(int64_t) == 0) {
        const int64_t* in_ptr =
            reinterpret_cast<const int64_t*>(in_data.data());
        std::vector<int64_t> sliced_vals;
        for (int64_t idx = start; (stride > 0) ? (idx < stop) : (idx > stop);
             idx += stride) {
          if (idx >= 0 && idx < dim_size) {
            sliced_vals.push_back(in_ptr[idx]);
          }
        }
        std::vector<uint8_t> out_bytes(sliced_vals.size() * sizeof(int64_t));
        std::memcpy(out_bytes.data(), sliced_vals.data(), out_bytes.size());
        result.propagated_data[0] = std::move(out_bytes);
        result.output_shapes[0] = {static_cast<int32_t>(sliced_vals.size())};
      } else if (!in_is_int64 && in_data.size() % sizeof(int32_t) == 0) {
        const int32_t* in_ptr =
            reinterpret_cast<const int32_t*>(in_data.data());
        std::vector<int32_t> sliced_vals;
        for (int64_t idx = start; (stride > 0) ? (idx < stop) : (idx > stop);
             idx += stride) {
          if (idx >= 0 && idx < dim_size) {
            sliced_vals.push_back(in_ptr[idx]);
          }
        }
        std::vector<uint8_t> out_bytes(sliced_vals.size() * sizeof(int32_t));
        std::memcpy(out_bytes.data(), sliced_vals.data(), out_bytes.size());
        result.propagated_data[0] = std::move(out_bytes);
        result.output_shapes[0] = {static_cast<int32_t>(sliced_vals.size())};
      }
    }
  }

  return kLiteRtStatusOk;
}

inline LiteRtStatus InferStridedSlice(const LiteRtOpT& op,
                                      absl::Span<const Dims> input_shapes,
                                      std::vector<Dims>& output_shapes) {
  StandaloneShapeInferenceContext ctx(op, input_shapes);
  InferenceResult result;
  result.output_shapes = std::move(output_shapes);
  auto status = InferStridedSlice(ctx, result);
  output_shapes = std::move(result.output_shapes);
  return status;
}

inline LiteRtStatus InferDynamicUpdateSlice(const LiteRtOpT& op,
                                            absl::Span<const Dims> input_shapes,
                                            std::vector<Dims>& output_shapes) {
  constexpr int kDynamicUpdateSliceNumArgs = 3;
  constexpr int kOperandArgIndex = 0;
  constexpr int kUpdateArgIndex = 1;
  constexpr int kStartIndicesArgIndex = 2;
  constexpr int kStartIndicesRank = 1;

  // Inputs: Operand, Update, StartIndices.
  if (input_shapes.size() != kDynamicUpdateSliceNumArgs) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  const auto& operand_shape = input_shapes[kOperandArgIndex];
  const auto& update_shape = input_shapes[kUpdateArgIndex];
  const auto& start_indices_shape = input_shapes[kStartIndicesArgIndex];

  // Rank check: start_indices must be 1D.
  if (start_indices_shape.size() != kStartIndicesRank) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  // Dimension check: size of start_indices must match rank of operand.
  if (start_indices_shape[0] != operand_shape.size()) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  // Rank check: update rank must match operand rank.
  if (update_shape.size() != operand_shape.size()) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  // Boundary check: update dims must be <= operand dims.
  for (size_t i = 0; i < operand_shape.size(); ++i) {
    if (update_shape[i] > operand_shape[i]) {
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
  }

  // Output shape is same as input (operand) shape.
  output_shapes[0] = operand_shape;
  return kLiteRtStatusOk;
}

inline void SliceRecursive(int dim, int rank, const char* input_data,
                           const int64_t* input_strides_bytes,
                           const int32_t* begins, const int32_t* output_dims,
                           size_t element_size, char*& output_data) {
  input_data += begins[dim] * input_strides_bytes[dim];
  if (dim == rank - 1) {
    const size_t copy_bytes = output_dims[dim] * element_size;
    std::memcpy(output_data, input_data, copy_bytes);
    output_data += copy_bytes;
    return;
  }
  for (int32_t i = 0; i < output_dims[dim]; ++i) {
    SliceRecursive(dim + 1, rank, input_data, input_strides_bytes, begins,
                   output_dims, element_size, output_data);
    input_data += input_strides_bytes[dim];
  }
}

template <typename T>
inline void ReferenceSlice(const T* input_data, const int32_t* input_dims,
                           const int32_t* begins, const int32_t* output_dims,
                           int rank, T* output_data) {
  if (rank <= 0) {
    if (rank == 0) {
      output_data[0] = input_data[0];
    }
    return;
  }
  for (int i = 0; i < rank; ++i) {
    if (output_dims[i] <= 0) {
      return;
    }
  }

  std::vector<int64_t> input_strides_bytes(rank);
  input_strides_bytes[rank - 1] = sizeof(T);
  for (int i = rank - 2; i >= 0; --i) {
    input_strides_bytes[i] = input_strides_bytes[i + 1] * input_dims[i + 1];
  }

  char* out_ptr = reinterpret_cast<char*>(output_data);
  SliceRecursive(0, rank, reinterpret_cast<const char*>(input_data),
                 input_strides_bytes.data(), begins, output_dims, sizeof(T),
                 out_ptr);
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_SLICE_H_
