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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_OPS_MATMUL_H_
#define ODML_LITERT_LITERT_CORE_MODEL_OPS_MATMUL_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"

namespace litert::internal {

inline LiteRtStatus InferBatchMatmul(const LiteRtOpT& op,
                                     absl::Span<const Dims> input_shapes,
                                     std::vector<Dims>& output_shapes) {
  constexpr size_t kBatchMatMulMinArgs = 2;
  constexpr size_t kLhsArgIndex = 0;
  constexpr size_t kRhsArgIndex = 1;
  constexpr size_t kMinRank = 2;

  if (input_shapes.size() != kBatchMatMulMinArgs) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto& lhs = input_shapes[kLhsArgIndex];
  const auto& rhs = input_shapes[kRhsArgIndex];
  if (lhs.size() < kMinRank || rhs.size() < kMinRank) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  const auto& opts = GetTflOptions(op);
  const auto* bmm_opts = opts.AsBatchMatMulOptions();
  bool adj_x = bmm_opts ? bmm_opts->adj_x : false;
  bool adj_y = bmm_opts ? bmm_opts->adj_y : false;

  constexpr size_t kOffset1 = 1;
  constexpr size_t kOffset2 = 2;

  int32_t m = adj_x ? lhs[lhs.size() - kOffset1] : lhs[lhs.size() - kOffset2];
  int32_t k_lhs =
      adj_x ? lhs[lhs.size() - kOffset2] : lhs[lhs.size() - kOffset1];
  int32_t k_rhs =
      adj_y ? rhs[rhs.size() - kOffset1] : rhs[rhs.size() - kOffset2];
  int32_t n = adj_y ? rhs[rhs.size() - kOffset2] : rhs[rhs.size() - kOffset1];

  if (k_lhs != -1 && k_rhs != -1 && k_lhs != k_rhs) {
    return kLiteRtStatusErrorInvalidArgument;  // Mismatch K
  }

  size_t rank = std::max(lhs.size(), rhs.size());
  Dims out_shape(rank);
  out_shape[rank - kOffset2] = m;
  out_shape[rank - kOffset1] = n;

  for (int i = 2; i < rank; ++i) {
    int idx_out = rank - kOffset1 - i;
    int idx_lhs = lhs.size() - kOffset1 - i;
    int idx_rhs = rhs.size() - kOffset1 - i;
    int32_t d1 = (idx_lhs >= 0) ? lhs[idx_lhs] : 1;
    int32_t d2 = (idx_rhs >= 0) ? rhs[idx_rhs] : 1;

    if (d1 == -1 || d2 == -1) {
      out_shape[idx_out] = -1;
    } else if (d1 == d2) {
      out_shape[idx_out] = d1;
    } else if (d1 == 1) {
      out_shape[idx_out] = d2;
    } else if (d2 == 1) {
      out_shape[idx_out] = d1;
    } else {
      return kLiteRtStatusErrorInvalidArgument;
    }
  }

  output_shapes[0] = std::move(out_shape);
  return kLiteRtStatusOk;
}

inline LiteRtStatus InferFullyConnected(const LiteRtOpT& op,
                                        absl::Span<const Dims> input_shapes,
                                        std::vector<Dims>& output_shapes) {
  constexpr size_t kFullyConnectedMinArgs = 2;
  constexpr size_t kInputArgIndex = 0;
  constexpr size_t kWeightsArgIndex = 1;
  constexpr size_t kWeightsRank = 2;
  constexpr size_t kOutputDimIndex = 0;

  if (input_shapes.size() < kFullyConnectedMinArgs) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto& input = input_shapes[kInputArgIndex];
  const auto& weights = input_shapes[kWeightsArgIndex];

  const auto& opts = GetTflOptions(op);
  const auto* fc_opts = opts.AsFullyConnectedOptions();
  bool keep_num_dims = fc_opts ? fc_opts->keep_num_dims : false;
  // TFLite FullyConnectedWeightsFormat_DEFAULT = 0.
  // We assume default: weights are [O, I].

  if (weights.size() != kWeightsRank) {
    // TFLite weights are usually 2D.
    // If not, we might need to handle sparse etc.
    // For now assume 2D.
    return kLiteRtStatusErrorInvalidArgument;
  }

  int32_t output_dim = weights[kOutputDimIndex];

  Dims out_shape;
  if (keep_num_dims) {
    out_shape = input;
    out_shape.back() = output_dim;
  } else {
    int64_t batch = 1;
    bool dynamic_batch = false;
    for (size_t i = 0; i < input.size() - 1; ++i) {
      if (input[i] == -1) {
        dynamic_batch = true;
      } else {
        batch *= input[i];
      }
    }
    // If we collapsed dims, output is 2D.
    out_shape.push_back(dynamic_batch ? -1 : batch);
    out_shape.push_back(output_dim);
  }

  output_shapes[0] = std::move(out_shape);
  return kLiteRtStatusOk;
}

inline void ReferenceBatchMatmul(const float* lhs_data, const int32_t* lhs_dims,
                                 size_t lhs_rank, const float* rhs_data,
                                 const int32_t* rhs_dims, size_t rhs_rank,
                                 float* out_data, const int32_t* out_dims,
                                 size_t out_rank, bool adj_x, bool adj_y) {
  if (out_rank < 2 || lhs_rank < 2 || rhs_rank < 2) {
    return;
  }
  int64_t m = out_dims[out_rank - 2];
  int64_t n = out_dims[out_rank - 1];
  int64_t k = adj_x ? lhs_dims[lhs_rank - 2] : lhs_dims[lhs_rank - 1];

  int64_t out_batch_size = 1;
  for (size_t i = 0; i < out_rank - 2; ++i) {
    out_batch_size *= out_dims[i];
  }

  for (int64_t b = 0; b < out_batch_size; ++b) {
    int64_t temp = b;
    int64_t lhs_batch_offset = 0;
    int64_t rhs_batch_offset = 0;

    int64_t lhs_stride = 1;
    int64_t rhs_stride = 1;

    for (size_t i = 1; i <= out_rank - 2; ++i) {
      int out_batch_idx = out_rank - 2 - i;
      int lhs_batch_idx = lhs_rank - 2 - i;
      int rhs_batch_idx = rhs_rank - 2 - i;

      int64_t out_dim = out_dims[out_batch_idx];
      int64_t coord = (temp % out_dim);
      temp /= out_dim;

      if (lhs_batch_idx >= 0) {
        int64_t lhs_dim = lhs_dims[lhs_batch_idx];
        if (lhs_dim > 1) {
          lhs_batch_offset += coord * lhs_stride;
        }
        lhs_stride *= lhs_dim;
      }

      if (rhs_batch_idx >= 0) {
        int64_t rhs_dim = rhs_dims[rhs_batch_idx];
        if (rhs_dim > 1) {
          rhs_batch_offset += coord * rhs_stride;
        }
        rhs_stride *= rhs_dim;
      }
    }

    const float* curr_lhs = lhs_data + lhs_batch_offset * (m * k);
    const float* curr_rhs = rhs_data + rhs_batch_offset * (k * n);
    float* curr_out = out_data + b * (m * n);

    for (int64_t row = 0; row < m; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        float sum = 0;
        for (int64_t i = 0; i < k; ++i) {
          int64_t lhs_idx = adj_x ? (i * m + row) : (row * k + i);
          int64_t rhs_idx = adj_y ? (col * k + i) : (i * n + col);
          sum += curr_lhs[lhs_idx] * curr_rhs[rhs_idx];
        }
        curr_out[row * n + col] = sum;
      }
    }
  }
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_MATMUL_H_
