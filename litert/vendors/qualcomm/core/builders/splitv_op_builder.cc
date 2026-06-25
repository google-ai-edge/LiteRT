// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "litert/vendors/qualcomm/core/builders/splitv_op_builder.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "QnnTypes.h"  // from @qairt
#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/builders/reshape_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/split_op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

namespace {
constexpr size_t kInputTensorIndex = 0;
constexpr size_t kSizeSplitsTensorIndex = 1;
constexpr size_t kAxisTensorIndex = 2;
}  // namespace

std::vector<OpWrapper> BuildSplitVOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, std::uint32_t num_splits) {
  if (inputs.size() <= kAxisTensorIndex || outputs.size() != num_splits ||
      num_splits == 0) {
    QNN_LOG_ERROR(
        "SplitV expects 3 inputs, outputs.size() == num_splits, and "
        "num_splits > 0.");
    return {};
  }

  const TensorWrapper& input_tensor = inputs[kInputTensorIndex];

  // QNN_OP_SPLIT requires a non-empty split_index (at least one cut), but
  // TFLite SPLIT_V permits num_splits == 1, where the sole size_splits entry
  // equals the axis dimension and the output is identical to the input. Lower
  // that case to reshape instead of QNN_OP_SPLIT.
  if (num_splits == 1) {
    return MakeVector(CreateReshapeOp(input_tensor, outputs[0]));
  }

  const TensorWrapper& size_splits_tensor = inputs[kSizeSplitsTensorIndex];
  const TensorWrapper& axis_tensor = inputs[kAxisTensorIndex];

  if (!axis_tensor.IsTensorStatic() || !size_splits_tensor.IsTensorStatic()) {
    QNN_LOG_ERROR("SplitV axis and size_splits tensors must be static.");
    return {};
  }

  // Resolve axis (TFLite SPLIT_V axis is always INT32).
  const auto axis_data = axis_tensor.GetTensorData<std::int32_t>();
  if (!axis_data.has_value() || axis_data->empty()) {
    QNN_LOG_ERROR("SplitV failed to read axis tensor data.");
    return {};
  }

  const std::int32_t raw_axis = (*axis_data)[0];
  const std::int32_t input_rank = input_tensor.GetRank();
  const std::int32_t adjusted_axis =
      raw_axis < 0 ? raw_axis + input_rank : raw_axis;
  if (adjusted_axis < 0 || adjusted_axis >= input_rank) {
    QNN_LOG_ERROR("SplitV axis is out of range.");
    return {};
  }
  const std::uint32_t axis = static_cast<std::uint32_t>(adjusted_axis);

  // Read size_splits — INT_32 or INT_64
  std::vector<std::int64_t> size_splits_values;
  size_splits_values.reserve(num_splits);
  if (size_splits_tensor.GetDataType() == QNN_DATATYPE_INT_32) {
    const auto size_splits_data =
        size_splits_tensor.GetTensorData<std::int32_t>();
    if (!size_splits_data.has_value() ||
        size_splits_data->size() != num_splits) {
      QNN_LOG_ERROR("SplitV failed to read int32 size_splits.");
      return {};
    }
    for (std::uint32_t i = 0; i < num_splits; ++i) {
      size_splits_values.emplace_back(
          static_cast<std::int64_t>((*size_splits_data)[i]));
    }
  } else if (size_splits_tensor.GetDataType() == QNN_DATATYPE_INT_64) {
    const auto size_splits_data =
        size_splits_tensor.GetTensorData<std::int64_t>();
    if (!size_splits_data.has_value() ||
        size_splits_data->size() != num_splits) {
      QNN_LOG_ERROR("SplitV failed to read int64 size_splits.");
      return {};
    }
    for (std::uint32_t i = 0; i < num_splits; ++i) {
      size_splits_values.emplace_back((*size_splits_data)[i]);
    }
  } else {
    QNN_LOG_ERROR("SplitV size_splits tensor must be INT32 or INT64.");
    return {};
  }

  // Resolve at most one -1 entry per TFLite spec. First count the -1 entries
  // (at most one is allowed) and sum the known sizes, then fill in the -1.
  const std::int64_t axis_dim =
      static_cast<std::int64_t>(input_tensor.GetDimension(axis));
  std::uint32_t num_neg = 0;
  std::int64_t known_sum = 0;
  for (std::uint32_t i = 0; i < num_splits; ++i) {
    if (size_splits_values[i] == -1) {
      ++num_neg;
    } else if (size_splits_values[i] < 0) {
      QNN_LOG_ERROR("SplitV size_splits contains a negative entry.");
      return {};
    } else {
      known_sum += size_splits_values[i];
    }
  }
  if (num_neg > 1) {
    QNN_LOG_ERROR("SplitV size_splits contains more than one -1.");
    return {};
  }
  if (num_neg == 1) {
    const std::int64_t inferred = axis_dim - known_sum;
    if (inferred < 0) {
      QNN_LOG_ERROR("SplitV inferred -1 entry is negative.");
      return {};
    }
    for (std::uint32_t i = 0; i < num_splits; ++i) {
      if (size_splits_values[i] == -1) {
        size_splits_values[i] = inferred;
        break;
      }
    }
  } else if (known_sum != axis_dim) {
    QNN_LOG_ERROR("SplitV size_splits sum does not match axis dimension.");
    return {};
  }

  // N segments are produced by N-1 internal cut points (QNN_OP_SPLIT takes the
  // cut points, not the per-segment sizes).
  const std::uint32_t num_cuts = num_splits - 1;

  std::vector<std::uint32_t> split_index;
  split_index.reserve(num_cuts);
  std::uint32_t running = 0;
  for (std::uint32_t i = 0; i < num_cuts; ++i) {
    running += static_cast<std::uint32_t>(size_splits_values[i]);
    split_index.emplace_back(running);
  }

  TensorWrapper& split_index_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, {}, {num_cuts},
      sizeof(std::uint32_t) * split_index.size(), split_index.data());

  // QNN has no dedicated SPLIT_V op; reuse the QNN_OP_SPLIT builder with the
  // cumulative split_index derived above.
  return MakeVector(CreateSplitOp(
      input_tensor,
      std::vector<ConstTensorWrapperRef>(outputs.begin(), outputs.end()), axis,
      split_index_tensor));
}

}  // namespace qnn
