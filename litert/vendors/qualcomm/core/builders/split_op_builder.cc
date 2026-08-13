// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "litert/vendors/qualcomm/core/builders/split_op_builder.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/builders/slice_op_builder.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnOpDef.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt

namespace qnn {

namespace {
constexpr size_t kInputIndex = 1;
constexpr size_t kAxisIndex = 0;
}  // namespace

std::vector<OpWrapper> BuildSplitOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs,
    const std::uint32_t num_splits) {
  const TensorWrapper& axis_tensor = inputs[kAxisIndex];
  if (!axis_tensor.IsTensorStatic()) {
    return {};
  }

  const TensorWrapper& input_tensor = inputs[kInputIndex];
  auto axis_data = axis_tensor.GetTensorData<int32_t>();
  if (!axis_data.has_value()) {
    QNN_LOG_ERROR("Get axis_data failed.");
    return {};
  }
  const std::int32_t raw_axis = (*axis_data)[0];
  const std::int32_t adjusted_axis =
      raw_axis >= 0 ? raw_axis : raw_axis + input_tensor.GetRank();
  if (adjusted_axis < 0 || adjusted_axis >= input_tensor.GetRank()) {
    QNN_LOG_ERROR("Split axis is out of range.");
    return {};
  }
  const std::uint32_t axis = static_cast<std::uint32_t>(adjusted_axis);

  if (num_splits == 0 || outputs.size() != num_splits ||
      input_tensor.GetDimension(axis) % num_splits != 0) {
    QNN_LOG_ERROR("Split dimensions are not evenly divisible.");
    return {};
  }
  const std::uint32_t slice_size = input_tensor.GetDimension(axis) / num_splits;
  return BuildSplitSlices(tensor_pool, input_tensor, outputs, axis,
                          std::vector<std::uint32_t>(num_splits, slice_size));
}

std::vector<OpWrapper> BuildSplitSlices(
    TensorPool& tensor_pool, const TensorWrapper& input,
    const std::vector<TensorWrapperRef>& outputs, std::uint32_t axis,
    const std::vector<std::uint32_t>& split_sizes) {
  if (axis >= input.GetRank() || outputs.size() != split_sizes.size()) {
    QNN_LOG_ERROR("Invalid axis or output count for split slices.");
    return {};
  }

  std::uint32_t total_size = 0;
  for (const auto size : split_sizes) {
    total_size += size;
  }
  if (total_size != input.GetDimension(axis)) {
    QNN_LOG_ERROR("Split sizes do not cover the input dimension.");
    return {};
  }

  std::vector<OpWrapper> result;
  result.reserve(outputs.size());
  std::uint32_t axis_begin = 0;
  for (size_t output_index = 0; output_index < outputs.size(); ++output_index) {
    std::vector<std::int32_t> ranges;
    ranges.reserve(input.GetRank() * 3);
    for (size_t dimension = 0; dimension < input.GetRank(); ++dimension) {
      if (dimension == axis) {
        ranges.emplace_back(static_cast<std::int32_t>(axis_begin));
        ranges.emplace_back(static_cast<std::int32_t>(
            axis_begin + split_sizes[output_index]));
      } else {
        ranges.emplace_back(0);
        ranges.emplace_back(
            static_cast<std::int32_t>(input.GetDimension(dimension)));
      }
      ranges.emplace_back(1);
    }
    auto& ranges_tensor = tensor_pool.CreateStaticTensor(
        QNN_DATATYPE_INT_32, {}, {input.GetRank(), 3},
        sizeof(std::int32_t) * ranges.size(), ranges.data());
    result.emplace_back(
        CreateSliceOp(input, outputs[output_index], ranges_tensor));
    axis_begin += split_sizes[output_index];
  }
  return result;
}

OpWrapper CreateSplitOp(const TensorWrapper& input_0,
                        const std::vector<ConstTensorWrapperRef>& outputs,
                        std::uint32_t axis, const TensorWrapper& split_index) {
  OpWrapper op(GetUniqueOpName(QNN_OP_SPLIT), QNN_OP_SPLIT, QnnOpCode::kSplit);
  op.AddInputTensor(input_0);
  for (const auto& output : outputs) {
    op.AddOutputTensor(output);
  }
  op.AddScalarParam<std::uint32_t>(QNN_OP_SPLIT_PARAM_AXIS, axis);
  op.AddTensorParam(QNN_OP_SPLIT_PARAM_SPLIT_INDEX, split_index);
  return op;
}

}  // namespace qnn
