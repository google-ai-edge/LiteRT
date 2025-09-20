// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/builders/strided_slice_op_builder.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnOpDef.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt

namespace qnn {

namespace {
constexpr size_t kInputIndex = 0;
constexpr size_t kBeginIndex = 1;
constexpr size_t kEndIndex = 2;
constexpr size_t kStridesIndex = 3;
constexpr size_t kOutputIndex = 0;
constexpr std::int32_t kDefaultMask = 0;
}  // namespace

std::vector<OpWrapper> BuildStridedSliceOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, const std::int32_t begin_mask,
    const std::int32_t end_mask, const std::int32_t ellipsis_mask,
    const std::int32_t shrink_axis_mask, const std::int32_t new_axis_mask,
    const bool offset) {
  if (ellipsis_mask != kDefaultMask) {
    QNN_LOG_ERROR("Only ellipsis_mask with a value of 0 is supported.");
    return {};
  }

  const TensorWrapper& begin_tensor = inputs[kBeginIndex];
  const TensorWrapper& end_tensor = inputs[kEndIndex];
  const TensorWrapper& strides_tensor = inputs[kStridesIndex];
  if (!begin_tensor.IsTensorStatic() || !end_tensor.IsTensorStatic() ||
      !strides_tensor.IsTensorStatic()) {
    QNN_LOG_ERROR(
        "Only static tensors are accepted for the begin, end, and strides "
        "tensors.");
    return {};
  }

  const auto opt_begin_data = begin_tensor.GetTensorData<std::int32_t>();
  const auto opt_end_data = end_tensor.GetTensorData<std::int32_t>();
  const auto opt_strides_data =
      strides_tensor.GetTensorData<std::int32_t>();
  if (!opt_begin_data.has_value() || !opt_end_data.has_value() ||
      !opt_strides_data.has_value()) {
    QNN_LOG_ERROR(
        "Unable to retrieve data from the begin, end, or strides tensors.");
    return {};
  }

  const auto begin_data = opt_begin_data.value();
  const auto end_data = opt_end_data.value();
  const auto strides_data = opt_strides_data.value();

  const TensorWrapper& input_tensor = inputs[kInputIndex];
  const auto input_rank = input_tensor.GetRank();
  std::vector<std::int32_t> range_data;
  for (size_t i = 0; i < input_rank; ++i) {
    std::int32_t begin = begin_data[i];
    if (begin < 0) {
      begin += input_tensor.GetDim(i);
    }

    std::int32_t stride = strides_data[i];
    std::int32_t end = end_data[i];
    if (offset) {
      end += begin;
    } else {
      // for stride > 0, end should be in [0, dimensions[i]]
      // for stride < 0, end should be in [-1, dimensions[i] - 1]
      if ((stride > 0 && end < 0) || (stride < 0 && end < -1)) {
        end += input_tensor.GetDim(i);
      }
    }

    const bool is_shrink = (shrink_axis_mask & (1 << i)) != 0U;
    if (is_shrink) {
      end = begin + 1;
    }

    range_data.emplace_back(begin);
    range_data.emplace_back(end);
    range_data.emplace_back(stride);
  }

  auto& range_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {input_rank, 3},
      sizeof(decltype(range_data)::value_type) * range_data.size(),
      range_data.data());

  std::vector<OpWrapper> res;
  auto& op = CreateOpWrapper(res, QNN_OP_STRIDED_SLICE);
  op.AddInputTensor(input_tensor);
  op.AddOutputTensor(outputs[kOutputIndex]);
  op.AddTensorParam(QNN_OP_STRIDED_SLICE_PARAM_RANGES, range_tensor);
  op.AddScalarParam<std::uint32_t>(QNN_OP_STRIDED_SLICE_PARAM_BEGIN_MASK,
                                   static_cast<std::uint32_t>(begin_mask));
  op.AddScalarParam<std::uint32_t>(QNN_OP_STRIDED_SLICE_PARAM_END_MASK,
                                   static_cast<std::uint32_t>(end_mask));
  op.AddScalarParam<std::uint32_t>(
      QNN_OP_STRIDED_SLICE_PARAM_SHRINK_AXES,
      static_cast<std::uint32_t>(shrink_axis_mask));
  op.AddScalarParam<std::uint32_t>(QNN_OP_STRIDED_SLICE_PARAM_NEW_AXES_MASK,
                                   static_cast<std::uint32_t>(new_axis_mask));
  return res;
}

}  // namespace qnn
