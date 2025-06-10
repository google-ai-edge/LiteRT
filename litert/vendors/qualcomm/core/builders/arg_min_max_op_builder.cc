// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/builders/arg_min_max_op_builder.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnOpDef.h"  // from @qairt

namespace qnn {

namespace {
constexpr size_t kInputIndex = 0;
constexpr size_t kAxisIndex = 1;
constexpr size_t kOutputIndex = 0;

bool GetAxis(std::uint32_t& axis, const TensorWrapper& axis_tensor,
             const std::uint32_t input_rank) {
  if (const auto opt_axis_data =
          axis_tensor.GetTensorData<std::int32_t>();
      opt_axis_data.has_value()) {
    const auto axis_data = opt_axis_data.value();
    axis = axis_data[0] >= 0 ? axis_data[0] : axis_data[0] + input_rank;
    return true;
  }

  if (const auto opt_axis_data =
          axis_tensor.GetTensorData<std::int64_t>();
      opt_axis_data.has_value()) {
    const auto axis_data = opt_axis_data.value();
    const auto axis_value = static_cast<std::int32_t>(axis_data[0]);
    axis = axis_value >= 0 ? axis_value : axis_value + input_rank;
    return true;
  }

  return false;
}

std::vector<OpWrapper> BuildArgOpImpl(
    const char* op_type, const char* axis_param, TensorPool& tensor_pool,
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  const TensorWrapper& axis_tensor = inputs[kAxisIndex];
  if (!axis_tensor.IsTensorStatic()) {
    QNN_LOG_ERROR("Unsupported non static axis tensor");
    return {};
  }

  const TensorWrapper& input_tensor = inputs[kInputIndex];
  std::uint32_t axis_value;
  if (!GetAxis(axis_value, axis_tensor, input_tensor.GetRank())) {
    QNN_LOG_ERROR(
        "Failed to get axis value from axis tensor, only support int32/int64 "
        "axis tensor.");
    return {};
  }

  std::vector<OpWrapper> res;
  auto& op = CreateOpWrapper(res, op_type);
  op.AddInputTensor(input_tensor);
  op.AddOutputTensor(outputs[kOutputIndex]);
  op.AddScalarParam<std::uint32_t>(axis_param, axis_value);
  return res;
}

}  // namespace

std::vector<OpWrapper> BuildArgMaxOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  return BuildArgOpImpl(QNN_OP_ARGMAX, QNN_OP_ARGMAX_PARAM_AXIS, tensor_pool,
                        inputs, outputs);
}

std::vector<OpWrapper> BuildArgMinOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  return BuildArgOpImpl(QNN_OP_ARGMIN, QNN_OP_ARGMIN_PARAM_AXIS, tensor_pool,
                        inputs, outputs);
}

}  // namespace qnn
