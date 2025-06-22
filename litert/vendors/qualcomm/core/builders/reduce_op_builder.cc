// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/builders/reduce_op_builder.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnOpDef.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt

namespace qnn {

std::vector<OpWrapper> BuildReduceOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, const bool keep_dims,
    const char* op_type, const char* axes_param, const char* keep_dims_param) {
  std::vector<OpWrapper> res;

  TensorWrapper& axis_tensor = inputs[1];
  if (!axis_tensor.IsTensorStatic() || axis_tensor.GetRank() != 1) {
    QNN_LOG_ERROR(
        "The axis tensor is not static, or the rank of axis tensor is not "
        "equal to 1.");
    return {};
  }

  TensorWrapper& input_tensor = inputs[0];

  auto axis_data = axis_tensor.GetStaticTensorData<std::int32_t>();
  if (!axis_data.has_value()) {
    QNN_LOG_ERROR("Get axis_data failed.");
    return {};
  }
  std::vector<std::uint32_t> adjusted_axis_data;
  for (size_t i = 0; i < axis_tensor.GetDim(0); ++i) {
    std::uint32_t adjusted_axis =
        (*axis_data)[i] >= 0 ? (*axis_data)[i]
                             : (*axis_data)[i] + input_tensor.GetRank();
    if (std::find(adjusted_axis_data.begin(), adjusted_axis_data.end(),
                  adjusted_axis) == adjusted_axis_data.end()) {
      adjusted_axis_data.emplace_back(adjusted_axis);
    }
  }
  TensorWrapper& adjusted_axis_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, axis_tensor.GetQuantParams(),
      {static_cast<const std::uint32_t>(adjusted_axis_data.size())},
      sizeof(std::uint32_t) * adjusted_axis_data.size(),
      adjusted_axis_data.data());

  OpWrapper& reduce_op = CreateOpWrapper(res, op_type);
  reduce_op.AddInputTensor(input_tensor);
  reduce_op.AddOutputTensor(outputs[0]);
  reduce_op.AddTensorParam(axes_param, adjusted_axis_tensor);
  reduce_op.AddScalarParam<bool>(keep_dims_param, keep_dims);

  return res;
}

std::vector<OpWrapper> BuildReduceSumOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, const bool keep_dims) {
  return BuildReduceOp(tensor_pool, inputs, outputs, keep_dims,
                       QNN_OP_REDUCE_SUM, QNN_OP_REDUCE_SUM_PARAM_AXES,
                       QNN_OP_REDUCE_SUM_PARAM_KEEP_DIMS);
}

std::vector<OpWrapper> BuildReduceMaxOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, const bool keep_dims) {
  return BuildReduceOp(tensor_pool, inputs, outputs, keep_dims,
                       QNN_OP_REDUCE_MAX, QNN_OP_REDUCE_MAX_PARAM_AXES,
                       QNN_OP_REDUCE_MAX_PARAM_KEEP_DIMS);
}

}  // namespace qnn
