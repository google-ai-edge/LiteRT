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

std::vector<OpWrapper> BuildReduceMeanOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, const bool keep_dims) {
  return BuildReduceOp(tensor_pool, inputs, outputs, keep_dims,
                       QNN_OP_REDUCE_MEAN, QNN_OP_REDUCE_MEAN_PARAM_AXES,
                       QNN_OP_REDUCE_MEAN_PARAM_KEEP_DIMS);
}

std::vector<OpWrapper> BuildReduceMaxOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, const bool keep_dims) {
  return BuildReduceOp(tensor_pool, inputs, outputs, keep_dims,
                       QNN_OP_REDUCE_MAX, QNN_OP_REDUCE_MAX_PARAM_AXES,
                       QNN_OP_REDUCE_MAX_PARAM_KEEP_DIMS);
}

std::vector<OpWrapper> BuildReduceMinOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, const bool keep_dims) {
  return BuildReduceOp(tensor_pool, inputs, outputs, keep_dims,
                       QNN_OP_REDUCE_MIN, QNN_OP_REDUCE_MIN_PARAM_AXES,
                       QNN_OP_REDUCE_MIN_PARAM_KEEP_DIMS);
}

std::vector<OpWrapper> BuildReduceAnyOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, const bool keep_dims) {
  std::vector<OpWrapper> res;

  auto& reduce_max_input = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_UFIXED_POINT_8, QuantizeParamsWrapperVariant{},
      inputs[0].get().GetDims());
  auto& reduce_max_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_UFIXED_POINT_8, QuantizeParamsWrapperVariant{},
      outputs[0].get().GetDims());

  auto& cast_to_uint8_op = CreateOpWrapper(res, QNN_OP_CAST);
  cast_to_uint8_op.AddInputTensor(inputs[0]);
  cast_to_uint8_op.AddOutputTensor(reduce_max_input);

  auto reduce_max_ops =
      BuildReduceMaxOp(tensor_pool, {reduce_max_input, inputs[1]},
                       {reduce_max_output}, keep_dims);
  std::move(reduce_max_ops.begin(), reduce_max_ops.end(),
            std::back_inserter(res));

  auto& cast_to_bool_op = CreateOpWrapper(res, QNN_OP_CAST);
  cast_to_bool_op.AddInputTensor(reduce_max_output);
  cast_to_bool_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildReduceAllOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, const bool keep_dims) {
  std::vector<OpWrapper> res;

  auto& reduce_min_input = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_UFIXED_POINT_8, QuantizeParamsWrapperVariant{},
      inputs[0].get().GetDims());
  auto& reduce_min_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_UFIXED_POINT_8, QuantizeParamsWrapperVariant{},
      outputs[0].get().GetDims());

  auto& cast_to_uint8_op = CreateOpWrapper(res, QNN_OP_CAST);
  cast_to_uint8_op.AddInputTensor(inputs[0]);
  cast_to_uint8_op.AddOutputTensor(reduce_min_input);

  auto reduce_min_ops =
      BuildReduceMinOp(tensor_pool, {reduce_min_input, inputs[1]},
                       {reduce_min_output}, keep_dims);
  std::move(reduce_min_ops.begin(), reduce_min_ops.end(),
            std::back_inserter(res));

  auto& cast_to_bool_op = CreateOpWrapper(res, QNN_OP_CAST);
  cast_to_bool_op.AddInputTensor(reduce_min_output);
  cast_to_bool_op.AddOutputTensor(outputs[0]);

  return res;
}

}  // namespace qnn
