// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "litert/vendors/qualcomm/core/builders/onehot_op_builder.h"

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

namespace {
constexpr size_t kIndicesTensorIndex = 0;
constexpr size_t kDepthTensorIndex = 1;
constexpr size_t kOnValueTensorIndex = 2;
constexpr size_t kOffValueTensorIndex = 3;
constexpr size_t kOutputTensorIndex = 0;

bool IsScalarStaticTensor(const TensorWrapper& tensor) {
  return tensor.IsTensorStatic() && tensor.GetTensorNumElements() == 1;
}

}  // namespace

std::vector<OpWrapper> BuildOneHotOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, const int32_t axis) {
  if (inputs.size() <= kOffValueTensorIndex || outputs.empty()) {
    QNN_LOG_ERROR("OneHot op expects 4 inputs and 1 output.");
    return {};
  }

  TensorWrapper& indices_tensor = inputs[kIndicesTensorIndex];
  TensorWrapper& depth_tensor = inputs[kDepthTensorIndex];
  TensorWrapper& on_value_tensor = inputs[kOnValueTensorIndex];
  TensorWrapper& off_value_tensor = inputs[kOffValueTensorIndex];
  TensorWrapper& output_tensor = outputs[kOutputTensorIndex];

  if (!IsScalarStaticTensor(depth_tensor) ||
      !IsScalarStaticTensor(on_value_tensor) ||
      !IsScalarStaticTensor(off_value_tensor)) {
    QNN_LOG_ERROR("Depth, on value, and off value must be static scalars.");
    return {};
  }

  if (on_value_tensor.GetDataType() != off_value_tensor.GetDataType() ||
      on_value_tensor.GetDataType() != output_tensor.GetDataType()) {
    QNN_LOG_ERROR(
        "On value and off value and output tensor must have the same data "
        "type.");
    return {};
  }

  std::uint32_t depth_value = 0;
  if (depth_tensor.GetDataType() == QNN_DATATYPE_UINT_32) {
    const auto depth_data = depth_tensor.GetTensorData<std::uint32_t>();
    if (!depth_data.has_value()) {
      QNN_LOG_ERROR("Failed to read OneHot depth value.");
      return {};
    }
    depth_value = (*depth_data)[0];
  } else if (depth_tensor.GetDataType() == QNN_DATATYPE_INT_32) {
    const auto depth_data = depth_tensor.GetTensorData<std::int32_t>();
    if (!depth_data.has_value()) {
      QNN_LOG_ERROR("Failed to read OneHot depth value.");
      return {};
    }
    if ((*depth_data)[0] < 0) {
      QNN_LOG_ERROR("OneHot depth must be non-negative.");
      return {};
    }
    depth_value = static_cast<std::uint32_t>((*depth_data)[0]);
  } else {
    QNN_LOG_ERROR("OneHot depth tensor must be INT32 or UINT32.");
    return {};
  }

  const std::int32_t rank = static_cast<std::int32_t>(indices_tensor.GetRank());
  const std::int32_t adjusted_axis = axis < 0 ? axis + rank + 1 : axis;
  if (adjusted_axis < 0 || adjusted_axis > rank) {
    QNN_LOG_ERROR("OneHot axis is out of range.");
    return {};
  }

  std::vector<OpWrapper> res;
  OpWrapper& one_hot_op = CreateOpWrapper(res, QNN_OP_ONE_HOT);
  one_hot_op.AddInputTensor(indices_tensor);
  one_hot_op.AddOutputTensor(output_tensor);
  one_hot_op.AddScalarParam<std::uint32_t>(QNN_OP_ONE_HOT_PARAM_DEPTH,
                                           depth_value);
  one_hot_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ONE_HOT_PARAM_AXIS, static_cast<std::uint32_t>(adjusted_axis));

  switch (on_value_tensor.GetDataType()) {
    case QNN_DATATYPE_UFIXED_POINT_8: {
      const auto on_value = on_value_tensor.GetTensorData<std::uint8_t>();
      const auto off_value = off_value_tensor.GetTensorData<std::uint8_t>();
      if (!on_value.has_value() || !off_value.has_value()) {
        QNN_LOG_ERROR("Failed to read OneHot on/off values.");
        return {};
      }
      one_hot_op.AddScalarParam<std::uint8_t>(QNN_OP_ONE_HOT_PARAM_ON_VALUE,
                                              (*on_value)[0], true);
      one_hot_op.AddScalarParam<std::uint8_t>(QNN_OP_ONE_HOT_PARAM_OFF_VALUE,
                                              (*off_value)[0], true);
      break;
    }
    case QNN_DATATYPE_SFIXED_POINT_8: {
      const auto on_value = on_value_tensor.GetTensorData<std::int8_t>();
      const auto off_value = off_value_tensor.GetTensorData<std::int8_t>();
      if (!on_value.has_value() || !off_value.has_value()) {
        QNN_LOG_ERROR("Failed to read OneHot on/off values.");
        return {};
      }
      one_hot_op.AddScalarParam<std::int8_t>(QNN_OP_ONE_HOT_PARAM_ON_VALUE,
                                             (*on_value)[0], true);
      one_hot_op.AddScalarParam<std::int8_t>(QNN_OP_ONE_HOT_PARAM_OFF_VALUE,
                                             (*off_value)[0], true);
      break;
    }
    case QNN_DATATYPE_UFIXED_POINT_16: {
      const auto on_value = on_value_tensor.GetTensorData<std::uint16_t>();
      const auto off_value = off_value_tensor.GetTensorData<std::uint16_t>();
      if (!on_value.has_value() || !off_value.has_value()) {
        QNN_LOG_ERROR("Failed to read OneHot on/off values.");
        return {};
      }
      one_hot_op.AddScalarParam<std::uint16_t>(QNN_OP_ONE_HOT_PARAM_ON_VALUE,
                                               (*on_value)[0], true);
      one_hot_op.AddScalarParam<std::uint16_t>(QNN_OP_ONE_HOT_PARAM_OFF_VALUE,
                                               (*off_value)[0], true);
      break;
    }
    case QNN_DATATYPE_FLOAT_32: {
      const auto on_value = on_value_tensor.GetTensorData<float>();
      const auto off_value = off_value_tensor.GetTensorData<float>();
      if (!on_value.has_value() || !off_value.has_value()) {
        QNN_LOG_ERROR("Failed to read OneHot on/off values.");
        return {};
      }
      one_hot_op.AddScalarParam<float>(QNN_OP_ONE_HOT_PARAM_ON_VALUE,
                                       (*on_value)[0]);
      one_hot_op.AddScalarParam<float>(QNN_OP_ONE_HOT_PARAM_OFF_VALUE,
                                       (*off_value)[0]);
      break;
    }
    default:
      QNN_LOG_ERROR("Unsupported data type for OneHot on/off values.");
      return {};
  }

  return res;
}

}  // namespace qnn
