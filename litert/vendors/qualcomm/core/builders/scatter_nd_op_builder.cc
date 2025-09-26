// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "litert/vendors/qualcomm/core/builders/scatter_nd_op_builder.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "QnnOpDef.h"
#include "QnnTypes.h"
#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

template <typename T>
TensorWrapper &CreateDataTensor(TensorPool &tensor_pool,
                                const Qnn_DataType_t data_type,
                                const QuantizeParamsWrapperVariant &quant_param,
                                std::vector<std::uint32_t> data_dims,
                                uint32_t num_data_element) {
  std::vector<T> data(num_data_element, 0);

  const std::uint32_t data_bytes =
      num_data_element * GetDataTypeSize(data_type);

  return tensor_pool.CreateStaticTensor(data_type, quant_param, data_dims,
                                        data_bytes, data.data());
}

std::vector<OpWrapper> BuildScatterNdOp(
    TensorPool &tensor_pool, const std::vector<TensorWrapperRef> &inputs,
    const std::vector<TensorWrapperRef> &outputs) {
  std::vector<OpWrapper> res;

  // indices, updates, shape
  const TensorWrapper &indices_tensor = inputs[0];
  const TensorWrapper &updates_tensor = inputs[1];
  const TensorWrapper &shape_tensor = inputs[2];

  // Shape tensor must be static to allow us to create data tensor
  if (!shape_tensor.IsTensorStatic()) {
    QNN_LOG_ERROR(
        "Failed to add build scatterNd, shape tensor must be static.");
    return res;
  }

  OpWrapper &scatter_nd_op = CreateOpWrapper(res, QNN_OP_SCATTER_ND);

  // Create data tensor with zero to mimic TFLite' behavior
  auto shape_data = shape_tensor.GetStaticTensorData<int32_t>();

  std::uint32_t num_shape_element = shape_tensor.GetTensorNumElements();
  std::uint32_t num_data_element = 1;
  std::vector<std::uint32_t> data_dims(num_shape_element);

  for (std::size_t idx = 0; idx < num_shape_element; ++idx) {
    data_dims[idx] = static_cast<std::uint32_t>((*shape_data)[idx]);
    num_data_element *= static_cast<std::int32_t>((*shape_data)[idx]);
  }

  if (std::holds_alternative<UndefinedQuantizeParamsWrapper>(
          updates_tensor.GetQuantParams())) {
    TensorWrapper &data_tensor = CreateDataTensor<char>(
        tensor_pool, updates_tensor.GetDataType(),
        updates_tensor.GetQuantParams(), data_dims, num_data_element);
    scatter_nd_op.AddInputTensor(data_tensor);

  } else if (std::holds_alternative<ScaleOffsetQuantizeParamsWrapper>(
                 updates_tensor.GetQuantParams())) {
    switch (updates_tensor.GetDataType()) {
      case Qnn_DataType_t::QNN_DATATYPE_SFIXED_POINT_8: {
        TensorWrapper &data_tensor = CreateDataTensor<std::int8_t>(
            tensor_pool, updates_tensor.GetDataType(),
            updates_tensor.GetQuantParams(), data_dims, num_data_element);
        scatter_nd_op.AddInputTensor(data_tensor);
        break;
      }
      case Qnn_DataType_t::QNN_DATATYPE_UFIXED_POINT_8: {
        TensorWrapper &data_tensor = CreateDataTensor<std::uint8_t>(
            tensor_pool, updates_tensor.GetDataType(),
            updates_tensor.GetQuantParams(), data_dims, num_data_element);
        scatter_nd_op.AddInputTensor(data_tensor);
        break;
      }
      case Qnn_DataType_t::QNN_DATATYPE_SFIXED_POINT_16: {
        TensorWrapper &data_tensor = CreateDataTensor<std::int16_t>(
            tensor_pool, updates_tensor.GetDataType(),
            updates_tensor.GetQuantParams(), data_dims, num_data_element);
        scatter_nd_op.AddInputTensor(data_tensor);
        break;
      }
      case Qnn_DataType_t::QNN_DATATYPE_UFIXED_POINT_16: {
        TensorWrapper &data_tensor = CreateDataTensor<std::uint16_t>(
            tensor_pool, updates_tensor.GetDataType(),
            updates_tensor.GetQuantParams(), data_dims, num_data_element);
        scatter_nd_op.AddInputTensor(data_tensor);
        break;
      }
      case Qnn_DataType_t::QNN_DATATYPE_SFIXED_POINT_32: {
        TensorWrapper &data_tensor = CreateDataTensor<std::int32_t>(
            tensor_pool, updates_tensor.GetDataType(),
            updates_tensor.GetQuantParams(), data_dims, num_data_element);
        scatter_nd_op.AddInputTensor(data_tensor);
        break;
      }
      case Qnn_DataType_t::QNN_DATATYPE_UFIXED_POINT_32: {
        TensorWrapper &data_tensor = CreateDataTensor<std::uint32_t>(
            tensor_pool, updates_tensor.GetDataType(),
            updates_tensor.GetQuantParams(), data_dims, num_data_element);
        scatter_nd_op.AddInputTensor(data_tensor);
        break;
      }
      default: {
        QNN_LOG_ERROR(
            "Failed to add build scatterNd, Unsupported qnn data type %d",
            updates_tensor.GetDataType());
      }
    }
  } else {
    QNN_LOG_ERROR("Unsuported quantization encoding for scatterNd.");
  }

  scatter_nd_op.AddInputTensor(indices_tensor);
  scatter_nd_op.AddInputTensor(updates_tensor);

  // TFLite runtime's scatterNd behavior is equivalent to reduction = Add, when
  // indices contains duplicates
  scatter_nd_op.AddScalarParam<std::uint32_t>(QNN_OP_SCATTER_ND_PARAM_REDUCTION,
                                              QNN_OP_SCATTER_ND_REDUCTION_ADD);
  scatter_nd_op.AddOutputTensor(outputs[0]);

  return res;
}

}  // namespace qnn
