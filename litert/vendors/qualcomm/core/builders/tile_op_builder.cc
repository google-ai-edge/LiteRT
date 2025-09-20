// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "litert/vendors/qualcomm/core/builders/tile_op_builder.h"

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

std::vector<OpWrapper> BuildTileOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  TensorWrapper& multiples_tensor = inputs[1];
  if (!multiples_tensor.IsTensorStatic()) {
    QNN_LOG_ERROR("The multiples param of Tile OP is not static.");
    return {};
  }

  auto& tile_op = CreateOpWrapper(res, QNN_OP_TILE);
  tile_op.AddInputTensor(inputs[0]);
  tile_op.AddOutputTensor(outputs[0]);
  if (multiples_tensor.GetDataType() == QNN_DATATYPE_INT_64) {
    QNN_LOG_WARNING("Convert multiples param of Tile OP from INT64 to UINT32.");
    std::vector<std::uint32_t> uint32_data;
    size_t data_len = multiples_tensor.GetTensorNumElements();
    auto int64_data = multiples_tensor.GetTensorData<std::int64_t>();
    if (!int64_data.has_value()) {
      QNN_LOG_ERROR("Tile OP get int64 multiples param failed.");
      return {};
    }
    uint32_data.reserve(data_len);
    for (size_t i = 0; i < data_len; ++i) {
      uint32_data.emplace_back(static_cast<std::uint32_t>((*int64_data)[i]));
    }
    TensorWrapper& uint32_multiples_tensor = tensor_pool.CreateStaticTensor(
        QNN_DATATYPE_UINT_32, multiples_tensor.GetQuantParams(),
        multiples_tensor.GetDims(),
        sizeof(decltype(uint32_data)::value_type) * uint32_data.size(),
        reinterpret_cast<void*>(uint32_data.data()));

    tile_op.AddTensorParam(QNN_OP_TILE_PARAM_MULTIPLES,
                           uint32_multiples_tensor);
  } else if (multiples_tensor.GetDataType() == QNN_DATATYPE_INT_32) {
    tile_op.AddTensorParam(QNN_OP_TILE_PARAM_MULTIPLES,
                           tensor_pool.CloneStaticTensorFrom(
                               multiples_tensor, QNN_DATATYPE_UINT_32));
  } else {
    QNN_LOG_ERROR("Unsupported data type for multiples param.");
    return {};
  }

  return res;
}

}  // namespace qnn
