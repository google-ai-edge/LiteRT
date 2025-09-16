// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "litert/vendors/qualcomm/core/builders/scatter_nd_op_builder.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

#include "QnnOpDef.h"
#include "QnnTypes.h"
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
namespace qnn {

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
  auto shape_data = shape_tensor.GetTensorData<std::int32_t>();

  std::vector<std::uint32_t> data_dims(shape_data->begin(), shape_data->end());
  std::uint32_t num_data_element = std::accumulate(
      data_dims.begin(), data_dims.end(), 1, std::multiplies<>());

  TensorWrapper *data_tensor = tensor_pool.CreateStaticTensorWithValue(
      updates_tensor.GetDataType(), updates_tensor.GetQuantParams(), data_dims,
      0);

  if (data_tensor == nullptr) {
    QNN_LOG_ERROR("Failed to create data tensor for scatterNd op.");
    return res;
  }
  scatter_nd_op.AddInputTensor(*data_tensor);
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
