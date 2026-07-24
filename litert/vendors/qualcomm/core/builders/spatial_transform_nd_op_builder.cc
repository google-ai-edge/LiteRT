// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/builders/spatial_transform_nd_op_builder.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnOpDef.h"  // from @qairt

namespace qnn {

namespace {
constexpr size_t kInputIndex = 0;
constexpr size_t kBlockShapeIndex = 1;
constexpr size_t kCropsOrPaddingsIndex = 2;
constexpr size_t kOutputIndex = 0;

bool ConvertBlockAndSpatialParams(TensorPool& tensor_pool,
                                  const std::vector<TensorWrapperRef>& inputs,
                                  TensorWrapper*& block_param,
                                  TensorWrapper*& spatial_param) {
  if (inputs.size() <= kCropsOrPaddingsIndex) {
    QNN_LOG_ERROR("Invalid number of inputs for BatchToSpaceNd/SpaceToBatchNd.");
    return false;
  }
  const TensorWrapper& block_shape = inputs[kBlockShapeIndex];
  const TensorWrapper& spatial = inputs[kCropsOrPaddingsIndex];
  if (!block_shape.IsTensorStatic() || !spatial.IsTensorStatic()) {
    QNN_LOG_ERROR("QNN only supports static block_shape and crops/paddings.");
    return false;
  }
  block_param = tensor_pool.ConvertStaticTensorFrom<std::uint32_t>(block_shape);
  if (block_param == nullptr) {
    QNN_LOG_ERROR("Failed to convert block_shape to uint32.");
    return false;
  }
  spatial_param = tensor_pool.ConvertStaticTensorFrom<std::uint32_t>(spatial);
  if (spatial_param == nullptr) {
    QNN_LOG_ERROR("Failed to convert crops/paddings to uint32.");
    return false;
  }
  return true;
}
}  // namespace

std::vector<OpWrapper> BuildBatchToSpaceNdOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  TensorWrapper* block_size = nullptr;
  TensorWrapper* crops = nullptr;
  if (!ConvertBlockAndSpatialParams(tensor_pool, inputs, block_size, crops)) {
    return {};
  }
  return MakeVector(CreateBatchToSpaceNdOp(
      inputs[kInputIndex], outputs[kOutputIndex], *block_size, *crops));
}

std::vector<OpWrapper> BuildSpaceToBatchNdOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  TensorWrapper* block_size = nullptr;
  TensorWrapper* pad_amount = nullptr;
  if (!ConvertBlockAndSpatialParams(tensor_pool, inputs, block_size,
                                    pad_amount)) {
    return {};
  }
  return MakeVector(CreateSpaceToBatchNdOp(
      inputs[kInputIndex], outputs[kOutputIndex], *block_size, *pad_amount));
}

OpWrapper CreateBatchToSpaceNdOp(const TensorWrapper& input,
                                 const TensorWrapper& output,
                                 const TensorWrapper& block_size,
                                 const TensorWrapper& crops) {
  OpWrapper op(GetUniqueOpName(QNN_OP_BATCH_TO_SPACE), QNN_OP_BATCH_TO_SPACE,
               QnnOpCode::kBatchToSpace);
  op.AddInputTensor(input);
  op.AddOutputTensor(output);
  op.AddTensorParam(QNN_OP_BATCH_TO_SPACE_PARAM_BLOCK_SIZE, block_size);
  op.AddTensorParam(QNN_OP_BATCH_TO_SPACE_PARAM_CROPS, crops);
  return op;
}

OpWrapper CreateSpaceToBatchNdOp(const TensorWrapper& input,
                                 const TensorWrapper& output,
                                 const TensorWrapper& block_size,
                                 const TensorWrapper& pad_amount) {
  OpWrapper op(GetUniqueOpName(QNN_OP_SPACE_TO_BATCH), QNN_OP_SPACE_TO_BATCH,
               QnnOpCode::kSpaceToBatch);
  op.AddInputTensor(input);
  op.AddOutputTensor(output);
  op.AddTensorParam(QNN_OP_SPACE_TO_BATCH_PARAM_BLOCK_SIZE, block_size);
  op.AddTensorParam(QNN_OP_SPACE_TO_BATCH_PARAM_PAD_AMOUNT, pad_amount);
  return op;
}

}  // namespace qnn
