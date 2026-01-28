// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "litert/vendors/qualcomm/core/builders/split_op_builder.h"

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
constexpr size_t kInputIndex = 1;
constexpr size_t kAxisIndex = 0;
}  // namespace

std::vector<OpWrapper> BuildSplitOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs,
    const std::uint32_t num_splits) {
  std::vector<OpWrapper> res;

  const TensorWrapper& axis_tensor = inputs[kAxisIndex];
  if (!axis_tensor.IsTensorStatic()) {
    return res;
  }

  const TensorWrapper& input_tensor = inputs[kInputIndex];
  auto axis_data = axis_tensor.GetTensorData<int32_t>();
  if (!axis_data.has_value()) {
    QNN_LOG_ERROR("Get axis_data failed.");
    return res;
  }
  std::uint32_t axis = (*axis_data)[0] >= 0
                           ? (*axis_data)[0]
                           : (*axis_data)[0] + input_tensor.GetRank();

  const std::uint32_t slice_size = input_tensor.GetDim(axis) / num_splits;
  // The split_indice will do N cuts, split the dimension into N+1 clips
  // so 0 will not be included in the split_indice
  // for example, when we split 12 into 4 clip, the split index will be {3,6,9}
  std::vector<std::uint32_t> split_indice;
  split_indice.reserve(num_splits);
  for (int i = 1; i < num_splits; i++) {
    split_indice.emplace_back(static_cast<std::uint32_t>(i * slice_size));
  }
  TensorWrapper& split_indice_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, axis_tensor.GetQuantParams(), {num_splits - 1},
      sizeof(std::uint32_t) * split_indice.size(), split_indice.data());
  res.emplace_back(CreateSplitOp(
      input_tensor,
      std::vector<ConstTensorWrapperRef>(outputs.begin(), outputs.end()), axis,
      split_indice_tensor));
  return res;
}

OpWrapper CreateSplitOp(const TensorWrapper& input_0,
                        const std::vector<ConstTensorWrapperRef>& outputs,
                        std::uint32_t axis, const TensorWrapper& split_index) {
  auto name = GetUniqueOpName(QNN_OP_SPLIT);
  OpWrapper op;
  op.SetName(std::move(name));
  op.SetType(QNN_OP_SPLIT, QnnOpCode::kSplit);
  op.AddInputTensor(input_0);
  for (const auto& output : outputs) {
    op.AddOutputTensor(output);
  }
  op.AddScalarParam<std::uint32_t>(QNN_OP_SPLIT_PARAM_AXIS, axis);
  op.AddTensorParam(QNN_OP_SPLIT_PARAM_SPLIT_INDEX, split_index);
  return op;
}

}  // namespace qnn
