// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/builders/slice_op_builder.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "QnnOpDef.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt
#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

namespace {
constexpr int kDefaultStrideValue = 1;
constexpr int kSizeNegative = -1;
constexpr int kRangeNumElements = 3;
}  // namespace

std::vector<OpWrapper> BuildSliceOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  TensorWrapper& input_tensor = inputs[0];
  TensorWrapper& begin_tensor = inputs[1];
  TensorWrapper& size_tensor = inputs[2];
  if (!begin_tensor.IsTensorStatic() || !size_tensor.IsTensorStatic()) {
    QNN_LOG_ERROR(
        "The begin tensor and size tensor of Slice OP is not static.");
    return res;
  }

  const auto input_rank = input_tensor.GetRank();
  auto begin_data = begin_tensor.GetTensorData<int32_t>();
  if (!begin_data.has_value()) {
    QNN_LOG_ERROR("Get begin_data failed.");
    return res;
  }
  auto size_data = size_tensor.GetTensorData<int32_t>();
  if (!size_data.has_value()) {
    QNN_LOG_ERROR("Get size_data failed.");
    return res;
  }
  std::vector<std::int32_t> range_data;
  range_data.reserve(input_rank * kRangeNumElements);
  for (size_t i = 0; i < input_rank; ++i) {
    range_data.emplace_back((*begin_data)[i]);
    if ((*size_data)[i] == kSizeNegative) {
      range_data.emplace_back(input_tensor.GetDim(i));
    } else {
      range_data.emplace_back((*begin_data)[i] + (*size_data)[i]);
    }
    range_data.emplace_back(kDefaultStrideValue);
  }
  TensorWrapper& range_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, begin_tensor.GetQuantParams(),
      {input_rank, kRangeNumElements}, sizeof(std::int32_t) * range_data.size(),
      range_data.data());

  res.emplace_back(CreateSliceOp(input_tensor, outputs[0], range_tensor));
  return res;
}

OpWrapper CreateSliceOp(const TensorWrapper& input_0,
                        const TensorWrapper& output_0,
                        const TensorWrapper& ranges) {
  auto name = GetUniqueOpName(QNN_OP_STRIDED_SLICE);
  OpWrapper op;
  op.SetName(std::move(name));
  op.SetType(QNN_OP_STRIDED_SLICE, QnnOpCode::kStridedSlice);
  op.AddInputTensor(input_0);
  op.AddOutputTensor(output_0);
  op.AddTensorParam(QNN_OP_STRIDED_SLICE_PARAM_RANGES, ranges);
  return op;
}

}  // namespace qnn
