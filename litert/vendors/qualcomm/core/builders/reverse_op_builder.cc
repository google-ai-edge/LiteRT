// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "litert/vendors/qualcomm/core/builders/reverse_op_builder.h"

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

std::vector<OpWrapper> BuildReverseOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  // We use StridedSlice with stride = -1, end = -1 to mimic the effect of
  // reverse_v2
  constexpr int range_num_elements{3};

  const TensorWrapper& input_tensor = inputs[0];
  const TensorWrapper& axis_tensor = inputs[1];

  // axis tensor must be static because StridedSlice require begin/end static.
  if (!axis_tensor.IsTensorStatic()) {
    QNN_LOG_ERROR("ReverseV2 axis tensor must be static.");
    return {};
  }

  // TODO: Support more axis and remove this check
  if (axis_tensor.GetRank() != 1 || axis_tensor.GetDims()[0] != 1) {
    QNN_LOG_ERROR("Qnn supports ReverseV2 with a single axis for now.");
    return {};
  }

  if (axis_tensor.GetDataType() != QNN_DATATYPE_INT_32) {
    QNN_LOG_ERROR("ReverseV2 axis tensor must be int32 datatype.");
    return {};
  }

  auto axis_values = axis_tensor.GetTensorData<std::int32_t>();
  if (!axis_values) {
    QNN_LOG_ERROR("ReverseV2 axis tensor not contain static tensor.");
    return {};
  }
  std::int32_t axis_value = (*axis_values)[0];
  const auto input_rank = input_tensor.GetRank();

  if (axis_value >= static_cast<std::int32_t>(input_rank)) {
    QNN_LOG_ERROR("ReverseV2 axis_value %d larger than input rank %u",
                  axis_value, input_rank);
    return {};
  }

  if (axis_value < 0) {
    axis_value = static_cast<std::int32_t>(axis_value + input_rank);
  }

  // For the axis which is reversed, use (begin, end, stride) = (dim[axis]-1,
  // -1, -1) Otherwise, use (0, dim[axis], 1)
  auto& input_dims = input_tensor.GetDims();
  std::vector<std::int32_t> ranges(
      static_cast<uint32_t>(input_rank * range_num_elements));
  for (size_t i = 0; i < input_rank; ++i) {
    // ranges is organized as [(being0, end0, stride0), (begin1, end1, stride1),
    // ...]
    ranges[i * range_num_elements] = 0;
    ranges[i * range_num_elements + 1] = input_dims[i];
    ranges[i * range_num_elements + 2] = 1;
  }
  ranges[axis_value * range_num_elements] = input_dims[axis_value] - 1;
  ranges[axis_value * range_num_elements + 1] = -1;
  ranges[axis_value * range_num_elements + 2] = -1;

  const TensorWrapper& range_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {input_rank, range_num_elements},
      sizeof(ranges[0]) * ranges.size(), ranges.data());

  auto& slice_op = CreateOpWrapper(res, QNN_OP_STRIDED_SLICE);
  slice_op.AddTensorParam(QNN_OP_STRIDED_SLICE_PARAM_RANGES, range_tensor);
  slice_op.AddInputTensor(inputs[0]);
  slice_op.AddOutputTensor(outputs[0]);

  return res;
}

}  // namespace qnn
