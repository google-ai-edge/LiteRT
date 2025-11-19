// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "litert/vendors/qualcomm/core/builders/unpack_op_builder.h"

#include <cstdint>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnOpDef.h"  // from @qairt

namespace qnn {

std::vector<OpWrapper> BuildUnpackOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, const int32_t axis) {
  std::vector<OpWrapper> res;

  auto& unpack_op = CreateOpWrapper(res, QNN_OP_UN_PACK);

  unpack_op.AddInputTensor(inputs[0]);
  uint32_t adjusted_axis = axis < 0 ? axis + inputs[0].get().GetRank() : axis;
  unpack_op.AddScalarParam<uint32_t>(QNN_OP_UN_PACK_PARAM_AXIS, adjusted_axis);
  for (const auto& output : outputs) {
    unpack_op.AddOutputTensor(output);
  }

  return res;
}

}  // namespace qnn
