// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/builders/squeeze_op_builder.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnOpDef.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt

namespace qnn {

std::vector<OpWrapper> BuildSqueezeOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs,
    const std::vector<int32_t>& squeeze_dims) {
  OpWrapper op(GetUniqueOpName(QNN_OP_SQUEEZE), QNN_OP_SQUEEZE,
               QnnOpCode::kSqueeze);
  op.AddInputTensor(inputs[0]);
  op.AddOutputTensor(outputs[0]);

  if (!squeeze_dims.empty()) {
    const std::uint32_t input_rank = inputs[0].get().GetRank();
    std::vector<std::uint32_t> resolved_dims;
    resolved_dims.reserve(squeeze_dims.size());
    for (int32_t dim : squeeze_dims) {
      resolved_dims.emplace_back(
          dim < 0 ? static_cast<std::uint32_t>(dim + input_rank)
                  : static_cast<std::uint32_t>(dim));
    }
    TensorWrapper& axes_tensor = tensor_pool.CreateStaticTensor(
        QNN_DATATYPE_UINT_32, {},
        {static_cast<std::uint32_t>(resolved_dims.size())},
        sizeof(resolved_dims[0]) * resolved_dims.size(), resolved_dims.data());
    op.AddTensorParam(QNN_OP_SQUEEZE_PARAM_AXES, axes_tensor);
  }

  return MakeVector(std::move(op));
}

}  // namespace qnn
