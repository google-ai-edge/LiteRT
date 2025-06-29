// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/builders/gathernd_op_builder.h"

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

std::vector<OpWrapper> BuildGatherNdOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs,
    const std::uint32_t batch_dims) {
  std::vector<OpWrapper> res;

  OpWrapper& gathernd_op = CreateOpWrapper(res, QNN_OP_GATHER_ND);
  for (const auto& input : inputs) {
    gathernd_op.AddInputTensor(input);
  }
  gathernd_op.AddOutputTensor(outputs[0]);
  gathernd_op.AddScalarParam<std::uint32_t>(QNN_OP_GATHER_ND_PARAM_BATCH_DIMS,
                                            batch_dims);

  return res;
}

}  // namespace qnn
