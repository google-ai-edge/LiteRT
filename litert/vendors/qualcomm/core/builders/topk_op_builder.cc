// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/builders/topk_op_builder.h"

#include <cstdint>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnOpDef.h"  // from @qairt

namespace qnn {

namespace {
constexpr bool kIsLargest = true;

}  // namespace

std::vector<OpWrapper> BuildTopKOp(TensorPool& tensor_pool,
                                   const std::vector<TensorWrapperRef>& inputs,
                                   const std::vector<TensorWrapperRef>& outputs,
                                   std::uint32_t k) {
  std::vector<OpWrapper> res;

  auto& topk_op = CreateOpWrapper(res, QNN_OP_TOP_K);
  topk_op.AddInputTensor(inputs[0]);
  for (const auto& output : outputs) {
    topk_op.AddOutputTensor(output);
  }
  topk_op.AddScalarParam<std::uint32_t>(QNN_OP_TOP_K_PARAM_K, k);
  topk_op.AddScalarParam<bool>(QNN_OP_TOP_K_PARAM_LARGEST, kIsLargest);

  return res;
}

}  // namespace qnn
