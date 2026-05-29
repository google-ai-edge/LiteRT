// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnOpDef.h"  // from @qairt

namespace qnn {

OpWrapper CreateCustomOp(const char* package_name, const char* op_type,
                         const std::vector<ConstTensorWrapperRef>& inputs,
                         const std::vector<ConstTensorWrapperRef>& outputs) {
  OpWrapper op(GetUniqueOpName(op_type), package_name, op_type,
               QnnOpCode::kUnknown);
  for (const auto& input : inputs) {
    op.AddInputTensor(input);
  }
  for (const auto& output : outputs) {
    op.AddOutputTensor(output);
  }
  return op;
}

}  // namespace qnn
