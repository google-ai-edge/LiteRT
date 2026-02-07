// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "litert/vendors/qualcomm/core/builders/select_op_builder.h"

#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnOpDef.h"  // from @qairt

namespace qnn {

OpWrapper CreateSelectOp(const TensorWrapper& input_0,
                         const TensorWrapper& input_1,
                         const TensorWrapper& input_2,
                         const TensorWrapper& output_0) {
  auto name = GetUniqueOpName(QNN_OP_ELEMENT_WISE_SELECT);
  OpWrapper op;
  op.SetName(std::move(name));
  op.SetType(QNN_OP_ELEMENT_WISE_SELECT, QnnOpCode::kElementWiseSelect);
  op.AddInputTensor(input_0);
  op.AddInputTensor(input_1);
  op.AddInputTensor(input_2);
  op.AddOutputTensor(output_0);
  return op;
}

}  // namespace qnn
