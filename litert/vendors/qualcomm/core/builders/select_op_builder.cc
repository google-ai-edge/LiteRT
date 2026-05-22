// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "litert/vendors/qualcomm/core/builders/select_op_builder.h"

#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnOpDef.h"  // from @qairt

namespace qnn {

OpWrapper CreateSelectOp(const TensorWrapper& condition,
                         const TensorWrapper& input_1,
                         const TensorWrapper& input_2,
                         const TensorWrapper& output) {
  OpWrapper op(GetUniqueOpName(QNN_OP_ELEMENT_WISE_SELECT),
               QNN_OP_ELEMENT_WISE_SELECT, QnnOpCode::kElementWiseSelect);
  op.AddInputTensor(condition);
  op.AddInputTensor(input_1);
  op.AddInputTensor(input_2);
  op.AddOutputTensor(output);
  return op;
}

}  // namespace qnn
