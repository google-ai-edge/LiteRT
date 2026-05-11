// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "litert/vendors/qualcomm/core/builders/cast_op_builder.h"

#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnOpDef.h"  // from @qairt

namespace qnn {

OpWrapper CreateCastOp(const TensorWrapper& input,
                       const TensorWrapper& output) {
  OpWrapper op(GetUniqueOpName(QNN_OP_CAST), QNN_OP_CAST, QnnOpCode::kCast);
  op.AddInputTensor(input);
  op.AddOutputTensor(output);
  return op;
}

}  // namespace qnn
