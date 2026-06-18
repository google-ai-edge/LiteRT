// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "litert/vendors/qualcomm/core/builders/matmul_op_builder.h"

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnOpDef.h"  // from @qairt

namespace qnn {

OpWrapper CreateMatmulOp(const TensorWrapper& input_0,
                         const TensorWrapper& input_1,
                         const TensorWrapper& output_0, bool transpose_in0,
                         bool transpose_in1) {
  OpWrapper op(GetUniqueOpName(QNN_OP_MAT_MUL), QNN_OP_MAT_MUL,
               QnnOpCode::kMatMul);
  op.AddInputTensor(input_0);
  op.AddInputTensor(input_1);
  op.AddOutputTensor(output_0);
  op.AddScalarParam<bool>(QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0, transpose_in0);
  op.AddScalarParam<bool>(QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1, transpose_in1);
  return op;
}

}  // namespace qnn
