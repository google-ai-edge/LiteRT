// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "litert/vendors/qualcomm/core/builders/softmax_op_builder.h"

#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnOpDef.h"  // from @qairt

namespace qnn {

OpWrapper CreateSoftmaxOp(const TensorWrapper& input_0,
                          const TensorWrapper& output_0, float beta) {
  auto name = GetUniqueOpName(QNN_OP_SOFTMAX);
  OpWrapper op;
  op.SetName(std::move(name));
  op.SetType(QNN_OP_SOFTMAX, QnnOpCode::kSoftmax);
  op.AddInputTensor(input_0);
  op.AddOutputTensor(output_0);
  op.AddScalarParam<float>(QNN_OP_SOFTMAX_PARAM_BETA, beta);
  return op;
}

OpWrapper CreateSoftmaxOpWithSameParam(const OpWrapper& src,
                                       const TensorWrapper& input_0,
                                       const TensorWrapper& output_0) {
  auto name = GetUniqueOpName(QNN_OP_SOFTMAX);
  OpWrapper op(src);
  op.SetName(std::move(name));
  op.AddInputTensor(input_0);
  op.AddOutputTensor(output_0);
  return op;
}

}  // namespace qnn
