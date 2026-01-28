// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "litert/vendors/qualcomm/core/builders/cast_op_builder.h"

#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnOpDef.h"  // from @qairt

namespace qnn {

OpWrapper CreateCastOp(const TensorWrapper& input_0,
                       const TensorWrapper& output_0) {
  auto name = GetUniqueOpName(QNN_OP_CAST);
  OpWrapper op;
  op.SetName(std::move(name));
  op.SetType(QNN_OP_CAST, QnnOpCode::kCast);
  op.AddInputTensor(input_0);
  op.AddOutputTensor(output_0);
  return op;
}

}  // namespace qnn
