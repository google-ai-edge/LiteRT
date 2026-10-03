// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "litert/vendors/qualcomm/core/builders/local_response_norm_op_builder.h"

#include <cstdint>
#include <vector>

#include "QnnOpDef.h"  // from @qairt
#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

OpWrapper CreateLocalResponseNormOp(const TensorWrapper& input,
                                    const TensorWrapper& output,
                                    std::int32_t radius, float bias,
                                    float alpha, float beta) {
  OpWrapper op(GetUniqueOpName(QNN_OP_LRN), QNN_OP_LRN, QnnOpCode::kLrn);
  op.AddInputTensor(input);
  op.AddOutputTensor(output);

  op.AddScalarParam<std::int32_t>(QNN_OP_LRN_PARAM_RADIUS, radius);
  op.AddScalarParam<float>(QNN_OP_LRN_PARAM_BIAS, bias);
  op.AddScalarParam<float>(QNN_OP_LRN_PARAM_ALPHA, alpha);
  op.AddScalarParam<float>(QNN_OP_LRN_PARAM_BETA, beta);

  return op;
}

}  // namespace qnn
