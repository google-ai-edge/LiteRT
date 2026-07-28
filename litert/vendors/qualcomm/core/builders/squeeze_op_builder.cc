// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "litert/vendors/qualcomm/core/builders/squeeze_op_builder.h"

#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/builders/op_builder.h"

#include "QnnOpDef.h"

namespace qnn {

OpWrapper CreateSqueezeOp(
    const TensorWrapper& input,
    const TensorWrapper& output,
    const TensorWrapper& squeeze_dims) {
    OpWrapper op(GetUniqueOpName(QNN_OP_SQUEEZE), QNN_OP_SQUEEZE,
                 QnnOpCode::kSqueeze);

    op.AddInputTensor(input);
    op.AddOutputTensor(output);
    op.AddTensorParam(QNN_OP_SQUEEZE_PARAM_AXES, squeeze_dims);
    return op;
}
}  // namespace qnn
