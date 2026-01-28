// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_SOFTMAX_OP_BUILDER_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_SOFTMAX_OP_BUILDER_H_

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

OpWrapper CreateSoftmaxOp(const TensorWrapper& input_0,
                          const TensorWrapper& output_0, float beta);

OpWrapper CreateSoftmaxOpWithSameParam(const OpWrapper& src,
                                       const TensorWrapper& input_0,
                                       const TensorWrapper& output_0);

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_SOFTMAX_OP_BUILDER_H_
