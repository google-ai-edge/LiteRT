// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_PRELU_OP_BUILDER_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_PRELU_OP_BUILDER_H_

#include <vector>

#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

// Builds a Qnn.Prelu op from a LiteRT.Prelu op.
std::vector<OpWrapper> BuildPreluOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs);

}  // namespace qnn

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_PRELU_OP_BUILDER_H_
