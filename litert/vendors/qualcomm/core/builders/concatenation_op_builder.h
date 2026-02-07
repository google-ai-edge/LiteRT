// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_CONCATENATION_OP_BUILDER_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_CONCATENATION_OP_BUILDER_H_

#include <cstdint>
#include <vector>

#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

OpWrapper CreateConcatenationOp(
    const std::vector<ConstTensorWrapperRef>& inputs,
    const TensorWrapper& output, std::uint32_t axis);

OpWrapper CreateConcatenationOpWithSameParam(
    const OpWrapper& src, const std::vector<ConstTensorWrapperRef>& inputs,
    const TensorWrapper& output);

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_CONCATENATION_OP_BUILDER_H_
