// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_TRANSPOSE_OP_BUILDER_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_TRANSPOSE_OP_BUILDER_H_

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

std::vector<OpWrapper> BuildTransposeOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs);

OpWrapper CreateTransposeOp(const TensorWrapper& input_0,
                            const TensorWrapper& output_0,
                            const TensorWrapper& perm);

OpWrapper CreateTransposeOpWithSameParam(const OpWrapper& src,
                                        const TensorWrapper& input_0,
                                        const TensorWrapper& output_0);

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_TRANSPOSE_OP_BUILDER_H_
