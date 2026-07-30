// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_SPATIAL_TRANSFORM_ND_OP_BUILDER_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_SPATIAL_TRANSFORM_ND_OP_BUILDER_H_

#include <vector>

#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

std::vector<OpWrapper> BuildBatchToSpaceNdOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs);

std::vector<OpWrapper> BuildSpaceToBatchNdOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs);

OpWrapper CreateBatchToSpaceNdOp(const TensorWrapper& input,
                                 const TensorWrapper& output,
                                 const TensorWrapper& block_size,
                                 const TensorWrapper& crops);

OpWrapper CreateSpaceToBatchNdOp(const TensorWrapper& input,
                                 const TensorWrapper& output,
                                 const TensorWrapper& block_size,
                                 const TensorWrapper& pad_amount);

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_SPATIAL_TRANSFORM_ND_OP_BUILDER_H_
