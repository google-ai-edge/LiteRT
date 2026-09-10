// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_TRANSPOSE_CONV3D_OP_BUILDER_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_TRANSPOSE_CONV3D_OP_BUILDER_H_

#include <cstdint>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

std::vector<OpWrapper> BuildTransposeConv3dOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, const std::uint32_t stride_d,
    const std::uint32_t stride_h, const std::uint32_t stride_w,
    const std::uint32_t dilation_d, const std::uint32_t dilation_h,
    const std::uint32_t dilation_w, const PaddingType padding_type);

OpWrapper CreateTransposeConv3dOp(const TensorWrapper& input,
                                  const TensorWrapper& filter,
                                  const TensorWrapper* bias,
                                  const TensorWrapper& output,
                                  const TensorWrapper& stride,
                                  const TensorWrapper& pad_amount,
                                  const TensorWrapper& dilation);

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_TRANSPOSE_CONV3D_OP_BUILDER_H_
