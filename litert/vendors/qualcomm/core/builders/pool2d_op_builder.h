// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_POOL2D_OP_BUILDER_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_POOL2D_OP_BUILDER_H_

#include <cstdint>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

std::vector<OpWrapper> BuildMaxPoolOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs,
    const std::uint32_t stride_height, const std::uint32_t stride_width,
    const std::uint32_t filter_height, const std::uint32_t filter_width,
    const PaddingType padding_type);

std::vector<OpWrapper> BuildAveragePoolOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs,
    const std::uint32_t stride_height, const std::uint32_t stride_width,
    const std::uint32_t filter_height, const std::uint32_t filter_width,
    const PaddingType padding_type);

std::vector<OpWrapper> BuildL2PoolOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs,
    const std::uint32_t stride_height, const std::uint32_t stride_width,
    const std::uint32_t filter_height, const std::uint32_t filter_width,
    const PaddingType padding_type);

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_POOL2D_OP_BUILDER_H_
