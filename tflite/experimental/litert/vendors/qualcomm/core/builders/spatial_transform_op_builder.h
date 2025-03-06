// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_SPATIAL_TRANSFORM_OP_BUILDER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_SPATIAL_TRANSFORM_OP_BUILDER_H_

#include <cstdint>
#include <vector>

#include "tflite/experimental/litert/vendors/qualcomm/core/tensor_pool.h"
#include "tflite/experimental/litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "tflite/experimental/litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

std::vector<OpWrapper> BuildDepthToSpaceOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs,
    const std::uint32_t block_size);

std::vector<OpWrapper> BuildSpaceToDepthOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs,
    const std::uint32_t block_size);

}  // namespace qnn

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_SPATIAL_TRANSFORM_OP_BUILDER_H_
