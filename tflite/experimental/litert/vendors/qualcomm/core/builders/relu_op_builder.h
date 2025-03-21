// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_RELU_OP_BUILDER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_RELU_OP_BUILDER_H_

#include <vector>

#include "tflite/experimental/litert/vendors/qualcomm/core/tensor_pool.h"
#include "tflite/experimental/litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "tflite/experimental/litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

std::vector<OpWrapper> BuildReluOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs);

}  // namespace qnn

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_RELU_OP_BUILDER_H_
