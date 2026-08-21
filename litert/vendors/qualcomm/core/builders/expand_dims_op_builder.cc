// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "litert/vendors/qualcomm/core/builders/expand_dims_op_builder.h"

#include <cstddef>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/builders/reshape_op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

namespace {
constexpr size_t kInputTensorIndex = 0;
constexpr size_t kAxisTensorIndex = 1;
constexpr size_t kOutputTensorIndex = 0;
}  // namespace

std::vector<OpWrapper> BuildExpandDimsOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {

  if (!inputs[kAxisTensorIndex].get().IsTensorStatic()) {
    QNN_LOG_ERROR("ExpandDims axis tensor must be static.");
    return {};
  }

  return MakeVector(CreateReshapeOp(inputs[kInputTensorIndex],
                                    outputs[kOutputTensorIndex]));
}

}  // namespace qnn
