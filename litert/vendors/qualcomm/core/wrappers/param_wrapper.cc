// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/wrappers/param_wrapper.h"

#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "include/QNN/QnnTypes.h"  // from @qairt

namespace qnn {

void ScalarParamWrapper::CloneTo(Qnn_Param_t& dst) const {
  dst.name = name_;
  dst.paramType = QNN_PARAMTYPE_SCALAR;
  dst.scalarParam = qnn_scalar_;
}

TensorParamWrapper::TensorParamWrapper(const char* name,
                                       const TensorWrapper& tensor)
    : name_{name}, tensor_{tensor} {}

void TensorParamWrapper::CloneTo(Qnn_Param_t& dst) const {
  dst.name = name_;
  dst.paramType = QNN_PARAMTYPE_TENSOR;
  tensor_.CloneTo(dst.tensorParam);
}

}  // namespace qnn
