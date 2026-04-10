// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_CUSTOM_OP_BUILDER_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_CUSTOM_OP_BUILDER_H_

#include <vector>

#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

OpWrapper CreateCustomOp(const char* package_name, const char* op_type,
                         const std::vector<TensorWrapperRef>& inputs,
                         const std::vector<TensorWrapperRef>& outputs);

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_CUSTOM_OP_BUILDER_H_