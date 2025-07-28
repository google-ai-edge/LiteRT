// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_QNN_MODEL_TEST_UTILS_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_QNN_MODEL_TEST_UTILS_

#include "litert/cc/litert_model.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace litert::qnn {
bool ConvertLiteRtOp(
    litert::Op& op, ::qnn::TensorPool& tensor_pool,
    std::vector<::qnn::TensorWrapperRef>& input_tensors,
    std::vector<::qnn::TensorWrapperRef>& output_tensors,
    std::vector<::qnn::OpWrapper>& op_wrappers, bool use_htp_preference);

}
#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_QNN_MODEL_TEST_UTILS_
