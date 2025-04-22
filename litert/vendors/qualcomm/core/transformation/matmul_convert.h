// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_TRANSFORMATION_MATMUL_CONVERT_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_TRANSFORMATION_MATMUL_CONVERT_H_

#include <vector>

#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"

namespace qnn {

bool FuseMatMulConvertDecode(std::vector<OpWrapper>& ops, size_t start_id,
                             TensorPool& tensor_pool, size_t pattern_size);

bool FuseMatMulConvertPrefill(std::vector<OpWrapper>& ops, size_t start_id,
                              TensorPool& tensor_pool, size_t pattern_size);

}  // namespace qnn
#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_TRANSFORMATION_MATMUL_CONVERT_H_
