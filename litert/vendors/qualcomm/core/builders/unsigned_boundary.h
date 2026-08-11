// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_UNSIGNED_BOUNDARY_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_UNSIGNED_BOUNDARY_H_

#include <vector>

#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"

namespace qnn {

// `ops` is modified in place and grows by the number of inserted Converts.
void InsertUnsignedActivationBoundaries(BackendType backend,
                                        TensorPool& tensor_pool,
                                        std::vector<OpWrapper>& ops);

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_UNSIGNED_BOUNDARY_H_
