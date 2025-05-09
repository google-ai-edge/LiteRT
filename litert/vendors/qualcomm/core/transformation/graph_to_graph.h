// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_TRANSFORMATION_GRAPH_TO_GRAPH_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_TRANSFORMATION_GRAPH_TO_GRAPH_H_

#include <vector>

#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
namespace qnn {
void GraphToGraphTransform(std::vector<OpWrapper>& ops,
                           TensorPool& tensor_pool);
}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_TRANSFORMATION_GRAPH_TO_GRAPH_H_
