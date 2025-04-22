// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_TRANSFORMATION_GRAPH_TO_GRAPH_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_TRANSFORMATION_GRAPH_TO_GRAPH_H_

#include <vector>

#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "QnnInterface.h"  // from @qairt

namespace qnn {

enum class G2GConfig {
  // Disable G2G.
  kOff,
  // Enable G2G MatMul-convert fusion.
  kMatMulConvert,
  // Enable G2G MHA optimization for prefill only.
  kMHAOptPrefill,
  // Enable G2G MHA optimization for both decode and prefill.
  kMHAOpt,
};

void GraphToGraphTransform(const G2GConfig g2g_option,
                           std::vector<OpWrapper>& ops, TensorPool& tensor_pool,
                           std::function<bool(OpWrapper&)> validate_op_config);
}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_TRANSFORMATION_GRAPH_TO_GRAPH_H_
