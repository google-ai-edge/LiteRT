// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_HADAMARD_TRANSFORM_OP_BUILDER_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_HADAMARD_TRANSFORM_OP_BUILDER_H_

#include <cstdint>
#include <vector>

#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

std::vector<OpWrapper> BuildHadamardTransformOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, float scale = 1);

// Returns the corresponding scale factor if the given static tensor forms a
// valid Sylvester Hadamard matrix; otherwise returns std::nullopt.
std::optional<float> GetSylvesterHadamardScale(const TensorWrapper& weight);

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_HADAMARD_TRANSFORM_OP_BUILDER_H_
