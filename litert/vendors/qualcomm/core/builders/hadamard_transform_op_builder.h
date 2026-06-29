// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_HADAMARD_TRANSFORM_OP_BUILDER_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_HADAMARD_TRANSFORM_OP_BUILDER_H_

#include <optional>

#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

OpWrapper CreateHadamardTransformOp(const TensorWrapper& input,
                                    const TensorWrapper& output,
                                    float scale = 1.0f);

// Returns the corresponding scale factor if the given static tensor forms a
// valid Sylvester Hadamard matrix; otherwise returns std::nullopt.
std::optional<float> GetSylvesterHadamardScale(const TensorWrapper& weight);

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BUILDERS_HADAMARD_TRANSFORM_OP_BUILDER_H_
