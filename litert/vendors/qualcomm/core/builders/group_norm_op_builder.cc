// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/builders/group_norm_op_builder.h"

#include <cstdint>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnOpDef.h"  // from @qairt

namespace qnn {

std::vector<OpWrapper> BuildGroupNormOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, const float epsilon,
    const uint32_t num_groups) {
  std::vector<OpWrapper> res;

  auto& group_norm_op = CreateOpWrapper(res, QNN_OP_GROUP_NORM);
  for (const auto& input : inputs) {
    group_norm_op.AddInputTensor(input);
  }
  group_norm_op.AddOutputTensor(outputs[0]);

  group_norm_op.AddScalarParam<float>(QNN_OP_GROUP_NORM_PARAM_EPSILON, epsilon);
  group_norm_op.AddScalarParam<uint32_t>(QNN_OP_GROUP_NORM_PARAM_GROUP,
                                              num_groups);

  return res;
}

}  // namespace qnn

