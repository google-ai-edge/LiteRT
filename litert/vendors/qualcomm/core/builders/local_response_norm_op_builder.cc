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

#include "litert/vendors/qualcomm/core/builders/local_response_norm_op_builder.h"

#include <cstdint>
#include <vector>

#include "QnnOpDef.h"  // from @qairt
#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

namespace {
constexpr int kInputIndex = 0;
constexpr int kOutputIndex = 0;
}  // namespace

std::vector<OpWrapper> BuildLocalResponseNormOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, std::int32_t radius,
    float bias, float alpha, float beta) {
  if (inputs.size() != 1 || outputs.size() != 1) {
    QNN_LOG_ERROR(
        "LocalResponseNorm op expects exactly one input and one output.");
    return {};
  }

  return MakeVector(CreateLocalResponseNormOp(
      inputs[kInputIndex], outputs[kOutputIndex], radius, bias, alpha, beta));
}

OpWrapper CreateLocalResponseNormOp(const TensorWrapper& input,
                                    const TensorWrapper& output,
                                    std::int32_t radius, float bias,
                                    float alpha, float beta) {
  OpWrapper op(GetUniqueOpName(QNN_OP_LRN), QNN_OP_LRN, QnnOpCode::kLrn);
  op.AddInputTensor(input);
  op.AddOutputTensor(output);

  op.AddScalarParam<std::int32_t>(QNN_OP_LRN_PARAM_RADIUS, radius);
  op.AddScalarParam<float>(QNN_OP_LRN_PARAM_BIAS, bias);
  op.AddScalarParam<float>(QNN_OP_LRN_PARAM_ALPHA, alpha);
  op.AddScalarParam<float>(QNN_OP_LRN_PARAM_BETA, beta);

  return op;
}

}  // namespace qnn
