// Copyright (C) 2026 Samsung Electronics Co. LTD.
// SPDX-License-Identifier: Apache-2.0
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
#include "litert/vendors/samsung/compiler/builders/elementwise_op_builder.h"

#include <string>

#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_op_options.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/samsung/compiler/builders/utils.h"

namespace litert::samsung {

Expected<OpWrapper> BuildElementwiseOp(const Op &op, const std::string &type,
                                       uint32_t tfl_fused_activation) {
  OpWrapper op_wrapper(type);

  for (const auto &input : op.Inputs()) {
    op_wrapper.AddInput(input);
  }
  for (const auto &output : op.Outputs()) {
    op_wrapper.AddOutput(output);
  }
  auto activation = GetFusedActivationName(tfl_fused_activation);
  if (!activation) {
    return activation.Error();
  }
  op_wrapper.AddParam("activation", *activation);

  return op_wrapper;
}

Expected<OpWrapper> BuildAddOp(const Op &op) {
  uint32_t tfl_fused_activation;
  if (auto status =
          LiteRtGetAddFusedActivationOption(op.Get(), &tfl_fused_activation);
      status != kLiteRtStatusOk) {
    return Error(static_cast<litert::Status>(status),
                 "Fail to get fused activation");
  }

  return BuildElementwiseOp(op, "ADD", tfl_fused_activation);
}

Expected<OpWrapper> BuildMulOp(const Op &op) {
  uint32_t tfl_fused_activation;
  if (auto status =
          LiteRtGetMulFusedActivationOption(op.Get(), &tfl_fused_activation);
      status != kLiteRtStatusOk) {
    return Error(status, "Fail to get fused activation");
  }

  return BuildElementwiseOp(op, "MUL", tfl_fused_activation);
}

Expected<OpWrapper> BuildDivOp(const Op &op) {
  uint32_t tfl_fused_activation;
  if (auto status =
          LiteRtGetDivFusedActivationOption(op.Get(), &tfl_fused_activation);
      status != kLiteRtStatusOk) {
    return Error(status, "Fail to get fused activation");
  }

  return BuildElementwiseOp(op, "DIV", tfl_fused_activation);
}

Expected<OpWrapper> BuildExpOp(const Op &op) {
  return BuildElementwiseOp(op, "EXP", 0);
}

Expected<OpWrapper> BuildMaxOp(const Op &op) {
  return BuildElementwiseOp(op, "Maximum", 0);
}

Expected<OpWrapper> BuildMinOp(const Op &op) {
  return BuildElementwiseOp(op, "Minimum", 0);
}

Expected<OpWrapper> BuildSubOp(const Op &op) {
  uint32_t tfl_fused_activation;
  if (auto status =
          LiteRtGetSubFusedActivationOption(op.Get(), &tfl_fused_activation);
      status != kLiteRtStatusOk) {
    return Error(status, "Fail to get fused activation");
  }

  return BuildElementwiseOp(op, "SUB", tfl_fused_activation);
}

Expected<OpWrapper> BuildSqrtOp(const Op &op) {
  return BuildElementwiseOp(op, "SQRT", 0);
}

Expected<OpWrapper> BuildRsqrtOp(const Op &op) {
  return BuildElementwiseOp(op, "RSQRT", 0);
}

Expected<OpWrapper> BuildSquaredDifferenceOp(const Op &op) {
  return BuildElementwiseOp(op, "SquaredDifference", 0);
}

}  // namespace litert::samsung
