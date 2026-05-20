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

#include <cstdint>
#include <string>

#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_op_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/vendors/samsung/compiler/builders/op_wrapper.h"
#include "litert/vendors/samsung/compiler/builders/utils.h"

namespace litert::samsung {

Expected<OpWrapper> BuildElementwiseOp(const litert::compiler::Op& op,
                                       const std::string& type,
                                       uint32_t tfl_fused_activation) {
  OpWrapper op_wrapper(type);

  for (const auto& input : op.Inputs()) {
    op_wrapper.AddInput(input);
  }
  for (const auto& output : op.Outputs()) {
    op_wrapper.AddOutput(output);
  }
  auto activation = GetFusedActivationName(tfl_fused_activation);
  if (!activation) {
    return activation.Error();
  }
  op_wrapper.AddParam("activation", *activation);

  return op_wrapper;
}

Expected<OpWrapper> BuildAddOp(const LiteRtCompilerContext* ctx,
                               const litert::compiler::Op& op) {
  uint32_t tfl_fused_activation;
  if (auto status =
          ctx->get_add_fused_activation_option(op.Get(), &tfl_fused_activation);
      status != kLiteRtStatusOk) {
    return Error(static_cast<litert::Status>(status),
                 "Fail to get fused activation");
  }

  return BuildElementwiseOp(op, "ADD", tfl_fused_activation);
}

Expected<OpWrapper> BuildMulOp(const LiteRtCompilerContext* ctx,
                               const litert::compiler::Op& op) {
  uint32_t tfl_fused_activation;
  if (auto status =
          ctx->get_mul_fused_activation_option(op.Get(), &tfl_fused_activation);
      status != kLiteRtStatusOk) {
    return Error(status, "Fail to get fused activation");
  }

  return BuildElementwiseOp(op, "MUL", tfl_fused_activation);
}

Expected<OpWrapper> BuildDivOp(const LiteRtCompilerContext* ctx,
                               const litert::compiler::Op& op) {
  uint32_t tfl_fused_activation;
  if (auto status =
          ctx->get_div_fused_activation_option(op.Get(), &tfl_fused_activation);
      status != kLiteRtStatusOk) {
    return Error(status, "Fail to get fused activation");
  }

  return BuildElementwiseOp(op, "DIV", tfl_fused_activation);
}

Expected<OpWrapper> BuildExpOp(const LiteRtCompilerContext* ctx,
                               const litert::compiler::Op& op) {
  return BuildElementwiseOp(op, "EXP", 0);
}

Expected<OpWrapper> BuildGreaterOp(const LiteRtCompilerContext* ctx,
                                   const litert::compiler::Op& op) {
  return BuildElementwiseOp(op, "Greater", 0);
}

Expected<OpWrapper> BuildGreaterEqualOp(const LiteRtCompilerContext* ctx,
                                        const litert::compiler::Op& op) {
  return BuildElementwiseOp(op, "GreaterOrEqual", 0);
}

Expected<OpWrapper> BuildMaxOp(const LiteRtCompilerContext* ctx,
                               const litert::compiler::Op& op) {
  return BuildElementwiseOp(op, "Maximum", 0);
}

Expected<OpWrapper> BuildMinOp(const LiteRtCompilerContext* ctx,
                               const litert::compiler::Op& op) {
  return BuildElementwiseOp(op, "Minimum", 0);
}

Expected<OpWrapper> BuildCosOp(const LiteRtCompilerContext* ctx,
                               const litert::compiler::Op& op) {
  return BuildElementwiseOp(op, "Cos", 0);
}

Expected<OpWrapper> BuildSinOp(const LiteRtCompilerContext* ctx,
                               const litert::compiler::Op& op) {
  return BuildElementwiseOp(op, "Sin", 0);
}

Expected<OpWrapper> BuildSubOp(const LiteRtCompilerContext* ctx,
                               const litert::compiler::Op& op) {
  uint32_t tfl_fused_activation;
  if (auto status =
          ctx->get_sub_fused_activation_option(op.Get(), &tfl_fused_activation);
      status != kLiteRtStatusOk) {
    return Error(status, "Fail to get fused activation");
  }

  return BuildElementwiseOp(op, "SUB", tfl_fused_activation);
}

Expected<OpWrapper> BuildSqrtOp(const LiteRtCompilerContext* ctx,
                                const litert::compiler::Op& op) {
  return BuildElementwiseOp(op, "SQRT", 0);
}

Expected<OpWrapper> BuildRsqrtOp(const LiteRtCompilerContext* ctx,
                                 const litert::compiler::Op& op) {
  return BuildElementwiseOp(op, "RSQRT", 0);
}

Expected<OpWrapper> BuildSquaredDifferenceOp(const LiteRtCompilerContext* ctx,
                                             const litert::compiler::Op& op) {
  return BuildElementwiseOp(op, "SquaredDifference", 0);
}

Expected<OpWrapper> BuildAbsOp(const LiteRtCompilerContext* ctx,
                               const litert::compiler::Op& op) {
  return BuildElementwiseOp(op, "ABS", 0);
}

Expected<OpWrapper> BuildEqualOp(const LiteRtCompilerContext* ctx,
                                 const litert::compiler::Op& op) {
  return BuildElementwiseOp(op, "Equal", 0);
}

Expected<OpWrapper> BuildCeilOp(const LiteRtCompilerContext* ctx,
                                const litert::compiler::Op& op) {
  return BuildElementwiseOp(op, "Ceil", 0);
}

Expected<OpWrapper> BuildFloorOp(const LiteRtCompilerContext* ctx,
                                 const litert::compiler::Op& op) {
  return BuildElementwiseOp(op, "Floor", 0);
}

Expected<OpWrapper> BuildFloorDivOp(const LiteRtCompilerContext* ctx,
                                    const litert::compiler::Op& op) {
  return BuildElementwiseOp(op, "FloorDiv", 0);
}

Expected<OpWrapper> BuildLessOp(const LiteRtCompilerContext* ctx,
                                const litert::compiler::Op& op) {
  return BuildElementwiseOp(op, "Less", 0);
}

Expected<OpWrapper> BuildLogOp(const LiteRtCompilerContext* ctx,
                               const litert::compiler::Op& op) {
  return BuildElementwiseOp(op, "Log", 0);
}

Expected<OpWrapper> BuildPowOp(const LiteRtCompilerContext* ctx,
                               const litert::compiler::Op& op) {
  return BuildElementwiseOp(op, "Pow", 0);
}

Expected<OpWrapper> BuildLogicalAndOp(const LiteRtCompilerContext* ctx,
                                      const litert::compiler::Op& op) {
  return BuildElementwiseOp(op, "And", 0);
}

Expected<OpWrapper> BuildNotEqualOp(const LiteRtCompilerContext* ctx,
                                    const litert::compiler::Op& op) {
  return BuildElementwiseOp(op, "NotEqual", 0);
}

}  // namespace litert::samsung
