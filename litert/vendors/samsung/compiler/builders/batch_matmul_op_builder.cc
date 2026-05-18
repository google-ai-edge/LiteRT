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

#include "litert/vendors/samsung/compiler/builders/batch_matmul_op_builder.h"

#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_op_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/vendors/samsung/compiler/builders/op_wrapper.h"

namespace litert::samsung {

Expected<OpWrapper> BuildBatchMatMulOp(const LiteRtCompilerContext* ctx,
                                       const litert::compiler::Op& op) {
  OpWrapper op_wrapper("BATCH_MATMUL");

  for (const auto& input : op.Inputs()) {
    op_wrapper.AddInput(input);
  }
  for (const auto& output : op.Outputs()) {
    op_wrapper.AddOutput(output);
  }
  bool adj_x = false, adj_y = false, asymmetric_quantize_input = false;
  LITERT_RETURN_IF_ERROR(ctx->get_batch_matmul_adj_x_option(op.Get(), &adj_x));
  LITERT_RETURN_IF_ERROR(ctx->get_batch_matmul_adj_y_option(op.Get(), &adj_y));
  LITERT_RETURN_IF_ERROR(ctx->get_batch_matmul_asymmetric_quantize_input_option(
      op.Get(), &asymmetric_quantize_input));

  op_wrapper.AddParam("adj_x", adj_x);
  op_wrapper.AddParam("adj_y", adj_y);
  op_wrapper.AddParam("asymmetric_quantize_input", asymmetric_quantize_input);
  return op_wrapper;
}
}  // namespace litert::samsung
