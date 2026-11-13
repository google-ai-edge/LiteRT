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

#include "litert/vendors/samsung/compiler/builders/dequantize_op_builder.h"

#include "litert/cc/litert_expected.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/vendors/samsung/compiler/builders/op_wrapper.h"
#include "litert/vendors/samsung/compiler/builders/utils.h"

namespace litert::samsung {

Expected<OpWrapper> BuildDequantizeOp(const litert::compiler::Op& op) {
  OpWrapper op_wrapper("DequantizeLinear");

  for (const auto& input : op.Inputs()) {
    op_wrapper.AddInput(input);
  }
  for (const auto& output : op.Outputs()) {
    op_wrapper.AddOutput(output);
  }

  // The input tensor's declared quantization is the source of truth for a
  // DequantizeLinear node: copy it into the node params so they agree with
  // the tensor's quant_info.
  const auto inputs = op.Inputs();
  if (!inputs.empty()) {
    if (auto params = GetQuantizationParams(inputs[0])) {
      op_wrapper.AddParam("x_scale", params->scales);
      op_wrapper.AddParam("x_zero_point", params->zero_points);
    }
  }

  return op_wrapper;
}
}  // namespace litert::samsung
