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
#include <string>

#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_model.h"

#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_op_options.h"
#include "litert/vendors/samsung/compiler/builders/mul_op_builder.h"
namespace litert::samsung {

Expected<OpWrapper> BuildMulOp(const Op &op) {
  OpWrapper op_wrapper("", "MUL");

  for (const auto &input : op.Inputs()) {
    op_wrapper.AddInput(input);
  }
  for (const auto &output : op.Outputs()) {
    op_wrapper.AddOutput(output);
  }

  uint32_t tfl_fused_activation;
  if (auto status =
          LiteRtGetMulFusedActivationOption(op.Get(), &tfl_fused_activation);
      status != kLiteRtStatusOk) {
    return Error(status, "Fail to get fused activation");
  }
  if (tfl_fused_activation == 1) {
      op_wrapper.AddParam("activation", "Relu");
  } else if (tfl_fused_activation != 0) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Unsupported fused activation");
  }

  return op_wrapper;
}
} // namespace litert::samsung
