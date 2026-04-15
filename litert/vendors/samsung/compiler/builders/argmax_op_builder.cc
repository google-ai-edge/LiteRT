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

#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_model.h"

#include "litert/c/litert_op_options.h"
#include "litert/vendors/samsung/compiler/builders/argmax_op_builder.h"
#include "litert/vendors/samsung/compiler/builders/utils.h"

namespace litert::samsung {

constexpr int32_t kInputIndex = 0;
constexpr int32_t kAxisIndex = 1;
constexpr int32_t kOutputIndex = 0;

Expected<OpWrapper> BuildArgMaxOp(const Op& op) {
  OpWrapper op_wrapper("ArgMax");

  op_wrapper.AddInput(op.Inputs()[kInputIndex]);
  op_wrapper.AddOutput(op.Outputs()[kOutputIndex]);

  LITERT_ASSIGN_OR_RETURN(auto axis,
                          GetWeightDataAs<int32_t>(op.Inputs()[kAxisIndex]));
  if (axis.size() != 1) {
    return Error(
        kLiteRtStatusErrorUnsupported,
        absl::StrCat("Doesn't support argmax axis size : except 1, but get ",
                     axis.size()));
  }
  op_wrapper.AddParam("axis", axis[0]);

  return op_wrapper;
}
} // namespace litert::samsung
