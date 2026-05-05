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

#include "litert/vendors/samsung/compiler/builders/broadcastto_op_builder.h"

#include "litert/c/litert_op_options.h"
#include "litert/vendors/samsung/compiler/builders/utils.h"

namespace litert::samsung {

constexpr int32_t kInputIndex = 0;
constexpr int32_t kShapeIndex = 1;

Expected<OpWrapper> BuildBroadcastToOp(const Op& op) {
  OpWrapper op_wrapper("Expand");

  op_wrapper.AddInput(op.Inputs()[kInputIndex]);
  for (const auto& output : op.Outputs()) {
    op_wrapper.AddOutput(output);
  }
  LITERT_ASSIGN_OR_RETURN(auto shape,
                          GetWeightDataAs<int64_t>(op.Inputs()[kShapeIndex]));
  op_wrapper.AddParam("shape", shape);

  return op_wrapper;
}
}  // namespace litert::samsung
