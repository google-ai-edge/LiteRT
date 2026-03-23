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

#include "litert/vendors/samsung/compiler/builders/gather_op_builder.h"

#include "litert/c/litert_op_options.h"

namespace litert::samsung {

constexpr int kOutIndex = 0;

Expected<OpWrapper> BuildGatherOp(const Op &op) {
  OpWrapper op_wrapper("GATHER");
  for (const auto &input : op.Inputs()) {
    op_wrapper.AddInput(input);
  }
  op_wrapper.AddOutput(op.Outputs()[kOutIndex]);

  int32_t axis = 0, batch_dims = 0;
  LITERT_RETURN_IF_ERROR(LiteRtGetGatherAxisOption(op.Get(), &axis));
  LITERT_RETURN_IF_ERROR(LiteRtGetGatherBatchDimsOption(op.Get(), &batch_dims));
  op_wrapper.AddParam("axis", axis);
  op_wrapper.AddParam("batch_dims", batch_dims);

  return op_wrapper;
}
}  // namespace litert::samsung
