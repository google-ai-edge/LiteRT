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
#include "litert/vendors/samsung/compiler/builders/rms_norm_op_builder.h"

#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/internal/litert_op_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/samsung/compiler/builders/utils.h"

namespace litert::samsung {

constexpr int kInputIndex = 0;

Expected<OpWrapper> BuildRmsNormOp(const Op& op) {
  OpWrapper op_wrapper("RMSNORM");

  for (const auto& input : op.Inputs()) {
    op_wrapper.AddInput(input);
  }
  for (const auto& output : op.Outputs()) {
    op_wrapper.AddOutput(output);
  }

  auto input_dimensions = GetDimensions(op.Inputs()[kInputIndex]);

  op_wrapper.AddParam("axis", input_dimensions.size() - 1);

  LITERT_ASSIGN_OR_RETURN(auto rms_norm_options,
                          GetOptionsAs<RmsNormOpts>(op.Get()));
  op_wrapper.AddParam("epsilon", rms_norm_options.epsilon);

  return op_wrapper;
}

} // namespace litert::samsung
