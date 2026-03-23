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

#include "litert/vendors/samsung/compiler/builders/softmax_op_builder.h"

#include "litert/c/litert_common.h"
#include "litert/c/litert_op_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_model.h"

namespace litert::samsung {

Expected<OpWrapper> BuildSoftmaxOp(const Op &op) {
  OpWrapper op_wrapper("Softmax");

  for (const auto &input : op.Inputs()) {
    op_wrapper.AddInput(input);
  }
  for (const auto &output : op.Outputs()) {
    op_wrapper.AddOutput(output);
    int32_t axis{};

    auto ranked_tensor_type = output.RankedTensorType();
    if (!ranked_tensor_type) {
      return ranked_tensor_type.Error();
    }

    axis = ranked_tensor_type->Layout().Dimensions().size() - 1;
    op_wrapper.AddParam("axis", axis);
  }

  return op_wrapper;
}
}  // namespace litert::samsung
