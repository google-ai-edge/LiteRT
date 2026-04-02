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

#include "litert/vendors/samsung/compiler/builders/cast_op_builder.h"

#include "litert/c/litert_op_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/samsung/compiler/builders/utils.h"

namespace litert::samsung {
constexpr int32_t kInputIndex = 0;
constexpr int32_t kOutputIndex = 0;

Expected<OpWrapper> BuildCastOp(const Op& op) {
  OpWrapper op_wrapper("CAST");

  op_wrapper.AddInput(op.Inputs()[kInputIndex]);
  auto output = std::move(op.Outputs()[kOutputIndex]);
  op_wrapper.AddOutput(output);

  auto ranked_tensor_type = output.RankedTensorType();
  if (!ranked_tensor_type) {
    return ranked_tensor_type.Error();
  }

  auto element_type = ranked_tensor_type->ElementType();
  LITERT_ASSIGN_OR_RETURN(auto element_type_mapping,
                          MapToElementTypeStr(element_type));
  op_wrapper.AddParam("to", element_type_mapping);

  return op_wrapper;
}
}  // namespace litert::samsung
