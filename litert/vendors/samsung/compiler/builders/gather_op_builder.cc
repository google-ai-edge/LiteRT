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

Expected<OpWrapper> BuildGeneralGatherOp(
    const Op& op, const std::vector<int32_t>& input_order, int32_t& axis) {
  OpWrapper op_wrapper("GATHER");
  for (const auto& index : input_order) {
    op_wrapper.AddInput(op.Inputs()[index]);
  }
  op_wrapper.AddOutput(op.Outputs()[kOutIndex]);
  op_wrapper.AddParam("axis", axis);

  return op_wrapper;
}

Expected<OpWrapper> BuildGatherOp(const Op& op) {
  int32_t axis = 0;
  std::vector<int32_t> gather_input_order = {0, 1};

  LITERT_RETURN_IF_ERROR(LiteRtGetGatherAxisOption(op.Get(), &axis));
  LITERT_ASSIGN_OR_RETURN(auto op_wrapper,
                          BuildGeneralGatherOp(op, gather_input_order, axis));

  return op_wrapper;
}

Expected<OpWrapper> BuildEmbeddingLookupOp(const Op& op) {
  int32_t axis = 0;
  std::vector<int32_t> embeding_lookup_input_order = {1, 0};
  LITERT_ASSIGN_OR_RETURN(
      auto op_wrapper,
      BuildGeneralGatherOp(op, embeding_lookup_input_order, axis));
  return op_wrapper;
}

}  // namespace litert::samsung
