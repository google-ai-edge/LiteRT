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

#include "litert/vendors/samsung/compiler/builders/slice_op_builder.h"

#include "litert/vendors/samsung/compiler/builders/utils.h"

namespace litert::samsung {

constexpr int32_t kIOIndex = 0;
constexpr int32_t kBeginIndex = 1;
constexpr int32_t kSizeIndex = 2;

Expected<OpWrapper> BuildSliceOp(const Op& op) {
  OpWrapper op_wrapper("Slice");

  op_wrapper.AddInput(op.Inputs()[kIOIndex]);
  op_wrapper.AddOutput(op.Outputs()[kIOIndex]);

  LITERT_ASSIGN_OR_RETURN(auto begin,
                          GetWeightDataAs<int32_t>(op.Inputs()[kBeginIndex]));
  LITERT_ASSIGN_OR_RETURN(auto size,
                          GetWeightDataAs<int32_t>(op.Inputs()[kSizeIndex]));

  op_wrapper.AddParam("begin", begin);
  op_wrapper.AddParam("size", size);

  return op_wrapper;
}
}  // namespace litert::samsung
