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
#include "litert/vendors/samsung/compiler/builders/reshape_op_builder.h"

namespace litert::samsung {

Expected<OpWrapper> BuildReshapeOp(const Op &op) {
  OpWrapper op_wrapper("", "Reshape");

  for (const auto &input : op.Inputs()) {
    op_wrapper.AddInput(input);
  }
  for (const auto &output : op.Outputs()) {
    op_wrapper.AddOutput(output);
  }
  const int32_t *reshape_new_shape;
  int32_t new_shape_size;
  if (auto status = LiteRtGetReshapeNewShapeOption(op.Get(), &reshape_new_shape,
                                                   &new_shape_size);
      status != kLiteRtStatusOk) {
    return Error(status, "Fail to get new shape.");
  }
  std::vector<int32_t> new_shape(reshape_new_shape,
                                 reshape_new_shape + new_shape_size);
  op_wrapper.AddParam("new_shape", new_shape);

  return op_wrapper;
}
} // namespace litert::samsung
