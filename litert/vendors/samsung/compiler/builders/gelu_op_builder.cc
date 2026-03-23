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

#include "litert/vendors/samsung/compiler/builders/gelu_op_builder.h"

#include "litert/c/litert_common.h"
#include "litert/c/litert_op_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_model.h"

namespace litert::samsung {

Expected<OpWrapper> BuildGeluOp(const Op &op) {
  OpWrapper op_wrapper("Gelu");

  for (const auto &input : op.Inputs()) {
    op_wrapper.AddInput(input);
  }
  for (const auto &output : op.Outputs()) {
    op_wrapper.AddOutput(output);
  }

  bool approximate{};
  if (auto status = LiteRtGetGeluApproximateOption(op.Get(), &approximate);
      status != kLiteRtStatusOk) {
    return Error(status, "Fail to get approximate.");
  }

  if (approximate) {
    op_wrapper.AddParam("approximate", "tanh");
  } else {
    op_wrapper.AddParam("approximate", "none");
  }

  return op_wrapper;
}
}  // namespace litert::samsung
