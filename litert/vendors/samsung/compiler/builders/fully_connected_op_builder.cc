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

#include "litert/vendors/samsung/compiler/builders/fully_connected_op_builder.h"

#include "litert/c/litert_op_options.h"
#include "litert/vendors/samsung/compiler/builders/utils.h"

namespace litert::samsung {

constexpr int kKernelIndex = 1;

Expected<OpWrapper> BuildFullyConnectedOp(const Op &op) {
  OpWrapper op_wrapper("FC");

  for (const auto &input : op.Inputs()) {
    op_wrapper.AddInput(input);
  }
  for (const auto &output : op.Outputs()) {
    op_wrapper.AddOutput(output);
  }

  auto kernel_dimensions = GetDimensions(op.Inputs()[kKernelIndex]);
  if (kernel_dimensions.size() != 2) {
    return Error(
        kLiteRtStatusErrorUnsupported,
        absl::StrCat(
            "Doesn't support Fully connected kernel dimension size : except "
            "2, but get ",
            kernel_dimensions.size()));
  }
  op_wrapper.AddParam("in_channels", kernel_dimensions[1]);
  op_wrapper.AddParam("out_channels", kernel_dimensions[0]);

  uint32_t tfl_fused_activation;
  if (auto status = LiteRtGetFullyConnectedFusedActivationOption(
          op.Get(), &tfl_fused_activation);
      status != kLiteRtStatusOk) {
    return Error(status, "Fail to get fused activation");
  }
  auto activation = GetFusedActivationName(tfl_fused_activation);
  if (!activation) {
    return activation.Error();
  }
  op_wrapper.AddParam("activation", *activation);

  return op_wrapper;
}
}  // namespace litert::samsung
