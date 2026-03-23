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
#include "litert/vendors/samsung/compiler/builders/reduce_op_builder.h"

#include <algorithm>

#include "litert/c/litert_common.h"
#include "litert/c/litert_op_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "litert/vendors/samsung/compiler/builders/utils.h"

namespace litert::samsung {

constexpr int32_t kIOIndex = 0;
constexpr int32_t kAxisIndex = 1;

Expected<OpWrapper> BuildGeneralReduceOp(const Op &op, const char *type,
                                         bool keep_dims) {
  OpWrapper op_wrapper(type);

  op_wrapper.AddInput(op.Inputs()[kIOIndex]);
  op_wrapper.AddOutput(op.Outputs()[kIOIndex]);

  LITERT_ASSIGN_OR_RETURN(auto reduce_axes,
                          GetWeightDataAs<int32_t>(op.Inputs()[kAxisIndex]));
  std::sort(reduce_axes.begin(), reduce_axes.end());
  reduce_axes.erase(std::unique(reduce_axes.begin(), reduce_axes.end()),
                    reduce_axes.end());
  op_wrapper.AddParam("axes", reduce_axes);
  op_wrapper.AddParam("keep_dims", static_cast<int32_t>(keep_dims));

  return op_wrapper;
}

Expected<OpWrapper> BuildReduceSumOp(const Op &op) {
  bool keep_dims{};
  if (auto status = LiteRtGetSumKeepDimsOption(op.Get(), &keep_dims);
      status != kLiteRtStatusOk) {
    return Error(status, "Fail to get keep_dims of sum");
  }

  return BuildGeneralReduceOp(op, "ReduceSum", keep_dims);
}

Expected<OpWrapper> BuildReduceMeanOp(const Op &op) {
  bool keep_dims{};
  if (auto status = LiteRtGetMeanKeepDimsOption(op.Get(), &keep_dims);
      status != kLiteRtStatusOk) {
    return Error(status, "Fail to get keep_dims of mean");
  }

  return BuildGeneralReduceOp(op, "ReduceMean", keep_dims);
}

Expected<OpWrapper> BuildReduceMaxOp(const Op &op) {
  bool keep_dims{};
  if (auto status = LiteRtGetReduceMaxKeepDimsOption(op.Get(), &keep_dims);
      status != kLiteRtStatusOk) {
    return Error(status, "Fail to get keep_dims of max");
  }

  return BuildGeneralReduceOp(op, "ReduceMax", keep_dims);
}

Expected<OpWrapper> BuildReduceMinOp(const Op &op) {
  bool keep_dims{};
  if (auto status = LiteRtGetReduceMinKeepDimsOption(op.Get(), &keep_dims);
      status != kLiteRtStatusOk) {
    return Error(status, "Fail to get keep_dims of min");
  }

  return BuildGeneralReduceOp(op, "ReduceMin", keep_dims);
}

}  // namespace litert::samsung
