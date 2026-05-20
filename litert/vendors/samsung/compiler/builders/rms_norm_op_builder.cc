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

#include <cstdint>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/vendors/samsung/compiler/builders/utils.h"

namespace litert::samsung {

constexpr int kInputIndex = 0;

Expected<OpWrapper> BuildRmsNormOp(const LiteRtCompilerContext* ctx,
                                   const litert::compiler::Op& op) {
  OpWrapper op_wrapper("RMSNORM");

  for (const auto& input : op.Inputs()) {
    op_wrapper.AddInput(input);
  }
  for (const auto& output : op.Outputs()) {
    op_wrapper.AddOutput(output);
  }

  auto input_dimensions = GetDimensions(op.Inputs()[kInputIndex]);

  op_wrapper.AddParam("axis", input_dimensions.size() - 1);

  const uint8_t* impl_attributes = nullptr;
  int32_t impl_attributes_size = 0;
  LITERT_RETURN_IF_ERROR(ctx->get_shlo_composite_op_attributes(
      op.Get(), &impl_attributes, &impl_attributes_size));

  if (impl_attributes_size <= 0) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Missing attributes for RMSNorm");
  }

  auto attributes_map =
      flexbuffers::GetRoot(impl_attributes, impl_attributes_size).AsMap();
  constexpr char kEpsilonKey[] = "epsilon";
  flexbuffers::Reference raw_epsilon = attributes_map[kEpsilonKey];
  if (raw_epsilon.IsNull()) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Missing epsilon attribute for RMSNorm");
  }

  op_wrapper.AddParam("epsilon", raw_epsilon.AsFloat());

  return op_wrapper;
}

}  // namespace litert::samsung
