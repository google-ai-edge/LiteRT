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
#include "litert/vendors/samsung/compiler/builders/pad_op_builder.h"

#include "litert/c/litert_common.h"
#include "litert/c/litert_op_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "litert/vendors/samsung/compiler/builders/utils.h"
#include "tflite/schema/schema_generated.h"

constexpr int32_t kIOIndex = 0;
constexpr int32_t kPaddingIndex = 1;
constexpr int32_t kConstantValueIndex = 2;
constexpr int32_t kPadDirection = 2;

namespace litert::samsung {
template <typename T>
Expected<std::vector<int32_t>> ConvertPaddings(int32_t num_of_pad_axes,
                                               absl::Span<const T> paddings) {
  if (num_of_pad_axes * kPadDirection != paddings.size()) {
    return Error(kLiteRtStatusErrorRuntimeFailure, "Invalid padding.");
  }

  std::vector<int32_t> converted_paddings(paddings.size());
  for (int32_t index = 0; index < paddings.size(); index++) {
    int32_t new_index =
        (index % kPadDirection) * num_of_pad_axes + (index / kPadDirection);
    converted_paddings[new_index] = paddings[index];
  }
  return converted_paddings;
}

Expected<OpWrapper> BuildGeneralPadOp(const Op &op) {
  OpWrapper op_wrapper("Pad");

  op_wrapper.AddInput(op.Inputs()[kIOIndex]);
  op_wrapper.AddOutput(op.Outputs()[kIOIndex]);

  auto num_of_pad_axes = GetDimensions(op.Inputs()[kPaddingIndex]).at(0);
  std::vector<int32_t> converted_paddings;
  LITERT_ASSIGN_OR_RETURN(auto pad_ranked_tensor_type,
                          op.Inputs()[kPaddingIndex].RankedTensorType());
  if (pad_ranked_tensor_type.ElementType() == ElementType::Int32) {
    LITERT_ASSIGN_OR_RETURN(auto paddings,
                            op.Inputs()[kPaddingIndex].WeightsData<int32_t>());
    LITERT_ASSIGN_OR_RETURN(converted_paddings,
                            ConvertPaddings(num_of_pad_axes, paddings));
  } else if (pad_ranked_tensor_type.ElementType() == ElementType::Int64) {
    LITERT_ASSIGN_OR_RETURN(auto paddings,
                            op.Inputs()[kPaddingIndex].WeightsData<int64_t>());
    LITERT_ASSIGN_OR_RETURN(converted_paddings,
                            ConvertPaddings(num_of_pad_axes, paddings));
  } else {
    return Error(kLiteRtStatusErrorUnsupported, "Unsupported type of paddings");
  }
  op_wrapper.AddParam("pads", converted_paddings);

  return op_wrapper;
}

Expected<OpWrapper> BuildPadOp(const Op &op) {
  LITERT_ASSIGN_OR_RETURN(auto op_wrapper, BuildGeneralPadOp(op));

  const auto input = std::move(op.Inputs()[kIOIndex]);
  float constant_pad_val = 0;
  if (op.Inputs().size() > kConstantValueIndex) {
    LITERT_ASSIGN_OR_RETURN(
        auto constant,
        GetWeightDataAs<float>(op.Inputs()[kConstantValueIndex]));
    constant_pad_val = constant[0];
  } else if (input.HasQuantization()) {
    if (input.QTypeId() != kLiteRtQuantizationPerTensor) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Unsupported quantization type for pad input.");
    }
    auto zero_point = input.PerTensorQuantization().zero_point;
    constant_pad_val = static_cast<float>(zero_point);
  }

  op_wrapper.AddParam("constant_value", constant_pad_val);
  return op_wrapper;
}

Expected<OpWrapper> BuildMirrorPadOp(const Op &op) {
  LITERT_ASSIGN_OR_RETURN(auto op_wrapper, BuildGeneralPadOp(op));

  uint32_t mode = 0;
  LITERT_RETURN_IF_ERROR(LiteRtGetMirrorPadModeOption(op.Get(), &mode));
  const char *pad_mode = tflite::EnumNamesMirrorPadMode()[mode];
  op_wrapper.AddParam("mode", pad_mode);

  return op_wrapper;
}

}  // namespace litert::samsung
