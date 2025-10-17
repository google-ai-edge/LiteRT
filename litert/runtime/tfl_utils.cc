// Copyright 2024 Google LLC.
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

#include "litert/runtime/tfl_utils.h"

#include <cstddef>
#include <cstring>

#include "absl/strings/str_format.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/core/options.h"
#include "litert/runtime/tensor_identifier.h"
#include "tflite/c/c_api_opaque.h"
#include "tflite/c/c_api_types.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"

namespace litert::internal {

TfLiteStatus SetCustomAllocationForInputTensor(
    tflite::Interpreter* interpreter,
    const LiteRtExternalTensorBinding& binding) {
  if (!interpreter) {
    return kTfLiteError;
  }
  if (!binding.data) {
    return kTfLiteError;
  }

  const char* signature_name =
      binding.signature_name.empty() ? nullptr : binding.signature_name.c_str();
  auto* signature_runner = interpreter->GetSignatureRunner(signature_name);
  if (!signature_runner) {
    LITERT_LOG(LITERT_INFO, "Signature %s not found in interpreter",
               binding.signature_name.c_str());
    return kTfLiteError;
  }

  const TfLiteCustomAllocation custom_allocation = {binding.data,
                                                    binding.size_bytes};
  signature_runner->SetCustomAllocationForInputTensor(
      binding.tensor_name.c_str(), custom_allocation);

  return kTfLiteOk;
}

Expected<LiteRtLayout> ConvertTensorLayout(
    const TfLiteOpaqueTensor* tfl_opaque_tensor) {
  size_t rank = TfLiteOpaqueTensorNumDims(tfl_opaque_tensor);
  auto layout = LiteRtLayout{};
  layout.rank = rank;
  layout.has_strides = false;
  for (size_t i = 0; i < rank; ++i) {
    layout.dimensions[i] = TfLiteOpaqueTensorDim(tfl_opaque_tensor, i);
  }
  return layout;
}

Expected<LiteRtRankedTensorType> ConvertTensorType(
    const TfLiteOpaqueTensor* tfl_opaque_tensor) {
  auto tfl_type = TfLiteOpaqueTensorType(tfl_opaque_tensor);
  auto element_type = static_cast<LiteRtElementType>(tfl_type);
  auto layout = ConvertTensorLayout(tfl_opaque_tensor);
  return LiteRtRankedTensorType{/*element_type = */ element_type,
                                /* layout = */ layout.Value()};
}

Expected<void> ResizeTensor(const LiteRtLayout& layout,
                            TfLiteOpaqueContext* tfl_context,
                            TfLiteOpaqueTensor* tfl_opaque_tensor) {
  // TFL tensors don't support strides.
  if (layout.has_strides) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Unexpected layout with strides");
  }

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(layout.rank);
  for (auto i = 0; i < layout.rank; ++i) {
    output_size->data[i] = layout.dimensions[i];
  }
  if (auto status = TfLiteOpaqueContextResizeTensor(
          tfl_context, tfl_opaque_tensor, output_size);
      status != kTfLiteOk) {
    return Unexpected(
        kLiteRtStatusErrorRuntimeFailure,
        absl::StrFormat("Failed to resize TFL tensor %s: %d",
                        TfLiteOpaqueTensorName(tfl_opaque_tensor), status));
  }

  return {};
}

Expected<TfLiteTensorIdentifier> GetTensorIdentifier(
    const tflite::Interpreter& interpreter, const TfLiteTensor* target_tensor) {
  for (int i = 0; i < interpreter.subgraphs_size(); ++i) {
    const auto* subgraph = interpreter.subgraph(i);
    // Instead of iterating over all tensors, we can start from the first one
    // and look for the target tensor in the array of tensors. This assumes
    // that the target tensor is linear located after the base tensor in memory.
    const auto* base_tensor = subgraph->tensor(0);
    const std::ptrdiff_t diff = target_tensor - base_tensor;
    if (diff < 0 || diff >= subgraph->tensors_size()) {
      continue;
    }
    if (subgraph->tensor(diff) == target_tensor) {
      return TfLiteTensorIdentifier{i, static_cast<int>(diff)};
    }
  }
  return Unexpected(kLiteRtStatusErrorNotFound,
                    "Tensor not found in interpreter");
}

}  // namespace litert::internal
