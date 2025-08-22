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
#include <utility>

#include "absl/strings/str_format.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_logging.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
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

Expected<ElementType> ConvertElementType(TfLiteType tfl_type) {
  switch (tfl_type) {
    case kTfLiteNoType:
      return ElementType::None;
    case kTfLiteBool:
      return ElementType::Bool;
    case kTfLiteInt4:
      return ElementType::Int4;
    case kTfLiteInt8:
      return ElementType::Int8;
    case kTfLiteInt16:
      return ElementType::Int16;
    case kTfLiteInt32:
      return ElementType::Int32;
    case kTfLiteInt64:
      return ElementType::Int64;
    case kTfLiteUInt8:
      return ElementType::UInt8;
    case kTfLiteUInt16:
      return ElementType::UInt16;
    case kTfLiteUInt32:
      return ElementType::UInt32;
    case kTfLiteUInt64:
      return ElementType::UInt64;
    case kTfLiteFloat16:
      return ElementType::Float16;
    case kTfLiteBFloat16:
      return ElementType::BFloat16;
    case kTfLiteFloat32:
      return ElementType::Float32;
    case kTfLiteFloat64:
      return ElementType::Float64;
    case kTfLiteComplex64:
      return ElementType::Complex64;
    case kTfLiteComplex128:
      return ElementType::Complex128;
    case kTfLiteResource:
      return ElementType::TfResource;
    case kTfLiteString:
      return ElementType::TfString;
    case kTfLiteVariant:
      return ElementType::TfVariant;
    default:
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "Unsupported TfLiteType");
  }
}

Expected<Layout> ConvertTensorLayout(
    const TfLiteOpaqueTensor* tfl_opaque_tensor) {
  size_t rank = TfLiteOpaqueTensorNumDims(tfl_opaque_tensor);
  Dimensions dimensions(rank);
  for (size_t i = 0; i < rank; ++i) {
    dimensions[i] = TfLiteOpaqueTensorDim(tfl_opaque_tensor, i);
  }
  return Layout(std::move(dimensions));
}

Expected<RankedTensorType> ConvertTensorType(
    const TfLiteOpaqueTensor* tfl_opaque_tensor) {
  auto tfl_type = TfLiteOpaqueTensorType(tfl_opaque_tensor);
  LITERT_ASSIGN_OR_RETURN(auto element_type, ConvertElementType(tfl_type));
  LITERT_ASSIGN_OR_RETURN(auto layout, ConvertTensorLayout(tfl_opaque_tensor));
  return RankedTensorType(element_type, std::move(layout));
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
