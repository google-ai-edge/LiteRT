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
#include <utility>

#include "absl/strings/str_format.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model.h"
#include "litert/cc/litert_detail.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/core/util/tensor_type_util.h"
#include "tflite/c/c_api_opaque.h"
#include "tflite/c/c_api_types.h"
#include "tflite/c/common.h"

namespace litert::internal {

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

Expected<TensorBuffer> CreateHostTensorBufferFromTflTensor(
    TfLiteOpaqueContext* tfl_context,
    const TfLiteOpaqueTensor* tfl_opaque_tensor) {
  LITERT_ASSIGN_OR_RETURN(auto tensor_type,
                          ConvertTensorType(tfl_opaque_tensor));
  void* host_mem_addr = TfLiteOpaqueTensorData(tfl_opaque_tensor);
  size_t buffer_size = TfLiteOpaqueTensorByteSize(tfl_opaque_tensor);
  LITERT_ASSIGN_OR_RETURN(auto tensor_buffer,
                          TensorBuffer::CreateFromHostMemory(
                              tensor_type, host_mem_addr, buffer_size));
  return tensor_buffer;
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

}  // namespace litert::internal
