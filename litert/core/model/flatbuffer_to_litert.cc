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

#include "litert/core/model/flatbuffer_to_litert.h"

#include <cstdint>
#include <utility>

#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/core/model/model.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "tflite/schema/schema_generated.h"

namespace litert::internal {

LiteRtStatus IsOpSupported(const tflite::OperatorT& op) {
  // TODO: b/365299994 - Check for supported options.

  if (!op.intermediates.empty()) {
    // TODO: b/365299994 - Support intermediates.
    LITERT_LOG(LITERT_ERROR, "Intermediate tensors not yet supported.");
    return kLiteRtStatusErrorUnsupported;
  }

  if (op.large_custom_options_size != 0) {
    // TODO: b/365299994 - Support large custom options.
    LITERT_LOG(LITERT_ERROR, "Large custom options not yet supported.");
    return kLiteRtStatusErrorUnsupported;
  }

  for (auto m_input : op.mutating_variable_inputs) {
    if (m_input) {
      // TODO: b/365299994 - Support mutating variable inputs.
      LITERT_LOG(LITERT_ERROR, "Mutating variable inputs not yet supported.");
      return kLiteRtStatusErrorUnsupported;
    }
  }

  return kLiteRtStatusOk;
}

LiteRtStatus IsBufferSupported(const tflite::BufferT& buffer) {
  if (buffer.offset != 0) {
    // TODO: b/365299994 - Support buffer with offset.
    LITERT_LOG(LITERT_ERROR, "Buffers with offset not yet supported.");
    return kLiteRtStatusErrorUnsupported;
  }

  return kLiteRtStatusOk;
}

LiteRtStatus IsTensorSupported(const TflTensor& tensor) {
  if (tensor.is_variable) {
    // TODO: b/365299994 - Support variable tensors.
    LITERT_LOG(LITERT_ERROR, "Variable tensors not yet supported.");
    return kLiteRtStatusErrorUnsupported;
  }

  if (!tensor.variant_tensors.empty()) {
    // TODO: b/365299994 - Support variant tensors.
    LITERT_LOG(LITERT_ERROR, "Variant tensors not yet supported.");
    return kLiteRtStatusErrorUnsupported;
  }

  if (tensor.sparsity) {
    // TODO: b/365299994 - Support sparsity tensors.
    LITERT_LOG(LITERT_ERROR, "Sparsity tensors not yet supported.");
    return kLiteRtStatusErrorUnsupported;
  }

  return kLiteRtStatusOk;
}

LiteRtElementType MapElementType(TflElementType type) {
  switch (type) {
    case tflite::TensorType_FLOAT32:
      return kLiteRtElementTypeFloat32;
    case tflite::TensorType_FLOAT16:
      return kLiteRtElementTypeFloat16;
    case tflite::TensorType_BFLOAT16:
      return kLiteRtElementTypeBFloat16;
    case tflite::TensorType_COMPLEX64:
      return kLiteRtElementTypeComplex64;
    case tflite::TensorType_INT32:
      return kLiteRtElementTypeInt32;
    case tflite::TensorType_UINT32:
      return kLiteRtElementTypeUInt32;
    case tflite::TensorType_INT64:
      return kLiteRtElementTypeInt64;
    case tflite::TensorType_BOOL:
      return kLiteRtElementTypeBool;
    case tflite::TensorType_INT16:
      return kLiteRtElementTypeInt16;
    case tflite::TensorType_INT8:
      return kLiteRtElementTypeInt8;
    case tflite::TensorType_UINT8:
      return kLiteRtElementTypeUInt8;
    case tflite::TensorType_INT4:
      return kLiteRtElementTypeInt4;
    case tflite::TensorType_INT2:
      return kLiteRtElementTypeInt2;
    default:
      return kLiteRtElementTypeNone;
  }
}

Expected<TensorType> MapTensorType(const TflTensorType& tfl_tensor_type) {
  const auto& [element_type, shape] = tfl_tensor_type;
  auto ranked_shape = AsDynamicShape(shape);
  if (!ranked_shape) {
    LITERT_LOG(LITERT_ERROR, "Only ranked tensors currently supported");
    return Error(Status::kErrorUnsupported);
  }

  auto litert_element_type = MapElementType(element_type);
  if (litert_element_type == kLiteRtElementTypeNone) {
    LITERT_LOG(LITERT_ERROR, "Element type (%d) not currently supported",
               element_type);
    return Error(Status::kErrorUnsupported);
  }

  TensorTypeDetail detail;
  detail.ranked_tensor_type.element_type = litert_element_type;
  detail.ranked_tensor_type.layout = BuildLayout(*ranked_shape);

  return std::make_pair(kLiteRtRankedTensorType, detail);
}

Expected<Quantization> MapQuantization(
    const TflPackedQuantization* tfl_quantization) {
  // If quantization is not set, return empty quantization.
  if (!tfl_quantization) {
    return MakeEmptyQuantization();
  }
  const bool scale_empty =
      !tfl_quantization->scale() || tfl_quantization->scale()->empty();
  const bool zp_empty = !tfl_quantization->zero_point() ||
                        tfl_quantization->zero_point()->empty();
  if (scale_empty && zp_empty) {
    return MakeEmptyQuantization();
  }

  const bool is_per_channel =
      (!scale_empty && tfl_quantization->scale()->size() > 1) ||
      (!zp_empty && tfl_quantization->zero_point()->size() > 1);
  const bool is_per_tensor =
      (!scale_empty && tfl_quantization->scale()->size() == 1) ||
      (!zp_empty && tfl_quantization->zero_point()->size() == 1);

  // Per tensor quantization.
  if (is_per_tensor) {
    const auto* scale_fb = tfl_quantization->scale()->data();
    const auto* zp_fb = tfl_quantization->zero_point()->data();
    return MakePerTensorQuantization(scale_fb[0], zp_fb[0]);
  }

  // Per channel quantization.
  if (is_per_channel) {
    const auto* scales_fb = tfl_quantization->scale();
    const auto* zero_points_fb = tfl_quantization->zero_point();
    int32_t quantized_dimension = tfl_quantization->quantized_dimension();

    if (!scales_fb || !zero_points_fb || scales_fb->empty() ||
        scales_fb->size() != zero_points_fb->size()) {
      LITERT_LOG(LITERT_ERROR, "Invalid per-channel quantization parameters");
      return Error(Status::kErrorInvalidArgument);
    }
    Quantization litert_quantization;
    litert_quantization.first = kLiteRtQuantizationPerChannel;
    litert_quantization.second.per_channel.scales =
        const_cast<float*>(scales_fb->data());
    litert_quantization.second.per_channel.zero_points =
        const_cast<int64_t*>(zero_points_fb->data());
    litert_quantization.second.per_channel.num_channels = scales_fb->size();
    litert_quantization.second.per_channel.quantized_dimension =
        quantized_dimension;
    return litert_quantization;
  }

  // Unknown quantization type.
  LITERT_LOG(LITERT_ERROR, "Unknown tfl quantization type");
  return Error(Status::kErrorUnsupported);
}
}  // namespace litert::internal
