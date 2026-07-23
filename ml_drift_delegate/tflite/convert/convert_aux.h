// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This file provides auxiliary functions for the convert functions between
// TFLite and MLDrift IR.

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_CONVERT_CONVERT_AUX_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_CONVERT_CONVERT_AUX_H_

#include <cstdint>
#include <cstring>

#include "xnnpack.h"  // from @XNNPACK
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"
#include "tflite/kernels/internal/tensor_ctypes.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift::ir {

enum class PopulateTensorFlags {
  kNoExtraBytes,
  kExtraBytes,
};

struct SizedLayout {
  ::ml_drift::Layout layout_1d = ::ml_drift::Layout::BHWC;  // Bx1x1x1
  ::ml_drift::Layout layout_2d = ::ml_drift::Layout::BHWC;  // Bx1x1xC
  ::ml_drift::Layout layout_3d = ::ml_drift::Layout::BHWC;  // Bx1xWxC
  ::ml_drift::Layout layout_4d = ::ml_drift::Layout::BHWC;  // BxHxWxC
};

namespace convert_aux_internal {
void SetAllDimensions(const TfLiteIntArray* dims, ::ml_drift::Scalar* shape);
void SetAllDimensions(const TfLiteIntArray* dims, ::ml_drift::Linear* shape);
void SetAllDimensions(const TfLiteIntArray* dims, ::ml_drift::HWC* shape);
void SetAllDimensions(const TfLiteIntArray* dims, ::ml_drift::HW* shape);
void SetAllDimensions(const TfLiteIntArray* dims, ::ml_drift::OHWI* shape);
void SetAllDimensions(const TfLiteIntArray* dims, ::ml_drift::BHWC* shape);
void SetAllDimensions(const TfLiteIntArray* dims, ::ml_drift::BHWDC* shape);

template <typename T>
void CopyData(const TfLiteTensor& src, T* dst) {
  ABSL_QCHECK_EQ(src.bytes % sizeof(T), 0);
  const int n = ::tflite::NumElements(&src);
  if (n * sizeof(T) == src.bytes) {
    std::memcpy(dst, src.data.raw_const, src.bytes);
    return;
  }
  switch (src.type) {
    case kTfLiteNoType:
      ABSL_LOG(FATAL) << "src has no type.";
    case kTfLiteFloat32:
      for (int i = 0; i < n; ++i) {
        dst[i] = ::tflite::GetTensorData<float>(&src)[i];
      }
      return;
    case kTfLiteInt32:
      for (int i = 0; i < n; ++i) {
        dst[i] = ::tflite::GetTensorData<int32_t>(&src)[i];
      }
      return;
    case kTfLiteUInt8:
      for (int i = 0; i < n; ++i) {
        dst[i] = ::tflite::GetTensorData<uint8_t>(&src)[i];
      }
      return;
    case kTfLiteInt64:
      for (int i = 0; i < n; ++i) {
        dst[i] = ::tflite::GetTensorData<int64_t>(&src)[i];
      }
      return;
    case kTfLiteString:
      ABSL_LOG(FATAL) << "src can't be string.";
    case kTfLiteBool:
      for (int i = 0; i < n; ++i) {
        dst[i] = ::tflite::GetTensorData<bool>(&src)[i];
      }
      return;
    case kTfLiteInt16:
      for (int i = 0; i < n; ++i) {
        dst[i] = ::tflite::GetTensorData<int16_t>(&src)[i];
      }
      return;
    case kTfLiteComplex64:
      ABSL_LOG(FATAL) << "src can't be complex64.";
    case kTfLiteInt8:
      for (int i = 0; i < n; ++i) {
        dst[i] = ::tflite::GetTensorData<int8_t>(&src)[i];
      }
      return;
    case kTfLiteFloat16:
      ABSL_LOG(FATAL) << "src can't be float16.";
    case kTfLiteFloat64:
      for (int i = 0; i < n; ++i) {
        dst[i] = ::tflite::GetTensorData<double>(&src)[i];
      }
      return;
    case kTfLiteComplex128:
      ABSL_LOG(FATAL) << "src can't be complex128.";
    case kTfLiteUInt64:
      for (int i = 0; i < n; ++i) {
        dst[i] = ::tflite::GetTensorData<uint64_t>(&src)[i];
      }
      return;
    case kTfLiteResource:
      ABSL_LOG(FATAL) << "src can't be resource.";
    case kTfLiteVariant:
      ABSL_LOG(FATAL) << "src can't be variant.";
    case kTfLiteUInt32:
      for (int i = 0; i < n; ++i) {
        dst[i] = ::tflite::GetTensorData<uint32_t>(&src)[i];
      }
      return;
    case kTfLiteUInt16:
      for (int i = 0; i < n; ++i) {
        dst[i] = ::tflite::GetTensorData<uint16_t>(&src)[i];
      }
      return;
    case kTfLiteInt2:
      std::memcpy(dst, src.data.data, src.bytes);
      return;
    case kTfLiteInt4: {
      std::memcpy(dst, src.data.data, src.bytes);
      return;
    }
    case kTfLiteUInt4: {
      ABSL_LOG(FATAL) << "UInt 4 not yet handled in MLDrift.";
    }
    case kTfLiteBFloat16:
      ABSL_LOG(FATAL) << "src can't be bfloat16.";
  }
}

template <>
void CopyData<float>(const TfLiteTensor& src, float* dst);

template <typename ShapeT, ::ml_drift::DataType Type>
inline void SetQuantizationParams(
    const TfLiteTensor* tflite_tensor, bool enable_spanned_weights,
    int extra_elements, ::ml_drift::Tensor<ShapeT, Type>* tensor,
    ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32>* scale,
    ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::INT32>*
        zero_point) {
  const auto* quant_params = static_cast<TfLiteAffineQuantization*>(
      tflite_tensor->quantization.params);
  // TODO: b/378522761 - Support blockwise quantized tensors.
  ABSL_QCHECK_EQ(tflite_tensor->quantization.type, kTfLiteAffineQuantization);
  ABSL_QCHECK_EQ(quant_params->quantized_dimension, 0);
  ABSL_CHECK(quant_params->scale);
  ABSL_CHECK(quant_params->zero_point);
  scale->shape = ::ml_drift::OHWI(quant_params->scale->size, 1, 1, 1);
  zero_point->shape = ::ml_drift::OHWI(quant_params->zero_point->size, 1, 1, 1);
  if (quant_params->scale->size > 1) {
    if (enable_spanned_weights) {
      scale->spanned_data =
          absl::MakeSpan(quant_params->scale->data, quant_params->scale->size);
      zero_point->spanned_data = absl::MakeSpan(quant_params->zero_point->data,
                                                quant_params->zero_point->size);
    } else {
      scale->data.resize(quant_params->scale->size + extra_elements);
      std::memcpy(scale->data.data(), quant_params->scale->data,
                  quant_params->scale->size * sizeof(float));
      zero_point->data.resize(quant_params->zero_point->size + extra_elements);
      std::memcpy(zero_point->data.data(), quant_params->zero_point->data,
                  quant_params->zero_point->size * sizeof(int));
    }
  } else {
    scale->data = {tflite_tensor->params.scale};
    zero_point->data = {tflite_tensor->params.zero_point};
  }
}

template <typename ShapeT, ::ml_drift::DataType Type>
inline void PopulateTensorInternal(
    const TfLiteTensor* const tflite_tensor,
    ::ml_drift::Tensor<ShapeT, Type>* tensor, PopulateTensorFlags flags,
    bool enable_spanned_weights,
    ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32>* scale,
    ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::INT32>*
        zero_point) {
  const int extra_elements =
      flags == PopulateTensorFlags::kExtraBytes && !enable_spanned_weights
          ? XNN_EXTRA_BYTES / SizeOf(Type)
          : 0;
  // TODO: b/378522761 - Support other types.
  if constexpr (Type == ::ml_drift::DataType::FLOAT32) {
    if (enable_spanned_weights) {
      tensor->spanned_data = absl::MakeSpan(tflite_tensor->data.f,
                                            tflite::NumElements(tflite_tensor));
    } else {
      tensor->data.resize(tflite::NumElements(tflite_tensor) + extra_elements);
      CopyData(*tflite_tensor, &tensor->data[0]);
    }
  } else if constexpr (Type == ::ml_drift::DataType::INT2 ||
                       Type == ::ml_drift::DataType::INT4 ||
                       Type == ::ml_drift::DataType::INT8 ||
                       Type == ::ml_drift::DataType::UINT8) {
    ABSL_CHECK(scale);
    ABSL_CHECK(zero_point);
    if (enable_spanned_weights) {
      tensor->spanned_data = absl::MakeSpan(
          reinterpret_cast<
              typename ::ml_drift::Tensor<ShapeT, Type>::ValueType*>(
              tflite_tensor->data.raw),
          tflite_tensor->bytes /
              sizeof(typename ::ml_drift::Tensor<ShapeT, Type>::ValueType));
    } else {
      ABSL_QCHECK_EQ(tflite_tensor->bytes % SizeOf(Type), 0);
      tensor->data.resize(tflite_tensor->bytes / SizeOf(Type) + extra_elements);
      std::memcpy(tensor->data.data(), tflite_tensor->data.raw_const,
                  tflite_tensor->bytes);
    }
    SetQuantizationParams(tflite_tensor, enable_spanned_weights, extra_elements,
                          tensor, scale, zero_point);
  } else {
    if (enable_spanned_weights) {
      ABSL_LOG(FATAL) << "Unsupported type for zero-copy: " << ToString(Type);
    } else {
      tensor->data.resize(tflite::NumElements(tflite_tensor) + extra_elements);
      CopyData(*tflite_tensor, &tensor->data[0]);
    }
  }

  SetAllDimensions(tflite_tensor->dims, &tensor->shape);
}

}  // namespace convert_aux_internal

// If fused_activation is not kTfLiteActNone, adds an activation op to the
// IR model and links it to the output of the provided op.
// Note, make sure you haven't set the op producer as the output_id.
void HandleFusedActivation(
    TfLiteFusedActivation fused_activation,
    ::ml_drift::ir::IrModel& ir_model, ::ml_drift::ir::IrOp* op,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    int output_id);

// If the tensor referenced by `bias_id` is a shared constant, flags it for
// LINEAR materialization by the shared-memory manager and moves its 1-D length
// from the batch dimension to the channel dimension (parity with GraphFloat32).
// Returns true if the bias is a shared constant, in which case callers must
// pass it as a runtime input rather than embedding it into op attributes.
bool MarkSharedBias(::ml_drift::ir::IrTensorId bias_id,
                    ::ml_drift::ir::IrModel& ir_model);

// Adds a constant input to the IR model from a TfLite tensor.
::ml_drift::ir::IrTensor* AddConstInput(const TfLiteContext& context,
                                        int tensor_id,
                                        ::ml_drift::ir::IrModel& ir_model,
                                        const SizedLayout& layout);

// Populates an ml_drift tensor with data from a TfLite tensor.
// If enable_spanned_weights is true, tensor->spanned_data will be populated
// with a span pointing to the TfLite tensor's data. Otherwise, tensor->data
// will be populated with a copy of the TfLite tensor's data.
// If the tensor is quantized, scale and zero_point will be populated with
// quantization parameters. Should be used in tandem with CheckPopulateTensor.
// TODO(b/443752881): Pass an IrTensor instead of a TensorT.
template <typename TensorT>
inline void PopulateTensor(
    const TfLiteTensor* tflite_tensor, int tensor_id, TensorT* tensor,
    PopulateTensorFlags flags, bool enable_spanned_weights = false,
    ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32>* scale =
        nullptr,
    ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::INT32>*
        zero_point = nullptr) {
  tensor->id = tensor_id;
  convert_aux_internal::PopulateTensorInternal(
      tflite_tensor, tensor, flags, enable_spanned_weights, scale, zero_point);
}

// Extracts an ml_drift::Axis from a TFLite tensor and a given axis index.
::ml_drift::Axis ExtractAxisFromIndex(const TfLiteTensor& tflite_tensor,
                                      int index);

// Maps a shape array (from size 0 to 4) to a BHWC struct,
// right-aligning the dimensions so the innermost dimension always maps to C.
::ml_drift::BHWC GetRightAlignedBHWC(const std::vector<int32_t>& values,
                                     int32_t start_val);

// Maps a shape array (from size 0 to 5) to a BHWDC struct,
// right-aligning the dimensions so the innermost dimension always maps to C.
::ml_drift::BHWDC GetRightAlignedBHWDC(const std::vector<int32_t>& values,
                                       int32_t start_val);

}  // namespace litert::ml_drift::ir

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_CONVERT_CONVERT_AUX_H_
