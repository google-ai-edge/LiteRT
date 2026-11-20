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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_MODEL_BUILDER_HELPER_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_MODEL_BUILDER_HELPER_H_

#include <stddef.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"
#include "tflite/kernels/internal/reference/dequantize.h"
#include "tflite/kernels/internal/tensor_ctypes.h"
#include "tflite/kernels/internal/types.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift {

absl::Status GetNodeAndRegistration(TfLiteContext* context, int node_id,
                                    TfLiteNode** tflite_node,
                                    TfLiteRegistration** registration);

::ml_drift::DataType ToDataType(TfLiteType type);

::ml_drift::BHWC ExtractTensorShape(const TfLiteTensor* tflite_tensor);

// Must check PreCheckAxisFromIndex.
::ml_drift::Axis ExtractAxisFromIndex(const TfLiteTensor& tflite_tensor,
                                      int index);

// Populates quantization parameters for non-constant UInt8/Int8 tensors.
// This helps the delegate emulate quantized inference with
// QuantizeAndDequantize.
void PopulateQuantParams(const TfLiteTensor& tensor,
                         ::ml_drift::QuantizationParams* quant_params);

int GetNumberOfRuntimeInputsForNode(const TfLiteContext* context,
                                    const TfLiteNode* tflite_node);

int GetNumberOfConstInputsForNode(const TfLiteContext* context,
                                  const TfLiteNode* tflite_node);

absl::Status CheckInputsOutputs(const TfLiteContext* context,
                                const TfLiteNode* tflite_node,
                                int runtime_inputs, int outputs);

absl::Status CheckInputsConstsOutputs(const TfLiteContext* context,
                                      const TfLiteNode* tflite_node,
                                      int runtime_inputs, int const_inputs,
                                      int outputs);

void ConvertFloat16ToFloat32(size_t num_elements, const uint16_t* src,
                             float* dst);

template <typename T>
inline void DequantizeConstantTensor(const TfLiteTensor& tensor,
                                     const T* source_data,
                                     float* dequantized_data) {
  TfLiteAffineQuantization* quant_params =
      static_cast<TfLiteAffineQuantization*>(tensor.quantization.params);
  if (quant_params->scale->size > 1) {
    // Tensor is per-channel quantized.
    ::tflite::PerChannelDequantizationParams op_params;
    std::vector<int> zero_points;
    if (quant_params->zero_point->size == quant_params->scale->size) {
      op_params.zero_point = quant_params->zero_point->data;
    } else {
      zero_points.assign(
          quant_params->zero_point->data,
          quant_params->zero_point->data + quant_params->zero_point->size);
      zero_points.resize(quant_params->scale->size,
                         quant_params->zero_point->data[0]);
      op_params.zero_point = zero_points.data();
    }
    op_params.scale = quant_params->scale->data;
    op_params.quantized_dimension = quant_params->quantized_dimension;
    ::tflite::reference_ops::PerChannelDequantize(
        op_params, ::tflite::GetTensorShape(&tensor), source_data,
        ::tflite::GetTensorShape(&tensor), dequantized_data);
  } else {
    ::tflite::DequantizationParams op_params;
    op_params.zero_point = tensor.params.zero_point;
    op_params.scale = tensor.params.scale;
    ::tflite::reference_ops::Dequantize(
        op_params, ::tflite::GetTensorShape(&tensor), source_data,
        ::tflite::GetTensorShape(&tensor), dequantized_data);
  }
}

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
    case kTfLiteInt4: {
      std::memcpy(dst, src.data.data, src.bytes);
      return;
    }
    case kTfLiteUInt4: {
      ABSL_LOG(FATAL) << "UInt 4 not yet handled in MLDrift.";
    }
    case kTfLiteInt2:
      std::memcpy(dst, src.data.data, src.bytes);
      return;
    case kTfLiteBFloat16:
      ABSL_LOG(FATAL) << "src can't be bfloat16.";
    case kTfLiteFloat8E4M3FN:
      ABSL_LOG(FATAL) << "src can't be float8e4m3fn.";
    case kTfLiteFloat8E5M2:
      ABSL_LOG(FATAL) << "src can't be float8e5m2.";
  }
}

template <>
void CopyData<float>(const TfLiteTensor& src, float* dst);

void CopyFloat32Data(const TfLiteTensor* tfl_tensor, float* dst);

void SetAllDimensions(const TfLiteIntArray* dims, ::ml_drift::Scalar* shape);

bool IsLinearConvertible(const TfLiteIntArray* dims);

void SetAllDimensions(const TfLiteIntArray* dims, ::ml_drift::Linear* shape);

void SetAllDimensions(const TfLiteIntArray* dims, ::ml_drift::HWC* shape);

void SetAllDimensions(const TfLiteIntArray* dims, ::ml_drift::HW* shape);

void SetAllDimensions(const TfLiteIntArray* dims, ::ml_drift::OHWI* shape);

void SetAllDimensions(const TfLiteIntArray* dims, ::ml_drift::BHWC* shape);

void HandleFusedActivation(TfLiteFusedActivation fused_activation,
                           ::ml_drift::GraphFloat32* graph,
                           ::ml_drift::Node* node);

// Checks if the given tensor dimensions are broadcast-compatible.
bool IsBroadcastable(const TfLiteIntArray* dims1, const TfLiteIntArray* dims2);

// Scale and zero point tensors of a constant tensor with N-D blockwise
// quantization (kTfLiteBlockwiseQuantizationV2):
//
//   real_value = scale[block] * (quantized_value - zero_point[block])
//
// The zero points may be fractional (e.g. -0.5 for a centered int2 grid).
struct BlockwiseV2Params {
  const TfLiteTensor* scale = nullptr;
  // Null if the tensor has no zero point tensor, i.e. the zero point is 0.
  const TfLiteTensor* zero_point = nullptr;
};

// Looks up and validates the scale and zero point tensors of `tensor`, which
// must be an INT2, INT4 or INT8 tensor with N-D blockwise quantization, among
// the `tensors_size` tensors of the subgraph at `tensors`.
absl::StatusOr<BlockwiseV2Params> GetBlockwiseV2Params(
    const TfLiteTensor* tensors, size_t tensors_size,
    const TfLiteTensor& tensor);

// Dequantizes `tensor` into the NumElements(tensor) floats at `dst`, in the
// element order of `tensor`.
void DequantizeBlockwiseV2(const TfLiteTensor& tensor,
                           const BlockwiseV2Params& params, float* dst);

// How a constant tensor with N-D blockwise quantization is consumed as the
// weights of a quantized FullyConnected node that reads its scales and zero
// points from separate tensors.
struct BlockwiseV2WeightsInfo {
  // Weights shape as seen by the FullyConnected kernel:
  //  * 2-D FULLY_CONNECTED weights [O, I] map to (O, 1, 1, I).
  //  * 3-D or 4-D batched weights [H, I, O] / [1, H, I, O] (when
  //    `oi_transposed` is true) or [H, O, I] / [1, H, O, I] (when
  //    false) map to (O, H, 1, I).
  ::ml_drift::OHWI weights_shape;
  // Shape of the scales and zero points as seen by the kernel: one value per
  // output channel, weights batch entry and block of input channels.
  ::ml_drift::OHWI scale_shape;
  bool has_zero_point = false;
  // True when batched weights are stored in [H, I, O] order rather than
  // [H, O, I] order.
  bool oi_transposed = true;
};

// Fills `info` for the blockwise quantized weights `tensor`. Fails if the
// weights can't be consumed by a quantized FullyConnected node, e.g. when the
// blocks along the input channels don't all have the same size.
absl::Status GetBlockwiseV2WeightsInfo(const TfLiteTensor& tensor,
                                       const BlockwiseV2Params& params,
                                       BlockwiseV2WeightsInfo* info,
                                       bool oi_transposed = true);

// Returns the scales or zero points `param` of the weights `tensor` in the
// (o, h, 1, block) order of `info.scale_shape`, broadcasting them as needed.
std::vector<float> ReadBlockwiseV2WeightsParam(
    const TfLiteTensor& tensor, const TfLiteTensor& param,
    const BlockwiseV2WeightsInfo& info);

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_MODEL_BUILDER_HELPER_H_
