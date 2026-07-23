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

// This file provides helper functions for converting TFLite ops to MLDrift
// IR ops. Note that helper functions which are used in both support and convert
// are placed here. Support specific helpers are in support_aux while convert
// specific helpers are in convert_aux.

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_IR_MODEL_BUILDER_HELPER_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_IR_MODEL_BUILDER_HELPER_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"
#include "tflite/kernels/internal/reference/dequantize.h"
#include "tflite/kernels/internal/tensor_ctypes.h"
#include "tflite/kernels/internal/types.h"

namespace litert::ml_drift::ir {

using TensorIndexToBufferIdMap = std::unordered_map<size_t, size_t>;
using TensorIndexToExternalBufferIdMap = std::unordered_map<size_t, size_t>;

// Options for parsing tflite model with model builder.
struct IrModelBuilderOptions {
  // TFLite graph partitioner options; see GraphPartitionHelper for details.
  int start_node_index = 0;
  int end_node_index = std::numeric_limits<int>::max();
  int max_delegated_partitions = 1;

  // Set to enforce capping inf/-inf to max float values for the softmax input
  // and padding.
  bool enable_infinite_float_capping = false;
  // Set to enforce max f16 for the max float value when capping softmax input
  // and padding.
  bool enable_reduced_precision = false;
  // Set to propagate pointers to the tflite weights instead of copying them to
  // GraphFloat32 operations.
  bool enable_raw_weights_propagation = false;
  // Set to avoid copying model weights to the GraphFloat32 attributes.
  bool enable_spanned_weights = false;
  // Set to allow/disallow boolean tensors.
  // TODO - b/483403743: Remove this once OpenGL supports boolean tensors.
  bool allow_bool_tensors = true;
  bool allow_quant_ops = true;
  // Runs TransformIrModel (noop removal, pad/gemm fusion, and the interior
  // AddQuantAdjustments fake-quant pass).
  bool apply_model_transformations = true;
};

inline ::ml_drift::HW ToHW(int h, int w) {
  return ::ml_drift::HW(std::max(1, h), std::max(1, w));
}

// Resolves a potentially negative index to a positive one.
// e.g., (index=-1, span=4) -> 3.
inline int ResolveNegativeIndex(const int index, const int span) {
  return index < 0 ? index + span : index;
}

// Resolves negative indices for starts and ends vectors.
void ResolveNegativeIndices(const TfLiteIntArray& input_dims,
                            std::vector<int>& tensor);

// Applies begin and end masks to the starts and ends vectors.
void UpdateWithMask(int begin_mask, int end_mask,
                    const TfLiteIntArray& input_dims, std::vector<int>& starts,
                    std::vector<int>& ends);

// Infers concatenation axis from input and output shapes.
::ml_drift::Axis GetConcatAxis(
    const std::vector<::ml_drift::BHWDC>& input_shapes,
    const ::ml_drift::BHWDC& output_shape);

// Extracts a BHWDC shape from a TfLiteIntArray.
::ml_drift::BHWDC ExtractTensorShape(const TfLiteIntArray* dims);

// Updates padding attributes for SAME or VALID padding.
template <typename AttrT>
void UpdatePadding(TfLitePadding padding, const ::ml_drift::BHWDC& shape,
                   AttrT* attr) {
  if (padding == kTfLitePaddingSame) {
    attr->padding = CalculateSamePadding(
        ::ml_drift::BHWC(shape.b, shape.h, shape.w, shape.c), *attr);
  } else if (padding == kTfLitePaddingValid) {
    attr->padding.prepended = ::ml_drift::HW(0, 0);
    attr->padding.appended = ::ml_drift::HW(0, 0);
  }
}

bool IsBroadcastable(const TfLiteIntArray* dims1, const TfLiteIntArray* dims2);

void ConvertFloat16ToFloat32(size_t num_elements, const uint16_t* src,
                             float* dst);

void CopyFloat32Data(const TfLiteTensor* tfl_tensor, float* dst);

void PopulateQuantParams(const TfLiteTensor& tensor,
                         ::ml_drift::ir::IrQuantParams* quant_params);

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

bool IsLinearConvertible(const TfLiteIntArray* dims);

// Returns true if `tensor` is an 8-bit affine-quantized tensor (int8/uint8).
bool IsAffineQuantized8Bit(const TfLiteTensor& tensor);

// Fills the valid quantized value range for an 8-bit type.
void GetQuant8Bounds(TfLiteType type, float* qmin, float* qmax);

// Appends an in-graph dequantize adapter computing
//   dst_float = (src_int8 - zero_point) * scale
// composed from existing CAST/SUB/MUL ops. `dst_float` must already exist; it
// is the float activation consumed by the rest of the graph. This lets a
// consumer (e.g. FullyConnected) run in float while the producer/boundary
// tensor stays int8/uint8 (matching the TFLite model). Eventually this will
// be deprecated in favor of native int8 activation support.
void InsertDequantizeChain(::ml_drift::ir::IrModel& ir_model,
                           ::ml_drift::ir::IrTensorId src_int8,
                           ::ml_drift::ir::IrTensorId dst_float,
                           const ::ml_drift::BHWDC& shape, float scale,
                           float zero_point);

// Appends an in-graph quantize adapter computing
//   dst_int8 = cast<int8>(clamp(round(src_float / scale) + zero_point,
//                                qmin, qmax))
// composed from existing DIV/ROUND/ADD/MAXIMUM/MINIMUM/CAST ops. This mirrors
// tflite's AffineQuantize.
void InsertQuantizeChain(::ml_drift::ir::IrModel& ir_model,
                         ::ml_drift::ir::IrTensorId src_float,
                         ::ml_drift::ir::IrTensorId dst_int8,
                         const ::ml_drift::BHWDC& shape, float scale,
                         float zero_point, float qmin, float qmax);

}  // namespace litert::ml_drift::ir

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_TFLITE_IR_MODEL_BUILDER_HELPER_H_
