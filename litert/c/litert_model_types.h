// Copyright 2025 Google LLC.
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

#ifndef ODML_LITERT_LITERT_C_LITERT_MODEL_TYPES_H_
#define ODML_LITERT_LITERT_C_LITERT_MODEL_TYPES_H_

#include <stdbool.h>  // NOLINT: To use bool type in C
#include <stddef.h>
#include <stdint.h>

#include "litert/c/litert_common.h"
#include "litert/c/litert_layout.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//
// LiteRtTensor + Types
//

// Get the string name associated with this tensor. This is an optional
// attribute and if not set will return a zero-length string.
// The returned string pointer is owned by the LiteRtModel to which the given
// Tensor belongs. It becomes invalid when the LiteRtModel is destroyed.
LiteRtStatus LiteRtGetTensorName(LiteRtTensor tensor, const char** name);

// Get the index associated with this tensor.
LiteRtStatus LiteRtGetTensorIndex(LiteRtTensor tensor, uint32_t* tensor_index);

// TENSOR TYPES

// Primitive types for elements in a tensor.
// Matched to tensorflow/compiler/mlir/lite/core/c/tflite_types.h
typedef enum {
  kLiteRtElementTypeNone = 0,         // kTfLiteNoType,
  kLiteRtElementTypeBool = 6,         // kTfLiteBool,
  kLiteRtElementTypeInt2 = 20,        // kTfLiteInt2,
  kLiteRtElementTypeInt4 = 18,        // kTfLiteInt4,
  kLiteRtElementTypeInt8 = 9,         // kTfLiteInt8,
  kLiteRtElementTypeInt16 = 7,        // kTfLiteInt16,
  kLiteRtElementTypeInt32 = 2,        // kTfLiteInt32,
  kLiteRtElementTypeInt64 = 4,        // kTfLiteInt64,
  kLiteRtElementTypeUInt8 = 3,        // kTfLiteUInt8,
  kLiteRtElementTypeUInt16 = 17,      // kTfLiteUInt16,
  kLiteRtElementTypeUInt32 = 16,      // kTfLiteUInt32,
  kLiteRtElementTypeUInt64 = 13,      // kTfLiteUInt64,
  kLiteRtElementTypeFloat16 = 10,     // kTfLiteFloat16,
  kLiteRtElementTypeBFloat16 = 19,    // kTfLiteBFloat16,
  kLiteRtElementTypeFloat32 = 1,      // kTfLiteFloat32,
  kLiteRtElementTypeFloat64 = 11,     // kTfLiteFloat64,
  kLiteRtElementTypeComplex64 = 8,    // kTfLiteComplex64,
  kLiteRtElementTypeComplex128 = 12,  // kTfLiteComplex128,
  kLiteRtElementTypeTfResource = 14,  // kTfLiteResource,
  kLiteRtElementTypeTfString = 5,     // kTfLiteString,
  kLiteRtElementTypeTfVariant = 15,   // kTfLiteVariant,
} LiteRtElementType;

// Tensor whose rank is dynamic.
typedef struct {
  // The primitive element type of the constituent data.
  LiteRtElementType element_type;
} LiteRtUnrankedTensorType;

// Tensor whose rank is static but dimensions may be dynamic.
typedef struct {
  // The primitive element type of the constituent data.
  LiteRtElementType element_type;

  // Shape information.
  LiteRtLayout layout;
} LiteRtRankedTensorType;

inline bool LiteRtIsSameUnrankedTensorType(
    const LiteRtUnrankedTensorType* type1,
    const LiteRtUnrankedTensorType* type2) {
  return type1->element_type == type2->element_type;
}

// The identifier for tensor type union.
typedef enum {
  // Type with fixed rank and possibly dynamic dimensions.
  kLiteRtRankedTensorType = 0,

  // Type with dynamic rank.
  kLiteRtUnrankedTensorType = 1,
} LiteRtTensorTypeId;

// QUANTIZATION

// Schema for tensors quantized with one set of q-params.
typedef struct {
  // Scaling factor.
  float scale;

  // The value that float:0 maps to in q-space.
  int64_t zero_point;
} LiteRtQuantizationPerTensor;

// Schema for tensors quantized with one set of q-params per channel.
typedef struct {
  int32_t quantized_dimension;
  uint64_t num_channels;
  float* scales;
  int64_t* zero_points;
} LiteRtQuantizationPerChannel;

// The identifier for quantization scheme type union.
typedef enum {
  // Tag for tensors without quantization.
  kLiteRtQuantizationNone = 0,

  // Basic quantization, one set of q-params per tensor.
  kLiteRtQuantizationPerTensor = 1,

  // Q-params for each element across a single dimension.
  kLiteRtQuantizationPerChannel = 2,

  // [NOT IMPLEMENTED YET] Q-params across blocks of fixed size (e.g. 2048).
  kLiteRtQuantizationBlockWise = 3,
} LiteRtQuantizationTypeId;

// Get the identifier for the type of quantization for a given tensor.
LiteRtStatus LiteRtGetQuantizationTypeId(LiteRtTensor tensor,
                                         LiteRtQuantizationTypeId* q_type_id);

// Get the per-tensor quantization information for a given tensor if it has it.
LiteRtStatus LiteRtGetPerTensorQuantization(
    LiteRtTensor tensor, LiteRtQuantizationPerTensor* per_tensor_quantization);

// Get the per-channel quantization information for a given tensor if it has it.
LiteRtStatus LiteRtGetPerChannelQuantization(
    LiteRtTensor tensor,
    LiteRtQuantizationPerChannel* per_channel_quantization);

// EDGES

// Information about the graph node that defines a tensor.
typedef struct LiteRtTensorDefiningOp {
  // The defining op itself.
  LiteRtOp op;

  // The op output index that defines the specific tensor.
  LiteRtParamIndex op_output_index;
} LiteRtTensorDefiningOp;

// Information about a reference to a tensor in the graph.
typedef struct LiteRtTensorUserOp {
  // The referring op itself.
  LiteRtOp op;

  // Index of which operand the op refers to a specific tensor on.
  LiteRtParamIndex op_input_index;
} LiteRtTensorUserOp;

// Options for model serialization.
typedef struct LiteRtModelSerializationOptions {
  // Alignment for bytecode assets that are appended to the model.
  // Alignment is enforced relative to the first byte of the flatbuffer.
  size_t bytecode_alignment;
} LiteRtModelSerializationOptions;


#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // ODML_LITERT_LITERT_C_LITERT_MODEL_TYPES_H_
