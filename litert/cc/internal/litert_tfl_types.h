// Copyright 2026 Google LLC.
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

#ifndef ODML_LITERT_LITERT_CC_INTERNAL_LITERT_TFL_TYPES_H_
#define ODML_LITERT_LITERT_CC_INTERNAL_LITERT_TFL_TYPES_H_

#include <cstdint>

#include "litert/c/litert_model_types.h"

namespace litert {

// TFLite Tensor Type equivalent.
enum TfliteTensorType : uint32_t {
  TensorType_FLOAT32 = 0,
  TensorType_FLOAT16 = 1,
  TensorType_INT32 = 2,
  TensorType_UINT8 = 3,
  TensorType_INT64 = 4,
  TensorType_STRING = 5,
  TensorType_BOOL = 6,
  TensorType_INT16 = 7,
  TensorType_COMPLEX64 = 8,
  TensorType_INT8 = 9,
  TensorType_FLOAT64 = 10,
  TensorType_COMPLEX128 = 11,
  TensorType_UINT64 = 12,
  TensorType_RESOURCE = 13,
  TensorType_VARIANT = 14,
  TensorType_UINT32 = 15,
  TensorType_UINT16 = 16,
  TensorType_INT4 = 17,
  TensorType_BFLOAT16 = 18,
  TensorType_INT2 = 19,
  TensorType_UINT4 = 20,
  TensorType_FLOAT8_E4M3FN = 21,
  TensorType_FLOAT8_E5M2 = 22,
  TensorType_MIN = TensorType_FLOAT32,
  TensorType_MAX = TensorType_FLOAT8_E5M2
};

// C++ enums mapping to TFLite equivalents.
using ActivationFunction = uint32_t;
enum ActivationFunctionType : uint32_t {
  kActivationFunctionTypeNone = 0,
  kActivationFunctionTypeRelu = 1,
  kActivationFunctionTypeReluN1To1 = 2,
  kActivationFunctionTypeRelu6 = 3,
  kActivationFunctionTypeTanh = 4,
  kActivationFunctionTypeSignBit = 5,
  kActivationFunctionTypeMin = kActivationFunctionTypeNone,
  kActivationFunctionTypeMax = kActivationFunctionTypeSignBit,
};

using FullyConnectedOptionsWeightsFormat = uint32_t;
enum FullyConnectedOptionsWeightsFormatType : uint32_t {
  kFullyConnectedOptionsWeightsFormatDefault = 0,
  kFullyConnectedOptionsWeightsFormatShuffled4x16Int8 = 1,
  kFullyConnectedOptionsWeightsFormatMin =
      kFullyConnectedOptionsWeightsFormatDefault,
  kFullyConnectedOptionsWeightsFormatMax =
      kFullyConnectedOptionsWeightsFormatShuffled4x16Int8
};

using Padding = uint32_t;
enum PaddingType : uint32_t {
  kPaddingSame = 0,
  kPaddingValid = 1,
  kPaddingMin = kPaddingSame,
  kPaddingMax = kPaddingValid,
};

using MirrorPadMode = uint32_t;
enum MirrorPadModeType : uint32_t {
  kMirrorPadModeReflect = 0,
  kMirrorPadModeSymmetric = 1,
  kMirrorPadModeMin = kMirrorPadModeReflect,
  kMirrorPadModeMax = kMirrorPadModeSymmetric,
};

inline LiteRtElementType GetElementType(uint32_t tflite_element_type) {
  switch (tflite_element_type) {
    case TensorType_FLOAT32:
      return kLiteRtElementTypeFloat32;
    case TensorType_FLOAT16:
      return kLiteRtElementTypeFloat16;
    case TensorType_INT32:
      return kLiteRtElementTypeInt32;
    case TensorType_UINT8:
      return kLiteRtElementTypeUInt8;
    case TensorType_UINT4:
      return kLiteRtElementTypeUInt4;
    case TensorType_INT64:
      return kLiteRtElementTypeInt64;
    case TensorType_STRING:
      return kLiteRtElementTypeTfString;
    case TensorType_BOOL:
      return kLiteRtElementTypeBool;
    case TensorType_INT16:
      return kLiteRtElementTypeInt16;
    case TensorType_COMPLEX64:
      return kLiteRtElementTypeComplex64;
    case TensorType_INT8:
      return kLiteRtElementTypeInt8;
    case TensorType_FLOAT64:
      return kLiteRtElementTypeFloat64;
    case TensorType_COMPLEX128:
      return kLiteRtElementTypeComplex128;
    case TensorType_UINT64:
      return kLiteRtElementTypeUInt64;
    case TensorType_RESOURCE:
      return kLiteRtElementTypeTfResource;
    case TensorType_VARIANT:
      return kLiteRtElementTypeTfVariant;
    case TensorType_UINT32:
      return kLiteRtElementTypeUInt32;
    case TensorType_UINT16:
      return kLiteRtElementTypeUInt16;
    case TensorType_INT4:
      return kLiteRtElementTypeInt4;
    case TensorType_BFLOAT16:
      return kLiteRtElementTypeBFloat16;
    case TensorType_INT2:
      return kLiteRtElementTypeInt2;
    case TensorType_FLOAT8_E4M3FN:
      return kLiteRtElementTypeFloat8E4M3FN;
    case TensorType_FLOAT8_E5M2:
      return kLiteRtElementTypeFloat8E5M2;
    default:
      return kLiteRtElementTypeNone;
  }
}

inline uint32_t GetTfliteTensorType(LiteRtElementType element_type) {
  switch (element_type) {
    case kLiteRtElementTypeFloat32:
      return TensorType_FLOAT32;
    case kLiteRtElementTypeFloat16:
      return TensorType_FLOAT16;
    case kLiteRtElementTypeInt32:
      return TensorType_INT32;
    case kLiteRtElementTypeUInt8:
      return TensorType_UINT8;
    case kLiteRtElementTypeUInt4:
      return TensorType_UINT4;
    case kLiteRtElementTypeInt64:
      return TensorType_INT64;
    case kLiteRtElementTypeTfString:
      return TensorType_STRING;
    case kLiteRtElementTypeBool:
      return TensorType_BOOL;
    case kLiteRtElementTypeInt16:
      return TensorType_INT16;
    case kLiteRtElementTypeComplex64:
      return TensorType_COMPLEX64;
    case kLiteRtElementTypeInt8:
      return TensorType_INT8;
    case kLiteRtElementTypeFloat64:
      return TensorType_FLOAT64;
    case kLiteRtElementTypeComplex128:
      return TensorType_COMPLEX128;
    case kLiteRtElementTypeUInt64:
      return TensorType_UINT64;
    case kLiteRtElementTypeTfResource:
      return TensorType_RESOURCE;
    case kLiteRtElementTypeTfVariant:
      return TensorType_VARIANT;
    case kLiteRtElementTypeUInt32:
      return TensorType_UINT32;
    case kLiteRtElementTypeUInt16:
      return TensorType_UINT16;
    case kLiteRtElementTypeInt4:
      return TensorType_INT4;
    case kLiteRtElementTypeBFloat16:
      return TensorType_BFLOAT16;
    case kLiteRtElementTypeInt2:
      return TensorType_INT2;
    case kLiteRtElementTypeFloat8E4M3FN:
      return TensorType_FLOAT8_E4M3FN;
    case kLiteRtElementTypeFloat8E5M2:
      return TensorType_FLOAT8_E5M2;
    default:
      return TensorType_FLOAT32;
  }
}

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_INTERNAL_LITERT_TFL_TYPES_H_
