#ifndef TENSORFLOW_COMPILER_MLIR_LITE_CORE_C_TFLITE_TYPES_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_CORE_C_TFLITE_TYPES_H_

// Minimal stub header for TensorFlow Lite types
// This is a simplified version to enable CMake build

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// TensorFlow Lite type definitions
typedef enum TfLiteType {
  kTfLiteNoType = 0,
  kTfLiteFloat32 = 1,
  kTfLiteInt32 = 2,
  kTfLiteUInt8 = 3,
  kTfLiteInt64 = 4,
  kTfLiteString = 5,
  kTfLiteBool = 6,
  kTfLiteInt16 = 7,
  kTfLiteComplex64 = 8,
  kTfLiteInt8 = 9,
  kTfLiteFloat16 = 10,
  kTfLiteFloat64 = 11,
  kTfLiteComplex128 = 12,
  kTfLiteUInt64 = 13,
  kTfLiteResource = 14,
  kTfLiteVariant = 15,
  kTfLiteUInt32 = 16,
  kTfLiteUInt16 = 17,
  kTfLiteInt4 = 18,
  kTfLiteBFloat16 = 19,
} TfLiteType;

// Opaque handle types
struct TfLiteModel;
struct TfLiteInterpreter;
struct TfLiteInterpreterOptions;
struct TfLiteTensor;
struct TfLiteContext;
struct TfLiteNode;

#ifdef __cplusplus
}
#endif

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_CORE_C_TFLITE_TYPES_H_