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


#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_COMMON_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_COMMON_H_

#include "litert/c/litert_common.h"

/// @file
/// @brief Defines common types and enums for the LiteRT C++ API.

namespace litert {

// LINT.IfChange(status_codes)
enum class Status : int {
  kOk = kLiteRtStatusOk,
  /// Generic errors.
  kErrorInvalidArgument = kLiteRtStatusErrorInvalidArgument,
  kErrorMemoryAllocationFailure = kLiteRtStatusErrorMemoryAllocationFailure,
  kErrorRuntimeFailure = kLiteRtStatusErrorRuntimeFailure,
  kErrorMissingInputTensor = kLiteRtStatusErrorMissingInputTensor,
  kErrorUnsupported = kLiteRtStatusErrorUnsupported,
  kErrorNotFound = kLiteRtStatusErrorNotFound,
  kErrorTimeoutExpired = kLiteRtStatusErrorTimeoutExpired,
  kErrorWrongVersion = kLiteRtStatusErrorWrongVersion,
  kErrorUnknown = kLiteRtStatusErrorUnknown,
  kErrorAlreadyExists = kLiteRtStatusErrorAlreadyExists,
  /// Inference progression errors.
  kCancelled = kLiteRtStatusCancelled,
  /// File and loading related errors.
  kErrorFileIO = kLiteRtStatusErrorFileIO,
  kErrorInvalidFlatbuffer = kLiteRtStatusErrorInvalidFlatbuffer,
  kErrorDynamicLoading = kLiteRtStatusErrorDynamicLoading,
  kErrorSerialization = kLiteRtStatusErrorSerialization,
  kErrorCompilation = kLiteRtStatusErrorCompilation,
  /// IR related errors.
  kErrorIndexOOB = kLiteRtStatusErrorIndexOOB,
  kErrorInvalidIrType = kLiteRtStatusErrorInvalidIrType,
  kErrorInvalidGraphInvariant = kLiteRtStatusErrorInvalidGraphInvariant,
  kErrorGraphModification = kLiteRtStatusErrorGraphModification,
  /// Tool related errors.
  kErrorInvalidToolConfig = kLiteRtStatusErrorInvalidToolConfig,
  /// Legalization related errors.
  kLegalizeNoMatch = kLiteRtStatusLegalizeNoMatch,
  kErrorInvalidLegalization = kLiteRtStatusErrorInvalidLegalization,
  /// Transformation related errors.
  kPatternNoMatch = kLiteRtStatusPatternNoMatch,
  kInvalidTransformation = kLiteRtStatusInvalidTransformation,
};
// LINT.ThenChange(../c/litert_common.h:status_codes)

enum class HwAccelerators : int {
  kNone = kLiteRtHwAcceleratorNone,
  kCpu = kLiteRtHwAcceleratorCpu,
  kGpu = kLiteRtHwAcceleratorGpu,
  kNpu = kLiteRtHwAcceleratorNpu,
#if defined(__EMSCRIPTEN__)
  kWebNn = kLiteRtHwAcceleratorWebNn,
#endif  // __EMSCRIPTEN__
};

/// @brief A type-safe bit field for `HwAccelerators`.
struct HwAcceleratorSet {
  int value;

  explicit HwAcceleratorSet(int val) : value(val) {}

  explicit HwAcceleratorSet(HwAccelerators val)
      : value(static_cast<int>(val)) {}
};

inline HwAcceleratorSet operator|(HwAccelerators lhs, HwAccelerators rhs) {
  return HwAcceleratorSet(static_cast<int>(lhs) | static_cast<int>(rhs));
}

inline HwAcceleratorSet operator|(HwAcceleratorSet lhs, HwAccelerators rhs) {
  return HwAcceleratorSet(lhs.value | static_cast<int>(rhs));
}

inline int operator&(HwAccelerators lhs, HwAccelerators rhs) {
  return static_cast<int>(lhs) & static_cast<int>(rhs);
}

inline int operator&(HwAcceleratorSet lhs, HwAccelerators rhs) {
  return lhs.value & static_cast<int>(rhs);
}

inline HwAcceleratorSet& operator|=(HwAcceleratorSet& lhs, HwAccelerators rhs) {
  lhs.value |= static_cast<int>(rhs);
  return lhs;
}

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_COMMON_H_
