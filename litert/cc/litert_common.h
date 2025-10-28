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

namespace litert {

enum class HwAccelerators : int {
  Nnone = kLiteRtHwAcceleratorNone,
  Cpu = kLiteRtHwAcceleratorCpu,
  Gpu = kLiteRtHwAcceleratorGpu,
  Npu = kLiteRtHwAcceleratorNpu,
#if defined(__EMSCRIPTEN__)
  WebNn = kLiteRtHwAcceleratorWebNn,
#endif  // __EMSCRIPTEN__
};

// A bit field of `LiteRtHwAccelerators` values.
typedef int HwAcceleratorSet;

enum class DelegateBufferStorageType : int {
  kDefault = kLiteRtDelegateBufferStorageTypeDefault,
  kBuffer = kLiteRtDelegateBufferStorageTypeBuffer,
  kTexture2D = kLiteRtDelegateBufferStorageTypeTexture2D,
};

enum class GpuBackend : int {
  kAutomatic = kLiteRtGpuBackendAutomatic,
  kOpenCl = kLiteRtGpuBackendOpenCl,
  kWebGpu = kLiteRtGpuBackendWebGpu,
  kOpenGl = kLiteRtGpuBackendOpenGl,
};

enum class GpuPriority : int {
  kDefault = kLiteRtGpuPriorityDefault,
  kLow = kLiteRtGpuPriorityLow,
  kNormal = kLiteRtGpuPriorityNormal,
  kHigh = kLiteRtGpuPriorityHigh,
};

enum class DelegatePrecision : int {
  kDefault = kLiteRtDelegatePrecisionDefault,
  kFp16 = kLiteRtDelegatePrecisionFp16,
  kFp32 = kLiteRtDelegatePrecisionFp32,
};

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_COMMON_H_
