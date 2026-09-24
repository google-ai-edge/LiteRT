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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_ACCELERATORS_GPU_COMPATIBILITY_ML_DRIFT_COMPATIBILITY_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_ACCELERATORS_GPU_COMPATIBILITY_ML_DRIFT_COMPATIBILITY_H_

#include <cstddef>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "tflite/experimental/acceleration/compatibility/android_info.h"

namespace litert::ml_drift {

// Returns true if the specified device and GPU are supported for ML Drift GPU
// acceleration in the given compatibility binary.
bool IsMlDriftGpuSupported(
    const unsigned char* compatibility_binary, size_t compatibility_binary_len,
    const tflite::acceleration::AndroidInfo& android_info,
    absl::string_view gl_renderer, int gles_major, int gles_minor);

// Returns true if the specified device and GPU are supported for ML Drift GPU
// acceleration.
bool IsMlDriftGpuSupported(
    const tflite::acceleration::AndroidInfo& android_info,
    absl::string_view gl_renderer, int gles_major, int gles_minor);

// Returns true if the current device is supported for ML Drift GPU
// acceleration.
bool IsMlDriftGpuSupportedOnThisDevice();

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_ACCELERATORS_GPU_COMPATIBILITY_ML_DRIFT_COMPATIBILITY_H_
