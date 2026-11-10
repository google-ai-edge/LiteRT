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

#include "absl/base/call_once.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tflite/experimental/acceleration/compatibility/android_info.h"
#include "tflite/experimental/acceleration/compatibility/database_generated.h"

namespace litert::ml_drift {

// Checks whether the current device is supported for ML Drift GPU acceleration
// against the embedded mldrift_compatibility.bin allowlist. Thread-safe.
class GpuCompatibilityChecker {
 public:
  // Returns the process-wide checker for the embedded allowlist.
  static const GpuCompatibilityChecker& Instance();

  GpuCompatibilityChecker(const GpuCompatibilityChecker&) = delete;
  GpuCompatibilityChecker& operator=(const GpuCompatibilityChecker&) = delete;

  // Returns true if the current device is supported. Reads the device and GL
  // info on the first call and caches the result. Always true on non-Android.
  bool IsSupportedOnThisDevice() const;

 private:
  friend class GpuCompatibilityCheckerTest;

  // Verifies `compatibility_binary` once. If it is empty or invalid, no device
  // is supported.
  explicit GpuCompatibilityChecker(
      absl::Span<const unsigned char> compatibility_binary);

  // Returns true if the given device and GPU are supported in the allowlist.
  bool IsSupported(const tflite::acceleration::AndroidInfo& android_info,
                   absl::string_view gl_renderer, int gles_major,
                   int gles_minor) const;

  // Null if the compatibility binary is empty or invalid.
  const tflite::acceleration::DeviceDatabase* const database_;
  mutable absl::once_flag once_;
  mutable bool is_supported_on_this_device_ = false;
};

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_ACCELERATORS_GPU_COMPATIBILITY_ML_DRIFT_COMPATIBILITY_H_
