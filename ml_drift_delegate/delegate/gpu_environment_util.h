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

#ifndef ODML_LITERT_ML_DRIFT_DELEGATE_GPU_ENVIRONMENT_UTIL_H_
#define ODML_LITERT_ML_DRIFT_DELEGATE_GPU_ENVIRONMENT_UTIL_H_

#include "absl/status/status.h"  // from @com_google_absl
#include "litert/c/litert_common.h"

namespace litert {
namespace ml_drift {

// Updates the given LiteRT environment with GPU capabilities queried from
// MLDrift. Specifically, it checks for FP16 support and updates the
// environment's GPU properties.
absl::Status UpdateGpuEnvironmentWithMlDriftCapabilities(
    LiteRtEnvironment environment);

}  // namespace ml_drift
}  // namespace litert

#endif  // ODML_LITERT_ML_DRIFT_DELEGATE_GPU_ENVIRONMENT_UTIL_H_
