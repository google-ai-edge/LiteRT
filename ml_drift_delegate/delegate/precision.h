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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_PRECISION_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_PRECISION_H_

#ifndef THIRD_PARTY_ODML_INFRA_ML_DRIFT_DELEGATE_ML_DRIFT_DELEGATE_H_
#define THIRD_PARTY_ODML_INFRA_ML_DRIFT_DELEGATE_ML_DRIFT_DELEGATE_H_

#ifdef __cplusplus
extern "C" {
#endif

// Precision of the ML Drift OpenCl and WebGpu delegates.
// When the precision is `kDefault`, the delegate will check if FP16 is
// supported. If so, use Fp16. Otherwise, use Fp32.
typedef enum {
    // Use FP16 if available, otherwise use FP32.
    kDefault,
    // Use FP16; can result in wrong output.
    kFp16,
    // Use FP32; is slower than FP16.
    kFp32,
} MlDriftDelegatePrecision;

#ifdef __cplusplus
}
#endif

#endif  // THIRD_PARTY_ODML_INFRA_ML_DRIFT_DELEGATE_ML_DRIFT_DELEGATE_H_

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_PRECISION_H_
