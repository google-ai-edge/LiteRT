// Copyright 2024 Google LLC.
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

#ifndef ODML_LITERT_LITERT_C_LITERT_DISPATCH_DELEGATE_H_
#define ODML_LITERT_LITERT_C_LITERT_DISPATCH_DELEGATE_H_

#include <stddef.h>

#include "litert/c/litert_common.h"
#include "tflite/c/c_api_types.h"
#include "tflite/c/common.h"

typedef struct LiteRtDispatchDelegateOptions LiteRtDispatchDelegateOptions;

// Create a delegate that uses the Dispatch API for execution.
//
// [TODO] DOCUMENT OWNERSHIP
TfLiteOpaqueDelegate* LiteRtCreateDispatchDelegate(
    LiteRtEnvironmentOptions environment_options, LiteRtOptions options);

// Do any needed cleanup and delete 'delegate'.
void LiteRtDestroyDispatchDelegate(TfLiteOpaqueDelegate* delegate);

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//
// Metrics
//

// Start collection of HW-specific metrics at a specific level of detail (>= 0).
LiteRtStatus LiteRtDispatchDelegateStartMetricsCollection(
    TfLiteOpaqueDelegate* delegate, int detail_level);

// Stop collection of HW-specific metrics and report the collected metrics.
LiteRtStatus LiteRtDispatchDelegateStopMetricsCollection(
    TfLiteOpaqueDelegate* delegate, LiteRtMetrics metrics);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // ODML_LITERT_LITERT_C_LITERT_DISPATCH_DELEGATE_H_
