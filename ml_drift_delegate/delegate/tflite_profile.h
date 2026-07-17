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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_TFLITE_PROFILE_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_TFLITE_PROFILE_H_

#include <cstdint>

#include "ml_drift/common/task/profiling_info.h"  // from @ml_drift
#include "tflite/c/common.h"

namespace ml_drift {

// Returns if TFLite Profiler is active.
bool IsTfLiteProfilerActive(TfLiteContext* context);

// Returns saved TFLite Profiler object.
void* GetTfLiteProfiler(TfLiteContext* context);

// Generate TFLite Profiler events with the given ProfilingInfo object.
void AddTfLiteProfilerEvents(TfLiteContext* context,
                             ml_drift::ProfilingInfo* profiling_info);

// Generate a TFLite Profiler event with the given label and elapsed time.
void AddTfLiteProfilerEvent(TfLiteContext* context, const char* label,
                            uint64_t elapsed_time_us);

}  // namespace ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_TFLITE_PROFILE_H_
