
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

#include "ml_drift_delegate/delegate/tflite_profile.h"

#include <cstdint>

#include "absl/time/time.h"  // from @com_google_absl
#include "ml_drift/common/task/profiling_info.h"  // from @ml_drift
#include "tflite/c/common.h"
#include "tflite/core/api/profiler.h"

namespace ml_drift {

bool IsTfLiteProfilerActive(TfLiteContext* context) {
  return context->profiler != nullptr;
}

void* GetTfLiteProfiler(TfLiteContext* context) { return context->profiler; }

void AddTfLiteProfilerEvents(TfLiteContext* const context,
                             ml_drift::ProfilingInfo* profiling_info) {
  tflite::Profiler* profile =
      reinterpret_cast<tflite::Profiler*>(GetTfLiteProfiler(context));
  if (profile == nullptr) return;

  int node_index = 0;
  for (const auto& dispatch : profiling_info->dispatches) {
    profile->AddEvent(
        dispatch.label.c_str(),
        tflite::Profiler::EventType::DELEGATE_PROFILED_OPERATOR_INVOKE_EVENT,
        absl::ToDoubleMicroseconds(dispatch.duration), node_index++);
  }
}

void AddTfLiteProfilerEvent(TfLiteContext* const context, const char* label,
                            uint64_t elapsed_time_us) {
  tflite::Profiler* profile =
      reinterpret_cast<tflite::Profiler*>(GetTfLiteProfiler(context));
  if (profile == nullptr) return;
  profile->AddEvent(
      label,
      tflite::Profiler::EventType::DELEGATE_PROFILED_OPERATOR_INVOKE_EVENT,
      elapsed_time_us, 0);
}

}  // namespace ml_drift
