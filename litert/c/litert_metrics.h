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

#ifndef ODML_LITERT_LITERT_C_LITERT_METRICS_H_
#define ODML_LITERT_LITERT_C_LITERT_METRICS_H_

#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct LiteRtMetric {
  const char* name;
  LiteRtAny value;
} LiteRtMetric;

// Create a metrics object. The caller is responsible for deallocating the
// returned metrics by calling `LiteRtDestroyMetrics`.
LiteRtStatus LiteRtCreateMetrics(LiteRtMetrics* metrics);

// Get the number of metrics collected.
LiteRtStatus LiteRtGetNumMetrics(LiteRtMetrics metrics, int* num_metrics);

// Fetch a specific metric.
// NOTE The output object fields are owned by the `metrics` object and should
// not be used after the `metrics` object has been destroyed.
LiteRtStatus LiteRtGetMetric(LiteRtMetrics metrics, int metric_index,
                             LiteRtMetric* metric);

// Destroy the metrics object.
void LiteRtDestroyMetrics(LiteRtMetrics metrics);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // ODML_LITERT_LITERT_C_LITERT_METRICS_H_
