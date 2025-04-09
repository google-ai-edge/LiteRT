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

#include "litert/c/litert_metrics.h"

#include "litert/c/litert_common.h"
#include "litert/runtime/metrics.h"

#ifdef __cplusplus
extern "C" {
#endif

LiteRtStatus LiteRtCreateMetrics(LiteRtMetrics* metrics) {
  *metrics = new LiteRtMetricsT();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumMetrics(LiteRtMetrics metrics, int* num_metrics) {
  if (!metrics || !num_metrics) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_metrics = metrics->metrics.size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetMetric(LiteRtMetrics metrics, int metric_index,
                             LiteRtMetric* metric) {
  if (!metrics || !metric) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (metric_index < 0 || metric_index >= metrics->metrics.size()) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& litert_metric = metrics->metrics[metric_index];
  *metric = {.name = litert_metric.name.c_str(), .value = litert_metric.value};
  return kLiteRtStatusOk;
}

void LiteRtDestroyMetrics(LiteRtMetrics metrics) { delete metrics; }

#ifdef __cplusplus
}
#endif  // __cplusplus
