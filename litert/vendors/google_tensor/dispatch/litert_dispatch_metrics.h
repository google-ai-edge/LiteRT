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

#ifndef ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_LITERT_DISPATCH_METRICS_H_
#define ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_LITERT_DISPATCH_METRICS_H_

#include <cstdint>
#include <string>
#include <vector>

#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_metrics.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/google_tensor/dispatch/sb_api.h"

class LiteRtDispatchMetricsT {
 public:
  explicit LiteRtDispatchMetricsT(const ThrInvocationMetrics& thr_metrics)
      : metric_names_(thr_metrics.metric_keys,
                      thr_metrics.metric_keys + thr_metrics.num_metrics),
        metric_values_(thr_metrics.metric_values,
                       thr_metrics.metric_values + thr_metrics.num_metrics) {}

  int GetNumMetrics() const { return metric_names_.size(); }

  LiteRtStatus GetMetric(int metric_index, LiteRtMetric& metric) const {
    if (metric_index < 0 || metric_index >= GetNumMetrics()) {
      LITERT_LOG(LITERT_ERROR, "Metric index %d is out of bounds [0, %d)",
                 metric_index, GetNumMetrics());
      return kLiteRtStatusErrorInvalidArgument;
    }

    metric.name = metric_names_[metric_index].c_str();
    metric.value.type = kLiteRtAnyTypeInt;
    metric.value.int_value = metric_values_[metric_index];

    return kLiteRtStatusOk;
  }

 private:
  const std::vector<std::string> metric_names_;
  const std::vector<int64_t> metric_values_;
};

#endif  // ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_LITERT_DISPATCH_METRICS_H_
