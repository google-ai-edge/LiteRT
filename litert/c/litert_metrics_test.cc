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

#include "litert/c/litert_metrics.h"

#include <gtest/gtest.h>
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"

namespace {

TEST(LiteRtMetricsTest, CreateRejectsNullOutput) {
  EXPECT_EQ(LiteRtCreateMetrics(nullptr), kLiteRtStatusErrorInvalidArgument);
}

TEST(LiteRtMetricsTest, CreateReturnsEmptyMetrics) {
  LiteRtMetrics metrics = nullptr;
  EXPECT_EQ(LiteRtCreateMetrics(&metrics), kLiteRtStatusOk);
  ASSERT_NE(metrics, nullptr);

  int num_metrics = -1;
  EXPECT_EQ(LiteRtGetNumMetrics(metrics, &num_metrics), kLiteRtStatusOk);
  EXPECT_EQ(num_metrics, 0);

  LiteRtDestroyMetrics(metrics);
}

TEST(LiteRtMetricsTest, AppendMetricSuccess) {
  LiteRtMetrics metrics = nullptr;
  ASSERT_EQ(LiteRtCreateMetrics(&metrics), kLiteRtStatusOk);

  LiteRtMetric metric1 = {
      .name = "test_int_metric",
      .value = LiteRtAny{
          .type = kLiteRtAnyTypeInt,
          .int_value = 12345,
      },
  };
  EXPECT_EQ(LiteRtAppendMetric(metrics, &metric1), kLiteRtStatusOk);

  int num_metrics = 0;
  EXPECT_EQ(LiteRtGetNumMetrics(metrics, &num_metrics), kLiteRtStatusOk);
  EXPECT_EQ(num_metrics, 1);

  LiteRtMetric retrieved_metric;
  EXPECT_EQ(LiteRtGetMetric(metrics, 0, &retrieved_metric), kLiteRtStatusOk);
  EXPECT_STREQ(retrieved_metric.name, "test_int_metric");
  EXPECT_EQ(retrieved_metric.value.type, kLiteRtAnyTypeInt);
  EXPECT_EQ(retrieved_metric.value.int_value, 12345);

  LiteRtDestroyMetrics(metrics);
}

TEST(LiteRtMetricsTest, AppendMetricNullChecks) {
  LiteRtMetrics metrics = nullptr;
  ASSERT_EQ(LiteRtCreateMetrics(&metrics), kLiteRtStatusOk);

  LiteRtMetric valid_metric = {
      .name = "valid",
      .value = LiteRtAny{.type = kLiteRtAnyTypeInt, .int_value = 1},
  };
  EXPECT_EQ(LiteRtAppendMetric(nullptr, &valid_metric),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(LiteRtAppendMetric(metrics, nullptr),
            kLiteRtStatusErrorInvalidArgument);

  LiteRtMetric null_name_metric = {
      .name = nullptr,
      .value = LiteRtAny{.type = kLiteRtAnyTypeInt, .int_value = 1},
  };
  EXPECT_EQ(LiteRtAppendMetric(metrics, &null_name_metric),
            kLiteRtStatusErrorInvalidArgument);

  LiteRtDestroyMetrics(metrics);
}

}  // namespace
