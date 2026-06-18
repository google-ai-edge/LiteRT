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

}  // namespace
