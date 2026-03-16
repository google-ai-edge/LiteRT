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

#include "litert/tools/gpu_numerics_threshold.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace litert {
namespace {

TEST(GpuNumericsThresholdTest, PassesWhenAllMetricsWithinThreshold) {
  const BufferDiffStats stats{
      .buffer_idx = 0,
      .total_elements = 1000,
      .diff_elements = 0,
      .epsilon = 1e-4,
      .max_diff = 5e-5,
      .min_diff = 0.0,
      .mean_diff = 2e-7,
      .mse = 1e-9,
  };
  const DiffThresholdConfig config{
      .fail_on_threshold = true,
      .max_abs_diff_threshold = 1e-4,
      .mean_abs_diff_threshold = 1e-6,
      .diff_ratio_threshold = 1e-6,
  };

  const auto eval = EvaluateDiffThresholds({&stats, 1}, config);
  EXPECT_FALSE(eval.has_violation);
  EXPECT_FALSE(eval.should_fail);
  EXPECT_TRUE(eval.violation_messages.empty());
}

TEST(GpuNumericsThresholdTest, FailsOnMaxAbsDiffViolation) {
  const BufferDiffStats stats{
      .buffer_idx = 1,
      .total_elements = 100,
      .diff_elements = 1,
      .epsilon = 1e-4,
      .max_diff = 2e-4,
      .min_diff = 0.0,
      .mean_diff = 1e-7,
      .mse = 1e-9,
  };
  const DiffThresholdConfig config{
      .fail_on_threshold = true,
      .max_abs_diff_threshold = 1e-4,
      .mean_abs_diff_threshold = 1e-6,
      .diff_ratio_threshold = 1.0,
  };

  const auto eval = EvaluateDiffThresholds({&stats, 1}, config);
  EXPECT_TRUE(eval.has_violation);
  EXPECT_TRUE(eval.should_fail);
  EXPECT_EQ(eval.violation_messages.size(), 1);
}

TEST(GpuNumericsThresholdTest, FailsOnMeanDiffViolation) {
  const BufferDiffStats stats{
      .buffer_idx = 2,
      .total_elements = 100,
      .diff_elements = 0,
      .epsilon = 1e-4,
      .max_diff = 5e-5,
      .min_diff = 0.0,
      .mean_diff = 2e-6,
      .mse = 1e-8,
  };
  const DiffThresholdConfig config{
      .fail_on_threshold = true,
      .max_abs_diff_threshold = 1e-4,
      .mean_abs_diff_threshold = 1e-6,
      .diff_ratio_threshold = 1.0,
  };

  const auto eval = EvaluateDiffThresholds({&stats, 1}, config);
  EXPECT_TRUE(eval.has_violation);
  EXPECT_TRUE(eval.should_fail);
  EXPECT_EQ(eval.violation_messages.size(), 1);
}

TEST(GpuNumericsThresholdTest, FailsOnDiffRatioViolation) {
  const BufferDiffStats stats{
      .buffer_idx = 3,
      .total_elements = 100,
      .diff_elements = 5,
      .epsilon = 1e-4,
      .max_diff = 5e-5,
      .min_diff = 0.0,
      .mean_diff = 1e-7,
      .mse = 1e-9,
  };
  const DiffThresholdConfig config{
      .fail_on_threshold = true,
      .max_abs_diff_threshold = 1e-3,
      .mean_abs_diff_threshold = 1e-5,
      .diff_ratio_threshold = 0.01,
  };

  const auto eval = EvaluateDiffThresholds({&stats, 1}, config);
  EXPECT_TRUE(eval.has_violation);
  EXPECT_TRUE(eval.should_fail);
  EXPECT_EQ(eval.violation_messages.size(), 1);
}

TEST(GpuNumericsThresholdTest, ReportsViolationButDoesNotFailWhenDisabled) {
  const BufferDiffStats stats{
      .buffer_idx = 4,
      .total_elements = 100,
      .diff_elements = 10,
      .epsilon = 1e-4,
      .max_diff = 1e-3,
      .min_diff = 0.0,
      .mean_diff = 1e-4,
      .mse = 1e-6,
  };
  const DiffThresholdConfig config{
      .fail_on_threshold = false,
      .max_abs_diff_threshold = 1e-4,
      .mean_abs_diff_threshold = 1e-6,
      .diff_ratio_threshold = 0.01,
  };

  const auto eval = EvaluateDiffThresholds({&stats, 1}, config);
  EXPECT_TRUE(eval.has_violation);
  EXPECT_FALSE(eval.should_fail);
  EXPECT_EQ(eval.violation_messages.size(), 3);
}

}  // namespace
}  // namespace litert
