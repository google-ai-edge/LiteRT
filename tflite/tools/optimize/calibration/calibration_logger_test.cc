/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tflite/tools/optimize/calibration/calibration_logger.h"

#include <cmath>
#include <cstddef>
#include <limits>

#include <gtest/gtest.h>
#include "tflite/core/c/c_api_types.h"

namespace tflite {
namespace optimize {
namespace calibration {
namespace {

constexpr float kSignedValues[] = {-10.0f, -5.0f, 0.0f, 5.0f, 10.0f};

// Runs the first (min/max) pass over `values`, switches to histogram mode, and
// runs the second pass over `values` `repeats` times.
void RunTwoPasses(MinMax& minmax, const float* values, size_t size,
                  int num_bins, double lower_percentile,
                  double upper_percentile, int repeats = 10) {
  minmax.Update(values, size, nullptr);
  minmax.InitializeHistogram(num_bins, lower_percentile, upper_percentile);
  for (int i = 0; i < repeats; ++i) {
    minmax.Update(values, size, nullptr);
  }
}

TEST(MinMaxTest, GetWithoutValuesFails) {
  MinMax minmax;
  float min_val, max_val;
  EXPECT_EQ(minmax.Get(&min_val, &max_val), kTfLiteError);
}

TEST(MinMaxTest, ReturnsMinAndMax) {
  MinMax minmax;
  const float values[] = {-10.0f, -5.0f, 0.0f, 5.0f, 10.0f};
  ASSERT_EQ(minmax.Update(values, 5, nullptr), kTfLiteOk);

  float min_val, max_val;
  ASSERT_EQ(minmax.Get(&min_val, &max_val), kTfLiteOk);
  EXPECT_FLOAT_EQ(min_val, -10.0f);
  EXPECT_FLOAT_EQ(max_val, 10.0f);
}

TEST(MinMaxTest, HistogramClipsBothTailsOfSignedTensor) {
  MinMax minmax;
  // 10 bins of width 2 over [-10, 10]. The values land in bins 0, 2, 5, 7 and
  // 9 (10 is clamped into the last bin), 10 samples each.
  RunTwoPasses(minmax, kSignedValues, 5, /*num_bins=*/10,
               /*lower_percentile=*/25.0, /*upper_percentile=*/75.0);

  float min_val, max_val;
  ASSERT_EQ(minmax.Get(&min_val, &max_val), kTfLiteOk);
  // 25% of 50 samples is reached in bin 2 and 75% in bin 7.
  EXPECT_FLOAT_EQ(min_val, -6.0f);
  EXPECT_FLOAT_EQ(max_val, 6.0f);
}

TEST(MinMaxTest, HistogramWithFullPercentileRangeKeepsMinAndMax) {
  MinMax minmax;
  RunTwoPasses(minmax, kSignedValues, 5, /*num_bins=*/10,
               /*lower_percentile=*/0.0, /*upper_percentile=*/100.0);

  float min_val, max_val;
  ASSERT_EQ(minmax.Get(&min_val, &max_val), kTfLiteOk);
  // The lower cutoff is in bin 0, so the lower bound must stay at the min.
  EXPECT_FLOAT_EQ(min_val, -10.0f);
  EXPECT_FLOAT_EQ(max_val, 10.0f);
}

TEST(MinMaxTest, HistogramKeepsZeroLowerBoundForNonNegativeTensor) {
  MinMax minmax;
  const float values[] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  // 4 bins of width 1 over [0, 4].
  RunTwoPasses(minmax, values, 5, /*num_bins=*/4, /*lower_percentile=*/25.0,
               /*upper_percentile=*/75.0, /*repeats=*/1);

  float min_val, max_val;
  ASSERT_EQ(minmax.Get(&min_val, &max_val), kTfLiteOk);
  EXPECT_FLOAT_EQ(min_val, 0.0f);
  // 75% of 5 samples (rounded down to 3) is reached in bin 2.
  EXPECT_FLOAT_EQ(max_val, 3.0f);
}

TEST(MinMaxTest, HistogramRangeAlwaysIncludesZero) {
  MinMax minmax;
  const float values[] = {2.0f, 3.0f, 4.0f};
  RunTwoPasses(minmax, values, 3, /*num_bins=*/8, /*lower_percentile=*/0.0,
               /*upper_percentile=*/100.0);

  float min_val, max_val;
  ASSERT_EQ(minmax.Get(&min_val, &max_val), kTfLiteOk);
  EXPECT_FLOAT_EQ(min_val, 0.0f);
  EXPECT_FLOAT_EQ(max_val, 4.0f);
}

TEST(MinMaxTest, HistogramClampsOutOfRangeAndInfiniteValues) {
  MinMax minmax;
  RunTwoPasses(minmax, kSignedValues, 5, /*num_bins=*/10,
               /*lower_percentile=*/0.0, /*upper_percentile=*/100.0);
  const float outliers[] = {-100.0f, 100.0f,
                            -std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::infinity()};
  ASSERT_EQ(minmax.Update(outliers, 4, nullptr), kTfLiteOk);

  float min_val, max_val;
  ASSERT_EQ(minmax.Get(&min_val, &max_val), kTfLiteOk);
  EXPECT_FLOAT_EQ(min_val, -10.0f);
  EXPECT_FLOAT_EQ(max_val, 10.0f);
}

TEST(MinMaxTest, NonFiniteRangeFallsBackToMinMax) {
  MinMax minmax;
  const float values[] = {-1.0f, std::numeric_limits<float>::infinity()};
  RunTwoPasses(minmax, values, 2, /*num_bins=*/10, /*lower_percentile=*/0.0,
               /*upper_percentile=*/100.0, /*repeats=*/1);

  float min_val, max_val;
  ASSERT_EQ(minmax.Get(&min_val, &max_val), kTfLiteOk);
  EXPECT_FLOAT_EQ(min_val, -1.0f);
  EXPECT_TRUE(std::isinf(max_val));
}

TEST(MinMaxTest, InvalidHistogramArgumentsAreClamped) {
  MinMax minmax;
  // num_bins becomes 1 and the percentiles become [0, 0].
  RunTwoPasses(minmax, kSignedValues, 5, /*num_bins=*/0,
               /*lower_percentile=*/150.0, /*upper_percentile=*/-5.0);

  float min_val, max_val;
  ASSERT_EQ(minmax.Get(&min_val, &max_val), kTfLiteOk);
  EXPECT_FLOAT_EQ(min_val, -10.0f);
  EXPECT_FLOAT_EQ(max_val, 10.0f);
}

TEST(MinMaxTest, NanPercentilesUseFullRange) {
  MinMax minmax;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  RunTwoPasses(minmax, kSignedValues, 5, /*num_bins=*/10,
               /*lower_percentile=*/nan, /*upper_percentile=*/nan);

  float min_val, max_val;
  ASSERT_EQ(minmax.Get(&min_val, &max_val), kTfLiteOk);
  EXPECT_FLOAT_EQ(min_val, -10.0f);
  EXPECT_FLOAT_EQ(max_val, 10.0f);
}

TEST(MinMaxTest, HugeBinCountIsCappedAndStaysInBounds) {
  // Regression test: with more than 2^24 bins, a float upper bound for the bin
  // index rounded up to num_bins and indexed one past the end of the bins.
  MinMax minmax;
  const float outliers[] = {10.0f, std::numeric_limits<float>::infinity()};
  minmax.Update(kSignedValues, 5, nullptr);
  minmax.InitializeHistogram(/*num_bins=*/16777220, /*lower_percentile=*/0.0,
                             /*upper_percentile=*/100.0);
  ASSERT_EQ(minmax.Update(kSignedValues, 5, nullptr), kTfLiteOk);
  ASSERT_EQ(minmax.Update(outliers, 2, nullptr), kTfLiteOk);

  float min_val, max_val;
  ASSERT_EQ(minmax.Get(&min_val, &max_val), kTfLiteOk);
  EXPECT_FLOAT_EQ(min_val, -10.0f);
  EXPECT_FLOAT_EQ(max_val, 10.0f);
}

TEST(LoggerTest, InitializeHistogramsAppliesToLoggedTensors) {
  Logger logger;
  ASSERT_EQ(logger.LogTensorValue(/*subgraph_index=*/0, /*tensor_index=*/0,
                                  kSignedValues, 5, nullptr),
            kTfLiteOk);
  logger.InitializeHistograms(/*num_bins=*/10, /*lower_percentile=*/25.0,
                              /*upper_percentile=*/75.0);
  for (int i = 0; i < 10; ++i) {
    ASSERT_EQ(logger.LogTensorValue(0, 0, kSignedValues, 5, nullptr),
              kTfLiteOk);
  }

  const MinMax& minmax = logger.GetCalibrationValues().at({0, 0});
  float min_val, max_val;
  ASSERT_EQ(minmax.Get(&min_val, &max_val), kTfLiteOk);
  EXPECT_FLOAT_EQ(min_val, -6.0f);
  EXPECT_FLOAT_EQ(max_val, 6.0f);
}

}  // namespace
}  // namespace calibration
}  // namespace optimize
}  // namespace tflite
