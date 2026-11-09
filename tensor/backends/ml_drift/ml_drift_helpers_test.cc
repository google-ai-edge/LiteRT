/* Copyright 2025 Google LLC.

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

#include "tensor/backends/ml_drift/ml_drift_helpers.h"

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "tensor/internal/graph.h"

namespace litert::tensor {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::FloatEq;

TEST(AdditiveZeroPointsTest, UsesIntegerZeroPoints) {
  const graph::PerChannelAffineQuantization pcq(
      /*scales=*/{0.5f, 2.0f}, /*zero_points=*/std::vector<int64_t>{3, -1},
      /*quantized_dimension=*/0);
  EXPECT_THAT(AdditiveZeroPoints(pcq, 2),
              IsOkAndHolds(ElementsAre(FloatEq(-1.5f), FloatEq(2.0f))));
}

TEST(AdditiveZeroPointsTest, AllZeroFloatZeroPointsKeepIntegerZeroPoints) {
  const graph::PerChannelAffineQuantization pcq(
      /*scales=*/{0.5f, 2.0f}, /*zero_points=*/std::vector<int64_t>{3, -1},
      /*quantized_dimension=*/0, /*float_zero_points=*/{0.0f, 0.0f});
  EXPECT_THAT(AdditiveZeroPoints(pcq, 2),
              IsOkAndHolds(ElementsAre(FloatEq(-1.5f), FloatEq(2.0f))));
}

TEST(AdditiveZeroPointsTest, UsesNonZeroFloatZeroPoints) {
  const graph::PerChannelAffineQuantization pcq(
      /*scales=*/{0.5f, 2.0f}, /*zero_points=*/std::vector<int64_t>{},
      /*quantized_dimension=*/0, /*float_zero_points=*/{-0.5f, 1.25f});
  EXPECT_THAT(AdditiveZeroPoints(pcq, 2),
              IsOkAndHolds(ElementsAre(FloatEq(0.25f), FloatEq(-2.5f))));
}

TEST(AdditiveZeroPointsTest, BroadcastsPerTensorScaleAndZeroPoint) {
  const graph::PerChannelAffineQuantization pcq(
      /*scales=*/{0.5f}, /*zero_points=*/std::vector<int64_t>{4},
      /*quantized_dimension=*/0);
  EXPECT_THAT(AdditiveZeroPoints(pcq, 3),
              IsOkAndHolds(
                  ElementsAre(FloatEq(-2.0f), FloatEq(-2.0f), FloatEq(-2.0f))));
}

TEST(AdditiveZeroPointsTest, NoZeroPoints) {
  const graph::PerChannelAffineQuantization pcq(
      /*scales=*/{0.5f, 2.0f}, /*zero_points=*/std::vector<int64_t>{},
      /*quantized_dimension=*/0);
  EXPECT_THAT(AdditiveZeroPoints(pcq, 2),
              IsOkAndHolds(ElementsAre(FloatEq(0.0f), FloatEq(0.0f))));
}

TEST(AdditiveZeroPointsTest, RejectsNegativeChannelCount) {
  const graph::PerChannelAffineQuantization pcq(
      /*scales=*/{0.5f}, /*zero_points=*/std::vector<int64_t>{},
      /*quantized_dimension=*/0);
  EXPECT_THAT(AdditiveZeroPoints(pcq, -1),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(AdditiveZeroPointsTest, RejectsMismatchedScaleCount) {
  const graph::PerChannelAffineQuantization pcq(
      /*scales=*/{0.5f, 2.0f}, /*zero_points=*/std::vector<int64_t>{},
      /*quantized_dimension=*/0);
  EXPECT_THAT(AdditiveZeroPoints(pcq, 3),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(AdditiveZeroPointsTest, RejectsMismatchedZeroPointCount) {
  const graph::PerChannelAffineQuantization pcq(
      /*scales=*/{0.5f, 2.0f, 1.0f},
      /*zero_points=*/std::vector<int64_t>{1, 2},
      /*quantized_dimension=*/0);
  EXPECT_THAT(AdditiveZeroPoints(pcq, 3),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(AdditiveZeroPointsTest, RejectsMismatchedFloatZeroPointCount) {
  const graph::PerChannelAffineQuantization pcq(
      /*scales=*/{0.5f, 2.0f, 1.0f}, /*zero_points=*/std::vector<int64_t>{},
      /*quantized_dimension=*/0, /*float_zero_points=*/{0.5f, -0.5f});
  EXPECT_THAT(AdditiveZeroPoints(pcq, 3),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace litert::tensor
