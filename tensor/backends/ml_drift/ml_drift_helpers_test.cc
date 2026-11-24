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
#include "absl/time/time.h"  // from @com_google_absl
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

TEST(AdditiveZeroPointsTest, UsesFloatZeroPoints) {
  const graph::PerChannelAffineQuantization pcq(
      /*scales=*/{0.5f, 2.0f},
      /*zero_points=*/graph::ZeroPoints::Float({-0.5f, 1.25f}),
      /*quantized_dimension=*/0);
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
      /*scales=*/{0.5f, 2.0f, 1.0f},
      /*zero_points=*/graph::ZeroPoints::Float({0.5f, -0.5f}),
      /*quantized_dimension=*/0);
  EXPECT_THAT(AdditiveZeroPoints(pcq, 3),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(MlDriftBuildContextTest, AddWeightPackStatAccumulatesByKind) {
  MlDriftBuildContext ctx;
  ctx.AddWeightPackStat("fp", absl::Milliseconds(10), 1024);
  ctx.AddWeightPackStat("fp", absl::Milliseconds(25), 2048);
  ctx.AddWeightPackStat("int4", absl::Milliseconds(5), 512);
  ctx.AddWeightPackStat("gpu_convert", absl::Milliseconds(7), 4096,
                        /*count=*/3);

  ASSERT_TRUE(ctx.weight_pack_stats.contains("fp"));
  EXPECT_EQ(ctx.weight_pack_stats.at("fp").count, 2);
  EXPECT_EQ(ctx.weight_pack_stats.at("fp").bytes, 3072u);
  EXPECT_EQ(ctx.weight_pack_stats.at("fp").time, absl::Milliseconds(35));

  ASSERT_TRUE(ctx.weight_pack_stats.contains("int4"));
  EXPECT_EQ(ctx.weight_pack_stats.at("int4").count, 1);
  EXPECT_EQ(ctx.weight_pack_stats.at("int4").bytes, 512u);
  EXPECT_EQ(ctx.weight_pack_stats.at("int4").time, absl::Milliseconds(5));

  ASSERT_TRUE(ctx.weight_pack_stats.contains("gpu_convert"));
  EXPECT_EQ(ctx.weight_pack_stats.at("gpu_convert").count, 3);
  EXPECT_EQ(ctx.weight_pack_stats.at("gpu_convert").bytes, 4096u);
}

}  // namespace
}  // namespace litert::tensor
