// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/backends/backend_utils.h"

#include <array>
#include <optional>
#include <vector>

#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/schema/soc_table.h"

namespace qnn {
namespace {
TEST(BackendUtilsTest, SetNullTermPtrArrayTest) {
  std::vector<int> src = {1, 2, 3};
  std::array<const int*, 5> dst{};
  SetNullTermPtrArray(absl::MakeConstSpan(src), dst);

  ASSERT_TRUE(dst[0]);
  EXPECT_EQ(*dst[0], 1);
  ASSERT_TRUE(dst[1]);
  EXPECT_EQ(*dst[1], 2);
  ASSERT_TRUE(dst[2]);
  EXPECT_EQ(*dst[2], 3);
  EXPECT_FALSE(dst[3]);
}

TEST(BackendUtilsTest, SetNullTermPtrArrayTestOverflow) {
  std::vector<int> src = {1, 2, 3, 4, 5};
  std::array<const int*, 4> dst{};
  SetNullTermPtrArray(absl::MakeConstSpan(src), dst);

  ASSERT_TRUE(dst[0]);
  EXPECT_EQ(*dst[0], 1);
  ASSERT_TRUE(dst[1]);
  EXPECT_EQ(*dst[1], 2);
  ASSERT_TRUE(dst[2]);
  EXPECT_EQ(*dst[2], 3);
  EXPECT_FALSE(dst[3]);
}

TEST(BackendUtilsTest, FindSocInfoTest) {
  // Test finding a valid SoC (SDM865)
  auto res = FindSocInfo(SnapdragonModel::SDM865);
  ASSERT_TRUE(res.has_value());
  EXPECT_STREQ(res->soc_name, "SDM865");
  EXPECT_EQ(res->soc_model, SnapdragonModel::SDM865);
  EXPECT_EQ(res->dsp_arch, DspArch::V66);
  EXPECT_EQ(res->vtcm_size_in_mb, 0);

  // Test finding another valid SoC (SM8550)
  res = FindSocInfo(SnapdragonModel::SM8550);
  ASSERT_TRUE(res.has_value());
  EXPECT_STREQ(res->soc_name, "SM8550");
  EXPECT_EQ(res->soc_model, SnapdragonModel::SM8550);
  EXPECT_EQ(res->dsp_arch, DspArch::V73);
  EXPECT_EQ(res->vtcm_size_in_mb, 8);

  // Test finding an invalid SoC
  auto res_invalid = FindSocInfo(static_cast<SnapdragonModel>(99999));
  EXPECT_FALSE(res_invalid.has_value());
}

}  // namespace
}  // namespace qnn
