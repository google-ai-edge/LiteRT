// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/schema/soc_table.h"

#include <optional>

#include <gtest/gtest.h>

namespace qnn {
namespace {

TEST(SocTableTest, FindSocInfoTest) {
  // Test finding a valid SoC (SDM865).
  static constexpr auto kSdm865 = FindSocInfo("SDM865");
  ASSERT_TRUE(kSdm865.has_value());
  EXPECT_EQ(kSdm865->soc_name, "SDM865");
  EXPECT_EQ(kSdm865->soc_model, 21);

  // Test finding another valid SoC (SM8550).
  static constexpr auto kSm8550 = FindSocInfo("SM8550");
  ASSERT_TRUE(kSm8550.has_value());
  EXPECT_EQ(kSm8550->soc_name, "SM8550");
  EXPECT_EQ(kSm8550->soc_model, 43);

  // Test finding an invalid SoC.
  EXPECT_FALSE(FindSocInfo("NONEXISTENT").has_value());

  // Test the nullptr guard.
  EXPECT_FALSE(FindSocInfo(nullptr).has_value());
}

TEST(SocTableTest, CreateSocInfoTest) {
  static constexpr auto kSm8550 = FindSocInfo("SM8550");
  // Test creating an SoC with SoC name.
  auto res_soc_name = FindOrCreateSocInfo("SM8550");
  ASSERT_TRUE(res_soc_name.has_value());
  EXPECT_EQ(res_soc_name->soc_name, kSm8550->soc_name);
  EXPECT_EQ(res_soc_name->soc_model, kSm8550->soc_model);

  // Test creating an SoC with SoC model.
  auto res_soc_model = FindOrCreateSocInfo("43");
  ASSERT_TRUE(res_soc_model.has_value());
  EXPECT_EQ(res_soc_model->soc_model, kSm8550->soc_model);

  // Test the nullptr guard.
  EXPECT_FALSE(FindOrCreateSocInfo(nullptr).has_value());

  // Test that a partially-numeric string is rejected.
  EXPECT_FALSE(FindOrCreateSocInfo("43x").has_value());
  EXPECT_FALSE(FindOrCreateSocInfo("abc").has_value());

  // Test that an empty string is rejected (neither a name nor a number).
  EXPECT_FALSE(FindOrCreateSocInfo("").has_value());

  // Test that a numeric value that overflows uint32_t is rejected.
  EXPECT_FALSE(FindOrCreateSocInfo("99999999999").has_value());
}

}  // namespace
}  // namespace qnn
