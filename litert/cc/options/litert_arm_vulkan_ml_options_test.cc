// Copyright 2026 The LiteRT Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "litert/cc/options/litert_arm_vulkan_ml_options.h"

#include <string>

#include <gtest/gtest.h>

namespace litert::arm_vulkan_ml {
namespace {
TEST(ArmVulkanMLOptionsTest, DefaultAndSerializedModesRoundTrip) {
  auto options = ArmVulkanMLOptions::Create(); ASSERT_TRUE(options);
  auto default_mode = options->GetOptimizationMode(); ASSERT_TRUE(default_mode);
  EXPECT_EQ(*default_mode, kLrtArmVulkanMLOptimizationQualified);
  for (auto mode : {kLrtArmVulkanMLOptimizationOff, kLrtArmVulkanMLOptimizationExact,
                    kLrtArmVulkanMLOptimizationQualified}) {
    ASSERT_TRUE(options->SetOptimizationMode(mode));
    ASSERT_TRUE(options->SetEnableJustInTime(true));
    const char* identifier = nullptr; void* payload = nullptr; void (*deleter)(void*) = nullptr;
    ASSERT_EQ(options->GetOpaqueOptionsData(&identifier, &payload, &deleter), kLiteRtStatusOk);
    LrtArmVulkanMLOptions parsed = nullptr;
    ASSERT_EQ(LrtCreateArmVulkanMLOptionsFromToml(static_cast<const char*>(payload), &parsed), kLiteRtStatusOk);
    deleter(payload);
    ArmVulkanMLOptions copy(parsed);
    auto copied_mode = copy.GetOptimizationMode(); auto jit = copy.GetEnableJustInTime();
    ASSERT_TRUE(copied_mode); ASSERT_TRUE(jit);
    EXPECT_EQ(*copied_mode, mode); EXPECT_TRUE(*jit);
  }
}
TEST(ArmVulkanMLOptionsTest, SoftmaxTableTypesRoundTripAndRejectInvalidValues) {
  auto options = ArmVulkanMLOptions::Create(); ASSERT_TRUE(options);
  auto defaults = options->GetSoftmaxLutType(); ASSERT_TRUE(defaults);
  EXPECT_EQ(*defaults, kLrtArmVulkanMLSoftmaxLutAuto);
  for (auto type : {kLrtArmVulkanMLSoftmaxLutAuto, kLrtArmVulkanMLSoftmaxLutInt8, kLrtArmVulkanMLSoftmaxLutInt16}) {
    ASSERT_TRUE(options->SetSoftmaxLutType(type));
    const char* identifier = nullptr; void* payload = nullptr; void (*deleter)(void*) = nullptr;
    ASSERT_EQ(options->GetOpaqueOptionsData(&identifier, &payload, &deleter), kLiteRtStatusOk);
    LrtArmVulkanMLOptions parsed = nullptr;
    ASSERT_EQ(LrtCreateArmVulkanMLOptionsFromToml(static_cast<const char*>(payload), &parsed), kLiteRtStatusOk);
    deleter(payload); ArmVulkanMLOptions copy(parsed);
    auto actual = copy.GetSoftmaxLutType(); ASSERT_TRUE(actual); EXPECT_EQ(*actual, type);
  }
  EXPECT_FALSE(options->SetSoftmaxLutType(static_cast<LrtArmVulkanMLSoftmaxLutType>(77)));
  EXPECT_EQ(LrtArmVulkanMLOptionsGetSoftmaxLutType(options->Get(), nullptr), kLiteRtStatusErrorInvalidArgument);
  LrtArmVulkanMLOptions bad = nullptr;
  EXPECT_EQ(LrtCreateArmVulkanMLOptionsFromToml("softmax_lut_type='bad'", &bad), kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(bad, nullptr);
}

TEST(ArmVulkanMLOptionsTest, RejectsInvalidModesAndNullArguments) {
  auto options = ArmVulkanMLOptions::Create(); ASSERT_TRUE(options);
  EXPECT_FALSE(options->SetOptimizationMode(static_cast<LrtArmVulkanMLOptimizationMode>(77)));
  EXPECT_EQ(LrtArmVulkanMLOptionsSetOptimizationMode(nullptr, kLrtArmVulkanMLOptimizationOff), kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(LrtArmVulkanMLOptionsGetOptimizationMode(options->Get(), nullptr), kLiteRtStatusErrorInvalidArgument);
  for (const char* text : {"optimization_mode = 'typo'", "optimization_mode = 1"}) {
    LrtArmVulkanMLOptions parsed = nullptr;
    EXPECT_EQ(LrtCreateArmVulkanMLOptionsFromToml(text, &parsed), kLiteRtStatusErrorInvalidArgument);
    EXPECT_EQ(parsed, nullptr);
  }
}
}  // namespace
}  // namespace litert::arm_vulkan_ml
