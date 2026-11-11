// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/intel_openvino/openvino_version_info.h"

#include <string>

#include <gtest/gtest.h>
#include "openvino/core/any.hpp"

namespace litert::openvino {
namespace {

TEST(ResolveNpuPlatformForLogTest, CompilePlatformWinsOverGlobal) {
  const ov::AnyMap props = {{"NPU_PLATFORM", std::string("NPU5010")}};
  EXPECT_EQ(ResolveNpuPlatformForLog(props, "NPU4000"), "NPU5010");
}

TEST(ResolveNpuPlatformForLogTest, CompilePlatformWinsOverAutoDetectDefault) {
  const ov::AnyMap props = {{"NPU_PLATFORM", std::string("NPU5010")}};
  EXPECT_EQ(ResolveNpuPlatformForLog(props, "AUTO_DETECT"), "NPU5010");
}

// LiteRT stores the SoC platform as a C string in the configs map.
TEST(ResolveNpuPlatformForLogTest, AcceptsCStringValue) {
  const ov::AnyMap props = {{"NPU_PLATFORM", "NPU5010"}};
  EXPECT_EQ(ResolveNpuPlatformForLog(props, "AUTO_DETECT"), "NPU5010");
}

TEST(ResolveNpuPlatformForLogTest, DefaultIsNeverReportedAsPlatform) {
  EXPECT_EQ(ResolveNpuPlatformForLog({}, "AUTO_DETECT"), "");
  const ov::AnyMap props = {{"NPU_PLATFORM", std::string("AUTO_DETECT")}};
  EXPECT_EQ(ResolveNpuPlatformForLog(props, "AUTO_DETECT"), "");
}

TEST(ResolveNpuPlatformForLogTest, FallsBackToGlobalWhenNotInCompileProps) {
  EXPECT_EQ(ResolveNpuPlatformForLog({}, "NPU4000"), "NPU4000");
  const ov::AnyMap props = {{"NPU_PLATFORM", std::string()}};
  EXPECT_EQ(ResolveNpuPlatformForLog(props, "NPU4000"), "NPU4000");
}

}  // namespace
}  // namespace litert::openvino
