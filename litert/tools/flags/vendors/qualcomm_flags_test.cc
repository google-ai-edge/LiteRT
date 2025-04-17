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

#include "litert/tools/flags/vendors/qualcomm_flags.h"

#include <string>

#include <gtest/gtest.h>
#include "litert/c/options/litert_qualcomm_options.h"

namespace litert::qualcomm {
namespace {

TEST(LogLevelFlagTest, Malformed) {
  std::string error;
  LiteRtQualcommOptionsLogLevel value;

  EXPECT_FALSE(AbslParseFlag("oogabooga", &value, &error));
}

TEST(LogLevelFlagTest, Parse) {
  std::string error;
  LiteRtQualcommOptionsLogLevel value;

  {
    static constexpr absl::string_view kLevel = "off";
    static constexpr LiteRtQualcommOptionsLogLevel kLevelEnum =
        kLiteRtQualcommLogOff;
    EXPECT_TRUE(AbslParseFlag(kLevel, &value, &error));
    EXPECT_EQ(value, kLevelEnum);
    EXPECT_EQ(kLevel, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kLevel = "error";
    static constexpr LiteRtQualcommOptionsLogLevel kLevelEnum =
        kLiteRtQualcommLogLevelError;
    EXPECT_TRUE(AbslParseFlag(kLevel, &value, &error));
    EXPECT_EQ(value, kLevelEnum);
    EXPECT_EQ(kLevel, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kLevel = "warn";
    static constexpr LiteRtQualcommOptionsLogLevel kLevelEnum =
        kLiteRtQualcommLogLevelWarn;
    EXPECT_TRUE(AbslParseFlag(kLevel, &value, &error));
    EXPECT_EQ(value, kLevelEnum);
    EXPECT_EQ(kLevel, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kLevel = "info";
    static constexpr LiteRtQualcommOptionsLogLevel kLevelEnum =
        kLiteRtQualcommLogLevelInfo;
    EXPECT_TRUE(AbslParseFlag(kLevel, &value, &error));
    EXPECT_EQ(value, kLevelEnum);
    EXPECT_EQ(kLevel, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kLevel = "verbose";
    static constexpr LiteRtQualcommOptionsLogLevel kLevelEnum =
        kLiteRtQualcommLogLevelVerbose;
    EXPECT_TRUE(AbslParseFlag(kLevel, &value, &error));
    EXPECT_EQ(value, kLevelEnum);
    EXPECT_EQ(kLevel, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kLevel = "debug";
    static constexpr LiteRtQualcommOptionsLogLevel kLevelEnum =
        kLiteRtQualcommLogLevelDebug;
    EXPECT_TRUE(AbslParseFlag(kLevel, &value, &error));
    EXPECT_EQ(value, kLevelEnum);
    EXPECT_EQ(kLevel, AbslUnparseFlag(value));
  }
}

TEST(PowerModeFlagTest, Malformed) {
  std::string error;
  LiteRtQualcommOptionsPowerMode value;

  EXPECT_FALSE(AbslParseFlag("boogabooga", &value, &error));
}

TEST(PowerModeFlagTest, Parse) {
  std::string error;
  LiteRtQualcommOptionsPowerMode value;

  {
    static constexpr absl::string_view kMode = "unknown";
    static constexpr LiteRtQualcommOptionsPowerMode kModeEnum =
        kLiteRtQualcommPowerModeUnknown;
    EXPECT_TRUE(AbslParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kMode = "performance";
    static constexpr LiteRtQualcommOptionsPowerMode kModeEnum =
        kLiteRtQualcommPowerModePerformance;
    EXPECT_TRUE(AbslParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kMode = "power_saver";
    static constexpr LiteRtQualcommOptionsPowerMode kModeEnum =
        kLiteRtQualcommPowerModePowerSaver;
    EXPECT_TRUE(AbslParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, AbslUnparseFlag(value));
  }
}

}  // namespace
}  // namespace litert::qualcomm
