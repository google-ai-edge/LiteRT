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

#include "litert/tools/flags/vendors/google_tensor_flags.h"

#include <string>

#include <gtest/gtest.h>
#include "litert/c/options/litert_google_tensor_options.h"
#include "litert/c/options/litert_google_tensor_options_type.h"

namespace litert::google_tensor {
namespace {

TEST(TruncationTypeFlagTest, Malformed) {
  std::string error;
  LiteRtGoogleTensorOptionsTruncationType value;

  EXPECT_FALSE(AbslParseFlag("oogabooga", &value, &error));
}

TEST(TruncationTypeFlagTest, Parse) {
  std::string error;
  LiteRtGoogleTensorOptionsTruncationType value;

  {
    static constexpr absl::string_view kLevel = "auto";
    static constexpr LiteRtGoogleTensorOptionsTruncationType kLevelEnum =
        kLiteRtGoogleTensorFloatTruncationTypeAuto;
    EXPECT_TRUE(AbslParseFlag(kLevel, &value, &error));
    EXPECT_EQ(value, kLevelEnum);
    EXPECT_EQ(kLevel, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kLevel = "no_truncation";
    static constexpr LiteRtGoogleTensorOptionsTruncationType kLevelEnum =
        kLiteRtGoogleTensorFloatTruncationTypeNoTruncation;
    EXPECT_TRUE(AbslParseFlag(kLevel, &value, &error));
    EXPECT_EQ(value, kLevelEnum);
    EXPECT_EQ(kLevel, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kLevel = "bfloat16";
    static constexpr LiteRtGoogleTensorOptionsTruncationType kLevelEnum =
        kLiteRtGoogleTensorFloatTruncationTypeBfloat16;
    EXPECT_TRUE(AbslParseFlag(kLevel, &value, &error));
    EXPECT_EQ(value, kLevelEnum);
    EXPECT_EQ(kLevel, AbslUnparseFlag(value));
  }
  {
    static constexpr absl::string_view kLevel = "half";
    static constexpr LiteRtGoogleTensorOptionsTruncationType kLevelEnum =
        kLiteRtGoogleTensorFloatTruncationTypeHalf;
    EXPECT_TRUE(AbslParseFlag(kLevel, &value, &error));
    EXPECT_EQ(value, kLevelEnum);
    EXPECT_EQ(kLevel, AbslUnparseFlag(value));
  }
}
}  // namespace

}  // namespace litert::google_tensor
