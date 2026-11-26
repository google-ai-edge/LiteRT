// Copyright 2026 Amlogic, Inc.
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

#include "litert/tools/flags/vendors/amlogic_flags.h"

#include <string>

#include <gtest/gtest.h>
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/reflection.h"  // from @com_google_absl
#include "litert/c/options/litert_amlogic_options.h"
#include "litert/cc/options/litert_amlogic_options.h"
#include "litert/test/matchers.h"

namespace litert::amlogic {
namespace {

TEST(UpdateAmlogicOptionsFromFlagsTest, EmptyFlagsLeaveOptionsUnset) {
  absl::FlagSaver saver;
  absl::SetFlag(&FLAGS_amlogic_model_path, "");
  absl::SetFlag(&FLAGS_amlogic_model_name, "");
  absl::SetFlag(&FLAGS_amlogic_log_level, "");

  LITERT_ASSERT_OK_AND_ASSIGN(AmlogicOptions options, AmlogicOptions::Create());
  EXPECT_TRUE(UpdateAmlogicOptionsFromFlags(options).HasValue());
  EXPECT_FALSE(options.GetModelPath().HasValue());
  EXPECT_FALSE(options.GetModelName().HasValue());
  EXPECT_FALSE(options.GetLogLevel().HasValue());
}

TEST(UpdateAmlogicOptionsFromFlagsTest, AppliesCompileFlags) {
  absl::FlagSaver saver;
  absl::SetFlag(&FLAGS_amlogic_model_path, "/data/vendor/nn/");
  absl::SetFlag(&FLAGS_amlogic_model_name, "mobilenet");
  absl::SetFlag(&FLAGS_amlogic_log_level, "debug");

  LITERT_ASSERT_OK_AND_ASSIGN(AmlogicOptions options, AmlogicOptions::Create());
  EXPECT_TRUE(UpdateAmlogicOptionsFromFlags(options).HasValue());
  LITERT_ASSERT_OK_AND_ASSIGN(const char* model_path, options.GetModelPath());
  LITERT_ASSERT_OK_AND_ASSIGN(const char* model_name, options.GetModelName());
  LITERT_ASSERT_OK_AND_ASSIGN(const auto log_level, options.GetLogLevel());
  EXPECT_STREQ(model_path, "/data/vendor/nn/");
  EXPECT_STREQ(model_name, "mobilenet");
  EXPECT_EQ(log_level, kLrtAmlogicLogLevelDebug);
}

TEST(UpdateAmlogicOptionsFromFlagsTest, RejectsUnknownLogLevel) {
  absl::FlagSaver saver;
  absl::SetFlag(&FLAGS_amlogic_log_level, "trace");
  LITERT_ASSERT_OK_AND_ASSIGN(AmlogicOptions options, AmlogicOptions::Create());
  EXPECT_FALSE(UpdateAmlogicOptionsFromFlags(options).HasValue());
}

}  // namespace
}  // namespace litert::amlogic
