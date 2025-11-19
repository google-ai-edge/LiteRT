// Copyright 2024 Google LLC.
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

#include "litert/core/environment.h"

#include <any>
#include <array>
#include <utility>

#include <gtest/gtest.h>
#include "litert/c/litert_any.h"
#include "litert/c/litert_environment.h"
#include "litert/cc/litert_any.h"
#include "litert/test/matchers.h"

namespace litert::internal {
namespace {

TEST(LiteRtEnvironmentT, CreateWithOptions) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtAny any_path, ToLiteRtAny(litert::LiteRtVariant("sample path")));
  const std::array<LiteRtEnvOption, 1> environment_options = {
      LiteRtEnvOption{
          kLiteRtEnvOptionTagCompilerPluginLibraryDir,
          any_path,
      },
  };
  auto env = LiteRtEnvironmentT::CreateWithOptions(environment_options);
  ASSERT_TRUE(env);

  auto option = (*env)->GetOption(kLiteRtEnvOptionTagCompilerPluginLibraryDir);
  ASSERT_TRUE(option.has_value());
  ASSERT_EQ(option->type, kLiteRtAnyTypeString);
  ASSERT_STREQ(option->str_value, "sample path");
}

TEST(LiteRtEnvironmentT, CheckStringCopy) {
  LiteRtEnvironmentT::Ptr env;

  // The passed string becomes obsolete after the scope.
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        LiteRtAny any_path, ToLiteRtAny(litert::LiteRtVariant("sample path")));
    const std::array<LiteRtEnvOption, 1> environment_options = {
        LiteRtEnvOption{
            kLiteRtEnvOptionTagCompilerPluginLibraryDir,
            any_path,
        },
    };
    auto res = LiteRtEnvironmentT::CreateWithOptions(environment_options);
    ASSERT_TRUE(res);
    env = std::move(*res);
  }

  auto option = env->GetOption(kLiteRtEnvOptionTagCompilerPluginLibraryDir);
  ASSERT_TRUE(option.has_value());
  ASSERT_EQ(option->type, kLiteRtAnyTypeString);
  ASSERT_STREQ(option->str_value, "sample path");
}

}  // namespace
}  // namespace litert::internal
