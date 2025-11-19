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

#include "litert/cc/litert_environment_options.h"

#include <any>

#include <gtest/gtest.h>
#include "litert/c/litert_environment_options.h"
#include "litert/cc/litert_any.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"

namespace litert {
namespace {

TEST(EnvironmentOptionsTest, GetOption) {
  static constexpr auto kOptionTag =
      Environment::OptionTag::CompilerPluginLibraryDir;
  static constexpr char kOptionValue[] = "foo/bar";

  auto env = Environment::Create(
      {Environment::Option{kOptionTag, litert::LiteRtVariant(kOptionValue)}});
  ASSERT_TRUE(env);

  auto options = env->GetOptions();
  ASSERT_TRUE(options);

  auto option = options->GetOption(kLiteRtEnvOptionTagCompilerPluginLibraryDir);
  ASSERT_TRUE(option);
  EXPECT_STREQ(std::get<const char*>(*option), kOptionValue);
}

TEST(EnvironmentOptionsTest, OptionNotFound) {
  static constexpr auto kOptionTag =
      Environment::OptionTag::CompilerPluginLibraryDir;
  litert::LiteRtVariant kOptionValue = "foo/bar";

  auto env =
      Environment::Create({Environment::Option{kOptionTag, kOptionValue}});
  ASSERT_TRUE(env);

  auto options = env->GetOptions();
  ASSERT_TRUE(options);

  auto option = options->GetOption(kLiteRtEnvOptionTagDispatchLibraryDir);
  EXPECT_FALSE(option);
}
}  // namespace
}  // namespace litert
