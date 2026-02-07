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

#include <gtest/gtest.h>
#include "litert/cc/litert_any.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"

namespace litert {
namespace {

TEST(EnvironmentOptionsTest, GetOption) {
  static constexpr auto kOptionTag =
      EnvironmentOptions::Tag::kCompilerPluginLibraryDir;
  static constexpr char kOptionValue[] = "foo/bar";

  auto opts = EnvironmentOptions({{kOptionTag, kOptionValue}});
  auto env = Environment::Create(opts);
  ASSERT_TRUE(env);

  auto options = env->GetOptions();
  ASSERT_TRUE(options);

  auto option = options->GetOption(kOptionTag);
  ASSERT_TRUE(option);
  EXPECT_STREQ(std::get<const char*>(*option), kOptionValue);
}

TEST(EnvironmentOptionsTest, OptionNotFound) {
  static constexpr auto kOptionTag =
      EnvironmentOptions::Tag::kCompilerPluginLibraryDir;
  litert::LiteRtVariant kOptionValue = "foo/bar";

  auto env = Environment::Create(EnvironmentOptions(
      {EnvironmentOptions::Option{kOptionTag, kOptionValue}}));
  ASSERT_TRUE(env);

  auto options = env->GetOptions();
  ASSERT_TRUE(options);

  auto option =
      options->GetOption(EnvironmentOptions::Tag::kDispatchLibraryDir);
  EXPECT_FALSE(option);
}

TEST(EnvironmentOptionsTest, GetOptions) {
  static constexpr auto kOptionTag1 =
      EnvironmentOptions::Tag::kCompilerPluginLibraryDir;
  static constexpr char kOptionValue1[] = "foo/bar";
  static constexpr auto kOptionTag2 =
      EnvironmentOptions::Tag::kDispatchLibraryDir;
  static constexpr char kOptionValue2[] = "baz/qux";

  EnvironmentOptions options(
      {EnvironmentOptions::Option{kOptionTag1,
                                  litert::LiteRtVariant(kOptionValue1)},
       EnvironmentOptions::Option{kOptionTag2,
                                  litert::LiteRtVariant(kOptionValue2)}});

  auto all_options = options.GetOptions();
  ASSERT_EQ(all_options.size(), 2);

  EXPECT_EQ(all_options[0].tag, kOptionTag1);
  EXPECT_STREQ(std::get<const char*>(all_options[0].value), kOptionValue1);

  EXPECT_EQ(all_options[1].tag, kOptionTag2);
  EXPECT_STREQ(std::get<const char*>(all_options[1].value), kOptionValue2);
}

}  // namespace
}  // namespace litert
