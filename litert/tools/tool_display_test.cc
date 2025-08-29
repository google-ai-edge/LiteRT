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

#include "litert/tools/tool_display.h"

#include <sstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace {

using ::litert::tools::ToolDisplay;
using ::testing::EndsWith;
using ::testing::StartsWith;

static constexpr absl::string_view kToolName = "test-tool";
static constexpr absl::string_view kLabel = "[LITERT_TOOLS:test-tool]";
static constexpr absl::string_view kStartLabel = "Test Routine";
static constexpr absl::string_view kDisplayInfo = "info";

TEST(TestToolDisplay, Display) {
  std::stringstream out;
  ToolDisplay display(out, kToolName);
  display.Display() << kDisplayInfo;
  const auto out_str = out.str();
  EXPECT_EQ(out_str, kDisplayInfo);
}

TEST(TestToolDisplay, Indented) {
  std::stringstream out;
  ToolDisplay display(out, kToolName);
  display.Indented() << kDisplayInfo;
  const auto out_str = out.str();
  EXPECT_EQ(out_str, absl::StrFormat("\t%s", kDisplayInfo));
}

TEST(TestToolDisplay, Labeled) {
  std::stringstream out;
  ToolDisplay display(out, kToolName);
  display.Labeled() << kDisplayInfo;
  const auto out_str = out.str();
  EXPECT_EQ(out_str, absl::StrFormat("%s %s", kLabel, kDisplayInfo));
}

TEST(TestToolDisplay, LabeledNoToolName) {
  std::stringstream out;
  ToolDisplay display(out);
  display.Labeled() << kDisplayInfo;
  const auto out_str = out.str();
  EXPECT_EQ(out_str, absl::StrFormat("%s %s", "[LITERT_TOOLS]", kDisplayInfo));
}

TEST(TestToolDisplay, Start) {
  std::stringstream out;
  ToolDisplay display(out, kToolName);
  display.Start(kStartLabel);
  const auto out_str = out.str();
  EXPECT_EQ(out_str,
            absl::StrFormat("%s Starting %s...\n", kLabel, kStartLabel));
}

TEST(TestToolDisplay, Done) {
  std::stringstream out;
  ToolDisplay display(out, kToolName);
  display.Done(kStartLabel);
  const auto out_str = out.str();
  EXPECT_EQ(out_str, absl::StrFormat("%s \t%s Done!\n", kLabel, kStartLabel));
}

TEST(TestToolDisplay, Fail) {
  std::stringstream out;
  ToolDisplay display(out, kToolName);
  display.Fail();
  const auto out_str = out.str();
  EXPECT_EQ(out_str, absl::StrFormat("%s \tFailed\n", kLabel));
}

TEST(TestLoggedScope, EnterExit) {
  std::stringstream out;
  ToolDisplay display(out, kToolName);
  {
    auto s = display.StartS(kStartLabel);
  }
  const auto out_str = out.str();
  EXPECT_THAT(out_str, StartsWith(absl::StrFormat("%s Starting %s...\n", kLabel,
                                                  kStartLabel)));
  EXPECT_THAT(out_str, EndsWith(absl::StrFormat("%s \t%s Done!\n", kLabel,
                                                kStartLabel)));
}

}  // namespace
