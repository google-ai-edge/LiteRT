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

#include "litert/cc/internal/litert_logging.h"

#include <sstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"

using ::testing::ContainsRegex;
using ::testing::Eq;

namespace litert {
namespace {

TEST(LiteRtLogginTest, LogInterceptionWorks) {
  InterceptLogs intercepted_logs;
  LITERT_LOG(LITERT_WARNING, "This is a warning log.");
  LITERT_LOG(LITERT_INFO, "This is an info log.");
  std::stringstream sstr;
  sstr << intercepted_logs;
  EXPECT_THAT(
      sstr.str(),
      ContainsRegex(
          "WARNING.*This is a warning log.*\n.*INFO.*This is an info log"));
}

TEST(LiteRtLogginTest, LogInterceptionRestoresPreviousLogger) {
  LiteRtLogger previous_logger = LiteRtGetDefaultLogger();
  {
    InterceptLogs intercepted_logs;
    LITERT_LOG(LITERT_WARNING, "This is a warning log.");
  }
  EXPECT_THAT(LiteRtGetDefaultLogger(), Eq(previous_logger));
}

TEST(LiteRtLogginTest, LogInterceptionClearsPreviousInterceptions) {
  {
    InterceptLogs intercepted_logs;
    LITERT_LOG(LITERT_WARNING, "This is a warning log.");
  }
  {
    InterceptLogs intercepted_logs;
    LITERT_LOG(LITERT_ERROR, "This is an error log.");
    std::stringstream sstr;
    sstr << intercepted_logs;
    EXPECT_THAT(sstr.str(),
                AllOf(Not(ContainsRegex("WARNING.*This is a warning log.*")),
                      ContainsRegex("ERROR.*This is an error log.*")));
  }
}

}  // namespace
}  // namespace litert
