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

#include "litert/c/internal/litert_logging.h"

#include <cstddef>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/test/matchers.h"

namespace {

using ::testing::Eq;
using ::testing::Ge;
using ::testing::HasSubstr;
using ::testing::Ne;
using ::testing::StrEq;
using ::testing::litert::IsError;

TEST(Layout, Creation) {
  LiteRtLogger logger;
  ASSERT_EQ(LiteRtCreateLogger(&logger), kLiteRtStatusOk);
  LiteRtDestroyLogger(logger);
}

TEST(Layout, MinLogging) {
  LiteRtLogger logger;
  ASSERT_EQ(LiteRtCreateLogger(&logger), kLiteRtStatusOk);
  ASSERT_EQ(LiteRtSetMinLoggerSeverity(logger, LITERT_SILENT), kLiteRtStatusOk);
  LiteRtLogSeverity min_severity;
  ASSERT_EQ(LiteRtGetMinLoggerSeverity(logger, &min_severity), kLiteRtStatusOk);
  ASSERT_EQ(min_severity, LITERT_SILENT);
  LiteRtDestroyLogger(logger);
}

TEST(DebugMacros, DebugOnlyCode) {
  bool val = false;
  LITERT_DEBUG_CODE(val = true);

  bool val2 = false;
  LITERT_DEBUG_CODE({ val2 = true; });

#ifndef NDEBUG
  EXPECT_TRUE(val2 && val);

#else
  EXPECT_FALSE(val2 || val);

#endif
}

TEST(LiteRtLoggerTest, LiteRtGetLoggerIdentifierFailsWithInvalidInput) {
  const char* identifier = nullptr;
  EXPECT_THAT(LiteRtGetLoggerIdentifier(nullptr, &identifier), IsError());
  EXPECT_THAT(LiteRtGetLoggerIdentifier(LiteRtGetDefaultLogger(), nullptr),
              IsError());
}

class LiteRtSinkLoggerTest : public testing::Test {
 public:
  void SetUp() override {
    LITERT_ASSERT_OK(LiteRtCreateSinkLogger(&sink_logger_));
    LITERT_ASSERT_OK(
        LiteRtLoggerLog(sink_logger_, LITERT_ERROR, "A log message"));
  }
  void TearDown() override { LiteRtDestroyLogger(sink_logger_); }

  static std::unique_ptr<LiteRtLoggerT, decltype(&LiteRtDestroyLogger)>
  CreateDefaultLogger() {
    LiteRtLogger logger;
    LiteRtCreateLogger(&logger);
    return std::unique_ptr<LiteRtLoggerT, decltype(&LiteRtDestroyLogger)>(
        logger, LiteRtDestroyLogger);
  }

  LiteRtLogger sink_logger_ = nullptr;
};

TEST_F(LiteRtSinkLoggerTest, IdentifierIsCorrect) {
  const char* identifier = nullptr;
  LITERT_ASSERT_OK(LiteRtGetLoggerIdentifier(sink_logger_, &identifier));
  EXPECT_THAT(identifier, AllOf(Ne(nullptr), StrEq("LiteRtSinkLogger")));
}

TEST_F(LiteRtSinkLoggerTest, GetSizeFailsWithInvalidParameters) {
  EXPECT_THAT(LiteRtGetSinkLoggerSize(sink_logger_, nullptr),
              IsError(kLiteRtStatusErrorInvalidArgument));
  size_t size;
  EXPECT_THAT(LiteRtGetSinkLoggerSize(nullptr, &size),
              IsError(kLiteRtStatusErrorInvalidArgument));
  auto default_logger = CreateDefaultLogger();
  EXPECT_THAT(LiteRtGetSinkLoggerSize(default_logger.get(), &size),
              IsError(kLiteRtStatusErrorInvalidArgument));
}

TEST_F(LiteRtSinkLoggerTest, GetMessageFailsWithInvalidParameters) {
  const char* message;
  EXPECT_THAT(LiteRtGetSinkLoggerMessage(sink_logger_, 0, nullptr),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(LiteRtGetSinkLoggerMessage(nullptr, 0, &message),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(LiteRtGetSinkLoggerMessage(sink_logger_, -1, &message),
              IsError(kLiteRtStatusErrorNotFound));
  size_t size = 0;
  LITERT_EXPECT_OK(LiteRtGetSinkLoggerSize(sink_logger_, &size));
  EXPECT_THAT(LiteRtGetSinkLoggerMessage(sink_logger_, size, &message),
              IsError(kLiteRtStatusErrorNotFound));
  auto default_logger = CreateDefaultLogger();
  EXPECT_THAT(LiteRtGetSinkLoggerMessage(default_logger.get(), 0, &message),
              IsError(kLiteRtStatusErrorInvalidArgument));
}

TEST_F(LiteRtSinkLoggerTest, AccessingMessagesWorks) {
  const char* message = nullptr;
  size_t size;

  LITERT_EXPECT_OK(LiteRtGetSinkLoggerSize(sink_logger_, &size));
  EXPECT_THAT(size, Ge(1));

  LITERT_EXPECT_OK(LiteRtGetSinkLoggerMessage(sink_logger_, 0, &message));
  EXPECT_THAT(message, HasSubstr("A log message"));
}

TEST_F(LiteRtSinkLoggerTest, ClearingMessagesWorks) {
  LITERT_EXPECT_OK(LiteRtClearSinkLogger(sink_logger_));

  size_t size;
  LITERT_EXPECT_OK(LiteRtGetSinkLoggerSize(sink_logger_, &size));
  EXPECT_THAT(size, Eq(0));
}

TEST_F(LiteRtSinkLoggerTest, ClearingMessagesFailsWithInvalidArguments) {
  EXPECT_THAT(LiteRtClearSinkLogger(nullptr),
              IsError(kLiteRtStatusErrorInvalidArgument));
}

}  // namespace
