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

#include "litert/c/litert_error_reporter.h"

#include <gtest/gtest.h>

namespace {

TEST(LiteRtErrorReporterTest, CreateStderrReporter) {
  LiteRtErrorReporter reporter = nullptr;
  ASSERT_EQ(LiteRtCreateStderrErrorReporter(&reporter), kLiteRtStatusOk);
  ASSERT_NE(reporter, nullptr);
  
  // Test reporting
  EXPECT_EQ(LiteRtErrorReporterReport(reporter, "Test error %d", 42),
            kLiteRtStatusOk);
  
  // Stderr reporter doesn't support GetMessage
  const char* message = nullptr;
  EXPECT_EQ(LiteRtErrorReporterGetMessage(reporter, &message),
            kLiteRtStatusErrorUnsupported);
  
  LiteRtDestroyErrorReporter(reporter);
}

TEST(LiteRtErrorReporterTest, CreateBufferReporter) {
  LiteRtErrorReporter reporter = nullptr;
  ASSERT_EQ(LiteRtCreateBufferErrorReporter(&reporter), kLiteRtStatusOk);
  ASSERT_NE(reporter, nullptr);
  
  // Test reporting
  EXPECT_EQ(LiteRtErrorReporterReport(reporter, "Error %d", 1),
            kLiteRtStatusOk);
  EXPECT_EQ(LiteRtErrorReporterReport(reporter, "Error %d", 2),
            kLiteRtStatusOk);
  
  // Get message
  const char* message = nullptr;
  EXPECT_EQ(LiteRtErrorReporterGetMessage(reporter, &message),
            kLiteRtStatusOk);
  ASSERT_NE(message, nullptr);
  EXPECT_STREQ(message, "Error 1\nError 2\n");
  
  // Message should be cleared after GetMessage
  EXPECT_EQ(LiteRtErrorReporterGetMessage(reporter, &message),
            kLiteRtStatusOk);
  EXPECT_STREQ(message, "");
  
  LiteRtDestroyErrorReporter(reporter);
}

TEST(LiteRtErrorReporterTest, BufferReporterClear) {
  LiteRtErrorReporter reporter = nullptr;
  ASSERT_EQ(LiteRtCreateBufferErrorReporter(&reporter), kLiteRtStatusOk);
  
  EXPECT_EQ(LiteRtErrorReporterReport(reporter, "Error"), kLiteRtStatusOk);
  
  const char* message = nullptr;
  EXPECT_EQ(LiteRtErrorReporterGetMessage(reporter, &message),
            kLiteRtStatusOk);
  EXPECT_STRNE(message, "");
  
  // Clear should work on buffer reporter
  EXPECT_EQ(LiteRtErrorReporterClear(reporter), kLiteRtStatusOk);
  
  EXPECT_EQ(LiteRtErrorReporterGetMessage(reporter, &message),
            kLiteRtStatusOk);
  EXPECT_STREQ(message, "");
  
  LiteRtDestroyErrorReporter(reporter);
}

TEST(LiteRtErrorReporterTest, DefaultErrorReporter) {
  LiteRtErrorReporter reporter = LiteRtGetDefaultErrorReporter();
  ASSERT_NE(reporter, nullptr);
  
  // Should be the same instance
  EXPECT_EQ(reporter, LiteRtGetDefaultErrorReporter());
  
  // Should be able to report
  EXPECT_EQ(LiteRtErrorReporterReport(reporter, "Default error %s", "test"),
            kLiteRtStatusOk);
  
  // Should not be destroyed (it's a singleton)
  LiteRtDestroyErrorReporter(reporter);
  
  // Should still work after "destroy"
  EXPECT_EQ(LiteRtErrorReporterReport(reporter, "Still works"),
            kLiteRtStatusOk);
}

TEST(LiteRtErrorReporterTest, InvalidArguments) {
  // Null reporter pointer
  EXPECT_EQ(LiteRtCreateStderrErrorReporter(nullptr),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(LiteRtCreateBufferErrorReporter(nullptr),
            kLiteRtStatusErrorInvalidArgument);
  
  // Null reporter
  EXPECT_EQ(LiteRtErrorReporterReport(nullptr, "test"),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(LiteRtErrorReporterGetMessage(nullptr, nullptr),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(LiteRtErrorReporterClear(nullptr),
            kLiteRtStatusErrorInvalidArgument);
  
  // Null format
  LiteRtErrorReporter reporter = LiteRtGetDefaultErrorReporter();
  EXPECT_EQ(LiteRtErrorReporterReport(reporter, nullptr),
            kLiteRtStatusErrorInvalidArgument);
  
  // Null message pointer
  EXPECT_EQ(LiteRtErrorReporterGetMessage(reporter, nullptr),
            kLiteRtStatusErrorInvalidArgument);
}

TEST(LiteRtErrorReporterTest, ClearOnNonBufferReporter) {
  LiteRtErrorReporter reporter = nullptr;
  ASSERT_EQ(LiteRtCreateStderrErrorReporter(&reporter), kLiteRtStatusOk);
  
  // Clear should fail on non-buffer reporter
  EXPECT_EQ(LiteRtErrorReporterClear(reporter),
            kLiteRtStatusErrorUnsupported);
  
  LiteRtDestroyErrorReporter(reporter);
}

}  // namespace