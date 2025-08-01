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

#include "litert/core/error_reporter.h"

#include <string>

#include <gtest/gtest.h>
#include "litert/core/buffer_error_reporter.h"

namespace litert {
namespace {

TEST(ErrorReporterTest, StderrReporter) {
  StderrReporter reporter;

  // This should not crash, but we can't easily test stderr output
  reporter.Report("Test error: %d", 42);
  reporter.Report("Test error with string: %s", "hello");
}

TEST(ErrorReporterTest, BufferErrorReporter) {
  BufferErrorReporter reporter;

  EXPECT_EQ(reporter.NumErrors(), 0);
  EXPECT_EQ(reporter.message(), "");

  reporter.Report("Error %d", 1);
  EXPECT_EQ(reporter.NumErrors(), 1);

  reporter.Report("Error %d", 2);
  EXPECT_EQ(reporter.NumErrors(), 2);

  std::string message = reporter.message();
  EXPECT_EQ(message, "Error 1\nError 2\n");

  // message() should clear the buffer
  EXPECT_EQ(reporter.NumErrors(), 0);
  EXPECT_EQ(reporter.message(), "");
}

TEST(ErrorReporterTest, BufferErrorReporterClear) {
  BufferErrorReporter reporter;

  reporter.Report("Error 1");
  reporter.Report("Error 2");
  EXPECT_EQ(reporter.NumErrors(), 2);

  reporter.Clear();
  EXPECT_EQ(reporter.NumErrors(), 0);
  EXPECT_EQ(reporter.message(), "");
}

TEST(ErrorReporterTest, BufferErrorReporterFormattedOutput) {
  BufferErrorReporter reporter;

  reporter.Report("Integer: %d, String: %s, Float: %.2f", 42, "test", 3.14);

  std::string message = reporter.message();
  EXPECT_EQ(message, "Integer: 42, String: test, Float: 3.14\n");
}

TEST(ErrorReporterTest, TemplateReport) {
  BufferErrorReporter reporter;

  // Test the template version of Report
  reporter.Report("Test %d %s", 123, "template");

  std::string message = reporter.message();
  EXPECT_EQ(message, "Test 123 template\n");
}

}  // namespace
}  // namespace litert
