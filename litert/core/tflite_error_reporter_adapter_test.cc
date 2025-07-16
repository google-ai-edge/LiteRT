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

#include "litert/core/tflite_error_reporter_adapter.h"

#include <string>

#include <gtest/gtest.h>
#include "litert/core/buffer_error_reporter.h"
#include "tflite/stderr_reporter.h"

namespace litert {
namespace {

TEST(TfliteErrorReporterAdapterTest, AdaptLiteRtToTflite) {
  BufferErrorReporter litert_reporter;
  TfliteErrorReporterAdapter adapter(&litert_reporter);
  
  // Report through TFLite interface
  adapter.Report("Error from TFLite: %d", 42);
  
  // Check it was captured by LiteRT reporter
  std::string message = litert_reporter.message();
  EXPECT_EQ(message, "Error from TFLite: 42\n");
}

TEST(TfliteErrorReporterAdapterTest, AdaptTfliteToLiteRt) {
  auto* tflite_reporter = ::tflite::DefaultErrorReporter();
  LiteRtErrorReporterAdapter adapter(tflite_reporter);
  
  // Report through LiteRT interface
  adapter.Report("Error from LiteRT: %s", "test");
  
  // We can't easily verify stderr output, but this shouldn't crash
}

TEST(TfliteErrorReporterAdapterTest, GetTfliteCompatibleReporter) {
  auto* reporter = GetTfliteCompatibleErrorReporter();
  ASSERT_NE(reporter, nullptr);
  
  // Should be the same instance
  EXPECT_EQ(reporter, GetTfliteCompatibleErrorReporter());
  
  // Should be able to report
  reporter->Report("Test error through compatibility layer");
}

TEST(TfliteErrorReporterAdapterTest, NullReporterHandling) {
  // Adapter should handle null reporters gracefully
  TfliteErrorReporterAdapter adapter1(nullptr);
  EXPECT_EQ(adapter1.Report("Test"), 0);
  
  LiteRtErrorReporterAdapter adapter2(nullptr);
  EXPECT_EQ(adapter2.Report("Test"), 0);
}

}  // namespace
}  // namespace litert