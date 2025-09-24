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

#include "litert/ats/compile_capture.h"

#include <sstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace litert::testing {
namespace {

using ::testing::HasSubstr;

TEST(AtsCompileCaptureTest, Basic) {
  CompileCapture cap;
  cap.NewEntry();

  std::ostringstream s;
  cap.Print(s);

  EXPECT_THAT(s.str(), HasSubstr("CompileCapture"));
}

}  // namespace
}  // namespace litert::testing
