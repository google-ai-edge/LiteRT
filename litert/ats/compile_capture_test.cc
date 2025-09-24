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

#include <limits>
#include <sstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/ats/common.h"

namespace litert::testing {
namespace {

using ::testing::HasSubstr;

TEST(AtsCompileCaptureTest, Basic) {
  CompileCapture cap;
  auto& e = cap.NewEntry();
  e.compilation_time.Stop(e.compilation_time.Start());
  e.model.name = "FOO";

  ASSERT_NE(e.compilation_time.Nanos(),
            std::numeric_limits<Nanoseconds>::max());

  std::ostringstream s;
  cap.Print(s);

  EXPECT_THAT(s.str(), HasSubstr("FOO"));
}

}  // namespace
}  // namespace litert::testing
