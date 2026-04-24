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

#include "litert/cc/internal/litert_source_location.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace litert {
namespace {

using testing::StrEq;

TEST(SourceLocation, BuildAtCurrentLocation) {
#if LITERT_HAS_BUILTIN(__builtin_FILE) && LITERT_HAS_BUILTIN(__builtin_LINE)
  EXPECT_THAT(SourceLocation::current().line(), __LINE__);
  EXPECT_THAT(SourceLocation::current().file_name(), StrEq(__FILE__));
#else
  EXPECT_THAT(SourceLocation::current().line(), 0);
  EXPECT_THAT(SourceLocation::current().file_name(), StrEq("unknown"));
#endif
}

}  // namespace
}  // namespace litert
