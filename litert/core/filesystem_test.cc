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

#include "litert/core/filesystem.h"

#include <gtest/gtest.h>
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::internal {
namespace {

static constexpr absl::string_view kPrefix = "a/prefix";
static constexpr absl::string_view kInfix = "an/infix";
static constexpr absl::string_view kSuffix = "suffix.ext";
static constexpr absl::string_view kPath = "a/prefix.ext";
static constexpr absl::string_view kStem = "prefix";

TEST(FilesystemTest, JoinTwo) {
  const auto path = Join({kPrefix, kSuffix});
  EXPECT_EQ(path, absl::StrFormat("%s/%s", kPrefix, kSuffix));
}

TEST(FilesystemTest, JoinMany) {
  const auto path = Join({kPrefix, kInfix, kSuffix});
  EXPECT_EQ(path, absl::StrFormat("%s/%s/%s", kPrefix, kInfix, kSuffix));
}

TEST(FilesystemTest, Stem){
  const auto stem = Stem(kPath);
  EXPECT_EQ(stem, kStem);
}

}  // namespace
}  // namespace litert::internal
