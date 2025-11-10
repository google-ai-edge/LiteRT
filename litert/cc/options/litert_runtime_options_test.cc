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

#include "litert/cc/options/litert_runtime_options.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/c/options/litert_runtime_options.h"
#include "litert/test/matchers.h"
namespace litert {
namespace {
using ::testing::StrEq;
using ::testing::litert::IsOkAndHolds;

TEST(RuntimeOptions, IdentifierIsCorrect) {
  EXPECT_THAT(RuntimeOptions::Identifier(),
              StrEq(LiteRtGetRuntimeOptionsIdentifier()));
}

TEST(RuntimeOptions, CreateAndOwnedHandle) {
  LITERT_ASSERT_OK_AND_ASSIGN(RuntimeOptions options, RuntimeOptions::Create());
  EXPECT_TRUE(options.IsOwned());
}

}  // namespace
}  // namespace litert
