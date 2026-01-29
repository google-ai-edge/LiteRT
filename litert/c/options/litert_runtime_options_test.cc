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

#include "litert/c/options/litert_runtime_options.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/test/matchers.h"

namespace {

using ::testing::StrEq;
using ::testing::litert::IsError;

TEST(LiteRtRuntimeOptionsTest, CreateErrorsOutWithNullptrParam) {
  EXPECT_THAT(LiteRtCreateRuntimeOptions(nullptr),
              IsError(kLiteRtStatusErrorInvalidArgument));
}

TEST(LiteRtRuntimeOptionsTest, CreateWorks) {
  LiteRtOpaqueOptions options = nullptr;
  LITERT_ASSERT_OK(LiteRtCreateRuntimeOptions(&options));

  const char* id;
  LITERT_ASSERT_OK(LiteRtGetOpaqueOptionsIdentifier(options, &id));
  EXPECT_THAT(id, StrEq(LiteRtGetRuntimeOptionsIdentifier()));

  LiteRtDestroyOpaqueOptions(options);
}

}  // namespace
