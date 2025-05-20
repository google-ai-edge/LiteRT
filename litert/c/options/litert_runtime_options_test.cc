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

namespace {

TEST(LiteRtRuntimeOptionsTest, CreateLitertRuntimeOptionsWorks) {
  LiteRtRuntimeOptions options;
  EXPECT_THAT(LiteRtCreateRuntimeOptions(&options), kLiteRtStatusOk);

  LiteRtDestroyRuntimeOptions(options);
}

TEST(LiteRtRuntimeOptionsTest, SetAndGetShloCompositeInliningWorks) {
  LiteRtRuntimeOptions options;
  EXPECT_THAT(LiteRtCreateRuntimeOptions(&options), kLiteRtStatusOk);

  bool shlo_composite_inlining = false;
  EXPECT_THAT(LiteRtSetRuntimeOptionsShloCompositeInlining(options, true),
              kLiteRtStatusOk);
  EXPECT_THAT(LiteRtGetRuntimeOptionsShloCompositeInlining(
                  options, &shlo_composite_inlining),
              kLiteRtStatusOk);
  EXPECT_TRUE(shlo_composite_inlining);

  LiteRtDestroyRuntimeOptions(options);
}

}  // namespace
