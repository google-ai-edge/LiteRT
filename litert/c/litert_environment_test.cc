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

#include "litert/c/litert_environment.h"

#include <gtest/gtest.h>
#include "litert/c/litert_common.h"

namespace {

TEST(LiteRtEnvironmentTest, CreateAndDestroy) {
  LiteRtEnvironment env;
  ASSERT_EQ(LiteRtCreateEnvironment(0, nullptr, &env), kLiteRtStatusOk);
  ASSERT_NE(env, nullptr);
  LiteRtDestroyEnvironment(env);
}

TEST(LiteRtEnvironmentTest, SupportsFP16) {
  LiteRtEnvironment env = nullptr;
  ASSERT_EQ(LiteRtCreateEnvironment(0, nullptr, &env), kLiteRtStatusOk);
  bool supports_fp16;
  EXPECT_EQ(LiteRtEnvironmentSupportsFP16(env, &supports_fp16), kLiteRtStatusOk);
  LiteRtDestroyEnvironment(env);
}

TEST(LiteRtEnvironmentTest, HasGpuEnvironment) {
  LiteRtEnvironment env = nullptr;
  ASSERT_EQ(LiteRtCreateEnvironment(0, nullptr, &env), kLiteRtStatusOk);
  bool has_gpu;
  LiteRtEnvironmentHasGpuEnvironment(env, &has_gpu);
  LiteRtDestroyEnvironment(env);
}

}  // namespace
