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

#include "litert/cc/litert_event.h"

#include <gtest/gtest.h>
#include "litert/c/litert_event_type.h"
#include "litert/cc/internal/litert_platform_support.h"
#include "litert/cc/litert_environment.h"
#include "litert/core/environment.h"
#include "litert/test/matchers.h"

namespace litert {
namespace {

using ::testing::Not;
using ::testing::litert::IsOk;

TEST(Event, DupFdOnNegativeFd) {
#if !LITERT_HAS_SYNC_FENCE_SUPPORT
  GTEST_SKIP() << "Skipping test on platform without sync fence support.";
#endif  // !LITERT_HAS_SYNC_FENCE_SUPPORT
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      Event event, Event::CreateFromSyncFenceFd(env.Get(), -1, true));
  EXPECT_THAT(event.DupFd(), Not(IsOk()));
}

TEST(Event, IsSignaledOnNegativeFd) {
#if !LITERT_HAS_SYNC_FENCE_SUPPORT
  GTEST_SKIP() << "Skipping test on platform without sync fence support.";
#endif  // !LITERT_HAS_SYNC_FENCE_SUPPORT
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      Event event, Event::CreateFromSyncFenceFd(env.Get(), -1, true));
  EXPECT_THAT(event.IsSignaled(), Not(IsOk()));
}

TEST(Event, CreateManagedEglSyncFence) {
  if (!HasOpenGlSupport()) {
    GTEST_SKIP() << "Skipping test for platforms without OpenGL support.";
  }
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  if (!env.Get()->GetGpuEnvironment().HasValue()) {
    GTEST_SKIP()
        << "Skipping test because the GPU environment is not available.";
  }

  LITERT_ASSERT_OK_AND_ASSIGN(
      Event event,
      Event::CreateManaged(env.Get(), LiteRtEventTypeEglSyncFence));
  EXPECT_EQ(event.Type(), LiteRtEventTypeEglSyncFence);
}

TEST(Event, CreateManagedEglNativeSyncFence) {
  if (!HasOpenGlSupport()) {
    GTEST_SKIP() << "Skipping test for platforms without OpenGL support.";
  }
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  if (!env.Get()->GetGpuEnvironment().HasValue()) {
    GTEST_SKIP()
        << "Skipping test because the GPU environment is not available.";
  }

  LITERT_ASSERT_OK_AND_ASSIGN(
      Event event,
      Event::CreateManaged(env.Get(), LiteRtEventTypeEglNativeSyncFence));
  EXPECT_EQ(event.Type(), LiteRtEventTypeEglNativeSyncFence);
}

}  // namespace
}  // namespace litert
