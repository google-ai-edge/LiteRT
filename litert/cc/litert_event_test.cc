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
#include "litert/test/matchers.h"

namespace litert {
namespace {

TEST(Event, DupFdOnNegativeFd) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  LITERT_ASSERT_OK_AND_ASSIGN(Event event,
                              Event::CreateFromSyncFenceFd(env, -1, true));
  EXPECT_FALSE(event.DupFd());
}

TEST(Event, IsSignaledOnNegativeFd) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  LITERT_ASSERT_OK_AND_ASSIGN(Event event,
                              Event::CreateFromSyncFenceFd(env, -1, true));
  EXPECT_FALSE(event.IsSignaled());
}

TEST(Event, CreateManagedEglSyncFence) {
  if (!HasOpenGlSupport()) {
    GTEST_SKIP() << "Skipping test for platforms without OpenGL support.";
  }
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));

  LITERT_ASSERT_OK_AND_ASSIGN(
      Event event, Event::CreateManaged(env, Event::Type::kEglSyncFence));
  EXPECT_EQ(event.Type(), Event::Type::kEglSyncFence);
}

TEST(Event, CreateManagedEglNativeSyncFence) {
  if (!HasOpenGlSupport()) {
    GTEST_SKIP() << "Skipping test for platforms without OpenGL support.";
  }
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));

  LITERT_ASSERT_OK_AND_ASSIGN(
      Event event, Event::CreateManaged(env, Event::Type::kEglNativeSyncFence));
  EXPECT_EQ(event.Type(), Event::Type::kEglNativeSyncFence);
}

}  // namespace
}  // namespace litert
