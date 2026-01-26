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

#include "litert/cc/internal/litert_handle.h"

#include <type_traits>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/c/litert_common.h"

namespace litert::internal {
namespace {
using testing::Eq;
using testing::IsNull;
using testing::Not;

LITERT_DEFINE_HANDLE(LiteRtTestResource);

struct LiteRtTestResourceT {
  int val;
};

void LiteRtDestroyTestResource(LiteRtTestResource handle) { delete handle; }

using TestHandle = Handle<LiteRtTestResource, LiteRtDestroyTestResource>;

TEST(HandleTest, HandlesAreNotCopiable) {
  static_assert(!std::is_copy_constructible_v<TestHandle>,
                "Handle must not be copy constructible.");
  static_assert(!std::is_copy_assignable_v<TestHandle>,
                "Handle must not be copy assignable.");
}

TEST(HandleTest, BuildAnOwningHandle) {
  const LiteRtTestResource res = new LiteRtTestResourceT;
  const TestHandle h(res, OwnHandle::kYes);
  EXPECT_TRUE(h.IsOwned());
  EXPECT_THAT(h.Get(), Eq(res));
}

TEST(HandleTest, BuildANonOwningHandle) {
  const LiteRtTestResource res = new LiteRtTestResourceT;
  const TestHandle h(res, OwnHandle::kNo);
  EXPECT_FALSE(h.IsOwned());
  EXPECT_THAT(h.Get(), Eq(res));
  LiteRtDestroyTestResource(res);
}

TEST(HandleTest, BuildANullOwningHandle) {
  const TestHandle h(nullptr, OwnHandle::kYes);
  EXPECT_TRUE(h.IsOwned());
  EXPECT_THAT(h.Get(), IsNull());
}

TEST(HandleTest, BuildANullNonOwningHandle) {
  const TestHandle h(nullptr, OwnHandle::kNo);
  EXPECT_FALSE(h.IsOwned());
  EXPECT_THAT(h.Get(), IsNull());
}

TEST(HandleTest, DefaultConstructorBuildsANullNonOwningHandle) {
  const TestHandle h;
  EXPECT_FALSE(h.IsOwned());
  EXPECT_THAT(h.Get(), IsNull());
}

TEST(HandleTest, MoveConstructAnOwningHandle) {
  TestHandle h1(new LiteRtTestResourceT, OwnHandle::kYes);
  TestHandle h2(std::move(h1));
  EXPECT_TRUE(h2.IsOwned());
  EXPECT_THAT(h2.Get(), Not(IsNull()));
}

TEST(HandleTest, MoveAssignAnOwningHandle) {
  TestHandle h1(new LiteRtTestResourceT, OwnHandle::kYes);
  // Default constructed handles are null and non owning.
  TestHandle h2;
  h2 = std::move(h1);
  EXPECT_TRUE(h2.IsOwned());
  EXPECT_THAT(h2.Get(), Not(IsNull()));
}

// This will double delete is Release or GetDeleter don't work.
TEST(HandleTest, ReleaseAndGetDeleterWork) {
  TestHandle h(new LiteRtTestResourceT, OwnHandle::kYes);
  auto deleter = h.GetDeleter();
  LiteRtTestResource res = h.Release();
  EXPECT_THAT(res, Not(IsNull()));
  deleter(res);
}

using TestBaseHandle = BaseHandle<LiteRtTestResource>;

TEST(BaseHandleTest, HandlesAreNotCopiable) {
  static_assert(!std::is_copy_constructible_v<TestBaseHandle>,
                "Handle must not be copy constructible.");
  static_assert(!std::is_copy_assignable_v<TestBaseHandle>,
                "Handle must not be copy assignable.");
}

TEST(BaseHandleTest, BuildAnOwningHandle) {
  const LiteRtTestResource res = new LiteRtTestResourceT;
  const TestBaseHandle h(res, LiteRtDestroyTestResource, OwnHandle::kYes);
  EXPECT_TRUE(h.IsOwned());
  EXPECT_THAT(h.Get(), Eq(res));
}

TEST(BaseHandleTest, BuildANonOwningHandle) {
  const LiteRtTestResource res = new LiteRtTestResourceT;
  const TestBaseHandle h(res, LiteRtDestroyTestResource, OwnHandle::kNo);
  EXPECT_FALSE(h.IsOwned());
  EXPECT_THAT(h.Get(), Eq(res));
  LiteRtDestroyTestResource(res);
}

TEST(BaseHandleTest, BuildANullOwningHandle) {
  const TestBaseHandle h(nullptr, LiteRtDestroyTestResource, OwnHandle::kYes);
  EXPECT_TRUE(h.IsOwned());
  EXPECT_THAT(h.Get(), IsNull());
}

TEST(BaseHandleTest, BuildANullNonOwningHandle) {
  const TestBaseHandle h(nullptr, LiteRtDestroyTestResource, OwnHandle::kNo);
  EXPECT_FALSE(h.IsOwned());
  EXPECT_THAT(h.Get(), IsNull());
}

TEST(BaseHandleTest, DefaultConstructorBuildsANullNonOwningHandle) {
  const TestBaseHandle h;
  EXPECT_FALSE(h.IsOwned());
  EXPECT_THAT(h.Get(), IsNull());
}

TEST(BaseHandleTest, MoveConstructAnOwningHandle) {
  TestBaseHandle h1(new LiteRtTestResourceT, LiteRtDestroyTestResource,
                    OwnHandle::kYes);
  TestBaseHandle h2(std::move(h1));
  EXPECT_TRUE(h2.IsOwned());
  EXPECT_THAT(h2.Get(), Not(IsNull()));
}

TEST(BaseHandleTest, MoveAssignAnOwningHandle) {
  TestBaseHandle h1(new LiteRtTestResourceT, LiteRtDestroyTestResource,
                    OwnHandle::kYes);
  // Default constructed handles are null and non owning.
  TestBaseHandle h2;
  h2 = std::move(h1);
  EXPECT_TRUE(h2.IsOwned());
  EXPECT_THAT(h2.Get(), Not(IsNull()));
}

// This will double delete is Release or GetDeleter don't work.
TEST(BaseHandleTest, ReleaseAndGetDeleterWork) {
  TestBaseHandle h(new LiteRtTestResourceT, LiteRtDestroyTestResource,
                   OwnHandle::kYes);
  auto deleter = h.GetDeleter();
  LiteRtTestResource res = h.Release();
  EXPECT_THAT(res, Not(IsNull()));
  deleter(res);
}

}  // namespace
}  // namespace litert::internal
