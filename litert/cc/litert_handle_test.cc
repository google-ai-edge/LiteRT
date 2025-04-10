#include "litert/cc/litert_handle.h"

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
  void (*deleter)(LiteRtTestResource) = h.GetDeleter();
  EXPECT_THAT(deleter, Eq(LiteRtDestroyTestResource));
  LiteRtTestResource res = h.Release();
  EXPECT_THAT(res, Not(IsNull()));
  deleter(res);
}

}  // namespace
}  // namespace litert::internal
