// Copyright 2026 MediaTek Inc.
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

#include "litert/vendors/mediatek/dispatch/shared_weight_cache.h"

#include <memory>

#include <gtest/gtest.h>
#include "litert/cc/litert_expected.h"

namespace litert::mediatek {
namespace {

using Cache = SharedWeightCache<int>;

// Returns a creator that counts its calls and makes an int holding `value`.
Cache::Creator Counting(int* calls, int value) {
  return [calls, value]() -> litert::Expected<std::shared_ptr<int>> {
    ++*calls;
    return std::make_shared<int>(value);
  };
}

TEST(SharedWeightCacheTest, SameRangeSharesOneObject) {
  Cache cache;
  const char blob[16] = {};
  int calls = 0;
  auto a = cache.GetOrCreate(blob, sizeof(blob), Counting(&calls, 1));
  auto b = cache.GetOrCreate(blob, sizeof(blob), Counting(&calls, 2));
  ASSERT_TRUE(a && b);
  EXPECT_EQ(a->get(), b->get());
  EXPECT_EQ(**b, 1);
  EXPECT_EQ(calls, 1);
}

TEST(SharedWeightCacheTest, DifferentAddressOrSizeDoesNotCollide) {
  Cache cache;
  const char blob[16] = {};
  int calls = 0;
  auto a = cache.GetOrCreate(blob, 16, Counting(&calls, 1));
  auto b = cache.GetOrCreate(blob, 8, Counting(&calls, 2));
  auto c = cache.GetOrCreate(blob + 8, 8, Counting(&calls, 3));
  ASSERT_TRUE(a && b && c);
  EXPECT_NE(a->get(), b->get());
  EXPECT_NE(b->get(), c->get());
  EXPECT_EQ(calls, 3);
}

TEST(SharedWeightCacheTest, RecreatesOnceTheLastUserDrops) {
  Cache cache;
  const char blob[16] = {};
  int calls = 0;
  {
    auto a = cache.GetOrCreate(blob, sizeof(blob), Counting(&calls, 1));
    ASSERT_TRUE(a);
  }
  auto b = cache.GetOrCreate(blob, sizeof(blob), Counting(&calls, 2));
  ASSERT_TRUE(b);
  EXPECT_EQ(**b, 2);
  EXPECT_EQ(calls, 2);
}

TEST(SharedWeightCacheTest, FailedCreateIsNotCached) {
  Cache cache;
  const char blob[16] = {};
  auto failed = cache.GetOrCreate(
      blob, sizeof(blob), []() -> litert::Expected<std::shared_ptr<int>> {
        return litert::Error(kLiteRtStatusErrorRuntimeFailure, "no dmabuf");
      });
  EXPECT_FALSE(failed);
  int calls = 0;
  auto ok = cache.GetOrCreate(blob, sizeof(blob), Counting(&calls, 7));
  ASSERT_TRUE(ok);
  EXPECT_EQ(**ok, 7);
  EXPECT_EQ(calls, 1);
}

TEST(SharedWeightCacheTest, ObjectOutlivesTheCache) {
  const char blob[16] = {};
  int freed = 0;
  std::shared_ptr<int> held;
  {
    Cache cache;
    auto a = cache.GetOrCreate(
        blob, sizeof(blob), [&freed]() -> litert::Expected<std::shared_ptr<int>> {
          return std::shared_ptr<int>(new int(5), [&freed](int* p) {
            ++freed;
            delete p;
          });
        });
    ASSERT_TRUE(a);
    held = *a;
  }
  // The device context (and its cache) may be destroyed before the
  // invocation contexts that hold the memory.
  EXPECT_EQ(freed, 0);
  EXPECT_EQ(*held, 5);
  held.reset();
  EXPECT_EQ(freed, 1);
}

}  // namespace
}  // namespace litert::mediatek
