/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensor/examples/gemma4/host_kv_cache.h"

#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/examples/utils/ring_buffer.h"

namespace litert::tensor::examples::gemma4 {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

constexpr int kDim = 2;

// Value of element `d` of token `t` of head `h`.
float Value(int h, int t, int d) { return 1000 * h + 10 * t + d; }

// Returns the K or V of token `t` for every head: [num_heads, kDim].
std::vector<float> Token(int num_heads, int t) {
  std::vector<float> token;
  for (int h = 0; h < num_heads; ++h) {
    for (int d = 0; d < kDim; ++d) {
      token.push_back(Value(h, t, d));
    }
  }
  return token;
}

// Returns tokens [first, end) of every head: [num_heads, end - first, kDim].
std::vector<float> Tokens(int num_heads, int first, int end) {
  std::vector<float> tokens;
  for (int h = 0; h < num_heads; ++h) {
    for (int t = first; t < end; ++t) {
      for (int d = 0; d < kDim; ++d) {
        tokens.push_back(Value(h, t, d));
      }
    }
  }
  return tokens;
}

TEST(AppendTokenToKvCacheTest, InsertsTheTokenAtTheEndOfEveryHead) {
  // [2 heads, 2 tokens, kDim] with room for one more token.
  std::vector<float> cache = Tokens(2, 0, 2);
  cache.resize(2 * 3 * kDim);

  AppendTokenToKvCache(absl::MakeSpan(cache), Token(2, 2), /*num_kv_heads=*/2,
                       /*cache_len=*/2, kDim);

  EXPECT_THAT(cache, ElementsAreArray(Tokens(2, 0, 3)));
}

TEST(HostKvCacheTest, RejectsInvalidDimensions) {
  EXPECT_THAT(HostKvCache::Create(/*num_kv_heads=*/0, kDim,
                                  /*max_cache_len=*/4, /*window_len=*/0),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(HostKvCacheTest, KeepsEveryTokenWithoutAWindow) {
  absl::StatusOr<HostKvCache> cache =
      HostKvCache::Create(/*num_kv_heads=*/2, kDim, /*max_cache_len=*/5,
                          /*window_len=*/0);
  ASSERT_THAT(cache, IsOk());

  ASSERT_THAT(cache->InitFromPrefill(Tokens(2, 0, 3), /*seq_len=*/3), IsOk());
  ASSERT_THAT(cache->Push(Token(2, 3)), IsOk());
  ASSERT_THAT(cache->Push(Token(2, 4)), IsOk());

  EXPECT_EQ(cache->len(), 5);
  EXPECT_THAT(cache->data(), ElementsAreArray(Tokens(2, 0, 5)));
  EXPECT_THAT(cache->Push(Token(2, 5)),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(HostKvCacheTest, RejectsAPrefillLongerThanTheCache) {
  absl::StatusOr<HostKvCache> cache =
      HostKvCache::Create(/*num_kv_heads=*/1, kDim, /*max_cache_len=*/2,
                          /*window_len=*/0);
  ASSERT_THAT(cache, IsOk());

  EXPECT_THAT(cache->InitFromPrefill(Tokens(1, 0, 3), /*seq_len=*/3),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

struct WindowParams {
  HostKvCache::RingMode ring_mode;
  int num_heads;
};

class HostKvCacheWindowTest : public ::testing::TestWithParam<WindowParams> {
 protected:
  void SetUp() override {
    if (GetParam().ring_mode == HostKvCache::RingMode::kMirrored &&
        !RingBufferStorage::IsMirroredModeSupported()) {
      GTEST_SKIP() << "The mirrored mode isn't supported on this platform.";
    }
  }

  static HostKvCache Create(int max_cache_len, int window_len) {
    absl::StatusOr<HostKvCache> cache =
        HostKvCache::Create(GetParam().num_heads, kDim, max_cache_len,
                            window_len, GetParam().ring_mode);
    ABSL_CHECK_OK(cache);
    return *std::move(cache);
  }
};

INSTANTIATE_TEST_SUITE_P(
    RingModesAndHeads, HostKvCacheWindowTest,
    ::testing::Values(WindowParams{HostKvCache::RingMode::kMirrored, 1},
                      WindowParams{HostKvCache::RingMode::kMirrored, 2},
                      WindowParams{HostKvCache::RingMode::kCompacting, 1},
                      WindowParams{HostKvCache::RingMode::kCompacting, 2}),
    [](const ::testing::TestParamInfo<WindowParams>& info) {
      return std::string(info.param.ring_mode ==
                                 HostKvCache::RingMode::kMirrored
                             ? "Mirrored"
                             : "Compacting") +
             (info.param.num_heads == 1 ? "OneHead" : "TwoHeads");
    });

TEST_P(HostKvCacheWindowTest, KeepsTheLastTokensInOrder) {
  const int num_heads = GetParam().num_heads;
  HostKvCache cache = Create(/*max_cache_len=*/20, /*window_len=*/3);
  EXPECT_EQ(cache.capacity(), 3);

  // Shorter than the window: everything is kept.
  ASSERT_THAT(cache.InitFromPrefill(Tokens(num_heads, 0, 2), /*seq_len=*/2),
              IsOk());
  EXPECT_EQ(cache.len(), 2);
  EXPECT_THAT(cache.data(), ElementsAreArray(Tokens(num_heads, 0, 2)));

  for (int t = 2; t < 12; ++t) {
    ASSERT_THAT(cache.Push(Token(num_heads, t)), IsOk());
    const int first = t + 1 > 3 ? t + 1 - 3 : 0;
    EXPECT_EQ(cache.len(), t + 1 - first) << "token " << t;
    EXPECT_THAT(cache.data(), ElementsAreArray(Tokens(num_heads, first, t + 1)))
        << "token " << t;
  }
}

TEST_P(HostKvCacheWindowTest, KeepsTheEndOfALongPrefill) {
  const int num_heads = GetParam().num_heads;
  HostKvCache cache = Create(/*max_cache_len=*/20, /*window_len=*/3);

  ASSERT_THAT(cache.InitFromPrefill(Tokens(num_heads, 0, 5), /*seq_len=*/5),
              IsOk());
  EXPECT_EQ(cache.len(), 3);
  EXPECT_THAT(cache.data(), ElementsAreArray(Tokens(num_heads, 2, 5)));

  for (int t = 5; t < 12; ++t) {
    ASSERT_THAT(cache.Push(Token(num_heads, t)), IsOk());
    EXPECT_THAT(cache.data(), ElementsAreArray(Tokens(num_heads, t - 2, t + 1)))
        << "token " << t;
  }
}

TEST_P(HostKvCacheWindowTest, InitFromPrefillReplacesTheContents) {
  const int num_heads = GetParam().num_heads;
  HostKvCache cache = Create(/*max_cache_len=*/20, /*window_len=*/3);
  ASSERT_THAT(cache.InitFromPrefill(Tokens(num_heads, 0, 5), /*seq_len=*/5),
              IsOk());
  ASSERT_THAT(cache.Push(Token(num_heads, 5)), IsOk());

  ASSERT_THAT(cache.InitFromPrefill(Tokens(num_heads, 7, 9), /*seq_len=*/2),
              IsOk());

  EXPECT_EQ(cache.len(), 2);
  EXPECT_THAT(cache.data(), ElementsAreArray(Tokens(num_heads, 7, 9)));
}

TEST_P(HostKvCacheWindowTest, WindowLargerThanTheCacheNeverDropsTokens) {
  const int num_heads = GetParam().num_heads;
  HostKvCache cache = Create(/*max_cache_len=*/4, /*window_len=*/10);
  EXPECT_EQ(cache.capacity(), 4);

  ASSERT_THAT(cache.InitFromPrefill(Tokens(num_heads, 0, 2), /*seq_len=*/2),
              IsOk());
  ASSERT_THAT(cache.Push(Token(num_heads, 2)), IsOk());
  ASSERT_THAT(cache.Push(Token(num_heads, 3)), IsOk());

  EXPECT_THAT(cache.data(), ElementsAreArray(Tokens(num_heads, 0, 4)));
}

TEST(HostKvCacheTest, RejectsATooShortToken) {
  absl::StatusOr<HostKvCache> cache =
      HostKvCache::Create(/*num_kv_heads=*/2, kDim, /*max_cache_len=*/4,
                          /*window_len=*/2);
  ASSERT_THAT(cache, IsOk());

  EXPECT_THAT(cache->Push(Token(1, 0)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(cache->data(), ElementsAre());
}

}  // namespace
}  // namespace litert::tensor::examples::gemma4
