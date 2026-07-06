// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/backends/backend_utils.h"

#include <array>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl

namespace qnn {
namespace {
TEST(BackendUtilsTest, SetNullTermPtrArrayTest) {
  std::vector<int> src = {1, 2, 3};
  std::array<const int*, 5> dst{};
  SetNullTermPtrArray(absl::MakeConstSpan(src), dst);

  ASSERT_TRUE(dst[0]);
  EXPECT_EQ(*dst[0], 1);
  ASSERT_TRUE(dst[1]);
  EXPECT_EQ(*dst[1], 2);
  ASSERT_TRUE(dst[2]);
  EXPECT_EQ(*dst[2], 3);
  EXPECT_FALSE(dst[3]);
}

TEST(BackendUtilsTest, SetNullTermPtrArrayTestOverflow) {
  std::vector<int> src = {1, 2, 3, 4, 5};
  std::array<const int*, 4> dst{};
  SetNullTermPtrArray(absl::MakeConstSpan(src), dst);

  ASSERT_TRUE(dst[0]);
  EXPECT_EQ(*dst[0], 1);
  ASSERT_TRUE(dst[1]);
  EXPECT_EQ(*dst[1], 2);
  ASSERT_TRUE(dst[2]);
  EXPECT_EQ(*dst[2], 3);
  EXPECT_FALSE(dst[3]);
}

// VOTING THREAD /////////////////////////////////////////////////////////
TEST(VotingThreadTest, UpvoteCancelsPendingDownvote) {
  std::atomic<int> up_count{0}, down_count{0};
  VotingThread vt([&](VotingThread::VoteType v) {
    if (v == VotingThread::VoteType::kUpVote)
      up_count.fetch_add(1);
    else
      down_count.fetch_add(1);
  });

  vt.Enqueue(VotingThread::VoteType::kUpVote);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  vt.Enqueue(VotingThread::VoteType::kDownVote, /*debounce=*/true);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  vt.Enqueue(VotingThread::VoteType::kUpVote);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  EXPECT_EQ(up_count.load(), 2);
  EXPECT_EQ(down_count.load(), 0);
}

TEST(VotingThreadTest, DebounceDownvoteFiresAfterTimeout) {
  std::atomic<int> down_count{0};
  VotingThread vt([&](VotingThread::VoteType v) {
    if (v == VotingThread::VoteType::kDownVote) down_count.fetch_add(1);
  });

  vt.Enqueue(VotingThread::VoteType::kDownVote, /*debounce=*/true);
  std::this_thread::sleep_for(std::chrono::milliseconds(400));

  EXPECT_EQ(down_count.load(), 1);
}

TEST(VotingThreadTest, DestructWithPendingVoteJoinsCleanly) {
  {
    VotingThread vt([&](VotingThread::VoteType) {});
    vt.Enqueue(VotingThread::VoteType::kDownVote, /*debounce=*/true);
  }
  SUCCEED();
}

}  // namespace
}  // namespace qnn
