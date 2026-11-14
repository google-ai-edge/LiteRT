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

#include "tensor/utils/parallel.h"

#include <atomic>
#include <cstddef>
#include <initializer_list>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace litert::tensor {
namespace {

using ::testing::Each;
using ::testing::Eq;

TEST(ParallelForTest, ZeroCountDoesNotInvokeCallback) {
  std::atomic<int> calls{0};
  ParallelFor(
      /*count=*/0, /*min_grain=*/1,
      [&](size_t /*begin*/, size_t /*end*/) {
        calls.fetch_add(1, std::memory_order_relaxed);
      },
      /*max_threads=*/8);
  EXPECT_EQ(calls.load(std::memory_order_relaxed), 0);
}

TEST(ParallelForTest, CountAtOrBelowMinGrainRunsSynchronouslyOnCallerThread) {
  thread_local bool is_caller_thread = false;
  is_caller_thread = true;

  std::vector<int> out(7, 0);
  std::atomic<int> calls{0};
  std::atomic<bool> ran_on_caller{false};
  ParallelFor(
      out.size(), /*min_grain=*/7,
      [&](size_t begin, size_t end) {
        calls.fetch_add(1, std::memory_order_relaxed);
        ran_on_caller.store(is_caller_thread, std::memory_order_relaxed);
        for (size_t i = begin; i < end; ++i) {
          out[i] += 1;
        }
      },
      /*max_threads=*/8);

  EXPECT_EQ(calls.load(std::memory_order_relaxed), 1);
  EXPECT_TRUE(ran_on_caller.load(std::memory_order_relaxed));
  EXPECT_THAT(out, Each(Eq(1)));
}

TEST(ParallelForTest, ZeroMinGrainIsClampedAndCoversRange) {
  constexpr size_t kCount = 65;
  std::vector<std::atomic<int>> visited(kCount);
  for (std::atomic<int>& v : visited) {
    v.store(0, std::memory_order_relaxed);
  }

  ParallelFor(
      kCount, /*min_grain=*/0,
      [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
          visited[i].fetch_add(1, std::memory_order_relaxed);
        }
      },
      /*max_threads=*/4);

  for (size_t i = 0; i < kCount; ++i) {
    EXPECT_EQ(visited[i].load(std::memory_order_relaxed), 1) << "i=" << i;
  }
}

TEST(ParallelForTest, MaxThreadsAtMostOneRunsSynchronouslyOnCallerThread) {
  thread_local bool is_caller_thread = false;
  is_caller_thread = true;

  for (size_t max_threads : {size_t{0}, size_t{1}}) {
    std::vector<int> out(64, 0);
    std::atomic<int> calls{0};
    std::atomic<bool> all_on_caller{true};
    auto fn = [&](size_t begin, size_t end) {
      calls.fetch_add(1, std::memory_order_relaxed);
      if (!is_caller_thread) {
        all_on_caller.store(false, std::memory_order_relaxed);
      }
      for (size_t i = begin; i < end; ++i) {
        out[i] += 1;
      }
    };
    ParallelFor(out.size(), /*min_grain=*/1, fn, max_threads);
    EXPECT_EQ(calls.load(std::memory_order_relaxed), 1)
        << "max_threads=" << max_threads;
    EXPECT_TRUE(all_on_caller.load(std::memory_order_relaxed))
        << "max_threads=" << max_threads;
    EXPECT_THAT(out, Each(Eq(1)));

    // Also exercise `ParallelForImpl` directly in case a caller bypasses the
    // header fast path.
    calls.store(0, std::memory_order_relaxed);
    all_on_caller.store(true, std::memory_order_relaxed);
    using FnType = decltype(fn);
    internal::ParallelForImpl(out.size(), /*min_grain=*/1, max_threads, &fn,
                              [](void* ctx, size_t begin, size_t end) {
                                (*static_cast<FnType*>(ctx))(begin, end);
                              });
    EXPECT_EQ(calls.load(std::memory_order_relaxed), 1)
        << "impl max_threads=" << max_threads;
    EXPECT_TRUE(all_on_caller.load(std::memory_order_relaxed))
        << "impl max_threads=" << max_threads;
    EXPECT_THAT(out, Each(Eq(2)));
  }
}

TEST(ParallelForTest, EnforcesMaxThreadsCapAndCoversUnevenRangesExactlyOnce) {
  constexpr size_t kCount = 257;
  constexpr size_t kMinGrain = 5;
  constexpr size_t kMaxThreads = 2;

  std::vector<std::atomic<int>> visited(kCount);
  for (std::atomic<int>& v : visited) {
    v.store(0, std::memory_order_relaxed);
  }
  std::atomic<size_t> active{0};
  std::atomic<size_t> peak_active{0};
  std::atomic<int> invalid_ranges{0};

  ParallelFor(
      kCount, kMinGrain,
      [&](size_t begin, size_t end) {
        if (begin >= end || end > kCount) {
          invalid_ranges.fetch_add(1, std::memory_order_relaxed);
          return;
        }
        const size_t cur = active.fetch_add(1, std::memory_order_acq_rel) + 1;
        size_t prev_peak = peak_active.load(std::memory_order_relaxed);
        while (cur > prev_peak &&
               !peak_active.compare_exchange_weak(prev_peak, cur,
                                                  std::memory_order_relaxed)) {
        }
        for (size_t i = begin; i < end; ++i) {
          visited[i].fetch_add(1, std::memory_order_relaxed);
        }
        active.fetch_sub(1, std::memory_order_acq_rel);
      },
      kMaxThreads);

  EXPECT_EQ(invalid_ranges.load(std::memory_order_relaxed), 0);
  EXPECT_LE(peak_active.load(std::memory_order_relaxed), kMaxThreads);
  for (size_t i = 0; i < kCount; ++i) {
    EXPECT_EQ(visited[i].load(std::memory_order_relaxed), 1) << "i=" << i;
  }
}

TEST(ParallelForTest, NestedParallelForDoesNotDeadlockAndComputesAllCells) {
  constexpr size_t kOuter = 37;
  constexpr size_t kInner = 29;
  std::vector<size_t> grid(kOuter * kInner, 0);

  ParallelFor(
      kOuter, /*min_grain=*/2,
      [&](size_t r_begin, size_t r_end) {
        for (size_t r = r_begin; r < r_end; ++r) {
          ParallelFor(
              kInner, /*min_grain=*/3,
              [&](size_t c_begin, size_t c_end) {
                for (size_t c = c_begin; c < c_end; ++c) {
                  grid[r * kInner + c] = (r + 1) * 100 + (c + 1);
                }
              },
              /*max_threads=*/8);
        }
      },
      /*max_threads=*/8);

  for (size_t r = 0; r < kOuter; ++r) {
    for (size_t c = 0; c < kInner; ++c) {
      EXPECT_EQ(grid[r * kInner + c], (r + 1) * 100 + (c + 1))
          << "r=" << r << ", c=" << c;
    }
  }
}

}  // namespace
}  // namespace litert::tensor
