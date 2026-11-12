// Copyright 2026 Google LLC.
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

#include "litert/core/util/memory_advice.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif  // defined(__linux__) || defined(__APPLE__)

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/test/matchers.h"

namespace litert::internal {
namespace {

TEST(MemoryAdviceTest, GetPageSizeIsPowerOfTwo) {
  const size_t page = GetPageSize();
  EXPECT_GT(page, 0);
  EXPECT_EQ(page & (page - 1), 0);
}

TEST(MemoryAdviceTest, AdviseWillNeedOnEmptyReportsAvailability) {
#if defined(__linux__) || defined(__APPLE__) || defined(LITERT_WINDOWS_OS)
  LITERT_EXPECT_OK(AdviseWillNeed({}));
#else
  EXPECT_THAT(AdviseWillNeed({}),
              ::testing::litert::IsError(kLiteRtStatusErrorUnsupported));
#endif
}

TEST(MemoryAdviceTest, SplitPageRangesPureMath) {
  auto make_ranges = [](size_t n) {
    std::vector<PageRange> ranges(n);
    for (size_t i = 0; i < n; ++i) ranges[i] = {i * 4096, (i + 1) * 4096};
    return ranges;
  };
  // Checks that `slices` partition `ranges` contiguously and in order into
  // pieces of `expected_sizes`.
  auto expect_partition =
      [](absl::Span<const PageRange> ranges,
         const std::vector<absl::Span<const PageRange>>& slices,
         const std::vector<size_t>& expected_sizes) {
        ASSERT_EQ(slices.size(), expected_sizes.size());
        const PageRange* next = ranges.data();
        for (size_t i = 0; i < slices.size(); ++i) {
          EXPECT_EQ(slices[i].data(), next);
          EXPECT_EQ(slices[i].size(), expected_sizes[i]);
          next += slices[i].size();
        }
        EXPECT_EQ(next, ranges.data() + ranges.size());
      };

  EXPECT_TRUE(SplitPageRanges({}, 2).empty());
  {
    // Fewer than 16 ranges per slice: one slice.
    const auto ranges = make_ranges(31);
    expect_partition(ranges, SplitPageRanges(ranges, 2), {31});
  }
  {
    const auto ranges = make_ranges(32);
    expect_partition(ranges, SplitPageRanges(ranges, 2), {16, 16});
  }
  {
    // Uneven split keeps order: [0, 32) then [32, 65).
    const auto ranges = make_ranges(65);
    expect_partition(ranges, SplitPageRanges(ranges, 2), {32, 33});
  }
  {
    // Capped at 8 slices however many are requested.
    const auto ranges = make_ranges(256);
    expect_partition(ranges, SplitPageRanges(ranges, 100),
                     {32, 32, 32, 32, 32, 32, 32, 32});
  }
  {
    // Non-positive slice counts mean "no fan-out".
    const auto ranges = make_ranges(256);
    expect_partition(ranges, SplitPageRanges(ranges, 0), {256});
  }
}

#if defined(__linux__) || defined(__APPLE__)
// The hint reports the OS's verdict instead of dropping it: Ok for mapped
// pages, RuntimeFailure for a page that is not mapped, so a wrong table
// address cannot pass the LITERT_EXPECT_OK checks of the prefetch tests
// silently. The middle page of a three-page anonymous mapping is unmapped, so
// madvise on it must fail with ENOMEM, and the pages on both sides stay mapped
// so `mixed` covers mapped-hole-mapped. The test is single-threaded between
// the munmap and the madvise calls, so nothing is mapped into the hole.
TEST(MemoryAdviceTest, AdviseWillNeedReportsFailures) {
  const size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  void* const mapping = mmap(nullptr, 3 * page_size, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(mapping, MAP_FAILED);
  const uintptr_t base = reinterpret_cast<uintptr_t>(mapping);
  ASSERT_EQ(munmap(reinterpret_cast<void*>(base + page_size), page_size), 0);

  const PageRange mapped[] = {{base, base + page_size}};
  LITERT_EXPECT_OK(AdviseWillNeed(mapped));
  LITERT_EXPECT_OK(AdviseWillNeed({}));

  const PageRange hole[] = {{base + page_size, base + 2 * page_size}};
  EXPECT_THAT(AdviseWillNeed(hole),
              ::testing::litert::IsError(kLiteRtStatusErrorRuntimeFailure));
  // A failure anywhere in the list is reported, not only in the first range.
  const PageRange mixed[] = {{base, base + page_size},
                             {base + page_size, base + 2 * page_size},
                             {base + 2 * page_size, base + 3 * page_size}};
  EXPECT_THAT(AdviseWillNeed(mixed),
              ::testing::litert::IsError(kLiteRtStatusErrorRuntimeFailure));

  ASSERT_EQ(munmap(mapping, page_size), 0);
  ASSERT_EQ(munmap(reinterpret_cast<void*>(base + 2 * page_size), page_size),
            0);
}
#endif  // defined(__linux__) || defined(__APPLE__)

#if defined(__APPLE__)
// IsReadOnlyMapping() walks every VM region of [data, data + bytes) and fails
// closed on a writable region or a hole.
TEST(MemoryAdviceTest, IsReadOnlyMapping) {
  const size_t page_size = GetPageSize();
  EXPECT_FALSE(IsReadOnlyMapping(nullptr, page_size));

  // Two read-only pages; then the second page is made writable.
  void* const two_pages = mmap(nullptr, 2 * page_size, PROT_READ,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(two_pages, MAP_FAILED);
  EXPECT_FALSE(IsReadOnlyMapping(two_pages, 0));
  EXPECT_TRUE(IsReadOnlyMapping(two_pages, 2 * page_size));
  uint8_t* const second_page = static_cast<uint8_t*>(two_pages) + page_size;
  ASSERT_EQ(mprotect(second_page, page_size, PROT_READ | PROT_WRITE), 0);
  EXPECT_TRUE(IsReadOnlyMapping(two_pages, page_size));
  EXPECT_FALSE(IsReadOnlyMapping(two_pages, 2 * page_size));
  EXPECT_FALSE(IsReadOnlyMapping(second_page, page_size));
  ASSERT_EQ(munmap(two_pages, 2 * page_size), 0);

  // Three read-only pages with the middle one unmapped: the hole fails closed
  // when the whole span is queried.
  void* const three_pages = mmap(nullptr, 3 * page_size, PROT_READ,
                                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(three_pages, MAP_FAILED);
  uint8_t* const base = static_cast<uint8_t*>(three_pages);
  ASSERT_EQ(munmap(base + page_size, page_size), 0);
  EXPECT_FALSE(IsReadOnlyMapping(three_pages, 3 * page_size));
  ASSERT_EQ(munmap(three_pages, page_size), 0);
  ASSERT_EQ(munmap(base + 2 * page_size, page_size), 0);
}
#endif  // defined(__APPLE__)

}  // namespace
}  // namespace litert::internal
