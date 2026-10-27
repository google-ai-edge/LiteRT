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

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif  // defined(__linux__)

#include <gmock/gmock.h>
#include <gtest/gtest.h>
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
#if defined(__linux__) || defined(LITERT_WINDOWS_OS)
  LITERT_EXPECT_OK(AdviseWillNeed({}));
#else
  EXPECT_THAT(AdviseWillNeed({}),
              ::testing::litert::IsError(kLiteRtStatusErrorUnsupported));
#endif
}

#if defined(__linux__)
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
#endif  // defined(__linux__)

}  // namespace
}  // namespace litert::internal
