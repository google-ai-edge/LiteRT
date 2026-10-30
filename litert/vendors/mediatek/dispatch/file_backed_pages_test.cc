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

#include "litert/vendors/mediatek/dispatch/file_backed_pages.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace litert::mediatek {
namespace {

constexpr size_t kPage = 4096;

const void* Addr(uintptr_t value) {
  return reinterpret_cast<const void*>(value);
}

TEST(InnerPageRangeTest, AlignedRangeIsUnchanged) {
  const PageRange range = InnerPageRange(Addr(2 * kPage), 3 * kPage, kPage);
  EXPECT_EQ(range.begin, 2 * kPage);
  EXPECT_EQ(range.end, 5 * kPage);
  EXPECT_EQ(range.size(), 3 * kPage);
}

TEST(InnerPageRangeTest, ExcludesPartialPagesAtBothEnds) {
  const PageRange range =
      InnerPageRange(Addr(2 * kPage + 100), 3 * kPage, kPage);
  EXPECT_EQ(range.begin, 3 * kPage);
  EXPECT_EQ(range.end, 5 * kPage);
}

TEST(InnerPageRangeTest, EmptyWhenNoWholePageFits) {
  EXPECT_TRUE(InnerPageRange(Addr(kPage + 1), kPage, kPage).empty());
  EXPECT_TRUE(InnerPageRange(Addr(kPage), kPage - 1, kPage).empty());
  EXPECT_TRUE(InnerPageRange(Addr(kPage), 0, kPage).empty());
}

TEST(InnerPageRangeTest, EmptyForInvalidInput) {
  EXPECT_TRUE(InnerPageRange(Addr(kPage), 4 * kPage, 0).empty());
  EXPECT_TRUE(InnerPageRange(Addr(kPage), 4 * kPage, 3000).empty());
  EXPECT_TRUE(
      InnerPageRange(Addr(UINTPTR_MAX - kPage), 2 * kPage, kPage).empty());
}

TEST(OuterPageRangeTest, CoversPartialPagesAtBothEnds) {
  const PageRange range =
      OuterPageRange(Addr(2 * kPage + 100), 3 * kPage, kPage);
  EXPECT_EQ(range.begin, 2 * kPage);
  EXPECT_EQ(range.end, 6 * kPage);
}

TEST(OuterPageRangeTest, EmptyForInvalidInput) {
  EXPECT_TRUE(OuterPageRange(Addr(kPage), 0, kPage).empty());
  EXPECT_TRUE(OuterPageRange(Addr(kPage), kPage, 0).empty());
  EXPECT_TRUE(
      OuterPageRange(Addr(UINTPTR_MAX - kPage), 2 * kPage, kPage).empty());
}

// Returns the `Rss:` value in KiB of the mapping that starts at `addr`, or -1
// if it cannot be found.
int64_t MappingRssKb(const void* addr) {
  std::ifstream smaps("/proc/self/smaps");
  const uintptr_t target = reinterpret_cast<uintptr_t>(addr);
  std::string line;
  bool in_target = false;
  while (std::getline(smaps, line)) {
    uintptr_t begin = 0, end = 0;
    if (std::sscanf(line.c_str(), "%" SCNxPTR "-%" SCNxPTR " ", &begin, &end) ==
        2) {
      in_target = begin == target;
      continue;
    }
    int64_t rss_kb = 0;
    if (in_target &&
        std::sscanf(line.c_str(), "Rss: %" SCNd64 " kB", &rss_kb) == 1) {
      return rss_kb;
    }
  }
  return -1;
}

class FileBackedPagesTest : public ::testing::Test {
 protected:
  static constexpr size_t kNumPages = 16;

  void SetUp() override {
    page_size_ = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    size_ = kNumPages * page_size_;
    expected_.resize(size_);
    for (size_t i = 0; i < size_; ++i) {
      expected_[i] = static_cast<char>(i * 31 + 7);
    }
  }

  void TearDown() override {
    if (mapping_ != nullptr) {
      munmap(mapping_, size_);
    }
    if (fd_ >= 0) {
      close(fd_);
      unlink(path_.c_str());
    }
  }

  // Writes `expected_` to a temporary file and maps it read-only into
  // `mapping_`, like a model file loaded with mmap.
  void MapTestFile() {
    path_ = ::testing::TempDir() + "/file_backed_pages_XXXXXX";
    fd_ = mkstemp(path_.data());
    ASSERT_GE(fd_, 0);
    ASSERT_EQ(write(fd_, expected_.data(), size_), static_cast<ssize_t>(size_));
    void* mapping = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    ASSERT_NE(mapping, MAP_FAILED);
    mapping_ = mapping;
  }

  const char* data() const { return static_cast<const char*>(mapping_); }

  // Reads every page of the mapping so that it is resident.
  void TouchAll() const {
    volatile char sink = 0;
    for (size_t i = 0; i < size_; i += page_size_) {
      sink = sink + data()[i];
    }
  }

  size_t page_size_ = 0;
  size_t size_ = 0;
  std::vector<char> expected_;
  std::string path_;
  int fd_ = -1;
  void* mapping_ = nullptr;
};

TEST_F(FileBackedPagesTest, ReleasesInteriorPagesAndKeepsContents) {
  ASSERT_NO_FATAL_FAILURE(MapTestFile());
  TouchAll();
  const int64_t page_kb = page_size_ / 1024;
  ASSERT_EQ(MappingRssKb(mapping_), kNumPages * page_kb);

  // The region starts and ends 100 bytes inside the first and last pages, so
  // only the kNumPages - 2 pages in between are released.
  EXPECT_TRUE(ReleaseFileBackedPages(fd_, data() + 100, size_ - 200));
  EXPECT_EQ(MappingRssKb(mapping_), 2 * page_kb);

  // Released pages fault back in from the file with the original contents.
  EXPECT_EQ(std::memcmp(data(), expected_.data(), size_), 0);
}

TEST_F(FileBackedPagesTest, DoesNotTouchMemoryWithoutFileDescriptor) {
  std::vector<char> heap_copy = expected_;

  // `MADV_DONTNEED` would zero-fill heap memory, so the helper must refuse to
  // act without a backing file.
  EXPECT_FALSE(ReleaseFileBackedPages(/*fd=*/-1, heap_copy.data(), size_));
  EXPECT_EQ(heap_copy, expected_);
}

TEST_F(FileBackedPagesTest, ReturnsFalseForNullOrEmptyRegion) {
  ASSERT_NO_FATAL_FAILURE(MapTestFile());
  EXPECT_FALSE(ReleaseFileBackedPages(fd_, nullptr, size_));
  EXPECT_FALSE(ReleaseFileBackedPages(fd_, data(), 0));
  EXPECT_FALSE(ReleaseFileBackedPages(fd_, data() + 1, page_size_));
}

TEST_F(FileBackedPagesTest, AdviseNoHugePagesKeepsContents) {
  ASSERT_NO_FATAL_FAILURE(MapTestFile());
  AdviseNoHugePages(fd_, data() + 100, size_ - 200);
  AdviseNoHugePages(/*fd=*/-1, data(), size_);
  EXPECT_EQ(std::memcmp(data(), expected_.data(), size_), 0);
}

}  // namespace
}  // namespace litert::mediatek
