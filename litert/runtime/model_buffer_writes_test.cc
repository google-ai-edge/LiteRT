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

#include "litert/runtime/model_buffer_writes.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#ifdef _WIN32
#include <windows.h>
// psapi.h needs the declarations of windows.h.
#include <psapi.h>
#pragma comment(lib, "psapi.lib")

#include <string>
#include <thread>  // NOLINT(build/c++11)
#endif

namespace litert::internal {
namespace {

TEST(MakeModelBufferWritableTest, LeavesMemoryThatIsNotAFileViewAlone) {
  std::vector<uint8_t> buffer(100, 1);
  ASSERT_TRUE(MakeModelBufferWritable(buffer.data() + 10, 20).HasValue());
  buffer[15] = 2;
  EXPECT_EQ(buffer[15], 2);
  EXPECT_TRUE(MakeModelBufferWritable(buffer.data(), 0).HasValue());
  KeepModelBufferReadOnlyUntilWritten(buffer.data());
  KeepModelBufferReadOnlyUntilWritten(nullptr);
  buffer[16] = 3;
  EXPECT_EQ(buffer[16], 3);
}

#ifdef _WIN32
constexpr size_t kBytes = 64 << 20;
constexpr size_t kPage = 4096;

uint8_t Pattern(size_t i) { return static_cast<uint8_t>(i * 7 + (i >> 12)); }

int64_t Commit() {
  PROCESS_MEMORY_COUNTERS_EX counters{};
  counters.cb = sizeof(counters);
  EXPECT_TRUE(GetProcessMemoryInfo(
      GetCurrentProcess(),
      reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters), sizeof(counters)));
  return static_cast<int64_t>(counters.PrivateUsage);
}

DWORD Protection(const void* address) {
  MEMORY_BASIC_INFORMATION region{};
  EXPECT_EQ(VirtualQuery(address, &region, sizeof(region)), sizeof(region));
  return region.Protect;
}

// A file of known bytes, opened for reading only, and views of it.
class FileViewTest : public ::testing::Test {
 protected:
  void SetUp() override {
    path_ = ::testing::TempDir() + "\\model_buffer_writes_test_" +
            std::to_string(GetCurrentProcessId()) + "_" +
            ::testing::UnitTest::GetInstance()->current_test_info()->name();
    HANDLE writer = CreateFileA(path_.c_str(), GENERIC_WRITE, 0, nullptr,
                                CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    ASSERT_NE(writer, INVALID_HANDLE_VALUE);
    std::vector<uint8_t> bytes(kBytes);
    for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = Pattern(i);
    DWORD written = 0;
    ASSERT_TRUE(WriteFile(writer, bytes.data(), static_cast<DWORD>(kBytes),
                          &written, nullptr));
    ASSERT_EQ(written, kBytes);
    CloseHandle(writer);
    file_ = CreateFileA(path_.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    ASSERT_NE(file_, INVALID_HANDLE_VALUE);
  }

  void TearDown() override {
    for (void* view : views_) UnmapViewOfFile(view);
    for (HANDLE section : sections_) CloseHandle(section);
    CloseHandle(file_);
    DeleteFileA(path_.c_str());
  }

  // Maps the whole file as a runtime maps a model: copy-on-write, or read-only.
  uint8_t* Map(bool copy_on_write) {
    HANDLE section = CreateFileMappingA(
        file_, nullptr, copy_on_write ? PAGE_WRITECOPY : PAGE_READONLY, 0, 0,
        nullptr);
    EXPECT_NE(section, nullptr);
    sections_.push_back(section);
    void* view = MapViewOfFile(
        section, copy_on_write ? FILE_MAP_COPY : FILE_MAP_READ, 0, 0, 0);
    EXPECT_NE(view, nullptr);
    views_.push_back(view);
    return static_cast<uint8_t*>(view);
  }

  // Whether the file still holds the bytes it was written with.
  bool FileUnchanged() const {
    std::vector<uint8_t> bytes(kBytes);
    DWORD read = 0;
    SetFilePointer(file_, 0, nullptr, FILE_BEGIN);
    if (!ReadFile(file_, bytes.data(), static_cast<DWORD>(kBytes), &read,
                  nullptr) ||
        read != kBytes) {
      return false;
    }
    for (size_t i = 0; i < bytes.size(); ++i) {
      if (bytes[i] != Pattern(i)) return false;
    }
    return true;
  }

  std::string path_;
  HANDLE file_ = INVALID_HANDLE_VALUE;
  std::vector<HANDLE> sections_;
  std::vector<void*> views_;
};

TEST_F(FileViewTest, ChargesOnlyThePagesOfACopyOnWriteViewThatAreAskedFor) {
  const int64_t before = Commit();
  uint8_t* view = Map(/*copy_on_write=*/true);
  ASSERT_NE(view, nullptr);
  ASSERT_GE(Commit() - before, static_cast<int64_t>(kBytes))
      << "the view was not charged in full";

  uint8_t* target = view + 5 * (1 << 20) + 4000;  // Spans two pages.
  ASSERT_TRUE(MakeModelBufferWritable(target, 200).HasValue());
  EXPECT_LT(Commit() - before, static_cast<int64_t>(4 * kPage + (1 << 20)))
      << "the unwritten pages kept their charge";
  for (int i = 0; i < 200; ++i) target[i] = static_cast<uint8_t>(~target[i]);
  for (int i = 0; i < 200; ++i) {
    ASSERT_EQ(target[i], static_cast<uint8_t>(~Pattern(target - view + i)));
  }
  EXPECT_EQ(Protection(target), PAGE_READWRITE);
  EXPECT_EQ(Protection(target + 199), PAGE_READWRITE);
  EXPECT_EQ(Protection(view), PAGE_READONLY);
  EXPECT_EQ(Protection(view + kBytes - 1), PAGE_READONLY);
  EXPECT_EQ(view[kBytes - 1], Pattern(kBytes - 1));

  // A range asked for and not written to yet survives the view being made
  // read-only again, as another model's initialization would.
  uint8_t* pending = view + 9 * (1 << 20);
  ASSERT_TRUE(MakeModelBufferWritable(pending, 10).HasValue());
  KeepModelBufferReadOnlyUntilWritten(view);
  pending[0] = 0x5A;
  EXPECT_EQ(pending[0], 0x5A);

  // A later range: the page written to keeps its copy.
  uint8_t* other = view + kBytes - 100;
  ASSERT_TRUE(MakeModelBufferWritable(other, 100).HasValue());
  other[0] = static_cast<uint8_t>(~other[0]);
  EXPECT_EQ(target[0], static_cast<uint8_t>(~Pattern(target - view)));
  EXPECT_EQ(Protection(target), PAGE_READWRITE);
  EXPECT_TRUE(FileUnchanged());
}

TEST_F(FileViewTest, KeepsAModelThatIsNeverWrittenToReadOnly) {
  const int64_t before = Commit();
  uint8_t* view = Map(/*copy_on_write=*/true);
  ASSERT_NE(view, nullptr);
  KeepModelBufferReadOnlyUntilWritten(view + 12345);
  EXPECT_LT(Commit() - before, static_cast<int64_t>(1 << 20));
  EXPECT_EQ(Protection(view), PAGE_READONLY);
  EXPECT_EQ(view[kBytes / 2], Pattern(kBytes / 2));
  // Once is enough, and a later write still works.
  KeepModelBufferReadOnlyUntilWritten(view);
  ASSERT_TRUE(MakeModelBufferWritable(view + 7, 1).HasValue());
  view[7] = static_cast<uint8_t>(~view[7]);
  EXPECT_EQ(view[7], static_cast<uint8_t>(~Pattern(7)));
  EXPECT_TRUE(FileUnchanged());
}

// A view mapped where an earlier one was, as a runtime that maps one model
// after another may get: it is made read-only all the same.
TEST_F(FileViewTest, KeepsAViewMappedWhereAnotherWasReadOnly) {
  uint8_t* first = Map(/*copy_on_write=*/true);
  ASSERT_NE(first, nullptr);
  KeepModelBufferReadOnlyUntilWritten(first);
  ASSERT_TRUE(UnmapViewOfFile(first));
  views_.pop_back();
  HANDLE section =
      CreateFileMappingA(file_, nullptr, PAGE_WRITECOPY, 0, 0, nullptr);
  ASSERT_NE(section, nullptr);
  sections_.push_back(section);
  void* second = MapViewOfFileEx(section, FILE_MAP_COPY, 0, 0, 0, first);
  ASSERT_EQ(second, first) << "the address was taken in between";
  views_.push_back(second);
  EXPECT_EQ(Protection(second), PAGE_WRITECOPY);
  KeepModelBufferReadOnlyUntilWritten(second);
  EXPECT_EQ(Protection(second), PAGE_READONLY);
}

TEST_F(FileViewTest, LeavesAReadOnlyViewAsItIs) {
  uint8_t* view = Map(/*copy_on_write=*/false);
  ASSERT_NE(view, nullptr);
  KeepModelBufferReadOnlyUntilWritten(view);
  EXPECT_EQ(Protection(view), PAGE_READONLY);
}

TEST_F(FileViewTest, MakesARangeOfAReadOnlyViewCopyOnWrite) {
  uint8_t* view = Map(/*copy_on_write=*/false);
  ASSERT_NE(view, nullptr);
  uint8_t* target = view + 3 * kPage + 17;
  ASSERT_TRUE(MakeModelBufferWritable(target, 8).HasValue());
  for (int i = 0; i < 8; ++i) target[i] = 0xEE;
  EXPECT_EQ(target[7], 0xEE);
  EXPECT_EQ(Protection(view), PAGE_READONLY);
  EXPECT_TRUE(FileUnchanged());
}

TEST_F(FileViewTest, ThreadsWritingToOneViewKeepEachOthersWrites) {
  uint8_t* view = Map(/*copy_on_write=*/true);
  ASSERT_NE(view, nullptr);
  constexpr int kThreads = 8;
  constexpr int kPagesPerThread = 64;
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([view, t] {
      for (int p = 0; p < kPagesPerThread; ++p) {
        // Interleave the threads' pages, so neighbors belong to others.
        uint8_t* at = view + ((p * kThreads + t) * 3 + 1) * kPage + 5;
        ASSERT_TRUE(MakeModelBufferWritable(at, 1).HasValue());
        *at = static_cast<uint8_t>(t + 1);
      }
    });
  }
  for (auto& thread : threads) thread.join();
  for (int t = 0; t < kThreads; ++t) {
    for (int p = 0; p < kPagesPerThread; ++p) {
      EXPECT_EQ(view[((p * kThreads + t) * 3 + 1) * kPage + 5], t + 1);
    }
  }
  EXPECT_TRUE(FileUnchanged());
}
#endif  // _WIN32

}  // namespace
}  // namespace litert::internal
