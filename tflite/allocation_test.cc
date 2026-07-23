/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tflite/converter/allocation.h"

#include <fcntl.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

#if defined(_WIN32)
#include <io.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <gtest/gtest.h>
#include "tflite/testing/util.h"

namespace tflite {
namespace {

// The file-descriptor / offset tests below run on both POSIX and Windows so
// that the Win32 CreateFileMapping/MapViewOfFile path in mmap_allocation.cc is
// exercised. These helpers wrap the small platform differences (MSVC's CRT
// uses `_open`/`_close`/`_filelengthi64` and needs the `_O_BINARY` flag).
int OpenReadOnly(const char* path) {
#if defined(_WIN32)
  return _open(path, _O_RDONLY | _O_BINARY);
#else
  return open(path, O_RDONLY);
#endif
}

void CloseFd(int fd) {
#if defined(_WIN32)
  _close(fd);
#else
  close(fd);
#endif
}

// Returns the file size in bytes, or -1 on failure.
int64_t FileSize(int fd) {
#if defined(_WIN32)
  return _filelengthi64(fd);
#else
  struct stat fd_stat;
  if (fstat(fd, &fd_stat) != 0) {
    return -1;
  }
  return fd_stat.st_size;
#endif
}

// Writes `contents` to a fresh temporary file and returns its path. Used by the
// copy-on-write test so it never mutates checked-in testdata even if the
// map_private mapping is (incorrectly) backed by the file.
std::string WriteTempFile(const std::string& contents) {
  std::string path = ::testing::TempDir() + "/mmap_alloc_private_test.bin";
  FILE* f = fopen(path.c_str(), "wb");
  EXPECT_NE(f, nullptr);
  if (f != nullptr) {
    fwrite(contents.data(), 1, contents.size(), f);
    fclose(f);
  }
  return path;
}

}  // namespace

TEST(MMAPAllocation, TestInvalidFile) {
  if (!MMAPAllocation::IsSupported()) {
    return;
  }

  TestErrorReporter error_reporter;
  MMAPAllocation allocation("/tmp/tflite_model_1234", &error_reporter);
  EXPECT_FALSE(allocation.valid());
}

TEST(MMAPAllocation, TestValidFile) {
  if (!MMAPAllocation::IsSupported()) {
    return;
  }

  TestErrorReporter error_reporter;
  MMAPAllocation allocation(
      "tflite/testdata/empty_model.bin", &error_reporter);

  ASSERT_TRUE(allocation.valid());
  EXPECT_GT(allocation.fd(), 0);
  EXPECT_GT(allocation.bytes(), 0);
  EXPECT_NE(allocation.base(), nullptr);
}

TEST(MMAPAllocation, TestInvalidFileDescriptor) {
  if (!MMAPAllocation::IsSupported()) {
    return;
  }

  TestErrorReporter error_reporter;
  MMAPAllocation allocation(-1, &error_reporter);
  EXPECT_FALSE(allocation.valid());
}

TEST(MMAPAllocation, TestInvalidSizeAndOffset) {
  if (!MMAPAllocation::IsSupported()) {
    return;
  }

  int fd = OpenReadOnly("tflite/testdata/empty_model.bin");
  ASSERT_GT(fd, 0);

  int64_t file_size = FileSize(fd);
  ASSERT_GT(file_size, 0);

  TestErrorReporter error_reporter;
  MMAPAllocation allocation_invalid_offset(fd, /*offset=*/file_size + 100,
                                           /*length=*/1, &error_reporter);
  EXPECT_FALSE(allocation_invalid_offset.valid());

  MMAPAllocation allocation_invalid_length(fd, /*offset=*/0, /*length=*/0,
                                           &error_reporter);
  EXPECT_FALSE(allocation_invalid_length.valid());

  MMAPAllocation allocation_excessive_length(fd, /*offset=*/0,
                                             /*length=*/file_size + 1,
                                             &error_reporter);
  EXPECT_FALSE(allocation_excessive_length.valid());

  MMAPAllocation allocation_excessive_length_with_offset(
      fd, /*offset=*/10, /*length=*/file_size, &error_reporter);
  EXPECT_FALSE(allocation_excessive_length_with_offset.valid());

  // offset + length overflows size_t; the CheckedInt bounds guard must reject
  // this rather than wrapping to a small in-range value.
  MMAPAllocation allocation_integer_overflow(
      fd, /*offset=*/10, /*length=*/SIZE_MAX - 5, &error_reporter);
  EXPECT_FALSE(allocation_integer_overflow.valid());

  CloseFd(fd);
}

TEST(MMAPAllocation, TestValidFileDescriptor) {
  if (!MMAPAllocation::IsSupported()) {
    return;
  }

  int fd = OpenReadOnly("tflite/testdata/empty_model.bin");
  ASSERT_GT(fd, 0);

  TestErrorReporter error_reporter;
  MMAPAllocation allocation(fd, &error_reporter);
  EXPECT_TRUE(allocation.valid());
  EXPECT_GT(allocation.fd(), 0);
  EXPECT_GT(allocation.bytes(), 0);
  EXPECT_NE(allocation.base(), nullptr);

  CloseFd(fd);
}

TEST(MMAPAllocation, TestValidFileDescriptorWithOffset) {
  if (!MMAPAllocation::IsSupported()) {
    return;
  }

  int fd = OpenReadOnly("tflite/testdata/empty_model.bin");
  ASSERT_GT(fd, 0);

  int64_t file_size = FileSize(fd);
  ASSERT_GT(file_size, 0);

  TestErrorReporter error_reporter;
  MMAPAllocation allocation(fd, /*offset=*/10, /*length=*/file_size - 10,
                            &error_reporter);
  EXPECT_TRUE(allocation.valid());
  EXPECT_GT(allocation.fd(), 0);
  EXPECT_EQ(allocation.bytes(), static_cast<size_t>(file_size - 10));
  EXPECT_NE(allocation.base(), nullptr);

  // A non-zero offset that is smaller than the mapping granularity (page size
  // on POSIX, dwAllocationGranularity on Windows) cannot start the mapped view
  // exactly at the requested byte. The implementation rounds the view down to
  // the granularity boundary and exposes the requested byte via base(), which
  // is mmapped_buffer() advanced by offset_in_buffer(). Verify base() lands
  // `offset` bytes past the granularity-aligned start of the file.
  EXPECT_EQ(allocation.mmapped_buffer_offset_in_file(), 0u);
  EXPECT_EQ(
      static_cast<const char*>(allocation.base()) -
          static_cast<const char*>(allocation.mmapped_buffer()),
      10);
  EXPECT_EQ(allocation.mmapped_buffer_size(), static_cast<size_t>(file_size));

  CloseFd(fd);
}

// The default (map_private == false) mapping is read-only and shared. The
// map_private == true mapping is copy-on-write and writable: edits must be
// visible through the mapping but must NOT be flushed back to the file on
// disk. This exercises the PAGE_WRITECOPY / FILE_MAP_COPY Win32 path (and
// MAP_PRIVATE|PROT_WRITE on POSIX).
TEST(MMAPAllocation, TestMapPrivateIsWritableAndDoesNotModifyFile) {
  if (!MMAPAllocation::IsSupported()) {
    return;
  }

  const char original_first_byte = 0x11;
  const std::string contents(64, original_first_byte);
  const std::string path = WriteTempFile(contents);

  int fd = OpenReadOnly(path.c_str());
  ASSERT_GT(fd, 0);

  TestErrorReporter error_reporter;
  MMAPAllocation writable(fd, &error_reporter, /*map_private=*/true);
  ASSERT_TRUE(writable.valid());

  char* data = const_cast<char*>(static_cast<const char*>(writable.base()));
  const char kSentinel = static_cast<char>(original_first_byte ^ 0x5A);
  data[0] = kSentinel;
  // The write is visible through the copy-on-write mapping.
  EXPECT_EQ(data[0], kSentinel);

  // Re-map the file read-only and confirm the on-disk byte is unchanged (the
  // copy-on-write page was never flushed back to the file).
  TestErrorReporter verify_reporter;
  MMAPAllocation verify(fd, &verify_reporter);
  ASSERT_TRUE(verify.valid());
  EXPECT_EQ(static_cast<const char*>(verify.base())[0], original_first_byte);

  CloseFd(fd);
}

}  // namespace tflite
