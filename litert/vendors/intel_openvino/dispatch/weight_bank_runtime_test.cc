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

#include "litert/vendors/intel_openvino/dispatch/weight_bank_runtime.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#if !defined(LITERT_WINDOWS_OS)
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace litert::openvino {
namespace {

std::vector<uint8_t> MakePool(size_t size) {
  std::vector<uint8_t> pool(size);
  for (size_t i = 0; i < size; ++i) {
    pool[i] = static_cast<uint8_t>(i * 31 + 7);
  }
  return pool;
}

PoolSource FromMemory(const std::vector<uint8_t>& pool) {
  PoolSource src;
  src.data = pool.data();
  src.size = pool.size();
  return src;
}

#if !defined(LITERT_WINDOWS_OS)

// Reads the whole file behind `fd` with pread, so the fd's offset (shared with
// any dup()s handed to OpenVINO) is left untouched.
std::vector<uint8_t> ReadAll(int fd) {
  struct stat st;
  if (::fstat(fd, &st) != 0) return {};
  std::vector<uint8_t> out(static_cast<size_t>(st.st_size));
  size_t done = 0;
  while (done < out.size()) {
    const ssize_t n = ::pread(fd, out.data() + done, out.size() - done, done);
    if (n <= 0) return {};
    done += static_cast<size_t>(n);
  }
  return out;
}

TEST(NpuSharedBankTest, EnsureInMemoryStagesPoolFromMemory) {
  NpuSharedBank bank;
  // Larger than one 16 MiB staging chunk, and not a multiple of it.
  const std::vector<uint8_t> pool = MakePool((size_t{17} << 20) + 123);
  const int fd = bank.EnsureInMemory(FromMemory(pool));
  if (fd < 0) GTEST_SKIP() << "memfd_create is not available";
  EXPECT_EQ(ReadAll(fd), pool);
}

TEST(NpuSharedBankTest, EnsureInMemoryStagesPoolFromModelFile) {
  // A "model file" with the pool at a non-page-aligned offset, as in a
  // .litertlm whose weights section starts mid-page.
  const size_t kPoolOffset = 4096 * 3 + 100;
  const std::vector<uint8_t> pool = MakePool((size_t{1} << 20) + 7);
  std::vector<uint8_t> file_bytes(kPoolOffset, 0xAB);
  file_bytes.insert(file_bytes.end(), pool.begin(), pool.end());
  file_bytes.insert(file_bytes.end(), 512, 0xCD);

  std::FILE* model = std::tmpfile();
  ASSERT_NE(model, nullptr);
  ASSERT_EQ(std::fwrite(file_bytes.data(), 1, file_bytes.size(), model),
            file_bytes.size());
  ASSERT_EQ(std::fflush(model), 0);

  PoolSource src = FromMemory(pool);
  src.fd = ::fileno(model);
  src.file_offset = kPoolOffset;

  NpuSharedBank bank;
  const int fd = bank.EnsureInMemory(src);
  if (fd < 0) {
    std::fclose(model);
    GTEST_SKIP() << "memfd_create is not available";
  }
  EXPECT_EQ(ReadAll(fd), pool);
  std::fclose(model);
}

TEST(NpuSharedBankTest, EnsureInMemoryIsWriteOnce) {
  NpuSharedBank bank;
  const std::vector<uint8_t> pool = MakePool(4096);
  const int fd = bank.EnsureInMemory(FromMemory(pool));
  if (fd < 0) GTEST_SKIP() << "memfd_create is not available";

  // Later calls return the cached fd and ignore their arguments.
  const std::vector<uint8_t> other = MakePool(16);
  EXPECT_EQ(bank.EnsureInMemory(FromMemory(other)), fd);
  EXPECT_EQ(bank.EnsureInMemory(PoolSource{}), fd);
  EXPECT_EQ(ReadAll(fd), pool);
}

TEST(NpuSharedBankTest, EnsureInMemoryDupedFdOutlivesCaller) {
  NpuSharedBank bank;
  const std::vector<uint8_t> pool = MakePool(8192);
  const int fd = bank.EnsureInMemory(FromMemory(pool));
  if (fd < 0) GTEST_SKIP() << "memfd_create is not available";

  // This mirrors how the handle provider hands dup()s to OpenVINO, which owns
  // and closes them; closing a dup must not affect the bank's own fd.
  const int dup_fd = ::dup(fd);
  ASSERT_GE(dup_fd, 0);
  EXPECT_EQ(ReadAll(dup_fd), pool);
  ASSERT_EQ(::close(dup_fd), 0);
  EXPECT_EQ(ReadAll(fd), pool);
}

TEST(NpuSharedBankTest, EnsureInMemoryRejectsEmptyPool) {
  NpuSharedBank bank;
  const uint8_t byte = 0;
  PoolSource null_data;
  null_data.size = 16;
  PoolSource zero_size;
  zero_size.data = &byte;
  EXPECT_EQ(bank.EnsureInMemory(null_data), -1);
  EXPECT_EQ(bank.EnsureInMemory(zero_size), -1);
}

#else  // LITERT_WINDOWS_OS

TEST(NpuSharedBankTest, EnsureInMemoryIsUnsupportedOnWindows) {
  NpuSharedBank bank;
  const std::vector<uint8_t> pool = MakePool(16);
  EXPECT_EQ(bank.EnsureInMemory(FromMemory(pool)), -1);
}

#endif  // LITERT_WINDOWS_OS

}  // namespace
}  // namespace litert::openvino
