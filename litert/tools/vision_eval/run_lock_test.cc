// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "litert/tools/vision_eval/run_lock.h"

#include <filesystem>  // NOLINT(build/c++17)
#include <system_error>  // NOLINT(build/c++11)
#if !defined(_WIN32)
#include <unistd.h>
#endif

#include <gtest/gtest.h>
#include "litert/tools/vision_eval/data.h"
#include "litert/tools/vision_eval/image.h"

namespace litert::vision {
namespace {

TEST(RunLockTest, PreventsConcurrentLockAndReleasesOnDestructionOrRelease) {
  const auto dir = Utf8ToPath(testing::TempDir()) / "run_lock";
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  ASSERT_FALSE(ec);
  const auto lock_path = dir / "run.lock";

  {
    RunLock first;
    ASSERT_TRUE(first.Acquire(lock_path));

    RunLock second;
    EXPECT_FALSE(second.Acquire(lock_path));

    first.Release();
    EXPECT_TRUE(second.Acquire(lock_path));
  }

  RunLock after_scope;
  EXPECT_TRUE(after_scope.Acquire(lock_path));
  // Re-acquiring on the same instance releases the old descriptor cleanly.
  EXPECT_TRUE(after_scope.Acquire(lock_path));
}

TEST(RunLockTest, FailsWhenParentDirectoryDoesNotExist) {
  const auto missing_path =
      Utf8ToPath(testing::TempDir()) / "missing_parent_dir" / "run.lock";
  RunLock lock;
  EXPECT_FALSE(lock.Acquire(missing_path));
}

TEST(RunLockTest, FailsWhenLockPathCannotBeOpenedForWrite) {
  const auto base_dir = Utf8ToPath(testing::TempDir()) / "run_lock_denied";
  std::error_code ec;
  std::filesystem::create_directories(base_dir, ec);
  ASSERT_FALSE(ec);

  // Opening an existing directory as a writable lock file fails on all OSes.
  RunLock dir_lock;
  const auto dir_res = dir_lock.Acquire(base_dir);
  ASSERT_FALSE(dir_res);
  EXPECT_FALSE(dir_res.Error().Message().empty());

  const auto ro_file = base_dir / "readonly.lock";
  const char byte = 'x';
  ASSERT_TRUE(WriteBytes(ro_file, &byte, sizeof(byte)));
  std::filesystem::permissions(ro_file, std::filesystem::perms::owner_read, ec);
  ASSERT_FALSE(ec);

  const auto ro_dir = base_dir / "readonly_dir";
  std::filesystem::create_directories(ro_dir, ec);
  ASSERT_FALSE(ec);
  std::filesystem::permissions(
      ro_dir,
      std::filesystem::perms::owner_read | std::filesystem::perms::owner_exec,
      ec);
  ASSERT_FALSE(ec);

#if !defined(_WIN32)
  const bool enforces_dac = (geteuid() != 0);
#else
  const bool enforces_dac = true;
#endif
  if (enforces_dac) {
    RunLock file_lock;
    EXPECT_FALSE(file_lock.Acquire(ro_file));
    RunLock subdir_lock;
    EXPECT_FALSE(subdir_lock.Acquire(ro_dir / "run.lock"));
  }

  std::filesystem::permissions(ro_file, std::filesystem::perms::owner_all, ec);
  std::filesystem::permissions(ro_dir, std::filesystem::perms::owner_all, ec);
}

}  // namespace
}  // namespace litert::vision
