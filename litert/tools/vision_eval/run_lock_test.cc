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

#include <gtest/gtest.h>
#include "litert/tools/vision_eval/image.h"

namespace litert::vision {
namespace {

TEST(RunLockTest, PreventsConcurrentLockAndReleasesOnDestructionOrRelease) {
  const auto dir = Utf8ToPath(testing::TempDir()) / Utf8ToPath("锁_run_lock");
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

}  // namespace
}  // namespace litert::vision
