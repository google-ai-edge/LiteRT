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

#include "ml_drift_delegate/delegate/webgpu_dawn_loader.h"

#include <gmock/gmock.h>  // copybara:comment_replace(gmock) #include "testing/base/public/gmock.h"
#include <gtest/gtest.h>  // copybara:comment_replace(gunit) #include "testing/base/public/gunit.h"

#if defined(__linux__) && !defined(__ANDROID__) && \
    defined(LITERT_WEBGPU_DAWN_DYNAMIC_LOADER)
#include <dlfcn.h>
#include <limits.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "dawn/webgpu.h"  // from @dawn

namespace litert::ml_drift {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;

constexpr char kFakeDawnSoPath[] =
    "ml_drift_delegate/delegate/libfake_webgpu_dawn.so";
constexpr char kIncompleteDawnSoPath[] =
    "ml_drift_delegate/delegate/libincomplete_webgpu_dawn.so";

TEST(WebGpuDawnLoaderDeathTest,
     BootstrapCreateInstanceResolvesPreloadedGlobalScopeDawnSymbols) {
  char resolved_fake_dawn[PATH_MAX];
  ASSERT_NE(realpath(kFakeDawnSoPath, resolved_fake_dawn), nullptr)
      << kFakeDawnSoPath;
  EXPECT_EXIT(
      {
        void* preloaded = dlopen(resolved_fake_dawn, RTLD_NOW | RTLD_GLOBAL);
        if (preloaded == nullptr) {
          _exit(1);
        }
        // Calling wgpuCreateInstance directly without EnsureWebGpuDawnLoaded()
        // exercises BootstrapCreateInstance and empty-dir fallback resolution.
        (void)wgpuCreateInstance(nullptr);
        using GetCallsFn = int (*)();
        auto get_calls = reinterpret_cast<GetCallsFn>(
            dlsym(preloaded, "FakeDawnGetCreateInstanceCalls"));
        _exit((get_calls != nullptr && get_calls() == 1) ? 0 : 2);
      },
      testing::ExitedWithCode(0), "");
}

TEST(WebGpuDawnLoaderTest, DynamicLoaderHandlesMissingIncompleteAndValidLibs) {
  char resolved_fake_dawn[PATH_MAX];
  ASSERT_NE(realpath(kFakeDawnSoPath, resolved_fake_dawn), nullptr)
      << kFakeDawnSoPath;
  char resolved_incomplete_dawn[PATH_MAX];
  ASSERT_NE(realpath(kIncompleteDawnSoPath, resolved_incomplete_dawn), nullptr)
      << kIncompleteDawnSoPath;

  const std::string base_temp = testing::TempDir();

  // 0. Before Dawn is available, calling wgpuCreateInstance directly returns
  // nullptr cleanly via the bootstrap proc table without caching failure, and
  // empty runtime_library_dir returns NotFoundError.
  EXPECT_EQ(wgpuCreateInstance(nullptr), nullptr);
  EXPECT_THAT(EnsureWebGpuDawnLoaded(), StatusIs(absl::StatusCode::kNotFound));

  // 1. Missing library in runtime_library_dir returns NotFoundError without
  // caching failure.
  const std::string empty_dir = absl::StrCat(base_temp, "/empty_dawn_dir");
  ASSERT_EQ(mkdir(empty_dir.c_str(), 0755), 0);
  EXPECT_THAT(EnsureWebGpuDawnLoaded(empty_dir),
              StatusIs(absl::StatusCode::kNotFound));

  // 2. Library missing required WebGPU symbols returns FailedPreconditionError
  // and unloads it.
  const std::string incomplete_dir =
      absl::StrCat(base_temp, "/incomplete_dawn_dir");
  ASSERT_EQ(mkdir(incomplete_dir.c_str(), 0755), 0);
  const std::string incomplete_so =
      absl::StrCat(incomplete_dir, "/libwebgpu_dawn.so");
  ASSERT_EQ(symlink(resolved_incomplete_dawn, incomplete_so.c_str()), 0);
  EXPECT_THAT(EnsureWebGpuDawnLoaded(incomplete_dir),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_EQ(dlopen(incomplete_so.c_str(), RTLD_LAZY | RTLD_NOLOAD), nullptr);

  // 3. Valid Dawn library in runtime_library_dir (with trailing slash) loads
  // cleanly under concurrent threads and populates libdawn_proc's
  // DawnProcTable.
  const std::string valid_dir = absl::StrCat(base_temp, "/valid_dawn_dir");
  ASSERT_EQ(mkdir(valid_dir.c_str(), 0755), 0);
  const std::string valid_so = absl::StrCat(valid_dir, "/libwebgpu_dawn.so");
  ASSERT_EQ(symlink(resolved_fake_dawn, valid_so.c_str()), 0);

  constexpr int kNumThreads = 4;
  std::vector<absl::Status> thread_statuses(kNumThreads);
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  const std::string valid_dir_with_slash = absl::StrCat(valid_dir, "/");
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([&, i]() {
      thread_statuses[i] = EnsureWebGpuDawnLoaded(valid_dir_with_slash);
    });
  }
  for (std::thread& t : threads) {
    t.join();
  }
  for (const absl::Status& status : thread_statuses) {
    ASSERT_THAT(status, IsOk());
  }
  // Subsequent calls hit the lock-free fast path.
  EXPECT_THAT(EnsureWebGpuDawnLoaded(), IsOk());

  EXPECT_EQ(wgpuCreateInstance(nullptr), nullptr);
  EXPECT_EQ(wgpuInstanceRequestAdapter(nullptr, nullptr, {}).id, 0u);
  EXPECT_EQ(wgpuAdapterRequestDevice(nullptr, nullptr, {}).id, 0u);
  EXPECT_EQ(wgpuDeviceCreateShaderModule(nullptr, nullptr), nullptr);
  EXPECT_EQ(wgpuDeviceCreateComputePipeline(nullptr, nullptr), nullptr);
  EXPECT_EQ(wgpuDeviceCreateBuffer(nullptr, nullptr), nullptr);
  EXPECT_EQ(wgpuDeviceGetQueue(nullptr), nullptr);

  void* loaded_dawn = dlopen(valid_so.c_str(), RTLD_LAZY | RTLD_NOLOAD);
  ASSERT_NE(loaded_dawn, nullptr);
  using GetCallsFn = int (*)();
  auto get_calls = reinterpret_cast<GetCallsFn>(
      dlsym(loaded_dawn, "FakeDawnGetCreateInstanceCalls"));
  ASSERT_NE(get_calls, nullptr) << dlerror();
  EXPECT_EQ(get_calls(), 1);
  dlclose(loaded_dawn);
}

}  // namespace
}  // namespace litert::ml_drift
#endif
