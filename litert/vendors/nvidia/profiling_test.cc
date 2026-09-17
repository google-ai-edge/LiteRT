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

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>

#include <gtest/gtest.h>
#include "litert/vendors/nvidia/dispatch/dispatch_profiler.h"
#include "litert/vendors/nvidia/memory_profile.h"

namespace litert::nvidia {
namespace {

class ScopedEnvironmentVariable {
 public:
  explicit ScopedEnvironmentVariable(const char* name) : name_(name) {
    if (const char* value = std::getenv(name); value != nullptr) {
      original_value_ = value;
    }
  }

  ~ScopedEnvironmentVariable() {
    if (original_value_.has_value()) {
      setenv(name_.c_str(), original_value_->c_str(), /*overwrite=*/1);
    } else {
      unsetenv(name_.c_str());
    }
  }

  void Set(const char* value) { setenv(name_.c_str(), value, /*overwrite=*/1); }

  void Unset() { unsetenv(name_.c_str()); }

 private:
  std::string name_;
  std::optional<std::string> original_value_;
};

TEST(ProfilingTest, EnvironmentFlagsUseTheExistingNonzeroSemantics) {
  ScopedEnvironmentVariable memory("LITERT_NVIDIA_MEMORY_PROFILE");
  ScopedEnvironmentVariable dispatch("LITERT_NVIDIA_DISPATCH_PROFILE");
  ScopedEnvironmentVariable layer("LITERT_NVIDIA_DISPATCH_LAYER_PROFILE");

  memory.Unset();
  dispatch.Unset();
  layer.Unset();
  EXPECT_FALSE(MemoryProfilingEnabled());
  EXPECT_FALSE(DispatchProfilingEnabled());
  EXPECT_FALSE(DispatchLayerProfilingEnabled());

  memory.Set("1");
  dispatch.Set("enabled");
  layer.Set("1");
  EXPECT_TRUE(MemoryProfilingEnabled());
  EXPECT_TRUE(DispatchProfilingEnabled());
  EXPECT_TRUE(DispatchLayerProfilingEnabled());

  memory.Set("0");
  dispatch.Set("0");
  layer.Set("0");
  EXPECT_FALSE(MemoryProfilingEnabled());
  EXPECT_FALSE(DispatchProfilingEnabled());
  EXPECT_FALSE(DispatchLayerProfilingEnabled());
}

TEST(ProfilingTest, DisabledCpuTimerDoesNotReportTime) {
  DispatchCpuTimer timer(/*enabled=*/false);
  for (int i = 0; i < 16; ++i) EXPECT_EQ(timer.ElapsedMs(), 0.0);
}

TEST(ProfilingTest, CpuTimerElapsedMillisecondsAreNondecreasing) {
  const auto before = std::chrono::steady_clock::now();
  DispatchCpuTimer timer(/*enabled=*/true);
  const auto after_start = std::chrono::steady_clock::now();
  double previous_ms = 0.0;
  for (int i = 0; i < 16; ++i) {
    const auto before_sample = std::chrono::steady_clock::now();
    const double elapsed_ms = timer.ElapsedMs();
    const auto after_sample = std::chrono::steady_clock::now();
    const double lower_bound_ms =
        std::chrono::duration<double, std::milli>(before_sample - after_start)
            .count();
    const double upper_bound_ms =
        std::chrono::duration<double, std::milli>(after_sample - before)
            .count();
    EXPECT_GE(elapsed_ms, previous_ms);
    EXPECT_GE(elapsed_ms, lower_bound_ms);
    EXPECT_LE(elapsed_ms, upper_bound_ms);
    previous_ms = elapsed_ms;
  }
}

TEST(ProfilingTest, DisabledMemoryProfileDoesNotLog) {
  ScopedEnvironmentVariable memory("LITERT_NVIDIA_MEMORY_PROFILE");
  memory.Set("0");
  testing::internal::CaptureStderr();
  LogMemoryProfile("test", "disabled");
  const std::string output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(output.empty()) << output;
}

TEST(ProfilingTest, MemoryLogTimestampsUseTheMonotonicClock) {
  ScopedEnvironmentVariable memory("LITERT_NVIDIA_MEMORY_PROFILE");
  memory.Set("1");
  const auto now_ns = [] {
    return static_cast<unsigned long long>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
  };
  testing::internal::CaptureStderr();
  const auto before = now_ns();
  LogMemoryProfile("test", "first");
  const auto between = now_ns();
  LogMemoryProfile("test", "second");
  const auto after = now_ns();
  const std::string output = testing::internal::GetCapturedStderr();

  const auto first_pos = output.find("monotonic_ns=");
  ASSERT_NE(first_pos, std::string::npos) << output;
  const auto second_pos = output.find("monotonic_ns=", first_pos + 1);
  ASSERT_NE(second_pos, std::string::npos) << output;
  unsigned long long first_ns = 0;
  unsigned long long second_ns = 0;
  ASSERT_EQ(
      std::sscanf(output.c_str() + first_pos, "monotonic_ns=%llu", &first_ns),
      1);
  ASSERT_EQ(
      std::sscanf(output.c_str() + second_pos, "monotonic_ns=%llu", &second_ns),
      1);
  EXPECT_GE(first_ns, before);
  EXPECT_LE(first_ns, between);
  EXPECT_GE(second_ns, between);
  EXPECT_LE(second_ns, after);
}

}  // namespace
}  // namespace litert::nvidia
