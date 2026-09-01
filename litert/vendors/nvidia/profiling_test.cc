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
  EXPECT_EQ(timer.ElapsedMs(), 0.0);
}

}  // namespace
}  // namespace litert::nvidia
