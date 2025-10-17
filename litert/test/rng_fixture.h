// Copyright 2024 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_RNG_FIXTURE_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_RNG_FIXTURE_H_

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/cc/internal/litert_rng.h"
#include "litert/test/fuzz.h"

// Basic litert rng integration with gtest.

namespace litert::testing {

// Fixture wrapper of the code in cc/litert_rng.h. This will use the seed
// set from gtest, which can be configured from command line for
// reproducibility. It can also be used to set up repeated blocks of code for
// fuzzing.
class RngTest : public ::testing::Test {
 public:
  void TearDown() override {
    static constexpr auto kMsg =
        "The minimum number of iterations was not reached in the alloted "
        "time.";

    if (TimedOut()) {
      if (fail_on_timeout_) {
        ADD_FAILURE() << kMsg;
      } else {
        LITERT_LOG(LITERT_WARNING, "%s", kMsg);
      }
    }
  }

 protected:
  bool TimedOut() const {
    for (const auto& block : fuzz_blocks_) {
      if (!block.ReachedIters()) {
        return true;
      }
    }
    return false;
  }

  template <typename Device = DefaultDevice>
  auto TracedDevice(std::optional<int> seed = std::nullopt) {
    const auto seed_to_use = seed ? *seed : CurrentSeed();
    return TraceSeedInfo(Device(seed_to_use));
  }

  template <typename... Args>
  auto& FuzzBlock(Args&&... args) {
    return fuzz_blocks_.emplace_back(std::forward<Args>(args)...);
  }

  explicit RngTest(bool fail_on_timeout = true)
      : fail_on_timeout_(fail_on_timeout) {}

 private:
  using ScopedTrace = ::testing::ScopedTrace;
  using UnitTest = ::testing::UnitTest;

  UnitTest* GetUnitTest() { return UnitTest::GetInstance(); }

  int CurrentSeed() { return GetUnitTest()->random_seed(); }

  template <typename Device>
  auto TraceSeedInfo(Device&& device) {
    const char* file = "<unknown>";
    int line = -1;
    const auto* unit_test = GetUnitTest();
    if (unit_test) {
      if (const auto* test_info = unit_test->current_test_info()) {
        file = test_info->file();
        line = test_info->line();
      }
    }
    traces_.push_back(std::make_unique<ScopedTrace>(
        file, line,
        absl::StrFormat("litert_rng %lu{%v}", traces_.size(), device)));
    return device;
  }

  bool fail_on_timeout_ = true;
  std::vector<std::unique_ptr<ScopedTrace>> traces_;
  std::vector<RepeatedBlock> fuzz_blocks_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_RNG_FIXTURE_H_
