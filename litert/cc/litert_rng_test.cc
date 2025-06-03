// Copyright 2025 Google LLC.
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

#include "litert/cc/litert_rng.h"

#include <cstdint>
#include <random>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert {
namespace {

using ::testing::HasSubstr;

struct DummyRng {
  using result_type = uint64_t;

  explicit DummyRng(result_type seed) : seed(seed) {}
  DummyRng() = default;

  static constexpr absl::string_view kName = "DummyRng";

  result_type operator()() { return 0; }

  static constexpr result_type min() { return 0; }
  static constexpr result_type max() { return 0; }

  result_type seed;
};

TEST(LitertRngTestWithStdRng, Seed) {
  RandomDevice<std::mt19937> lite_rng(1234);
  EXPECT_THAT(absl::StrFormat("%v", lite_rng), HasSubstr("seed=1234,"));
}

TEST(LitertRngTestWithStdRng, NoSeed) {
  RandomDevice<std::mt19937> lite_rng;
  EXPECT_THAT(absl::StrFormat("%v", lite_rng), HasSubstr("seed=<default>,"));
}

TEST(LitertRngTestWithCustomRng, Seed) {
  RandomDevice<DummyRng> lite_rng(1234);
  EXPECT_THAT(absl::StrFormat("%v", lite_rng),
              HasSubstr("DummyRng(seed=1234,"));
}

TEST(LitertRngTestWithCustomRng, NoSeed) {
  RandomDevice<DummyRng> lite_rng;
  EXPECT_THAT(absl::StrFormat("%v", lite_rng),
              HasSubstr("DummyRng(seed=<default>,"));
}

}  // namespace

}  // namespace litert
