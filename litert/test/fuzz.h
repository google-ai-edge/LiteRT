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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_FUZZ_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_FUZZ_H_

// Utilities for setting up "fuzzed" test code.

#include <chrono>  // NOLINT
#include <cstddef>
#include <limits>
#include <variant>

namespace litert::testing {

// Class that will repeatadly run a block of code until max iterations, or time
// limit is reached.
class RepeatedBlock final {
 private:
  using Clock = std::chrono::steady_clock;
  static constexpr auto kDefaultMaxMs = std::chrono::milliseconds(50);

 public:
  class Iterator {
   public:
    explicit Iterator(RepeatedBlock& parent) : parent_(parent) {}

    void operator++() { parent_.cur_iter_++; }

    bool operator!=(const Iterator& other) const { return !parent_.Done(); }

    auto operator*() const { return std::monostate{}; }

   private:
    RepeatedBlock& parent_;
  };

  template <typename Duration>
  explicit RepeatedBlock(size_t iters, Duration max_duration = kDefaultMaxMs)
      : expire_time_(AddDuration(Clock::now(), max_duration)), iters_(iters) {}

  auto begin() { return Iterator(*this); }
  auto end() { return Iterator(*this); }

  bool Done() const {
    return cur_iter_ >= iters_ || Clock::now() >= expire_time_;
  }

  bool ReachedIters() const { return cur_iter_ >= iters_; }

 private:
  // Check overflow.
  static Clock::time_point AddDuration(typename Clock::time_point tp,
                                       std::chrono::milliseconds m) {
    using Dur = Clock::time_point::duration::rep;
    if (tp.time_since_epoch().count() >
        std::numeric_limits<Dur>::max() - m.count()) {
      return tp;
    }
    return tp + m;
  }

  Clock::time_point expire_time_;
  size_t iters_;
  size_t cur_iter_ = 0;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_FUZZ_H_
