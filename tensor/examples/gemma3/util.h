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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA3_UTIL_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA3_UTIL_H_

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl

namespace litert::tensor::examples {

class Timer {
 public:
  struct Lap {
    absl::Time start;
    absl::Time stop;

    Lap(absl::Time start_time, absl::Time stop_time)
        : start(start_time), stop(stop_time) {}
    // NOLINTNEXTLINE: this is intended to implicitly convert.
    operator absl::Duration() const { return stop - start; }
  };

  class [[nodiscard]] LapScope {
   public:
    explicit LapScope(Timer& timer) : timer_(&timer) { timer_->StartLap(); }
    LapScope(const LapScope&) = delete;
    LapScope& operator=(const LapScope&) = delete;
    LapScope(LapScope&& scope)
        : timer_(std::exchange(scope.timer_, &Get("unused_timer"))) {}
    LapScope& operator=(LapScope&& scope) {
      timer_ = std::exchange(scope.timer_, &Get("unused_timer"));
      return *this;
    }
    ~LapScope() { timer_->StopLap(); }

   private:
    Timer* timer_;
  };

  explicit Timer(std::string name) : name_(std::move(name)) {}
  void StartLap() { start_ = absl::Now(); }
  Lap StopLap() {
    laps_.emplace_back(start_, absl::Now());
    return laps_.back();
  }
  void Reset() { laps_.clear(); }
  LapScope Lap() { return LapScope(*this); }

  absl::Duration Duration() const {
    return absl::c_accumulate(
        laps_, absl::ZeroDuration(),
        [](auto accu, auto lap) -> absl::Duration { return accu + lap; });
  }

  const std::string& Name() const { return name_; }
  absl::Duration AverageDuration() const { return Duration() / laps_.size(); }
  const std::vector<struct Lap>& Laps() const { return laps_; }

  static Timer& Get(absl::string_view name) {
    static auto* timers = new std::map<std::string, std::unique_ptr<Timer>>();
    static auto* mutex = new absl::Mutex();

    const std::string timer_name(name);
    absl::MutexLock lock(*mutex);
    auto [it, inserted] = timers->try_emplace(timer_name);
    if (inserted) {
      it->second = std::make_unique<Timer>(timer_name);
    }
    return *it->second;
  }

  static LapScope Lap(absl::string_view name) { return Get(name).Lap(); }

  std::string Stats() const {
    return absl::StrCat(name_, "=",
                        absl::ToDoubleMilliseconds(Duration() / laps_.size()));
  }

  std::string PerSec() const {
    return absl::StrCat(
        name_, "_per_sec=", laps_.size() / absl::ToDoubleSeconds(Duration()));
  }

 private:
  std::string name_;
  absl::Time start_;
  std::vector<struct Lap> laps_;
};

struct PrefillTiming {
  Timer prefill{"prefill"};
  Timer cpu_prep{"cpu_prep"};
  Timer uploads{"uploads"};
  Timer run{"run"};
  Timer readback{"readback"};
  Timer cache_readback{"cache_readback"};

  std::string Stats() const {
    std::stringstream os;
    os << "Prefill breakdown " << prefill.PerSec() << " <=> " << prefill.Stats()
       << " ms/tok { " << cpu_prep.Stats() << ", " << uploads.Stats() << ", "
       << run.Stats() << ", " << readback.Stats() << ", "
       << cache_readback.Stats() << " }";
    return os.str();
  }
};

struct DecodeTiming {
  Timer decode{"decode"};
  Timer cpu_prep{"cpu_prep"};
  Timer uploads{"uploads"};
  Timer run{"run"};
  Timer readback{"readback"};
  Timer argmax{"argmax"};
  Timer cache_readback{"cache_readback"};
  Timer cache_upload{"cache_upload"};

  std::string Stats() const {
    std::stringstream os;
    os << "Decode breakdown " << decode.PerSec() << " <=> " << decode.Stats()
       << " ms/tok { " << cpu_prep.Stats() << ", " << uploads.Stats() << ", "
       << run.Stats() << ", " << readback.Stats() << ", " << argmax.Stats()
       << ", " << cache_readback.Stats() << ", " << cache_upload.Stats()
       << " }";
    return os.str();
  }
};

class TokenPrinter {
 public:
  enum class Kind { kTokens, kProgress };
  explicit TokenPrinter(Kind kind, size_t max_seq_length)
      : kind_(kind), max_seq_length_(max_seq_length + 2 /*for prefill token*/) {
    buffer_.reserve(256);
  }

  void Push(absl::string_view token) {
    switch (kind_) {
      case Kind::kTokens:
        buffer_ += token;
        break;
      case Kind::kProgress:
        ++tokens_;
    }
    if (const absl::Time now = absl::Now();
        now - last_print_ > absl::Milliseconds(100)) {
      Print();
      last_print_ = now;
    }
  }

  void Flush() {
    Print();
    std::cout << std::endl;
    buffer_.clear();
  }

 private:
  void Print() {
    switch (kind_) {
      case Kind::kTokens:
        std::cout << buffer_ << std::flush;
        buffer_.clear();
        break;
      case Kind::kProgress:
        const int width = 50;
        const int fill = tokens_ * width / max_seq_length_;
        std::cout << "\r[" << std::setw(fill) << std::setfill('#') << ""
                  << std::setfill(' ') << std::setw(width - fill + 1) << "]"
                  << " " << std::fixed << std::setprecision(2)
                  << (tokens_ * 100 / max_seq_length_) << "% " << tokens_
                  << " tokens\033[0K" << std::flush;
        break;
    }
  }

  Kind kind_;
  absl::Time last_print_ = absl::Now();
  std::string buffer_;
  size_t tokens_ = 0;
  size_t max_seq_length_;
};

inline bool AbslParseFlag(absl::string_view text, TokenPrinter::Kind* mode,
                          std::string* error) {
  if (text == "tokens") {
    *mode = TokenPrinter::Kind::kTokens;
    return true;
  }
  if (text == "progress") {
    *mode = TokenPrinter::Kind::kProgress;
    return true;
  }
  *error = "unknown value for enumeration";
  return false;
}

// AbslUnparseFlag converts from an OutputMode to a string.
// Must be in same namespace as OutputMode.

// Returns a textual flag value corresponding to the OutputMode `mode`.
inline std::string AbslUnparseFlag(TokenPrinter::Kind mode) {
  switch (mode) {
    case TokenPrinter::Kind::kTokens:
      return "tokens";
    case TokenPrinter::Kind::kProgress:
      return "progress";
  }
}

}  // namespace litert::tensor::examples

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA3_UTIL_H_
