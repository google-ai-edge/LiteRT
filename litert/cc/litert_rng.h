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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_RNG_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_RNG_H_

#include <iostream>
#include <optional>
#include <ostream>
#include <type_traits>

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

// Various utilities and types for random number generation.

namespace litert {

// Wraps variants of rng devices w/ seeds. Provides interop with logging and
// other common litert/absl features.
template <typename RandomDeviceBase>
class RandomDevice {
 public:
  using Base = RandomDeviceBase;
  using ResultType = typename Base::result_type;

  // Construct from given int seed.
  explicit RandomDevice(int seed)
      : seed_(seed), rng_(seed), repr_(MakeRepr(seed)) {}

  // Construct with implementation defined seed.
  RandomDevice() : seed_(std::nullopt), rng_(), repr_(MakeRepr(std::nullopt)) {}

  // Wrapped method to return the next random value.
  ResultType operator()() { return rng_(); }

  // Wrapped static methods to return the min and max values.
  static constexpr ResultType Min() { return Base::min(); }
  static constexpr ResultType Max() { return Base::max(); }

  // Returns the string representation of the rng.
  absl::string_view Repr() const { return repr_; }

  // Support absl::StrFormat.
  template <typename Sink>
  friend void AbslStringify(Sink& sink, const RandomDevice& rng) {
    absl::Format(&sink, "%s", rng.Repr());
  }

  // Support printing to std out streams.
  friend std::ostream& operator<<(std::ostream& os, const RandomDevice& rng) {
    return os << rng.Repr();
  }

 private:
  template <typename T, typename = int>
  struct BaseHasName : std::false_type {};

  template <typename T>
  struct BaseHasName<T, decltype((void)T::kName, 0)> : std::true_type {};

  // Use kName if the base defines it, otherwise use the typeid name.
  static constexpr absl::string_view Name() {
    constexpr auto kMaxTypeNameLen = 24;
    if constexpr (BaseHasName<Base>::value) {
      return Base::kName;
    } else {
      return absl::NullSafeStringView(typeid(Base).name())
          .substr(/*start=*/0, kMaxTypeNameLen);
    }
  }

  static std::string MakeRepr(std::optional<int> seed) {
    constexpr absl::string_view kReprFmt = "%s(seed=%s, min=%d, max=%d)";
    const auto seed_str = seed.has_value() ? absl::StrCat(*seed) : "<default>";
    auto res =
        absl::StrFormat(kReprFmt, Name(), seed_str, Base::min(), Base::max());
    return res;
  }

  const std::optional<int> seed_;
  Base rng_;
  const std::string repr_;
};

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_RNG_H_
