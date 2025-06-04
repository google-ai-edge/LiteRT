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
#include <random>
#include <type_traits>

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_detail.h"
#include "litert/cc/litert_numerics.h"

// Various utilities and types for random number generation.

namespace litert {

// STATEFUL RNG INTERFACE //////////////////////////////////////////////////////

// Wraps variants of rng devices w/ seeds. Provides interop with logging and
// other common litert/absl features.
template <typename RngEngine>
class RandomDevice {
 public:
  using Engine = RngEngine;
  using ResultType = typename Engine::result_type;

  RandomDevice(const RandomDevice&) = delete;
  RandomDevice& operator=(const RandomDevice&) = delete;
  RandomDevice(RandomDevice&&) = default;
  RandomDevice& operator=(RandomDevice&&) = default;

  // Construct from given int seed.
  explicit RandomDevice(int seed)
      : seed_(seed), rng_(seed), repr_(MakeRepr(seed)) {}

  // Construct with implementation defined seed.
  RandomDevice() : seed_(std::nullopt), rng_(), repr_(MakeRepr(std::nullopt)) {}

  // Wrapped method to return the next random value.
  ResultType operator()() { return rng_(); }

  // Wrapped static methods to return the min and max values.
  static constexpr ResultType Min() { return Engine::min(); }

  static constexpr ResultType Max() { return Engine::max(); }

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
    if constexpr (BaseHasName<Engine>::value) {
      return Engine::kName;
    } else {
      return absl::NullSafeStringView(typeid(Engine).name())
          .substr(/*start=*/0, kMaxTypeNameLen);
    }
  }

  static std::string MakeRepr(std::optional<int> seed) {
    constexpr absl::string_view kReprFmt = "%s(seed=%s, min=%d, max=%d)";
    const auto seed_str = seed.has_value() ? absl::StrCat(*seed) : "<default>";
    auto res = absl::StrFormat(kReprFmt, Name(), seed_str, Engine::min(),
                               Engine::max());
    return res;
  }

  const std::optional<int> seed_;
  Engine rng_;
  const std::string repr_;

 public:
  // Interop with the std interface which is not google style compliant.
  using result_type = ResultType;
  static constexpr ResultType min() { return Min(); }
  static constexpr ResultType max() { return Max(); }
};

// TENSOR DATA GENERATOR ///////////////////////////////////////////////////////

// Abstract base class for generating data of a certain type from a given rng
// device, e.g. populating tensors and the like.
template <typename D, template <typename> typename Dist, typename DeviceBase>
class DataGenerator {
 public:
  using DataType = D;
  using Wide = WideType<D>;
  using Device = RandomDevice<DeviceBase>;

  virtual DataType operator()(Device& rng) = 0;

  // Bounds of distribution.
  DataType Max() const { return dist_.max(); }
  DataType Min() const { return dist_.min(); }

 protected:
  Dist<DataType> dist_;
};

// A data generator that generates data within a given range.
template <typename D, template <typename> typename Dist, typename DeviceBase>
class RangedGenerator final : public DataGenerator<D, Dist, DeviceBase> {
 private:
  using Base = DataGenerator<D, Dist, DeviceBase>;

 public:
  using typename Base::DataType;
  using typename Base::Device;
  using typename Base::Wide;

  RangedGenerator() = default;
  RangedGenerator(Wide min, Wide max = NumericLimits<D>::Max()) {
    ConstructAt(&this->dist_, LowerBoundary<DataType>(min),
                UpperBoundary<DataType>(max));
  }

  RangedGenerator(const RangedGenerator&) = default;
  RangedGenerator& operator=(const RangedGenerator&) = default;
  RangedGenerator(RangedGenerator&&) = default;
  RangedGenerator& operator=(RangedGenerator&&) = default;

  DataType operator()(Device& rng) override { return this->dist_(rng); }
};

// A rangeless float generator that casts random bits to the given float type.
// This generally produces higher quality floats more repersentative of the
// target distribution than a ranged generator. Particularly covers more values
// around zero and infinities.
template <typename D, template <typename> typename Dist, typename DeviceBase,
          typename Enable = void>
class ReinterpretGenerator final : public DataGenerator<D, Dist, DeviceBase> {};

template <typename D, template <typename> typename Dist, typename DeviceBase>
class ReinterpretGenerator<D, Dist, DeviceBase,
                           std::enable_if_t<std::is_floating_point_v<D>>>
    final : public DataGenerator<D, Dist, DeviceBase> {
 private:
  using Base = DataGenerator<D, Dist, DeviceBase>;

 public:
  using typename Base::DataType;
  using typename Base::Device;
  using typename Base::Wide;

  DataType operator()(Device& rng) override {
    DataType res;
    auto bits = rng();
    memcpy(&res, &bits, sizeof(res));
    return GetOrFlush(res);
  }

  ReinterpretGenerator() {
    ConstructAt(&this->dist_, NumericLimits<DataType>::Lowest(),
                NumericLimits<DataType>::Max());
  }
  ReinterpretGenerator(const ReinterpretGenerator&) = default;
  ReinterpretGenerator& operator=(const ReinterpretGenerator&) = default;
  ReinterpretGenerator(ReinterpretGenerator&&) = default;
  ReinterpretGenerator& operator=(ReinterpretGenerator&&) = default;
};

// Factory for creating data generators from just a data type with recommended
// defaults.
template <typename D>
class DataGenerators {
  // Exotic types not yet supported (e.g. quant, complex, half-precision etc).
  static_assert(std::is_floating_point_v<D> || std::is_integral_v<D>);

 public:
  using Engine = std::mt19937_64;
  template <typename T>
  using Uniform =
      SelectT<std::is_floating_point<T>, std::uniform_real_distribution<T>,
              std::is_integral<T>, std::uniform_int_distribution<T>>;

 private:
  using GeneratorBase = DataGenerator<D, Uniform, Engine>;

 public:
  using Reinterpret = ReinterpretGenerator<D, Uniform, Engine>;
  using Ranged = RangedGenerator<D, Uniform, Engine>;
  using Dataype = GeneratorBase::DataType;
  using Wide = GeneratorBase::Wide;
  using RandomDevice = GeneratorBase::Device;

  DataGenerators() = default;
  DataGenerators(const DataGenerators&) = default;
  DataGenerators& operator=(const DataGenerators&) = default;
  DataGenerators(DataGenerators&&) = default;
  DataGenerators& operator=(DataGenerators&&) = default;

  // Create a ranged generator with the given limits.
  static auto Generator(Wide min, Wide max) { return Ranged(min, max); }

  // Create a rangeless generator. Floating point types will leverage the
  // reinterpretation generator, which is recommended.
  static auto Generator() {
    if constexpr (std::is_floating_point_v<D>) {
      return Reinterpret();
    } else {
      return Ranged();
    }
  }

  // Initialize a random device with the proper types to work with generators.
  template <typename... Args>
  static auto Device(Args&&... args) {
    return RandomDevice(std::forward<Args>(args)...);
  }

  // Convenience method(s) to create a generator and device in a pair.
  static auto GeneratorAndDevice() {
    return std::make_pair(Generator(), Device());
  }
  static auto GeneratorAndDevice(int seed) {
    return std::make_pair(Generator(), Device(seed));
  }
  static auto GeneratorAndDevice(Wide min, Wide max) {
    return std::make_pair(Generator(min, max), Device());
  }
  static auto GeneratorAndDevice(int seed, Wide min, Wide max) {
    return std::make_pair(Generator(min, max), Device(seed));
  }
};

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_RNG_H_
