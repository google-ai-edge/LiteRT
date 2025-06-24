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
#include <variant>
#include <vector>

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model.h"
#include "litert/cc/litert_detail.h"
#include "litert/cc/litert_expected.h"
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

// PRIMITIVE DATA GENERATORS ///////////////////////////////////////////////////

// Abstract base class for generating data of a certain type from a given rng
// device, e.g. populating tensors and the like.
template <typename D, template <typename> typename Dist>
class DataGenerator {
 public:
  using DataType = D;
  using Wide = WideType<D>;

  // Bounds of distribution.
  DataType Max() const { return dist_.max(); }
  DataType Min() const { return dist_.min(); }

 protected:
  Dist<DataType> dist_;
};

// A data generator that generates data within a given range.
template <typename D, template <typename> typename Dist>
class RangedGenerator final : public DataGenerator<D, Dist> {
 private:
  using Base = DataGenerator<D, Dist>;

 public:
  using typename Base::DataType;
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

  template <typename Rng>
  DataType operator()(Rng& rng) {
    return this->dist_(rng);
  }
};

// A rangeless float generator that casts random bits to the given float type.
// This generally produces higher quality floats more repersentative of the
// target distribution than a ranged generator. Particularly covers more values
// around zero and infinities.
template <typename D, template <typename> typename Dist, typename Enable = void>
class ReinterpretGenerator final : public DataGenerator<D, Dist> {};

template <typename D, template <typename> typename Dist>
class ReinterpretGenerator<D, Dist,
                           std::enable_if_t<std::is_floating_point_v<D>>>
    final : public DataGenerator<D, Dist> {
 private:
  using Base = DataGenerator<D, Dist>;

 public:
  using typename Base::DataType;
  using typename Base::Wide;

  template <typename Rng>
  DataType operator()(Rng& rng) {
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

// DEFAULTS FOR DATA GENERATORS ////////////////////////////////////////////////

template <typename D>
using DefaultGenerator =
    SelectT<std::is_floating_point<D>,
            ReinterpretGenerator<D, std::uniform_real_distribution>,
            std::is_integral<D>,
            RangedGenerator<D, std::uniform_int_distribution>>;

template <typename D>
using DefaultRangedGenerator =
    SelectT<std::is_floating_point<D>,
            RangedGenerator<D, std::uniform_real_distribution>,
            std::is_integral<D>,
            RangedGenerator<D, std::uniform_int_distribution>>;

using DefaultDevice = RandomDevice<std::mt19937>;

// RANDOM TENSOR TYPES /////////////////////////////////////////////////////////

// This class composes the primitive data generators above to support
// generating randomized tensor types (and shapes).
class RandomTensorType {
 private:
  using DimSize = uint32_t;
  using DimGenerator = DefaultRangedGenerator<DimSize>;
  using ElementTypeInt = uint8_t;
  using ElementTypeGenerator = DefaultRangedGenerator<ElementTypeInt>;

 public:
  using DimRange = std::pair<DimSize, DimSize>;
  using DimSpec = std::variant<DimSize, DimRange>;
  using ElementTypeSpec = std::vector<LiteRtElementType>;

  // Max value of a random dimension.
  static constexpr auto kMaxDim = 1024;

  // Generate a random tensor type with a pre-determined rank given
  // in `shape_spec`.
  template <typename Rng>
  Expected<LiteRtRankedTensorType> Generate(
      Rng& rng, const std::vector<std::optional<DimSpec>>& shape_spec,
      const ElementTypeSpec& type = {kLiteRtElementTypeInt32,
                                     kLiteRtElementTypeFloat32}) {
    const auto rank = shape_spec.size();
    if (rank > LITERT_TENSOR_MAX_RANK) {
      return Error(kLiteRtStatusErrorInvalidArgument, "Rank too large");
    }
    LiteRtRankedTensorType res;
    res.layout.rank = rank;
    res.element_type = GenerateElementType(rng, type);
    for (auto i = 0; i < rank; ++i) {
      res.layout.dimensions[i] = GenerateDim(rng, shape_spec[i]);
    }
    return res;
  }

  // Generate a random tensor type with a random rank.
  template <typename Rng>
  Expected<LiteRtRankedTensorType> Generate(
      Rng& rng, size_t max_rank = LITERT_TENSOR_MAX_RANK,
      const ElementTypeSpec& type = {kLiteRtElementTypeInt32,
                                     kLiteRtElementTypeFloat32}) {
    DimGenerator rank_gen(0, max_rank);
    std::vector<std::optional<DimSpec>> shape_spec(rank_gen(rng), std::nullopt);
    return Generate(rng, shape_spec, type);
  }

 private:
  template <typename Rng>
  DimSize GenerateDim(Rng& rng, const std::optional<DimSpec>& dim) {
    if (!dim) {
      DimGenerator gen(0, kMaxDim);
      const auto res = gen(rng);
      return res;
    } else if (std::holds_alternative<DimSize>(*dim)) {
      auto d = std::get<DimSize>(*dim);
      return d;
    } else {
      auto d = std::get<DimRange>(*dim);
      DimGenerator gen(d.first, d.second);
      return gen(rng);
    }
  }

  template <typename Rng>
  LiteRtElementType GenerateElementType(Rng& rng, const ElementTypeSpec& type) {
    if (type.size() == 1) {
      return type.front();
    }
    ElementTypeGenerator gen(0, type.size() - 1);
    return type[gen(rng)];
  }
};

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_RNG_H_
