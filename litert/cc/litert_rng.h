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
#include "litert/cc/litert_macros.h"
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

// Base class for generating data of a certain type.
template <typename D>
class DataGeneratorBase {
 public:
  using DataType = D;

  // Bounds of distribution.
  virtual DataType Max() const = 0;
  virtual DataType Min() const = 0;

  virtual ~DataGeneratorBase() = default;
};

// Base class for generating data of a certain type from a specific
// distribution.
template <typename D, template <typename> typename Dist>
class DataGenerator : public DataGeneratorBase<D> {
 public:
  using DataType = D;
  using Wide = WideType<D>;

  // Bounds of distribution.
  DataType Max() const override { return dist_.max(); }
  DataType Min() const override { return dist_.min(); }

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

  RangedGenerator(Wide min = NumericLimits<D>::Lowest(),
                  Wide max = NumericLimits<D>::Max()) {
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
// TODO: Update this to separate the type and shape generation.
template <size_t Rank, size_t MaxSize, LiteRtElementType... ElementType>
class RandomTensorType {
 private:
  using DimSize = int32_t;
  using DimGenerator = DefaultRangedGenerator<DimSize>;
  using ElementTypeInt = uint8_t;
  using ElementTypeGenerator = DefaultRangedGenerator<ElementTypeInt>;
  static constexpr auto kNumElementTypes = sizeof...(ElementType);
  static_assert(kNumElementTypes > 0);
  static constexpr std::array<LiteRtElementType, kNumElementTypes>
      kElementTypes = {ElementType...};
  static_assert(Rank < LITERT_TENSOR_MAX_RANK);

  // std::pow not constexpr until c++26 sadly so no constexpr here.
  static DimSize MaxDimSize() {
    const double rank = std::max(1lu, Rank);
    const double exp = 1.0 / rank;
    const double max_flat = MaxSize;
    const double max_dim = std::pow(max_flat, exp);
    return static_cast<DimSize>(std::floor(max_dim));
  }

 public:
  using RandDim = std::monostate;
  using DimRange = std::pair<DimSize, DimSize>;
  using DimSpec = std::variant<DimSize, DimRange, RandDim>;
  using ShapeSpec = std::array<DimSpec, Rank>;

  // Max number of elements we want a tensor to have.
  static constexpr size_t kMaxFlatSize = MaxSize;

  // Cap single dimenions at the rankth root of the flat size to respect the
  // max.
  static const DimSize kMaxDimSize;

  // TODO: Explore need for 0 valued dims.
  static constexpr DimSize kMinDimSize = 1;

 private:
  static constexpr ShapeSpec DefaultShapeSpec() {
    ShapeSpec res;
    for (auto i = 0; i < Rank; ++i) {
      res[i] = RandDim();
    }
    return res;
  };

 public:
  // Generate a random tensor type from the specification provided. An
  // element type is taken randomly from the template parameter. The shape
  // spec must be of same rank as provided by the template parameter (this
  // is checked at compile time). An element of the shape spec can be an
  // explicit value, in which case it will not be random, a range, or a RandDim
  // which signifies a range over all possible values of that dimension.
  // `shuffle` can be used to permute the dimensions after generation.
  template <typename Rng>
  Expected<LiteRtRankedTensorType> operator()(
      Rng& rng, const ShapeSpec& spec = DefaultShapeSpec(),
      bool shuffle = false) {
    LITERT_ASSIGN_OR_RETURN(auto layout, Layout(rng, spec, shuffle));
    return LiteRtRankedTensorType{GenerateElementType(rng), std::move(layout)};
  }

 private:
  using ResolvedDimSpec = std::variant<DimSize, DimRange>;

  // LAYOUT ////////////////////////////////////////////////////////////////////

  // Overloads that check the value of the dim spec against the max and
  // handles any defaults.

  static Expected<ResolvedDimSpec> ResolveDimSpec(DimSize dim) {
    if (dim < kMinDimSize) {
      return Error(kLiteRtStatusErrorInvalidArgument, "Dimension must be > 0");
    }
    if (dim > kMaxDimSize) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Dimension must be <= kMaxDimSize");
    }
    return ResolvedDimSpec(dim);
  }

  static Expected<ResolvedDimSpec> ResolveDimSpec(DimRange dim) {
    LITERT_ASSIGN_OR_RETURN(auto l, ResolveDimSpec(dim.first));
    LITERT_ASSIGN_OR_RETURN(auto r, ResolveDimSpec(dim.second));
    if (l >= r) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Left dimension must be < right dimension");
    }
    return ResolvedDimSpec(
        std::make_pair(std::get<DimSize>(l), std::get<DimSize>(r)));
  }

  static Expected<ResolvedDimSpec> ResolveDimSpec(std::monostate) {
    return ResolvedDimSpec(std::make_pair(kMinDimSize, kMaxDimSize));
  }

  // Overloads that produce the final dimension from a resolved dim spec.

  template <typename Rng>
  static DimSize GenerateDim(Rng& rng, DimSize dim) {
    return dim;
  }

  template <typename Rng>
  static DimSize GenerateDim(Rng& rng, DimRange dim) {
    DimGenerator gen(dim.first, dim.second);
    auto res = gen(rng);
    return res;
  }

  template <typename Rng>
  Expected<LiteRtLayout> Layout(Rng& rng, const ShapeSpec& spec, bool shuffle) {
    LiteRtLayout res;
    res.rank = Rank;
    for (auto i = 0; i < Rank; ++i) {
      const DimSpec& dim_spec = spec[i];
      LITERT_ASSIGN_OR_RETURN(
          auto resolved_dim_spec,
          std::visit([](auto&& arg) { return ResolveDimSpec(arg); }, dim_spec));
      res.dimensions[i] =
          std::visit([&rng](auto&& arg) { return GenerateDim(rng, arg); },
                     resolved_dim_spec);
    }
    if (shuffle) {
      auto beg = std::begin(res.dimensions);
      std::shuffle(beg, beg + Rank, rng);
    }
    return res;
  }

  // ELEMENT TYPE //////////////////////////////////////////////////////////////

  template <typename Rng>
  static LiteRtElementType GenerateElementType(Rng& rng) {
    ElementTypeGenerator gen(0, kNumElementTypes - 1);
    return kElementTypes[gen(rng)];
  }
};

template <size_t Rank, size_t MaxSize, LiteRtElementType... ElementType>
const auto RandomTensorType<Rank, MaxSize, ElementType...>::kMaxDimSize =
    MaxDimSize();

// RANDOM TENSOR DATA //////////////////////////////////////////////////////////

// Base class for generating data for tensors.
template <typename D, typename Derived, typename Gen>
class RandomTensorDataBase {
 private:
  // TODO: Support on standard types.
  static_assert(std::is_integral_v<D> || std::is_floating_point_v<D>);

 public:
  // Fill out the pre-allocated buffer with random data.
  template <typename Rng, typename Iter>
  Expected<void> operator()(Rng& rng, Iter start, Iter end) {
    std::generate(start, end, [&]() { return gen_(rng); });
    return {};
  }

  // Fill out the pre-allocated buffer with random data.
  template <typename Rng>
  Expected<void> operator()(Rng& rng, absl::Span<D> data) {
    return operator()(rng, data.begin(), data.end());
  }

  // Allocate a new buffer with size matching the given layout and fill it with
  // random data.
  template <typename Rng>
  Expected<std::vector<D>> operator()(Rng& rng, const LiteRtLayout& layout) {
    size_t num_elements;
    LITERT_RETURN_IF_ERROR(LiteRtGetNumLayoutElements(&layout, &num_elements));
    return operator()(rng, num_elements);
  }

  // Allocate a new buffer with the given number of elements and fill it with
  // random data.
  template <typename Rng>
  Expected<std::vector<D>> operator()(Rng& rng, size_t num_elements) {
    std::vector<D> res(num_elements);
    LITERT_RETURN_IF_ERROR(operator()(rng, res.begin(), res.end()));
    return res;
  }

  D High() const { return gen_.Max(); }
  D Low() const { return gen_.Min(); }

 private:
  Gen gen_ = Derived::MakeGen();
};

// Generates random data for tensors using the implicit entire avaiable range of
// values. Uses the default data generator types.
// TODO: Decide if type configurability is useful for this.
template <typename D>
class RandomTensorData
    : public RandomTensorDataBase<D, RandomTensorData<D>, DefaultGenerator<D>> {
  friend class RandomTensorDataBase<D, RandomTensorData<D>,
                                    DefaultGenerator<D>>;
  using Gen = DefaultGenerator<D>;
  using DataType = D;
  static Gen MakeGen() { return Gen(); }
};

// Generates random data for tensors using the explicitly specified range of
// values.
template <typename D, int64_t Low = NumericLimits<int64_t>::Lowest(),
          int64_t High = NumericLimits<int64_t>::Max()>
class RangedRandomTensorData
    : public RandomTensorDataBase<D, RangedRandomTensorData<D, Low, High>,
                                  DefaultRangedGenerator<D>> {
  friend class RandomTensorDataBase<D, RangedRandomTensorData<D, Low, High>,
                                    DefaultRangedGenerator<D>>;
  using Gen = DefaultRangedGenerator<D>;
  using DataType = D;
  static Gen MakeGen() { return Gen(Low, High); }
};

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_RNG_H_
