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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_LITERT_RNG_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_LITERT_RNG_H_

#include <bitset>
#include <cstdlib>
#include <functional>
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
#include "litert/c/litert_model_types.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/internal/litert_numerics.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

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

template <typename D>
class F16InF32Generator final {};

// Generates valid f16 values stored as f32.
// TODO: Add first class support for f16 in litert.
template <>
class F16InF32Generator<float> final {
 public:
  // This works by generating 32 random bits and then masking and moving things
  // around so there are a maximum of 10 significant mantissa bits and an
  // an exponent component between 2^-14 and 2^15.
  // NOTE: We will replace this in the future with a proper f16 -> f32
  // converter.
  template <typename Rng>
  float operator()(Rng& rng) {
    std::bitset<32> b = rng();
    // Mask out trailing mantissa bits that an f16 doesn't have.
    std::bitset<32> mant_mask = ~((1u << 13) - 1);
    // Value for the final left 4 bits of the exponent. Uses the fourth
    // bit from left of the initial exponent componenet to determine which of
    // the 2 possible values the final exponent 4 bit prefix can take.
    std::bitset<32> exp_prefix = ((b[27] + 7u) << 27);
    // Masks out the top 4 bits of the exponent, after the final prefix has
    // been computed.
    std::bitset<32> exp_mask = ~(((1u << 4) - 1) << 27);

    // Apply the masks and prefix to return a valid f16 value stored as f32.
    b = (b & mant_mask & exp_mask) | exp_prefix;

    // Add or subtract one from the exponent to avoid super or sub normals.
    if ((b[30] == b[26]) && (b[30] == b[25]) && (b[30] == b[24]) &&
        (b[30] == b[23])) {
      b.flip(23);
    }

    // Reinterpret as float.
    const auto res = b.to_ulong();
    float f_res;
    memcpy(&f_res, &res, sizeof(f_res));

    return f_res;
  }
};

template <typename D>
class SinGenerator final : public DataGeneratorBase<D> {};

// Generates sin values in the range [-1, 1].
template <>
class SinGenerator<float> final : public DataGeneratorBase<float> {
 public:
  SinGenerator() = default;

  template <typename Rng>
  float operator()(Rng& rng) {
    return std::sin(rng() * 0.12345f);
  }
  float Max() const override { return 1.0f; }
  float Min() const override { return -1.0f; }
};

// Dummy primitive generator that returns a monotonically increasing sequence.
template <typename D>
class DummyGenerator final : public DataGeneratorBase<D> {
 public:
  using DataType = D;

  DummyGenerator() = default;

  template <typename Rng>
  DataType operator()(Rng& rng) {
    return val_++;
  }

  DataType Max() const override { return NumericLimits<D>::Max(); }
  DataType Min() const override { return 0; }

 private:
  D val_ = 0;
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
template <typename D, template <typename> typename Generator>
class RandomTensorData {
 private:
  // TODO: Support on standard types.
  static_assert(std::is_integral_v<D> || std::is_floating_point_v<D>);
  using Gen = Generator<D>;

 public:
  using DataType = D;

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

  template <typename DD,
            typename = std::enable_if_t<std::is_constructible_v<Gen, DD, DD>>>
  explicit RandomTensorData(DD min, DD max) : gen_(min, max) {}

  template <typename = std::enable_if<std::is_constructible_v<Gen>>::type>
  explicit RandomTensorData() : gen_() {}

 private:
  Gen gen_;
};

// Utility class to specify how tensor data generation should be performed
// per-datatype without templates.
class RandomTensorDataBuilder {
 public:
  RandomTensorDataBuilder() = default;

  RandomTensorDataBuilder& SetIntRange(int32_t min, int32_t max) {
    int_config_ = std::make_pair(min, max);
    return *this;
  }

  RandomTensorDataBuilder& SetIntDummy() {
    int_config_ = Dummy();
    return *this;
  }

  RandomTensorDataBuilder& SetFloatRange(float min, float max) {
    float_config_ = std::make_pair(min, max);
    return *this;
  }

  RandomTensorDataBuilder& SetFloatDummy() {
    float_config_ = Dummy();
    return *this;
  }

  RandomTensorDataBuilder& SetF16InF32() {
    float_config_ = F16InF32();
    return *this;
  }

  RandomTensorDataBuilder& SetSin() {
    float_config_ = Sin();
    return *this;
  }

  template <typename D>
  std::pair<double, double> Bounds() const {
    if constexpr (std::is_same_v<D, int32_t>) {
      if (std::holds_alternative<Dummy>(int_config_)) {
        return {0, static_cast<double>(std::numeric_limits<D>::max())};
      } else if (std::holds_alternative<NullOpt>(int_config_)) {
        return {std::numeric_limits<D>::min(),
                static_cast<double>(std::numeric_limits<D>::max())};
      } else {
        auto [min, max] = std::get<std::pair<D, D>>(int_config_);
        return {static_cast<double>(min), static_cast<double>(max)};
      }
    } else if constexpr (std::is_same_v<D, float>) {
      if (std::holds_alternative<Dummy>(float_config_)) {
        return {0, std::numeric_limits<float>::max()};
      } else if (std::holds_alternative<NullOpt>(float_config_)) {
        return {std::numeric_limits<float>::lowest(),
                std::numeric_limits<float>::max()};
      } else if (std::holds_alternative<F16InF32>(float_config_)) {
        return {std::numeric_limits<float>::lowest(), 65504.0};
      } else if (std::holds_alternative<Sin>(float_config_)) {
        return {-1.0f, 1.0f};
      } else {
        auto [min, max] = std::get<std::pair<float, float>>(float_config_);
        return {min, max};
      }
    } else if constexpr (std::is_same_v<D, int64_t>) {
      if (std::holds_alternative<Dummy>(int64_config_)) {
        return {0, static_cast<double>(std::numeric_limits<D>::max())};
      } else if (std::holds_alternative<NullOpt>(int64_config_)) {
        return {std::numeric_limits<D>::min(),
                static_cast<double>(std::numeric_limits<D>::max())};
      } else {
        auto [min, max] = std::get<std::pair<D, D>>(int64_config_);
        return {static_cast<double>(min), static_cast<double>(max)};
      }
    } else {
      static_assert(false, "Unsupported type");
    }
  }

  template <typename D, typename Functor, typename... Args>
  auto Call(Args&&... args) const {
    if constexpr (std::is_same_v<D, int32_t>) {
      if (std::holds_alternative<Dummy>(int_config_)) {
        RandomTensorData<D, DummyGenerator> data;
        return Functor()(data, std::forward<Args>(args)...);
      } else if (std::holds_alternative<NullOpt>(int_config_)) {
        RandomTensorData<D, DefaultGenerator> data;
        return Functor()(data, std::forward<Args>(args)...);
      } else {
        auto [min, max] = std::get<std::pair<D, D>>(int_config_);
        RandomTensorData<D, DefaultRangedGenerator> data(min, max);
        return Functor()(data, std::forward<Args>(args)...);
      }
    } else if constexpr (std::is_same_v<D, float>) {
      if (std::holds_alternative<Dummy>(float_config_)) {
        RandomTensorData<D, DummyGenerator> data;
        return Functor()(data, std::forward<Args>(args)...);
      } else if (std::holds_alternative<NullOpt>(float_config_)) {
        RandomTensorData<D, DefaultGenerator> data;
        return Functor()(data, std::forward<Args>(args)...);
      } else if (std::holds_alternative<F16InF32>(float_config_)) {
        RandomTensorData<D, F16InF32Generator> data;
        return Functor()(data, std::forward<Args>(args)...);
      } else if (std::holds_alternative<Sin>(float_config_)) {
        RandomTensorData<D, SinGenerator> data;
        return Functor()(data, std::forward<Args>(args)...);
      } else {
        auto [min, max] = std::get<std::pair<D, D>>(float_config_);
        RandomTensorData<D, DefaultRangedGenerator> data(min, max);
        return Functor()(data, std::forward<Args>(args)...);
      }
    } else if constexpr (std::is_same_v<D, int64_t>) {
      if (std::holds_alternative<Dummy>(int64_config_)) {
        RandomTensorData<D, DummyGenerator> data;
        return Functor()(data, std::forward<Args>(args)...);
      } else if (std::holds_alternative<NullOpt>(int64_config_)) {
        RandomTensorData<D, DefaultGenerator> data;
        return Functor()(data, std::forward<Args>(args)...);
      } else {
        auto [min, max] = std::get<std::pair<D, D>>(int64_config_);
        RandomTensorData<D, DefaultRangedGenerator> data(min, max);
        return Functor()(data, std::forward<Args>(args)...);
      }
    } else {
      static_assert(false, "Unsupported type");
    }
  }

 private:
  struct Dummy {};
  struct NullOpt {};
  struct F16InF32 {};
  struct Sin {};

  template <typename D>
  using IntConfig = std::variant<std::pair<D, D>, Dummy, NullOpt>;
  template <typename D>
  using FloatConfig =
      std::variant<std::pair<float, float>, Dummy, NullOpt, F16InF32, Sin>;

  IntConfig<int32_t> int_config_ = NullOpt();
  FloatConfig<float> float_config_ = NullOpt();
  IntConfig<int64_t> int64_config_ = NullOpt();
};

// Scale random data values down to prevent overflow.
template <typename It>
void ScaleDown(It start, It end, uint32_t scale) {
  using T = typename std::iterator_traits<It>::value_type;
  std::for_each(start, end, [scale](auto& x) { x /= static_cast<T>(scale); });
}
template <typename Itb>
void ScaleDown(Itb& itb, uint32_t scale) {
  ScaleDown(std::begin(itb), std::end(itb), scale);
}

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_LITERT_RNG_H_
