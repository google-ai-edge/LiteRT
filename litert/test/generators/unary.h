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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_UNARY_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_UNARY_H_

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_c_types_printing.h"  // IWYU pragma: keep
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/internal/litert_rng.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/model/model.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/graph_helpers.h"
#include "litert/test/simple_buffer.h"

namespace litert {
namespace testing {

template <typename T>
struct FloorReference {
  T operator()(T value) const { return std::floor(value); }
};

template <typename T>
struct LogisticReference {
  T operator()(T value) const {
    return static_cast<T>(1) / (static_cast<T>(1) + std::exp(-value));
  }
};

template <typename T>
struct ReluReference {
  T operator()(T value) const { return std::max(value, static_cast<T>(0)); }
};

template <typename T>
struct ReluN1To1Reference {
  T operator()(T value) const {
    return std::min(static_cast<T>(1), std::max(static_cast<T>(-1), value));
  }
};

template <typename T>
struct Relu6Reference {
  T operator()(T value) const {
    return std::min(static_cast<T>(6), std::max(static_cast<T>(0), value));
  }
};

template <typename T>
struct TanhReference {
  T operator()(T value) const { return std::tanh(value); }
};

template <typename T>
struct ExpReference {
  T operator()(T value) const { return std::exp(value); }
};

template <typename T>
struct NegReference {
  T operator()(T value) const { return -value; }
};

template <typename T>
struct SinReference {
  T operator()(T value) const { return std::sin(value); }
};

template <typename T>
struct LogReference {
  T operator()(T value) const { return std::log(value); }
};

template <typename T>
struct SqrtReference {
  T operator()(T value) const { return std::sqrt(value); }
};

template <typename T>
struct RsqrtReference {
  T operator()(T value) const { return 1 / std::sqrt(value); }
};

template <typename T>
struct SquareReference {
  T operator()(T value) const { return value * value; }
};

template <typename T>
struct ZerosLikeReference {
  T operator()(T value) const { return static_cast<T>(0); }
};

template <typename T>
struct AbsReference {
  T operator()(T value) const { return std::abs(value); }
};

template <typename T>
struct CeilReference {
  T operator()(T value) const { return std::ceil(value); }
};

template <typename T>
struct CosReference {
  T operator()(T value) const { return std::cos(value); }
};

template <typename T>
struct EluReference {
  T operator()(T value) const {
    return value > 0 ? value : std::expm1(value);
  }
};

template <typename T>
struct RoundReference {
  T operator()(T value) const { return std::rint(value); }
};

template <typename T>
struct HardSwishReference {
  T operator()(T value) const {
    // If T is double, compute in double. Otherwise, compute in float.
    using ComputeT = std::conditional_t<std::is_floating_point_v<T>, T, float>;

    const ComputeT x = static_cast<ComputeT>(value);
    const ComputeT relu6_x_plus_3 =
        std::min(ComputeT{6}, std::max(ComputeT{0}, x + ComputeT{3}));

    return static_cast<T>(x * relu6_x_plus_3 / ComputeT{6});
  }
};

template <typename T>
struct GeluReference {
  T operator()(T value) const {
    return static_cast<T>(0.5) * value *
           (static_cast<T>(1) + std::erf(value / std::sqrt(static_cast<T>(2))));
  }
};

template <typename T>
struct Relu0To1Reference {
  T operator()(T value) const {
    return std::min(static_cast<T>(1), std::max(static_cast<T>(0), value));
  }
};

template <typename T>
struct SignReference {
  T operator()(T value) const {
    if (value == 0) return T{0};

    if constexpr (std::is_floating_point_v<T>) {
      // preserves correct sign handling for floats (incl. sign-bit semantics)
      return std::signbit(value) ? T{-1} : T{1};
    } else if constexpr (std::is_signed_v<T>) {
      return value < 0 ? T{-1} : T{1};
    } else {
      // unsigned: never negative
      return T{1};
    }
  }
};

// clang-format off
template <
    typename Rank,
    typename T,
    typename OpCode,
    typename MaxTensorSize = SizeC<1024>
>
// clang-format on
class Unary : public TestGraph {
 private:
  static_assert(std::is_same_v<typename OpCode::value_type, LiteRtOpCode>);
  static constexpr LiteRtOpCode kOpCode = OpCode::value;

  static_assert(std::is_same_v<typename MaxTensorSize::value_type, size_t>);
  static constexpr size_t kMaxTensorSize = MaxTensorSize::value;

  static_assert(std::is_same_v<typename Rank::value_type, size_t>);
  static constexpr size_t kRank = Rank::value;

  // TODO finish support for other op codes and activation functions.
  static_assert(
      kOpCode == kLiteRtOpCodeTflFloor || kOpCode == kLiteRtOpCodeTflLogistic ||
      kOpCode == kLiteRtOpCodeTflRelu || kOpCode == kLiteRtOpCodeTflReluN1To1 ||
      kOpCode == kLiteRtOpCodeTflRelu6 || kOpCode == kLiteRtOpCodeTflTanh ||
      kOpCode == kLiteRtOpCodeTflExp || kOpCode == kLiteRtOpCodeTflNeg ||
      kOpCode == kLiteRtOpCodeTflSin || kOpCode == kLiteRtOpCodeTflLog ||
      kOpCode == kLiteRtOpCodeTflSqrt || kOpCode == kLiteRtOpCodeTflRsqrt ||
      kOpCode == kLiteRtOpCodeTflSquare ||
      kOpCode == kLiteRtOpCodeTflZerosLike || kOpCode == kLiteRtOpCodeTflAbs ||
      kOpCode == kLiteRtOpCodeTflCeil || kOpCode == kLiteRtOpCodeTflCos ||
      kOpCode == kLiteRtOpCodeTflElu || kOpCode == kLiteRtOpCodeTflRound ||
      kOpCode == kLiteRtOpCodeTflHardSwish || kOpCode == kLiteRtOpCodeTflGelu ||
      kOpCode == kLiteRtOpCodeTflRelu0To1 || kOpCode == kLiteRtOpCodeTflSign);

  static constexpr TensorNames<1> kInputNames = {"input"};
  static constexpr TensorNames<2> kOutputNames = {"output"};
  static constexpr absl::string_view kSignatureName = "default";

  // clang-format off
  using ReferenceOperator =
      SelectT<
          std::bool_constant<kOpCode == kLiteRtOpCodeTflFloor>,
            FloorReference<T>,
          std::bool_constant<kOpCode == kLiteRtOpCodeTflLogistic>,
            LogisticReference<T>,
          std::bool_constant<kOpCode == kLiteRtOpCodeTflRelu>,
            ReluReference<T>,
          std::bool_constant<kOpCode == kLiteRtOpCodeTflReluN1To1>,
            ReluN1To1Reference<T>,
          std::bool_constant<kOpCode == kLiteRtOpCodeTflRelu6>,
            Relu6Reference<T>,
          std::bool_constant<kOpCode == kLiteRtOpCodeTflTanh>,
            TanhReference<T>,
          std::bool_constant<kOpCode == kLiteRtOpCodeTflExp>,
            ExpReference<T>,
          std::bool_constant<kOpCode == kLiteRtOpCodeTflNeg>,
            NegReference<T>,
          std::bool_constant<kOpCode == kLiteRtOpCodeTflSin>,
            SinReference<T>,
          std::bool_constant<kOpCode == kLiteRtOpCodeTflLog>,
            LogReference<T>,
          std::bool_constant<kOpCode == kLiteRtOpCodeTflSqrt>,
            SqrtReference<T>,
          std::bool_constant<kOpCode == kLiteRtOpCodeTflRsqrt>,
            RsqrtReference<T>,
          std::bool_constant<kOpCode == kLiteRtOpCodeTflSquare>,
            SquareReference<T>,
          std::bool_constant<kOpCode == kLiteRtOpCodeTflZerosLike>,
            ZerosLikeReference<T>,
          std::bool_constant<kOpCode == kLiteRtOpCodeTflAbs>,
            AbsReference<T>,
          std::bool_constant<kOpCode == kLiteRtOpCodeTflCeil>,
            CeilReference<T>,
          std::bool_constant<kOpCode == kLiteRtOpCodeTflCos>,
            CosReference<T>,
          std::bool_constant<kOpCode == kLiteRtOpCodeTflElu>,
            EluReference<T>,
          std::bool_constant<kOpCode == kLiteRtOpCodeTflRound>,
            RoundReference<T>,
          std::bool_constant<kOpCode == kLiteRtOpCodeTflHardSwish>,
            HardSwishReference<T>,
          std::bool_constant<kOpCode == kLiteRtOpCodeTflGelu>,
            GeluReference<T>,
          std::bool_constant<kOpCode == kLiteRtOpCodeTflRelu0To1>,
            Relu0To1Reference<T>,
          std::bool_constant<kOpCode == kLiteRtOpCodeTflSign>,
            SignReference<T>
            >;
  // clang-format on

  static constexpr ElementType kElementType = GetElementType<T>();

  using FbTypes = FbOpTypes<kOpCode>;

  struct Params {
    std::array<Layout::Dim, kRank> shape;
  };

 public:
  using Traits = TestLogicTraits<TypeList<T>, TypeList<T>, Params>;
  using Ptr = std::unique_ptr<Unary>;

  static constexpr absl::string_view Name() { return "Unary"; }

  template <typename Rng>
  static Expected<Unary::Ptr> Create(Rng& rng) {
    LITERT_ASSIGN_OR_RETURN(auto params, GenerateParams(rng));
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<Unary>(std::move(params), std::move(model));
  }

  static Expected<Unary::Ptr> Create(Params params) {
    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<Unary>(std::move(params), std::move(model));
  }

  bool HasReference() const override { return true; }

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    VarBuffers inputs;
    LITERT_ASSIGN_OR_RETURN(auto input, SimpleBuffer::Create<T>(params_.shape));

    if constexpr (kOpCode == kLiteRtOpCodeTflLog ||
                  kOpCode == kLiteRtOpCodeTflSqrt ||
                  kOpCode == kLiteRtOpCodeTflRsqrt) {
      auto constrained_builder = data_builder;
      if constexpr (std::is_floating_point_v<T>) {
        if (!data_builder.IsFloatDummy()) {
          constrained_builder.SetFloatRange(std::numeric_limits<T>::epsilon(),
                                            1000.0);
        }
      }
      LITERT_RETURN_IF_ERROR(
          (input.template WriteRandom<T>(constrained_builder, device)));
    } else {
      LITERT_RETURN_IF_ERROR(
          (input.template WriteRandom<T>(data_builder, device)));
    }

    inputs.push_back(std::move(input));
    return inputs;
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    LITERT_ASSIGN_OR_RETURN(auto ref_inputs,
                            Traits::MakeReferenceInputs(inputs));
    LITERT_ASSIGN_OR_RETURN(auto ref_outputs,
                            Traits::MakeReferenceOutputs(outputs));
    return ReferenceImpl(ref_inputs, ref_outputs);
  }

  Unary(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

 private:
  template <typename Rng>
  static Expected<Params> GenerateParams(Rng& rng) {
    RandomTensorType<kRank, kMaxTensorSize, LiteRtElementType(kElementType)>
        type;
    LITERT_ASSIGN_OR_RETURN(const auto tensor_type, type(rng));
    Params p;
    std::copy(std::cbegin(tensor_type.layout.dimensions),
              std::cbegin(tensor_type.layout.dimensions) + kRank,
              std::begin(p.shape));
    return p;
  }
  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    const std::vector<int32_t> dims(params.shape.begin(), params.shape.end());

    std::vector<TensorDetails> inputs(1);
    std::vector<TensorDetails> outputs(1);

    inputs[0] = TensorDetails{dims, LiteRtElementType(kElementType),
                              std::string(kInputNames[0])};

    outputs[0] = TensorDetails{dims, LiteRtElementType(kElementType),
                               std::string(kOutputNames[0])};

    return SingleOpModel<kOpCode>(inputs, outputs);
  }

  Expected<void> ReferenceImpl(
      const typename Traits::ReferenceInputs& inputs,
      const typename Traits::ReferenceOutputs& outputs) const {
    auto [input] = inputs;
    auto [output] = outputs;
    if (input.dimensions != output.dimensions) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "input and output must have the same dimensions");
    }
    for (auto i = 0; i < input.NumElements(); ++i) {
      output.data[i] = ReferenceOperator()(input.data[i]);
    }
    return {};
  }

  Params params_;
};

}  // namespace testing
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_UNARY_H_
