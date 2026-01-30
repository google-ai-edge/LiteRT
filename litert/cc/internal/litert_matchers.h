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

#ifndef ODML_LITERT_LITERT_CC_INTERNAL_LITERT_MATCHERS_H_
#define ODML_LITERT_LITERT_CC_INTERNAL_LITERT_MATCHERS_H_

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/internal/litert_op_options.h"

namespace litert {

// Helper to check if a type has a Match method for a given type U.
template <typename T, typename U, typename = void>
struct has_match_method : std::false_type {};

template <typename T, typename U>
struct has_match_method<
    T, U, std::void_t<decltype(std::declval<T>().Match(std::declval<U>()))>>
    : std::true_type {};

template <typename T, typename U>
constexpr bool has_match_method_v = has_match_method<T, U>::value;

// Main Match entry point.
// Matches val against matcher m.
template <typename T, typename Matcher>
bool Match(const T& val, const Matcher& m) {
  return m.Match(val);
}

// Helper to match input at index I.
template <typename OpType, typename Matcher>
bool MatchInput(const OpType& op, size_t index, const Matcher& m) {
  auto input = op.Input(index);
  if (!input) return false;
  return m.Match(*input);
}

// OpCode matcher.
template <LiteRtOpCode Code>
struct OpCodeMatcher {
  bool Match(const Op& op) const { return op.Code() == Code; }
  bool Match(const Tensor& tensor) const {
    auto def_op = tensor.GetDefiningOp();
    if (!def_op) return false;
    return Match(*def_op);
  }
};

template <LiteRtOpCode Code>
inline auto m_OpCode() {
  return OpCodeMatcher<Code>{};
}

// OpMatcher implementation.
template <LiteRtOpCode Code, typename... InputMatchers>
struct OpMatcher {
  std::tuple<InputMatchers...> input_matchers;

  explicit OpMatcher(InputMatchers... m) : input_matchers(std::move(m)...) {}

  // Match against an Op.
  bool Match(const Op& op) const {
    if (op.Code() != Code) return false;
    // We check that we have exactly the number of inputs specified.
    // If you want to allow trailing inputs, this logic would need adjustment.
    // For exact match:
    if (op.Inputs().size() != sizeof...(InputMatchers)) return false;

    return MatchInputs(op, std::index_sequence_for<InputMatchers...>{});
  }

  // Match against a Tensor (checks defining op).
  bool Match(const Tensor& tensor) const {
    auto def_op = tensor.GetDefiningOp();
    if (!def_op) return false;
    return Match(*def_op);
  }

 private:
  template <size_t... I>
  bool MatchInputs(const Op& op, std::index_sequence<I...>) const {
    // Fold expression to match all inputs.
    return (MatchInput(op, I, std::get<I>(input_matchers)) && ...);
  }
};

// Builder function for OpMatcher.
template <LiteRtOpCode Code, typename... Args>
auto m_Op(Args&&... args) {
  return OpMatcher<Code, std::decay_t<Args>...>(std::forward<Args>(args)...);
}

// Op Matcher allowing extra trailing inputs.
template <LiteRtOpCode Code, typename... InputMatchers>
struct OpVariadicMatcher {
  std::tuple<InputMatchers...> input_matchers;

  explicit OpVariadicMatcher(InputMatchers... m)
      : input_matchers(std::move(m)...) {}

  bool Match(const Op& op) const {
    if (op.Code() != Code) return false;
    if (op.Inputs().size() < sizeof...(InputMatchers)) return false;

    return MatchInputs(op, std::index_sequence_for<InputMatchers...>{});
  }

 private:
  template <size_t... I>
  bool MatchInputs(const Op& op, std::index_sequence<I...>) const {
    return (MatchInput(op, I, std::get<I>(input_matchers)) && ...);
  }
};

template <LiteRtOpCode Code, typename... Args>
auto m_OpVariadic(Args&&... args) {
  return OpVariadicMatcher<Code, std::decay_t<Args>...>(
      std::forward<Args>(args)...);
}

// Commutative Op Matcher (e.g. Add, Mul).
template <LiteRtOpCode Code, typename Matcher1, typename Matcher2>
struct CommutativeOpMatcher {
  Matcher1 m1;
  Matcher2 m2;

  CommutativeOpMatcher(Matcher1 matcher1, Matcher2 matcher2)
      : m1(std::move(matcher1)), m2(std::move(matcher2)) {}

  bool Match(const Op& op) const {
    if (op.Code() != Code) return false;
    if (op.Inputs().size() != 2) return false;

    auto in0 = op.Input(0);
    auto in1 = op.Input(1);
    if (!in0 || !in1) return false;

    // Check permutation 1
    if (m1.Match(*in0) && m2.Match(*in1)) return true;

    // Check permutation 2
    if (m1.Match(*in1) && m2.Match(*in0)) return true;

    return false;
  }

  bool Match(const Tensor& tensor) const {
    auto def_op = tensor.GetDefiningOp();
    if (!def_op) return false;
    return Match(*def_op);
  }
};

template <LiteRtOpCode Code, typename Matcher1, typename Matcher2>
auto m_CommutativeOp(Matcher1&& m1, Matcher2&& m2) {
  return CommutativeOpMatcher<Code, std::decay_t<Matcher1>,
                              std::decay_t<Matcher2>>(
      std::forward<Matcher1>(m1), std::forward<Matcher2>(m2));
}

// Options Matcher.
template <typename OptionsT, typename Pred>
struct OptionsMatcher {
  Pred pred;

  explicit OptionsMatcher(Pred p) : pred(std::move(p)) {}

  bool Match(const Op& op) const {
    auto opts = GetOptionsAs<OptionsT>(op.Get());
    if (!opts) return false;
    return pred(*opts);
  }

  bool Match(const Tensor& tensor) const {
    auto def_op = tensor.GetDefiningOp();
    if (!def_op) return false;
    return Match(*def_op);
  }
};

template <typename OptionsT, typename Pred>
auto m_Options(Pred&& pred) {
  return OptionsMatcher<OptionsT, std::decay_t<Pred>>{std::forward<Pred>(pred)};
}

// Shape matcher.
struct ShapeMatcher {
  std::vector<int32_t> shape;

  explicit ShapeMatcher(std::vector<int32_t> s) : shape(std::move(s)) {}

  bool Match(const Tensor& tensor) const {
    auto ranked_type = tensor.RankedTensorType();
    if (!ranked_type) return false;
    auto dimensions = ranked_type->Layout().Dimensions();
    if (dimensions.size() != shape.size()) return false;
    for (size_t i = 0; i < shape.size(); ++i) {
      if (shape[i] != -1 && dimensions[i] != shape[i]) return false;
    }
    return true;
  }
};

// Returns a shape matcher. Use -1 for wildcard dimensions.
inline auto m_Shape(std::vector<int32_t> shape) {
  return ShapeMatcher(std::move(shape));
}

// Rank matcher.
struct RankMatcher {
  size_t rank;

  explicit RankMatcher(size_t r) : rank(r) {}

  bool Match(const Tensor& tensor) const {
    auto ranked_type = tensor.RankedTensorType();
    if (!ranked_type) return false;
    return ranked_type->Layout().Dimensions().size() == rank;
  }
};

inline auto m_Rank(size_t rank) { return RankMatcher(rank); }

// ElementType matcher.
struct ElementTypeMatcher {
  LiteRtElementType type;

  explicit ElementTypeMatcher(LiteRtElementType t) : type(t) {}

  bool Match(const Tensor& tensor) const {
    auto ranked_type = tensor.RankedTensorType();
    if (!ranked_type) return false;
    return static_cast<LiteRtElementType>(ranked_type->ElementType()) == type;
  }
};

inline auto m_ElementType(LiteRtElementType type) {
  return ElementTypeMatcher(type);
}

// Output Index Matcher.
// Matches if the tensor is the N-th output of its defining op, AND the defining
// op matches the sub-matcher.
template <typename OpMatcher>
struct OutputIndexMatcher {
  size_t index;
  OpMatcher op_matcher;

  OutputIndexMatcher(size_t i, OpMatcher m)
      : index(i), op_matcher(std::move(m)) {}

  bool Match(const Tensor& tensor) const {
    auto def_op_wrapper = tensor.DefiningOp();
    if (!def_op_wrapper.has_value()) return false;
    if (def_op_wrapper->op_output_index != index) return false;

    // We need to construct an Op wrapper to match against OpMatcher.
    Op op(def_op_wrapper->op);
    return op_matcher.Match(op);
  }
};

template <typename Matcher>
auto m_OutputIndex(size_t index, Matcher&& m) {
  return OutputIndexMatcher<std::decay_t<Matcher>>(index,
                                                   std::forward<Matcher>(m));
}

// Capture matcher.
// Captures the matched object into the provided pointer if the sub-matcher
// succeeds.
template <typename T, typename SubMatcher>
struct CaptureMatcher {
  T* storage;
  SubMatcher sub_matcher;

  CaptureMatcher(T* s, SubMatcher m) : storage(s), sub_matcher(std::move(m)) {}

  template <typename U>
  bool Match(const U& val) const {
    if (!sub_matcher.Match(val)) {
      return false;
    }
    if (storage) {
      // If T is Op and U is Tensor, we might want to capture the defining Op?
      // Or if T is Op and U is Op, simple assignment.
      // Let's handle the specific case where we match a Tensor but capture
      // the defining Op.
      if constexpr (std::is_same_v<T, Op> && std::is_same_v<U, Tensor>) {
        auto def_op = val.GetDefiningOp();
        if (!def_op) return false;
        *storage = std::move(*def_op);
      } else if constexpr (std::is_assignable_v<T&, const U&>) {
        *storage = val;
      } else {
        if constexpr (std::is_same_v<T, Op> && std::is_same_v<U, Op>) {
          *storage = Op(val.Get());
        } else if constexpr (std::is_same_v<T, Tensor> &&
                             std::is_same_v<U, Tensor>) {
          *storage = Tensor(val.Get());
        }
      }
    }
    return true;
  }
};

template <typename T, typename Matcher>
auto m_Capture(T* storage, Matcher&& m) {
  return CaptureMatcher<T, std::decay_t<Matcher>>(storage,
                                                  std::forward<Matcher>(m));
}

// Wildcard matcher for Tensor. Always true.
struct AnyTensorMatcher {
  bool Match(const Tensor&) const { return true; }
};

inline auto m_Any() { return AnyTensorMatcher{}; }

// Wildcard matcher for Op. Always true.
struct AnyOpMatcher {
  bool Match(const Op&) const { return true; }
  bool Match(const Tensor& tensor) const {
    auto def_op = tensor.GetDefiningOp();
    if (!def_op) return false;
    return Match(*def_op);
  }
};

inline auto m_AnyOp() { return AnyOpMatcher{}; }

// Constant tensor matcher.
struct IsConstantMatcher {
  bool Match(const Tensor& tensor) const { return tensor.IsConstant(); }
};

inline auto m_IsConstant() { return IsConstantMatcher{}; }

// Subgraph input tensor matcher.
struct IsSubgraphInputMatcher {
  bool Match(const Tensor& tensor) const { return tensor.IsSubgraphInput(); }
};

inline auto m_IsSubgraphInput() { return IsSubgraphInputMatcher{}; }

// Predicate matcher.
template <typename T, typename Pred>
struct PredicateMatcher {
  Pred pred;
  bool Match(const T& val) const { return pred(val); }
};

template <typename T, typename Pred>
auto m_Predicate(Pred&& pred) {
  return PredicateMatcher<T, std::decay_t<Pred>>{std::forward<Pred>(pred)};
}

// Generic Custom Matcher (Lambda).
template <typename Pred>
struct CustomMatcher {
  Pred pred;

  explicit CustomMatcher(Pred p) : pred(std::move(p)) {}

  template <typename T>
  bool Match(const T& val) const {
    return pred(val);
  }
};

template <typename Pred>
auto m_Custom(Pred&& pred) {
  return CustomMatcher<std::decay_t<Pred>>(std::forward<Pred>(pred));
}

// Logical AND matcher (AllOf).
template <typename... Matchers>
struct AllOfMatcher {
  std::tuple<Matchers...> matchers;

  explicit AllOfMatcher(Matchers... m) : matchers(std::move(m)...) {}

  template <typename T>
  bool Match(const T& val) const {
    return MatchImpl(val, std::index_sequence_for<Matchers...>{});
  }

 private:
  template <typename T, size_t... I>
  bool MatchImpl(const T& val, std::index_sequence<I...>) const {
    return (std::get<I>(matchers).Match(val) && ...);
  }
};

template <typename... Matchers>
auto m_AllOf(Matchers&&... matchers) {
  return AllOfMatcher<std::decay_t<Matchers>...>(
      std::forward<Matchers>(matchers)...);
}

// Logical OR matcher (AnyOf).
template <typename... Matchers>
struct AnyOfMatcher {
  std::tuple<Matchers...> matchers;

  explicit AnyOfMatcher(Matchers... m) : matchers(std::move(m)...) {}

  template <typename T>
  bool Match(const T& val) const {
    return MatchImpl(val, std::index_sequence_for<Matchers...>{});
  }

 private:
  template <typename T, size_t... I>
  bool MatchImpl(const T& val, std::index_sequence<I...>) const {
    return (std::get<I>(matchers).Match(val) || ...);
  }
};

template <typename... Matchers>
auto m_AnyOf(Matchers&&... matchers) {
  return AnyOfMatcher<std::decay_t<Matchers>...>(
      std::forward<Matchers>(matchers)...);
}

// Logical NOT matcher.
template <typename Matcher>
struct NotMatcher {
  Matcher matcher;

  explicit NotMatcher(Matcher m) : matcher(std::move(m)) {}

  template <typename T>
  bool Match(const T& val) const {
    return !matcher.Match(val);
  }
};

template <typename Matcher>
auto m_Not(Matcher&& matcher) {
  return NotMatcher<std::decay_t<Matcher>>(std::forward<Matcher>(matcher));
}

// Quantization Presence Matcher.
struct IsQuantizedMatcher {
  bool Match(const Tensor& tensor) const { return tensor.HasQuantization(); }
};

inline auto m_IsQuantized() { return IsQuantizedMatcher{}; }

// Quantization Type Matcher.
struct QuantizationTypeMatcher {
  LiteRtQuantizationTypeId type;
  explicit QuantizationTypeMatcher(LiteRtQuantizationTypeId t) : type(t) {}
  bool Match(const Tensor& tensor) const { return tensor.QTypeId() == type; }
};

inline auto m_QType(LiteRtQuantizationTypeId type) {
  return QuantizationTypeMatcher(type);
}

// User Count matcher.
// Matches if the tensor has exactly N uses.
struct HasUsersMatcher {
  size_t count;
  explicit HasUsersMatcher(size_t c) : count(c) {}
  bool Match(const Tensor& tensor) const {
    return tensor.Uses().size() == count;
  }
};

inline auto m_HasUsers(size_t count) { return HasUsersMatcher(count); }

// Convenience for single use.
inline auto m_HasOneUse() { return HasUsersMatcher(1); }

// SameAs Matcher.
// Matches if the value is the same as the one stored in the pointer.
// Useful for ensuring two inputs are the same tensor (e.g. Square = Mul(x, x)).
// Requires that the storage has been populated by a previous m_Capture within
// the same match expression (left-to-right evaluation).
template <typename T>
struct SameAsMatcher {
  T* storage;
  explicit SameAsMatcher(T* s) : storage(s) {}

  bool Match(const T& val) const {
    if (!storage) return false;
    // Compare underlying handles.
    return val.Get() == storage->Get();
  }
};

template <typename T>
auto m_SameAs(T* storage) {
  return SameAsMatcher<T>(storage);
}

// Constant Scalar matcher.
// Matches if the tensor is a scalar (or splat) constant with the specific
// value. Note: Currently only supports scalar constants of standard types.
template <typename T>
struct ConstantValueMatcher {
  T value;

  explicit ConstantValueMatcher(T v) : value(v) {}

  bool Match(const Tensor& tensor) const {
    if (!tensor.IsConstant()) return false;
    auto weights_data_res = tensor.WeightsData<T>();
    if (!weights_data_res) return false;
    auto weights_data = *weights_data_res;
    if (weights_data.empty()) return false;

    // Check if all elements match (splat)
    for (const auto& elem : weights_data) {
      if (elem != value) return false;
    }
    return true;
  }
};

template <typename T>
auto m_ConstantValue(T value) {
  return ConstantValueMatcher<T>(value);
}

// Custom Op Code Matcher.
struct CustomOpCodeMatcher {
  absl::string_view custom_code;

  explicit CustomOpCodeMatcher(absl::string_view c) : custom_code(c) {}

  bool Match(const Op& op) const {
    if (op.Code() != kLiteRtOpCodeTflCustom) return false;
    auto cc = op.CustomCode();
    return cc && *cc == custom_code;
  }

  bool Match(const Tensor& tensor) const {
    auto def_op = tensor.GetDefiningOp();
    if (!def_op) return false;
    return Match(*def_op);
  }
};

inline auto m_CustomOpCode(absl::string_view custom_code) {
  return CustomOpCodeMatcher(custom_code);
}

// Custom Op Matcher (Code + Inputs).
template <typename... InputMatchers>
struct CustomOpMatcher {
  absl::string_view custom_code;
  std::tuple<InputMatchers...> input_matchers;

  explicit CustomOpMatcher(absl::string_view c, InputMatchers... m)
      : custom_code(c), input_matchers(std::move(m)...) {}

  bool Match(const Op& op) const {
    if (op.Code() != kLiteRtOpCodeTflCustom) return false;
    auto cc = op.CustomCode();
    if (!cc || *cc != custom_code) return false;

    if (op.Inputs().size() != sizeof...(InputMatchers)) return false;
    return MatchInputs(op, std::index_sequence_for<InputMatchers...>{});
  }

  bool Match(const Tensor& tensor) const {
    auto def_op = tensor.GetDefiningOp();
    if (!def_op) return false;
    return Match(*def_op);
  }

 private:
  template <size_t... I>
  bool MatchInputs(const Op& op, std::index_sequence<I...>) const {
    return (MatchInput(op, I, std::get<I>(input_matchers)) && ...);
  }
};

template <typename... Args>
auto m_CustomOp(absl::string_view custom_code, Args&&... args) {
  return CustomOpMatcher<std::decay_t<Args>...>(custom_code,
                                                std::forward<Args>(args)...);
}

// Name Matcher.
struct NameMatcher {
  absl::string_view name;
  explicit NameMatcher(absl::string_view n) : name(n) {}
  bool Match(const Tensor& tensor) const { return tensor.Name() == name; }
};

inline auto m_Name(absl::string_view name) { return NameMatcher(name); }

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_INTERNAL_LITERT_MATCHERS_H_
