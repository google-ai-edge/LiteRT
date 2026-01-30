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
#include <tuple>
#include <type_traits>
#include <utility>

#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_extended_model.h"

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
struct OpCodeMatcher {
  LiteRtOpCode code;
  bool Match(const Op& op) const { return op.Code() == code; }
  bool Match(const Tensor& tensor) const {
    auto def_op = tensor.GetDefiningOp();
    if (!def_op) return false;
    return Match(*def_op);
  }
};

inline auto m_OpCode(LiteRtOpCode code) { return OpCodeMatcher{code}; }

// OpMatcher implementation.
template <typename... InputMatchers>
struct OpMatcher {
  LiteRtOpCode code;
  std::tuple<InputMatchers...> input_matchers;

  explicit OpMatcher(LiteRtOpCode c, InputMatchers... m)
      : code(c), input_matchers(std::move(m)...) {}

  // Match against an Op.
  bool Match(const Op& op) const {
    if (op.Code() != code) return false;
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
template <typename... Args>
auto m_Op(LiteRtOpCode code, Args&&... args) {
  return OpMatcher<std::decay_t<Args>...>(code, std::forward<Args>(args)...);
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
    if (sub_matcher.Match(val)) {
      if (storage) {
        // If T is Op and U is Tensor, we might want to capture the defining Op?
        // Or if T is Op and U is Op, simple assignment.
        // Let's handle the specific case where we match a Tensor but capture
        // the defining Op.
        if constexpr (std::is_same_v<T, Op> && std::is_same_v<U, Tensor>) {
          auto def_op = val.GetDefiningOp();
          if (def_op) *storage = std::move(*def_op);
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
    return false;
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

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_INTERNAL_LITERT_MATCHERS_H_
