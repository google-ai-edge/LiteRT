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
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/internal/litert_op_options.h"

namespace litert {

// Interface for tracing match operations.
class MatchTracer {
 public:
  virtual ~MatchTracer() = default;

  // Called when a matcher fails.
  // matcher_name: The label of the matcher that failed.
  // reason: A descriptive reason for the failure.
  virtual void LogFailure(absl::string_view matcher_name,
                          absl::string_view reason) = 0;

  // Called before a matcher starts matching.
  // scope_name: The label of the matcher entering scope.
  virtual void PushScope(absl::string_view scope_name) {}

  // Called after a matcher finishes matching (regardless of success/failure).
  virtual void PopScope() {}
};

// A tracer that captures logs and can dump them to LITERT_LOG.
class LoggingMatchTracer : public MatchTracer {
 public:
  struct LogEntry {
    std::string type;    // "Fail" or "Start"
    std::string name;    // Matcher name/label
    std::string reason;  // Failure reason (empty for Start)
    int depth;           // Depth level
  };

  LoggingMatchTracer() = default;

  void LogFailure(absl::string_view matcher_name,
                  absl::string_view reason) override {
    logs_.push_back(
        {"Fail", std::string(matcher_name), std::string(reason), depth_});
  }

  void PushScope(absl::string_view scope_name) override {
    logs_.push_back({"Start", std::string(scope_name), "", depth_});
    depth_++;
  }

  void PopScope() override {
    if (depth_ > 0) depth_--;
  }

  void LogToError() const {
    for (const auto& log : logs_) {
      std::string indent(log.depth * 2, ' ');

      if (log.type == "Start") {
        LITERT_LOG(LITERT_INFO, "%s[%s] %s", indent.c_str(), log.type.c_str(),
                   log.name.c_str());
      } else {
        LITERT_LOG(LITERT_INFO, "%s[%s] %s: %s", indent.c_str(),
                   log.type.c_str(), log.name.c_str(), log.reason.c_str());
      }
    }
  }

  const std::vector<LogEntry>& logs() const { return logs_; }

 private:
  int depth_ = 0;
  std::vector<LogEntry> logs_;
};

// Main Match entry point.
// Matches val against matcher m.
template <typename T, typename Matcher>
bool Match(const T& val, const Matcher& m, MatchTracer* tracer = nullptr) {
  return m.Match(val, tracer);
}

// Debug Match entry point.
// Uses a LoggingMatchTracer to print debug info to LITERT_LOG.
// log_depth_greater_than: Minimum log size required to print logs.
//                         Useful to filter out noise (short/shallow matches).
template <typename T, typename Matcher>
bool DebugMatch(const T& val, const Matcher& m,
                size_t log_depth_greater_than = 0) {
  LoggingMatchTracer tracer;
  bool res = m.Match(val, &tracer);
  if (tracer.logs().size() > log_depth_greater_than) {
    tracer.LogToError();
  }
  return res;
}

// Helper to match input at index I.
template <typename OpType, typename Matcher>
bool MatchInput(const OpType& op, size_t index, const Matcher& m,
                MatchTracer* tracer = nullptr) {
  auto input = op.Input(index);
  if (!input) {
    if (tracer)
      tracer->LogFailure("MatchInput", "Input index out of bounds or null");
    return false;
  }
  if (tracer) {
    std::string scope = "Input[" + std::to_string(index) + "]";
    tracer->PushScope(scope);
    bool res = m.Match(*input, tracer);
    tracer->PopScope();
    return res;
  }
  return m.Match(*input, nullptr);
}

// --- Helpers for optional label parsing ---

template <typename T>
struct is_string_like : std::false_type {};
template <>
struct is_string_like<const char*> : std::true_type {};
template <>
struct is_string_like<char*> : std::true_type {};
template <>
struct is_string_like<std::string> : std::true_type {};
template <>
struct is_string_like<absl::string_view> : std::true_type {};

template <typename T>
constexpr bool is_string_like_v =
    is_string_like<std::decay_t<std::remove_reference_t<T>>>::value;

// Helper to detect if the last argument in a pack is string-like.
template <typename... Args>
struct last_arg_is_string_like;

template <>
struct last_arg_is_string_like<> : std::false_type {};

template <typename T>
struct last_arg_is_string_like<T>
    : std::integral_constant<bool, is_string_like_v<T>> {};

template <typename Head, typename... Tail>
struct last_arg_is_string_like<Head, Tail...>
    : last_arg_is_string_like<Tail...> {};

template <typename... Args>
constexpr bool last_arg_is_string_like_v =
    last_arg_is_string_like<Args...>::value;

// --- Matchers ---

// OpCode matcher.
template <LiteRtOpCode Code>
struct OpCodeMatcher {
  absl::string_view label;
  explicit OpCodeMatcher(absl::string_view l) : label(l) {}

  bool Match(const Op& op, MatchTracer* tracer = nullptr) const {
    if (tracer) tracer->PushScope(label);
    if (op.Code() != Code) {
      if (tracer) {
        tracer->LogFailure(label, "OpCode mismatch");
        tracer->PopScope();
      }
      return false;
    }
    if (tracer) tracer->PopScope();
    return true;
  }
  bool Match(const Tensor& tensor, MatchTracer* tracer = nullptr) const {
    if (tracer) tracer->PushScope(label);
    auto def_op = tensor.GetDefiningOp();
    if (!def_op) {
      if (tracer) {
        tracer->LogFailure(label, "Tensor has no defining op");
        tracer->PopScope();
      }
      return false;
    }
    bool res = Match(*def_op, tracer);
    if (tracer) tracer->PopScope();
    return res;
  }
};

template <LiteRtOpCode Code>
inline auto m_OpCode(absl::string_view label = "OpCodeMatcher") {
  return OpCodeMatcher<Code>(label);
}

// OpMatcher implementation.
template <LiteRtOpCode Code, typename... InputMatchers>
struct OpMatcher {
  absl::string_view label;
  std::tuple<InputMatchers...> input_matchers;

  explicit OpMatcher(absl::string_view l, InputMatchers... m)
      : label(l), input_matchers(std::move(m)...) {}

  bool Match(const Op& op, MatchTracer* tracer = nullptr) const {
    if (tracer) tracer->PushScope(label);
    if (op.Code() != Code) {
      if (tracer) {
        tracer->LogFailure(label, "OpCode mismatch");
        tracer->PopScope();
      }
      return false;
    }
    if (op.Inputs().size() != sizeof...(InputMatchers)) {
      if (tracer) {
        tracer->LogFailure(label, "Input count mismatch");
        tracer->PopScope();
      }
      return false;
    }

    bool res =
        MatchInputs(op, tracer, std::index_sequence_for<InputMatchers...>{});
    if (tracer) tracer->PopScope();
    return res;
  }

  bool Match(const Tensor& tensor, MatchTracer* tracer = nullptr) const {
    if (tracer) tracer->PushScope(label);
    auto def_op = tensor.GetDefiningOp();
    if (!def_op) {
      if (tracer) {
        tracer->LogFailure(label, "Tensor has no defining op");
        tracer->PopScope();
      }
      return false;
    }
    bool res = Match(*def_op, tracer);
    if (tracer) tracer->PopScope();
    return res;
  }

 private:
  template <size_t... I>
  bool MatchInputs(const Op& op, MatchTracer* tracer,
                   std::index_sequence<I...>) const {
    return (MatchInput(op, I, std::get<I>(input_matchers), tracer) && ...);
  }
};

// Helper struct to build OpMatcher, peeling off the last argument if it's a
// label.
namespace internal {

// Helper to unpack N-1 args from tuple.
template <LiteRtOpCode Code, typename Tuple, size_t... I>
auto MakeOpMatcherImpl(absl::string_view label, Tuple&& tuple,
                       std::index_sequence<I...>) {
  return OpMatcher<
      Code, std::decay_t<std::tuple_element_t<I, std::decay_t<Tuple>>>...>(
      label, std::get<I>(std::forward<Tuple>(tuple))...);
}

// Case 1: Last arg is string-like.
template <LiteRtOpCode Code, typename... Args>
auto MakeOpMatcherHelper(std::true_type /* last_is_label */, Args&&... args) {
  // We need to separate the last arg.
  auto tuple = std::forward_as_tuple(std::forward<Args>(args)...);
  constexpr size_t N = sizeof...(Args);
  auto label = std::get<N - 1>(tuple);

  return MakeOpMatcherImpl<Code>(absl::string_view(label), std::move(tuple),
                                 std::make_index_sequence<N - 1>{});
}

// Case 2: Last arg is NOT string-like.
template <LiteRtOpCode Code, typename... Args>
auto MakeOpMatcherHelper(std::false_type /* last_is_label */, Args&&... args) {
  return OpMatcher<Code, std::decay_t<Args>...>("OpMatcher",
                                                std::forward<Args>(args)...);
}

}  // namespace internal

template <LiteRtOpCode Code, typename... Args>
auto m_Op(Args&&... args) {
  // Edge case: sizeof...(Args) == 0 handled by default param in internal helper
  // if implemented, but here we can just dispatch.
  if constexpr (sizeof...(Args) == 0) {
    return OpMatcher<Code>("OpMatcher");
  } else {
    return internal::MakeOpMatcherHelper<Code>(
        std::integral_constant<bool, last_arg_is_string_like_v<Args...>>{},
        std::forward<Args>(args)...);
  }
}

// Op Matcher allowing extra trailing inputs.
template <LiteRtOpCode Code, typename... InputMatchers>
struct OpVariadicMatcher {
  absl::string_view label;
  std::tuple<InputMatchers...> input_matchers;

  explicit OpVariadicMatcher(absl::string_view l, InputMatchers... m)
      : label(l), input_matchers(std::move(m)...) {}

  bool Match(const Op& op, MatchTracer* tracer = nullptr) const {
    if (tracer) tracer->PushScope(label);
    if (op.Code() != Code) {
      if (tracer) {
        tracer->LogFailure(label, "OpCode mismatch");
        tracer->PopScope();
      }
      return false;
    }
    if (op.Inputs().size() < sizeof...(InputMatchers)) {
      if (tracer) {
        tracer->LogFailure(label, "Input count mismatch (insufficient inputs)");
        tracer->PopScope();
      }
      return false;
    }

    bool res =
        MatchInputs(op, tracer, std::index_sequence_for<InputMatchers...>{});
    if (tracer) tracer->PopScope();
    return res;
  }

 private:
  template <size_t... I>
  bool MatchInputs(const Op& op, MatchTracer* tracer,
                   std::index_sequence<I...>) const {
    return (MatchInput(op, I, std::get<I>(input_matchers), tracer) && ...);
  }
};

namespace internal {
template <LiteRtOpCode Code, typename Tuple, size_t... I>
auto MakeOpVariadicMatcherImpl(absl::string_view label, Tuple&& tuple,
                               std::index_sequence<I...>) {
  return OpVariadicMatcher<
      Code, std::decay_t<std::tuple_element_t<I, std::decay_t<Tuple>>>...>(
      label, std::get<I>(std::forward<Tuple>(tuple))...);
}

template <LiteRtOpCode Code, typename... Args>
auto MakeOpVariadicMatcherHelper(std::true_type, Args&&... args) {
  auto tuple = std::forward_as_tuple(std::forward<Args>(args)...);
  constexpr size_t N = sizeof...(Args);
  auto label = std::get<N - 1>(tuple);
  return MakeOpVariadicMatcherImpl<Code>(absl::string_view(label),
                                         std::move(tuple),
                                         std::make_index_sequence<N - 1>{});
}

template <LiteRtOpCode Code, typename... Args>
auto MakeOpVariadicMatcherHelper(std::false_type, Args&&... args) {
  return OpVariadicMatcher<Code, std::decay_t<Args>...>(
      "OpVariadicMatcher", std::forward<Args>(args)...);
}
}  // namespace internal

template <LiteRtOpCode Code, typename... Args>
auto m_OpVariadic(Args&&... args) {
  if constexpr (sizeof...(Args) == 0) {
    return OpVariadicMatcher<Code>("OpVariadicMatcher");
  } else {
    return internal::MakeOpVariadicMatcherHelper<Code>(
        std::integral_constant<bool, last_arg_is_string_like_v<Args...>>{},
        std::forward<Args>(args)...);
  }
}

// Commutative Op Matcher (e.g. Add, Mul).
template <LiteRtOpCode Code, typename Matcher1, typename Matcher2>
struct CommutativeOpMatcher {
  absl::string_view label;
  Matcher1 m1;
  Matcher2 m2;

  CommutativeOpMatcher(absl::string_view l, Matcher1 matcher1,
                       Matcher2 matcher2)
      : label(l), m1(std::move(matcher1)), m2(std::move(matcher2)) {}

  bool Match(const Op& op, MatchTracer* tracer = nullptr) const {
    if (tracer) tracer->PushScope(label);
    if (op.Code() != Code) {
      if (tracer) {
        tracer->LogFailure(label, "OpCode mismatch");
        tracer->PopScope();
      }
      return false;
    }
    if (op.Inputs().size() != 2) {
      if (tracer) {
        tracer->LogFailure(label, "Input count mismatch (expected 2)");
        tracer->PopScope();
      }
      return false;
    }

    auto in0 = op.Input(0);
    auto in1 = op.Input(1);
    if (!in0 || !in1) {
      if (tracer) {
        tracer->LogFailure(label, "Inputs invalid");
        tracer->PopScope();
      }
      return false;
    }

    if (m1.Match(*in0, nullptr) && m2.Match(*in1, nullptr)) {
      if (tracer) tracer->PopScope();
      return true;
    }

    if (m1.Match(*in1, nullptr) && m2.Match(*in0, nullptr)) {
      if (tracer) tracer->PopScope();
      return true;
    }

    if (tracer) {
      tracer->LogFailure(label, "Both permutations failed");
      tracer->PopScope();
    }
    return false;
  }

  bool Match(const Tensor& tensor, MatchTracer* tracer = nullptr) const {
    if (tracer) tracer->PushScope(label);
    auto def_op = tensor.GetDefiningOp();
    if (!def_op) {
      if (tracer) {
        tracer->LogFailure(label, "Tensor has no defining op");
        tracer->PopScope();
      }
      return false;
    }
    bool res = Match(*def_op, tracer);
    if (tracer) tracer->PopScope();
    return res;
  }
};

template <LiteRtOpCode Code, typename M1, typename M2>
auto m_CommutativeOp(M1&& m1, M2&& m2,
                     absl::string_view label = "CommutativeOpMatcher") {
  return CommutativeOpMatcher<Code, std::decay_t<M1>, std::decay_t<M2>>(
      label, std::forward<M1>(m1), std::forward<M2>(m2));
}

// Options Matcher.
template <typename OptionsT, typename Pred>
struct OptionsMatcher {
  absl::string_view label;
  Pred pred;

  explicit OptionsMatcher(absl::string_view l, Pred p)
      : label(l), pred(std::move(p)) {}

  bool Match(const Op& op, MatchTracer* tracer = nullptr) const {
    if (tracer) tracer->PushScope(label);
    auto opts = GetOptionsAs<OptionsT>(op.Get());
    if (!opts) {
      if (tracer) {
        tracer->LogFailure(label, "Failed to retrieve options");
        tracer->PopScope();
      }
      return false;
    }
    if (!pred(*opts)) {
      if (tracer) {
        tracer->LogFailure(label, "Predicate returned false");
        tracer->PopScope();
      }
      return false;
    }
    if (tracer) tracer->PopScope();
    return true;
  }

  bool Match(const Tensor& tensor, MatchTracer* tracer = nullptr) const {
    if (tracer) tracer->PushScope(label);
    auto def_op = tensor.GetDefiningOp();
    if (!def_op) {
      if (tracer) {
        tracer->LogFailure(label, "Tensor has no defining op");
        tracer->PopScope();
      }
      return false;
    }
    bool res = Match(*def_op, tracer);
    if (tracer) tracer->PopScope();
    return res;
  }
};

template <typename OptionsT, typename Pred>
auto m_Options(Pred&& pred, absl::string_view label = "OptionsMatcher") {
  return OptionsMatcher<OptionsT, std::decay_t<Pred>>{label,
                                                      std::forward<Pred>(pred)};
}

// Shape matcher.
struct ShapeMatcher {
  absl::string_view label;
  std::vector<int32_t> shape;

  explicit ShapeMatcher(absl::string_view l, std::vector<int32_t> s)
      : label(l), shape(std::move(s)) {}

  bool Match(const Tensor& tensor, MatchTracer* tracer = nullptr) const {
    if (tracer) tracer->PushScope(label);
    auto ranked_type = tensor.RankedTensorType();
    if (!ranked_type) {
      if (tracer) {
        tracer->LogFailure(label, "Not a ranked tensor");
        tracer->PopScope();
      }
      return false;
    }
    auto dimensions = ranked_type->Layout().Dimensions();
    if (dimensions.size() != shape.size()) {
      if (tracer) {
        tracer->LogFailure(label, "Rank mismatch");
        tracer->PopScope();
      }
      return false;
    }
    for (size_t i = 0; i < shape.size(); ++i) {
      if (shape[i] != -1 && dimensions[i] != shape[i]) {
        if (tracer) {
          tracer->LogFailure(label, "Dimension mismatch");
          tracer->PopScope();
        }
        return false;
      }
    }
    if (tracer) tracer->PopScope();
    return true;
  }
};

inline auto m_Shape(std::vector<int32_t> shape,
                    absl::string_view label = "ShapeMatcher") {
  return ShapeMatcher(label, std::move(shape));
}

// Rank matcher.
struct RankMatcher {
  absl::string_view label;
  size_t rank;

  explicit RankMatcher(absl::string_view l, size_t r) : label(l), rank(r) {}

  bool Match(const Tensor& tensor, MatchTracer* tracer = nullptr) const {
    if (tracer) tracer->PushScope(label);
    auto ranked_type = tensor.RankedTensorType();
    if (!ranked_type) {
      if (tracer) {
        tracer->LogFailure(label, "Not a ranked tensor");
        tracer->PopScope();
      }
      return false;
    }
    if (ranked_type->Layout().Dimensions().size() != rank) {
      if (tracer) {
        tracer->LogFailure(label, "Rank mismatch");
        tracer->PopScope();
      }
      return false;
    }
    if (tracer) tracer->PopScope();
    return true;
  }
};

inline auto m_Rank(size_t rank, absl::string_view label = "RankMatcher") {
  return RankMatcher(label, rank);
}

// ElementType matcher.
struct ElementTypeMatcher {
  absl::string_view label;
  LiteRtElementType type;

  explicit ElementTypeMatcher(absl::string_view l, LiteRtElementType t)
      : label(l), type(t) {}

  bool Match(const Tensor& tensor, MatchTracer* tracer = nullptr) const {
    if (tracer) tracer->PushScope(label);
    auto ranked_type = tensor.RankedTensorType();
    if (!ranked_type) {
      if (tracer) {
        tracer->LogFailure(label, "Not a ranked tensor");
        tracer->PopScope();
      }
      return false;
    }
    if (static_cast<LiteRtElementType>(ranked_type->ElementType()) != type) {
      if (tracer) {
        tracer->LogFailure(label, "Type mismatch");
        tracer->PopScope();
      }
      return false;
    }
    if (tracer) tracer->PopScope();
    return true;
  }
};

inline auto m_ElementType(LiteRtElementType type,
                          absl::string_view label = "ElementTypeMatcher") {
  return ElementTypeMatcher(label, type);
}

// Output Index Matcher.
template <typename OpMatcher>
struct OutputIndexMatcher {
  absl::string_view label;
  size_t index;
  OpMatcher op_matcher;

  OutputIndexMatcher(absl::string_view l, size_t i, OpMatcher m)
      : label(l), index(i), op_matcher(std::move(m)) {}

  bool Match(const Tensor& tensor, MatchTracer* tracer = nullptr) const {
    if (tracer) tracer->PushScope(label);
    auto def_op_wrapper = tensor.DefiningOp();
    if (!def_op_wrapper.has_value()) {
      if (tracer) {
        tracer->LogFailure(label, "No defining op found");
        tracer->PopScope();
      }
      return false;
    }
    if (def_op_wrapper->op_output_index != index) {
      if (tracer) {
        tracer->LogFailure(label, "Index mismatch");
        tracer->PopScope();
      }
      return false;
    }

    Op op(def_op_wrapper->op);
    bool res = op_matcher.Match(op, tracer);
    if (tracer) tracer->PopScope();
    return res;
  }
};

template <typename Matcher>
auto m_OutputIndex(size_t index, Matcher&& m,
                   absl::string_view label = "OutputIndexMatcher") {
  return OutputIndexMatcher<std::decay_t<Matcher>>(label, index,
                                                   std::forward<Matcher>(m));
}

// Capture Or SameAs matcher.
template <typename T, typename SubMatcher>
struct CaptureOrSameAsMatcher {
  absl::string_view label;
  T* storage;
  SubMatcher sub_matcher;

  CaptureOrSameAsMatcher(absl::string_view l, T* s, SubMatcher m)
      : label(l), storage(s), sub_matcher(std::move(m)) {}

  template <typename U>
  bool Match(const U& val, MatchTracer* tracer = nullptr) const {
    if (tracer) tracer->PushScope(label);
    if (!storage) {
      if (tracer) {
        tracer->LogFailure(label, "Storage is null");
        tracer->PopScope();
      }
      return false;
    }

    if (storage->Get() == nullptr) {
      if (!sub_matcher.Match(val, tracer)) {
        if (tracer) tracer->PopScope();
        return false;
      }
      if constexpr (std::is_same_v<T, Op> && std::is_same_v<U, Tensor>) {
        auto def_op = val.GetDefiningOp();
        if (!def_op) {
          if (tracer) {
            tracer->LogFailure(label, "Tensor has no defining op for capture");
            tracer->PopScope();
          }
          return false;
        }
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
    } else {
      if constexpr (std::is_same_v<T, Op> && std::is_same_v<U, Tensor>) {
        auto def_op = val.GetDefiningOp();
        if (!def_op || def_op->Get() != storage->Get()) {
          if (tracer) {
            tracer->LogFailure(label,
                               "Object mismatch (Tensor's defining op != "
                               "storage)");
            tracer->PopScope();
          }
          return false;
        }
      } else {
        if (val.Get() != storage->Get()) {
          if (tracer) {
            tracer->LogFailure(label, "Object mismatch");
            tracer->PopScope();
          }
          return false;
        }
      }
    }

    if (tracer) tracer->PopScope();
    return true;
  }
};

template <typename T, typename Matcher>
auto m_CaptureOrSameAs(T* s, Matcher&& m,
                       absl::string_view label = "CaptureOrSameAsMatcher") {
  return CaptureOrSameAsMatcher<T, std::decay_t<Matcher>>(
      label, s, std::forward<Matcher>(m));
}

// Wildcard matcher for Tensor. Always true.
struct AnyTensorMatcher {
  bool Match(const Tensor&, MatchTracer* tracer = nullptr) const {
    return true;
  }
};

inline auto m_Any() { return AnyTensorMatcher{}; }

// Wildcard matcher for Op. Always true.
struct AnyOpMatcher {
  bool Match(const Op&, MatchTracer* tracer = nullptr) const { return true; }
  bool Match(const Tensor& tensor, MatchTracer* tracer = nullptr) const {
    auto def_op = tensor.GetDefiningOp();
    if (!def_op) {
      if (tracer)
        tracer->LogFailure("AnyOpMatcher", "Tensor has no defining op");
      return false;
    }
    return Match(*def_op, tracer);
  }
};

inline auto m_AnyOp() { return AnyOpMatcher{}; }

// Constant tensor matcher.
struct IsConstantMatcher {
  absl::string_view label;
  explicit IsConstantMatcher(absl::string_view l) : label(l) {}

  bool Match(const Tensor& tensor, MatchTracer* tracer = nullptr) const {
    if (tracer) tracer->PushScope(label);
    if (!tensor.IsConstant()) {
      if (tracer) {
        tracer->LogFailure(label, "Tensor is not constant");
        tracer->PopScope();
      }
      return false;
    }
    if (tracer) tracer->PopScope();
    return true;
  }
};

inline auto m_IsConstant(absl::string_view label = "IsConstantMatcher") {
  return IsConstantMatcher(label);
}

// Subgraph input tensor matcher.
struct IsSubgraphInputMatcher {
  absl::string_view label;
  explicit IsSubgraphInputMatcher(absl::string_view l) : label(l) {}

  bool Match(const Tensor& tensor, MatchTracer* tracer = nullptr) const {
    if (tracer) tracer->PushScope(label);
    if (!tensor.IsSubgraphInput()) {
      if (tracer) {
        tracer->LogFailure(label, "Tensor is not subgraph input");
        tracer->PopScope();
      }
      return false;
    }
    if (tracer) tracer->PopScope();
    return true;
  }
};

inline auto m_IsSubgraphInput(
    absl::string_view label = "IsSubgraphInputMatcher") {
  return IsSubgraphInputMatcher(label);
}

// Predicate matcher.
template <typename T, typename Pred>
struct PredicateMatcher {
  absl::string_view label;
  Pred pred;

  bool Match(const T& val, MatchTracer* tracer = nullptr) const {
    if (tracer) tracer->PushScope(label);
    bool res = pred(val);
    if (!res && tracer) {
      tracer->LogFailure(label, "Predicate returned false");
    }
    if (tracer) tracer->PopScope();
    return res;
  }
};

template <typename T, typename Pred>
auto m_Predicate(Pred&& pred, absl::string_view label = "PredicateMatcher") {
  return PredicateMatcher<T, std::decay_t<Pred>>{label,
                                                 std::forward<Pred>(pred)};
}

// Generic Custom Matcher (Lambda).
template <typename Pred>
struct CustomMatcher {
  absl::string_view label;
  Pred pred;

  explicit CustomMatcher(absl::string_view l, Pred p)
      : label(l), pred(std::move(p)) {}

  template <typename T>
  bool Match(const T& val, MatchTracer* tracer = nullptr) const {
    if (tracer) tracer->PushScope(label);
    bool res = pred(val);
    if (!res && tracer) {
      tracer->LogFailure(label, "Predicate returned false");
    }
    if (tracer) tracer->PopScope();
    return res;
  }
};

template <typename Pred>
auto m_Custom(Pred&& pred, absl::string_view label = "CustomMatcher") {
  return CustomMatcher<std::decay_t<Pred>>(label, std::forward<Pred>(pred));
}

// Logical AND matcher (AllOf).
template <typename... Matchers>
struct AllOfMatcher {
  absl::string_view label;
  std::tuple<Matchers...> matchers;

  explicit AllOfMatcher(absl::string_view l, Matchers... m)
      : label(l), matchers(std::move(m)...) {}

  template <typename T>
  bool Match(const T& val, MatchTracer* tracer = nullptr) const {
    if (tracer) tracer->PushScope(label);
    bool res = MatchImpl(val, tracer, std::index_sequence_for<Matchers...>{});
    if (tracer) tracer->PopScope();
    return res;
  }

 private:
  template <typename T, size_t... I>
  bool MatchImpl(const T& val, MatchTracer* tracer,
                 std::index_sequence<I...>) const {
    return (std::get<I>(matchers).Match(val, tracer) && ...);
  }
};

namespace internal {
template <typename Tuple, size_t... I>
auto MakeAllOfMatcherImpl(absl::string_view label, Tuple&& tuple,
                          std::index_sequence<I...>) {
  return AllOfMatcher<
      std::decay_t<std::tuple_element_t<I, std::decay_t<Tuple>>>...>(
      label, std::get<I>(std::forward<Tuple>(tuple))...);
}

template <typename... Args>
auto MakeAllOfMatcherHelper(std::true_type, Args&&... args) {
  auto tuple = std::forward_as_tuple(std::forward<Args>(args)...);
  constexpr size_t N = sizeof...(Args);
  auto label = std::get<N - 1>(tuple);
  return MakeAllOfMatcherImpl(absl::string_view(label), std::move(tuple),
                              std::make_index_sequence<N - 1>{});
}

template <typename... Args>
auto MakeAllOfMatcherHelper(std::false_type, Args&&... args) {
  return AllOfMatcher<std::decay_t<Args>...>("AllOfMatcher",
                                             std::forward<Args>(args)...);
}
}  // namespace internal

template <typename... Args>
auto m_AllOf(Args&&... args) {
  return internal::MakeAllOfMatcherHelper(
      std::integral_constant<bool, last_arg_is_string_like_v<Args...>>{},
      std::forward<Args>(args)...);
}

// Logical OR matcher (AnyOf).
template <typename... Matchers>
struct AnyOfMatcher {
  absl::string_view label;
  std::tuple<Matchers...> matchers;

  explicit AnyOfMatcher(absl::string_view l, Matchers... m)
      : label(l), matchers(std::move(m)...) {}

  template <typename T>
  bool Match(const T& val, MatchTracer* tracer = nullptr) const {
    if (tracer) tracer->PushScope(label);

    if (MatchImpl(val, nullptr, std::index_sequence_for<Matchers...>{})) {
      if (tracer) tracer->PopScope();
      return true;
    }

    if (tracer) {
      tracer->LogFailure(label, "All sub-matchers failed");
      tracer->PopScope();
    }
    return false;
  }

 private:
  template <typename T, size_t... I>
  bool MatchImpl(const T& val, MatchTracer* tracer,
                 std::index_sequence<I...>) const {
    return (std::get<I>(matchers).Match(val, tracer) || ...);
  }
};

namespace internal {
template <typename Tuple, size_t... I>
auto MakeAnyOfMatcherImpl(absl::string_view label, Tuple&& tuple,
                          std::index_sequence<I...>) {
  return AnyOfMatcher<
      std::decay_t<std::tuple_element_t<I, std::decay_t<Tuple>>>...>(
      label, std::get<I>(std::forward<Tuple>(tuple))...);
}

template <typename... Args>
auto MakeAnyOfMatcherHelper(std::true_type, Args&&... args) {
  auto tuple = std::forward_as_tuple(std::forward<Args>(args)...);
  constexpr size_t N = sizeof...(Args);
  auto label = std::get<N - 1>(tuple);
  return MakeAnyOfMatcherImpl(absl::string_view(label), std::move(tuple),
                              std::make_index_sequence<N - 1>{});
}

template <typename... Args>
auto MakeAnyOfMatcherHelper(std::false_type, Args&&... args) {
  return AnyOfMatcher<std::decay_t<Args>...>("AnyOfMatcher",
                                             std::forward<Args>(args)...);
}
}  // namespace internal

template <typename... Args>
auto m_AnyOf(Args&&... args) {
  return internal::MakeAnyOfMatcherHelper(
      std::integral_constant<bool, last_arg_is_string_like_v<Args...>>{},
      std::forward<Args>(args)...);
}

// Logical NOT matcher.
template <typename Matcher>
struct NotMatcher {
  absl::string_view label;
  Matcher matcher;

  explicit NotMatcher(absl::string_view l, Matcher m)
      : label(l), matcher(std::move(m)) {}

  template <typename T>
  bool Match(const T& val, MatchTracer* tracer = nullptr) const {
    if (tracer) tracer->PushScope(label);
    if (matcher.Match(val, nullptr)) {
      if (tracer) {
        tracer->LogFailure(label, "Sub-matcher matched (expected failure)");
        tracer->PopScope();
      }
      return false;
    }
    if (tracer) tracer->PopScope();
    return true;
  }
};

template <typename Matcher>
auto m_Not(Matcher&& m, absl::string_view label = "NotMatcher") {
  return NotMatcher<std::decay_t<Matcher>>(label, std::forward<Matcher>(m));
}

// Quantization Presence Matcher.
struct IsQuantizedMatcher {
  absl::string_view label;
  explicit IsQuantizedMatcher(absl::string_view l) : label(l) {}

  bool Match(const Tensor& tensor, MatchTracer* tracer = nullptr) const {
    if (tracer) tracer->PushScope(label);
    if (!tensor.HasQuantization()) {
      if (tracer) {
        tracer->LogFailure(label, "Tensor is not quantized");
        tracer->PopScope();
      }
      return false;
    }
    if (tracer) tracer->PopScope();
    return true;
  }
};

inline auto m_IsQuantized(absl::string_view label = "IsQuantizedMatcher") {
  return IsQuantizedMatcher(label);
}

// Quantization Type Matcher.
struct QuantizationTypeMatcher {
  absl::string_view label;
  LiteRtQuantizationTypeId type;
  explicit QuantizationTypeMatcher(absl::string_view l,
                                   LiteRtQuantizationTypeId t)
      : label(l), type(t) {}
  bool Match(const Tensor& tensor, MatchTracer* tracer = nullptr) const {
    if (tracer) tracer->PushScope(label);
    if (tensor.QTypeId() != type) {
      if (tracer) {
        tracer->LogFailure(label, "Quantization type mismatch");
        tracer->PopScope();
      }
      return false;
    }
    if (tracer) tracer->PopScope();
    return true;
  }
};

inline auto m_QType(LiteRtQuantizationTypeId type,
                    absl::string_view label = "QuantizationTypeMatcher") {
  return QuantizationTypeMatcher(label, type);
}

// User Count matcher.
struct HasUsersMatcher {
  absl::string_view label;
  size_t count;
  explicit HasUsersMatcher(absl::string_view l, size_t c)
      : label(l), count(c) {}

  bool Match(const Tensor& tensor, MatchTracer* tracer = nullptr) const {
    if (tracer) tracer->PushScope(label);
    if (tensor.Uses().size() != count) {
      if (tracer) {
        tracer->LogFailure(label, "User count mismatch");
        tracer->PopScope();
      }
      return false;
    }
    if (tracer) tracer->PopScope();
    return true;
  }
};

inline auto m_HasUsers(size_t count,
                       absl::string_view label = "HasUsersMatcher") {
  return HasUsersMatcher(label, count);
}

inline auto m_HasOneUse(absl::string_view label = "HasOneUseMatcher") {
  return m_HasUsers(1, label);
}

// Constant Scalar matcher.
template <typename T>
struct ConstantValueMatcher {
  absl::string_view label;
  T value;

  explicit ConstantValueMatcher(absl::string_view l, T v)
      : label(l), value(v) {}

  bool Match(const Tensor& tensor, MatchTracer* tracer = nullptr) const {
    if (tracer) tracer->PushScope(label);
    if (!tensor.IsConstant()) {
      if (tracer) {
        tracer->LogFailure(label, "Tensor is not constant");
        tracer->PopScope();
      }
      return false;
    }
    auto weights_data_res = tensor.WeightsData<T>();
    if (!weights_data_res) {
      if (tracer) {
        tracer->LogFailure(label, "Failed to get weights");
        tracer->PopScope();
      }
      return false;
    }
    auto weights_data = *weights_data_res;
    if (weights_data.empty()) {
      if (tracer) {
        tracer->LogFailure(label, "Weights empty");
        tracer->PopScope();
      }
      return false;
    }

    for (const auto& elem : weights_data) {
      if (elem != value) {
        if (tracer) {
          tracer->LogFailure(label, "Value mismatch");
          tracer->PopScope();
        }
        return false;
      }
    }
    if (tracer) tracer->PopScope();
    return true;
  }
};

template <typename T, typename Val>
auto m_ConstantValue(Val&& v,
                     absl::string_view label = "ConstantValueMatcher") {
  return ConstantValueMatcher<T>(label, std::forward<Val>(v));
}

// Custom Op Code Matcher.
struct CustomOpCodeMatcher {
  absl::string_view label;
  absl::string_view custom_code;

  explicit CustomOpCodeMatcher(absl::string_view l, absl::string_view c)
      : label(l), custom_code(c) {}

  bool Match(const Op& op, MatchTracer* tracer = nullptr) const {
    if (tracer) tracer->PushScope(label);
    if (op.Code() != kLiteRtOpCodeTflCustom) {
      if (tracer) {
        tracer->LogFailure(label, "Not a custom op");
        tracer->PopScope();
      }
      return false;
    }
    auto cc = op.CustomCode();
    if (!cc || *cc != custom_code) {
      if (tracer) {
        tracer->LogFailure(label, "Custom code mismatch");
        tracer->PopScope();
      }
      return false;
    }
    if (tracer) tracer->PopScope();
    return true;
  }

  bool Match(const Tensor& tensor, MatchTracer* tracer = nullptr) const {
    if (tracer) tracer->PushScope(label);
    auto def_op = tensor.GetDefiningOp();
    if (!def_op) {
      if (tracer) {
        tracer->LogFailure(label, "Tensor has no defining op");
        tracer->PopScope();
      }
      return false;
    }
    bool res = Match(*def_op, tracer);
    if (tracer) tracer->PopScope();
    return res;
  }
};

inline auto m_CustomOpCode(absl::string_view custom_code,
                           absl::string_view label = "CustomOpCodeMatcher") {
  return CustomOpCodeMatcher(label, custom_code);
}

// Custom Op Matcher (Code + Inputs).
template <typename... InputMatchers>
struct CustomOpMatcher {
  absl::string_view label;
  absl::string_view custom_code;
  std::tuple<InputMatchers...> input_matchers;

  explicit CustomOpMatcher(absl::string_view l, absl::string_view c,
                           InputMatchers... m)
      : label(l), custom_code(c), input_matchers(std::move(m)...) {}

  bool Match(const Op& op, MatchTracer* tracer = nullptr) const {
    if (tracer) tracer->PushScope(label);
    if (op.Code() != kLiteRtOpCodeTflCustom) {
      if (tracer) {
        tracer->LogFailure(label, "Not a custom op");
        tracer->PopScope();
      }
      return false;
    }
    auto cc = op.CustomCode();
    if (!cc || *cc != custom_code) {
      if (tracer) {
        tracer->LogFailure(label, "Custom code mismatch");
        tracer->PopScope();
      }
      return false;
    }

    if (op.Inputs().size() != sizeof...(InputMatchers)) {
      if (tracer) {
        tracer->LogFailure(label, "Input count mismatch");
        tracer->PopScope();
      }
      return false;
    }
    bool res =
        MatchInputs(op, tracer, std::index_sequence_for<InputMatchers...>{});
    if (tracer) tracer->PopScope();
    return res;
  }

  bool Match(const Tensor& tensor, MatchTracer* tracer = nullptr) const {
    if (tracer) tracer->PushScope(label);
    auto def_op = tensor.GetDefiningOp();
    if (!def_op) {
      if (tracer) {
        tracer->LogFailure(label, "Tensor has no defining op");
        tracer->PopScope();
      }
      return false;
    }
    bool res = Match(*def_op, tracer);
    if (tracer) tracer->PopScope();
    return res;
  }

 private:
  template <size_t... I>
  bool MatchInputs(const Op& op, MatchTracer* tracer,
                   std::index_sequence<I...>) const {
    return (MatchInput(op, I, std::get<I>(input_matchers), tracer) && ...);
  }
};

namespace internal {
template <typename Tuple, size_t... I>
auto MakeCustomOpMatcherImpl(absl::string_view label, absl::string_view code,
                             Tuple&& tuple, std::index_sequence<I...>) {
  return CustomOpMatcher<
      std::decay_t<std::tuple_element_t<I, std::decay_t<Tuple>>>...>(
      label, code, std::get<I>(std::forward<Tuple>(tuple))...);
}

template <typename... Args>
auto MakeCustomOpMatcherHelper(std::true_type, absl::string_view code,
                               Args&&... args) {
  auto tuple = std::forward_as_tuple(std::forward<Args>(args)...);
  constexpr size_t N = sizeof...(Args);
  auto label = std::get<N - 1>(tuple);
  return MakeCustomOpMatcherImpl(absl::string_view(label), code,
                                 std::move(tuple),
                                 std::make_index_sequence<N - 1>{});
}

template <typename... Args>
auto MakeCustomOpMatcherHelper(std::false_type, absl::string_view code,
                               Args&&... args) {
  return CustomOpMatcher<std::decay_t<Args>...>("CustomOpMatcher", code,
                                                std::forward<Args>(args)...);
}
}  // namespace internal

template <typename... Args>
auto m_CustomOp(absl::string_view code, Args&&... args) {
  return internal::MakeCustomOpMatcherHelper(
      std::integral_constant<bool, last_arg_is_string_like_v<Args...>>{}, code,
      std::forward<Args>(args)...);
}

// Name Matcher.
struct NameMatcher {
  absl::string_view label;
  absl::string_view name;
  explicit NameMatcher(absl::string_view l, absl::string_view n)
      : label(l), name(n) {}

  bool Match(const Tensor& tensor, MatchTracer* tracer = nullptr) const {
    if (tracer) tracer->PushScope(label);
    if (tensor.Name() != name) {
      if (tracer) {
        tracer->LogFailure(label, "Name mismatch");
        tracer->PopScope();
      }
      return false;
    }
    if (tracer) tracer->PopScope();
    return true;
  }
};

inline auto m_Name(absl::string_view name,
                   absl::string_view label = "NameMatcher") {
  return NameMatcher(label, name);
}

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_INTERNAL_LITERT_MATCHERS_H_
