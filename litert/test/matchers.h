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

#ifndef ODML_LITERT_LITERT_TEST_MATCHERS_H_
#define ODML_LITERT_LITERT_TEST_MATCHERS_H_

#include <cmath>
#include <cstddef>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/cc/internal/litert_c_types_printing.h"  // IWYU pragma: keep
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"

// Is equivalent to `ASSERT_THAT(expr, testing::litert::IsOk())`
#define LITERT_ASSERT_OK(EXPR) ASSERT_THAT((EXPR), ::testing::litert::IsOk())

// Is equivalent to `EXPECT_THAT(expr, testing::litert::IsOk())`
#define LITERT_EXPECT_OK(EXPR) EXPECT_THAT((EXPR), ::testing::litert::IsOk())

// Checks that the result of `EXPR` (a `litert::Expected` object) is not an
// error and assigns the value it holds to `DECL` as if:
// ```
// DECL = std::move(EXPR.Value());
// ```
//
// ```cpp
// Expected<Something> BuildSomething();
//
// Will fail the test if `BuildSomething()`'s returned value holds an error.
// Otherwise defines and assigns the returned `Something` value to `smth`
// ASSERT_OK_AND_ASSIGN(Something smth, BuildSomething());
// ```
#define LITERT_ASSERT_OK_AND_ASSIGN(DECL, EXPR) \
  LITERT_ASSERT_OK_AND_ASSIGN_HELPER2(__LINE__, DECL, EXPR)

#define LITERT_ASSERT_OK_AND_ASSIGN_HELPER1(LINE, DECL, EXPR) \
  auto&& litert_expected_value_or_error_##LINE = (EXPR);      \
  LITERT_ASSERT_OK(litert_expected_value_or_error_##LINE);    \
  _LITERT_STRIP_PARENS(DECL) =                                \
      ::litert::ErrorStatusBuilder::ForwardWrappedValue(      \
          litert_expected_value_or_error_##LINE)

#define LITERT_ASSERT_OK_AND_ASSIGN_HELPER2(LINE, DECL, EXPR) \
  LITERT_ASSERT_OK_AND_ASSIGN_HELPER1(LINE, DECL, EXPR)

// TODO: b/?????? - Deduplicate this from litert_macros.h when a common folder
// has been decided.
#ifndef _LITERT_STRIP_PARENS
#define _LITERT_STRIP_PARENS(X) _LITERT_ESC(_LITERT_ISH X)
#define _LITERT_ISH(...) _LITERT_ISH __VA_ARGS__
#define _LITERT_ESC(...) _LITERT_ESC_(__VA_ARGS__)
#define _LITERT_ESC_(...) _LITERT_VAN##__VA_ARGS__
#define _LITERT_VAN_LITERT_ISH
#endif

namespace testing::litert {

// Matches `litert::Expected` values that hold a success value and
// `LiteRtStatusOk`.
//
// See `IsOk()` function below for usage examples.
class IsOkMatcher {
 public:
  // Implicitly builds and wraps the matcher implementation in a GTest
  // Matcher object.
  template <class T>
  // NOLINTNEXTLINE(*-explicit-constructor): This needs to be implicit.
  operator testing::Matcher<T>() const {
    return testing::Matcher<T>(new Impl<const T&>());
  }

  template <class V>
  class Impl : public testing::MatcherInterface<V> {
    template <class T>
    bool MatchAndExplainImpl(const ::litert::Expected<T>& value,
                             testing::MatchResultListener* listener) const {
      return value.HasValue();
    }

    bool MatchAndExplainImpl(const ::litert::Unexpected& unexpected,
                             testing::MatchResultListener* listener) const {
      return false;
    }

    bool MatchAndExplainImpl(const ::litert::Error& e,
                             testing::MatchResultListener* listener) const {
      return false;
    }

    bool MatchAndExplainImpl(const LiteRtStatus& status,
                             testing::MatchResultListener* listener) const {
      if (status != kLiteRtStatusOk) {
        *listener << "status is " << LiteRtGetStatusString(status);
        return false;
      }
      return true;
    }

    bool MatchAndExplainImpl(const absl::Status& value,
                             testing::MatchResultListener* listener) const {
      return value.ok();
    }

    template <class T>
    bool MatchAndExplainImpl(const absl::StatusOr<T>& value,
                             testing::MatchResultListener* listener) const {
      return value.ok();
    }

   public:
    using is_gtest_matcher = void;

    bool MatchAndExplain(
        V value, testing::MatchResultListener* listener) const override {
      return MatchAndExplainImpl(value, listener);
    }

    void DescribeTo(std::ostream* os) const override {
      if (os) {
        *os << "is ok.";
      }
    }

    void DescribeNegationTo(std::ostream* os) const override {
      if (os) {
        *os << "is not ok.";
      }
    }
  };
};

// Matches `litert::Expected` values that hold a success value and
// `LiteRtStatusOk`.
//
// Note: you might want to use the convenience macros:
//   - `LITERT_EXPECT_OK(expr)`
//   - `LITERT_ASSERT_OK(expr)`
//   - `ASSERT_OK_AND_ASSIGN(type var, expr)`
//
// ```cpp
// LiteRtStatus DoSomething();
//
// // Will fail the test if DoSomething() doesn't return kLiteRtStatusOk.
// EXPECT_THAT(DoSomething(), IsOk());
// ```
//
// This also works for `Expected` objects.
//
// Note: You probably want `ASSERT_OK_AND_ASSIGN` when working with `Expected`.
//
// ```cpp
// Expected<Something> BuildSomething();
//
// // Will fail the test if BuildSomething()'s returned value holds an error.
// // Note that the returned value is discarded.
// EXPECT_THAT(BuildSomething(), IsOk());
// ```
inline IsOkMatcher IsOk() { return IsOkMatcher(); }

// Matches `litert::Expected` values that hold a value and which value matches
// `matcher`.
//
// ```cpp
// Expected<Something> BuildSomething();
//
// // Will fail the test if BuildSomething()'s returned value holds an error or
// // if the value doesn't match `Eq(expected_something)`.
// //
// // Note that the returned value is discarded.
// EXPECT_THAT(BuildSomething(), IsOkAndHolds(Eq(expected_something)));
// ```
MATCHER_P(IsOkAndHolds, matcher, "") {
  return testing::ExplainMatchResult(testing::litert::IsOk(), arg,
                                     result_listener) &&
         testing::ExplainMatchResult(matcher, arg.Value(), result_listener);
}

// Matches `litert::Expected` values that hold an error and
// `LiteRtStatusError*` values.
//
// See `IsError(...)` functions below for usage examples.
class IsErrorMatcher {
 public:
  IsErrorMatcher(std::optional<LiteRtStatus> status,
                 std::optional<std::string> msg)
      : impl_(status, msg) {}

  // Implicitly builds and wraps the matcher implementation in a GTest
  // Matcher object.
  template <class T>
  // NOLINTNEXTLINE(*-explicit-constructor): This needs to be implicit.
  operator testing::Matcher<T>() const {
    return testing::Matcher<T>(new Impl<const T&>(impl_));
  }

 private:
  class ImplBase {
   public:
    ImplBase() = default;

    explicit ImplBase(std::optional<LiteRtStatus> status,
                      std::optional<std::string> msg)
        : status_(status), msg_(std::move(msg)) {};

   protected:
    bool MatchAndExplainImpl(const LiteRtStatus status,
                             const absl::string_view msg,
                             testing::MatchResultListener* listener) const {
      if (status == kLiteRtStatusOk ||
          (status_.has_value() && status != status_.value())) {
        if (listener) {
          *listener << "status doesn't match";
        }
        return false;
      }
      if (msg_.has_value() && msg != msg_.value()) {
        if (listener) {
          *listener << "message doesn't match";
        }
        return false;
      }
      return true;
    }

    template <class T>
    bool MatchAndExplainImpl(const ::litert::Expected<T>& value,
                             testing::MatchResultListener* listener) const {
      if (value.HasValue()) {
        *listener << "expected holds a value (but should hold an error)";
        return false;
      }
      return MatchAndExplainImpl(value.Error(), listener);
    }

    bool MatchAndExplainImpl(const ::litert::Unexpected& e,
                             testing::MatchResultListener* listener) const {
      return MatchAndExplainImpl(e.Error().Status(), e.Error().Message(),
                                 listener);
    }

    bool MatchAndExplainImpl(const ::litert::Error& e,
                             testing::MatchResultListener* listener) const {
      return MatchAndExplainImpl(e.Status(), e.Message(), listener);
    }

    bool MatchAndExplainImpl(const LiteRtStatus& status,
                             testing::MatchResultListener* listener) const {
      return MatchAndExplainImpl(status, {}, listener);
    }

    void DescribeImpl(std::ostream* os, const bool negation) const {
      if (os) {
        *os << "is" << (negation ? " not" : "") << " an error";
        const char* sep = " with ";
        if (status_.has_value()) {
          *os << sep << "status " << LiteRtGetStatusString(status_.value());
          sep = " and ";
        }
        if (msg_.has_value()) {
          *os << sep << "message matching: '" << msg_.value() << "'";
        }
        *os << ".";
      }
    }

   private:
    std::optional<LiteRtStatus> status_;
    std::optional<std::string> msg_;
  };

  template <class V>
  class Impl : public testing::MatcherInterface<V>, ImplBase {
   public:
    using is_gtest_matcher = void;

    Impl() = default;
    explicit Impl(const ImplBase& base) : ImplBase(base) {}

    bool MatchAndExplain(
        V value, testing::MatchResultListener* listener) const override {
      return MatchAndExplainImpl(value, listener);
    }

    void DescribeTo(std::ostream* os) const override {
      DescribeImpl(os, /*negation=*/false);
    }

    void DescribeNegationTo(std::ostream* os) const override {
      DescribeImpl(os, /*negation=*/true);
    }
  };

  ImplBase impl_;
};

// Matches `litert::Expected`, `litert::Unexpected`, `litert::Error` and
// `LiteRtStatus` values that hold an error.
//
// Note: This will always match `true` for `litert::Unexpected` and
// `litert::Error`. This can be useful to test template code that might always
// return an error for certain specialisations.
//
// ```cpp
// LiteRtStatus DoSomething();
//
// // Will fail the test if `DoSomething()` returns `kLiteRtStatusOk`.
// EXPECT_THAT(DoSomething(), IsError());
// ```
//
// This also works for `Expected` objects.
//
// ```cpp
// Expected<Something> BuildSomething();
//
// // Will fail the test if BuildSomething()'s returned object holds a value.
// EXPECT_THAT(BuildSomething(), IsError());
// ```
inline IsErrorMatcher IsError() {
  return IsErrorMatcher(/*status=*/std::nullopt, /*msg=*/std::nullopt);
}

// Matches `litert::Expected`, `litert::Unexpected`, `litert::Error` and
// `LiteRtStatus` values that hold a specific error status.
//
// ```cpp
// Expected<Something> BuildSomething();
//
// // Will fail the test if BuildSomething()'s returned object holds a value or
// // if the error status is not `kLiteRtStatusErrorSystemError`.
// EXPECT_THAT(BuildSomething(), IsError(kLiteRtStatusErrorSystemError));
// ```
inline IsErrorMatcher IsError(LiteRtStatus status) {
  return IsErrorMatcher(status, /*msg=*/std::nullopt);
}

// Matches `litert::Expected` and `LiteRtStatus` values that have a specific
// error status and error message.
//
// Warning: This will always return `false` for `LiteRtStatus` objects as those
// do not convey a message.
//
// ```cpp
// Expected<Something> BuildSomething();
//
// // Will fail the test if BuildSomething()'s returned object holds a value.
// EXPECT_THAT(BuildSomething(), IsError(kLiteRtStatusErrorSystemError,
//                                       "System is not initialised"));
// ```
inline IsErrorMatcher IsError(LiteRtStatus status, std::string msg) {
  return IsErrorMatcher(status, std::move(msg));
}

}  // namespace testing::litert

// Teaches GTest how to print LiteRtStatus enum values.
//
// LiteRtSTatus lives in the global namespace. We try to avoid conflict by only
// defining this function in this file which is only pulled for tests.
inline void PrintTo(const LiteRtStatus status, std::ostream* os) {
  *os << LiteRtGetStatusString(status);
}

// GTest doesn't use `AbslStringify` if `GTEST_USE_ABSL` is not defined. This
// provides a fallback implementation.
//
// This is defined here instead of with `litert::Expected` because those
// functions should only be used for testing.
#if defined(LITERT_DEFINE_GTEST_STATUS_PRINTER) && !defined(GTEST_USE_ABSL)

// GTest documentation explicitly states that functions the those below must
// live in the same namespace as the classes they are used with so that GTest
// can find them through ADL.
namespace litert {

inline void PrintTo(const Error& error, std::ostream* os) {
  *os << absl::StrFormat("%v", error);
}

inline void PrintTo(const Unexpected& unexpected, std::ostream* os) {
  *os << absl::StrFormat("%v", unexpected);
}

template <class T>
void PrintTo(const Expected<T>& expected, std::ostream* os) {
  *os << absl::StrFormat("%v", expected);
}

}  // namespace litert

#endif

namespace testing::litert {

// Helper for providing polymorphic matcher impl in the below classes.
template <typename ImplBase, class V>
class PolyImpl : public ::testing::MatcherInterface<V>, ImplBase {
 private:
  using MatchResultListener = ::testing::MatchResultListener;

 public:
  using is_gtest_matcher = void;

  PolyImpl() = default;
  explicit PolyImpl(const ImplBase& base) : ImplBase(base) {}

  bool MatchAndExplain(const V& value,
                       MatchResultListener* listener) const override {
    return this->MatchAndExplainImpl(value, listener);
  }

  void DescribeTo(std::ostream* os) const override { this->DescribeToImpl(os); }

  void DescribeNegationTo(std::ostream* os) const override {
    this->DescribeNegationToImpl(os);
  }
};

// Polymorphic matcher for matching the element type for element type
// carrying types.
class HasElementTypeMatcher {
 private:
  template <typename T>
  using Matcher = ::testing::Matcher<T>;
  using MatchResultListener = ::testing::MatchResultListener;

 public:
  // Implicitly builds and wraps the matcher implementation in a GTest
  // Matcher object.
  template <class T>
  // NOLINTNEXTLINE(*-explicit-constructor): This needs to be implicit.
  operator Matcher<T>() const {
    return Matcher<T>(new Impl<const T&>(impl_));
  }

  explicit HasElementTypeMatcher(LiteRtElementType type) : impl_(type) {}

  class ImplBase {
   public:
    using is_gtest_matcher = void;

    ImplBase() = default;

    explicit ImplBase(LiteRtElementType type) : type_(type) {}

    void DescribeToImpl(std::ostream* os) const {
      if (os) {
        *os << absl::StreamFormat("has element type %v", type_);
      }
    }

    void DescribeNegationToImpl(std::ostream* os) const {
      if (os) {
        *os << absl::StreamFormat("does not have element type %v", type_);
      }
    }

   protected:
    bool MatchAndExplainImpl(const LiteRtRankedTensorType& value,
                             MatchResultListener* listener) const {
      return value.element_type == type_;
    }

    bool MatchAndExplainImpl(const ::litert::RankedTensorType& value,
                             MatchResultListener* listener) const {
      return MatchAndExplainImpl(LiteRtRankedTensorType(value), listener);
    }

   private:
    LiteRtElementType type_;
  };

  template <class V>
  using Impl = PolyImpl<ImplBase, V>;

 private:
  ImplBase impl_;
};

// Polymorphic matcher for matching the dims for dims carrying types.
class HasDimsMatcher {
 private:
  template <typename T>
  using Matcher = ::testing::Matcher<T>;
  using MatchResultListener = ::testing::MatchResultListener;

 public:
  // Implicitly builds and wraps the matcher implementation in a GTest
  // Matcher object.
  template <class T>
  // NOLINTNEXTLINE(*-explicit-constructor): This needs to be implicit.
  operator Matcher<T>() const {
    return Matcher<T>(new Impl<const T&>(impl_));
  }

  explicit HasDimsMatcher(absl::Span<const int> dims) : impl_(dims) {}

  class ImplBase {
   public:
    using is_gtest_matcher = void;

    ImplBase() = default;

    explicit ImplBase(absl::Span<const int> dims)
        : dims_str_(DimsStr(absl::MakeConstSpan(dims))),
          dims_(dims.cbegin(), dims.cend()) {}

    void DescribeToImpl(std::ostream* os) const {
      if (os) {
        *os << absl::StreamFormat("has dims %s", dims_str_);
      }
    }

    void DescribeNegationToImpl(std::ostream* os) const {
      if (os) {
        *os << absl::StreamFormat("does not have dims %s", dims_str_);
      }
    }

   protected:
    bool MatchAndExplainImpl(const LiteRtRankedTensorType& value,
                             MatchResultListener* listener) const {
      return absl::MakeConstSpan(value.layout.dimensions, value.layout.rank) ==
             dims_;
    }

    bool MatchAndExplainImpl(const ::litert::RankedTensorType& value,
                             MatchResultListener* listener) const {
      return MatchAndExplainImpl(LiteRtRankedTensorType(value), listener);
    }

   private:
    static std::string DimsStr(absl::Span<const int> dims) {
      return absl::StrFormat("[%s]", absl::StrJoin(dims, ", "));
    }

    const std::string dims_str_;
    std::vector<int> dims_;
  };

  template <class V>
  using Impl = PolyImpl<ImplBase, V>;

 private:
  ImplBase impl_;
};

// Matches the element type if type matched carries it.
inline auto HasTypeAspect(LiteRtElementType type) {
  return HasElementTypeMatcher(type);
}

// Matches the element type if type matched carries it.
inline auto HasTypeAspect(::litert::ElementType type) {
  return HasElementTypeMatcher(static_cast<LiteRtElementType>(type));
}

// Matches the dims if type matched  carries it.
inline auto HasTypeAspect(absl::Span<const int> dims) {
  return HasDimsMatcher(dims);
}

// Matches the element type and dims if matched type carries them.
template <typename ElementTy>
auto HasTypeAspect(ElementTy ty, absl::Span<const int> dims) {
  return ::testing::AllOf(HasTypeAspect(dims), HasTypeAspect(ty));
}

// Checks that the mean squared error between two spans is less than the
// provided tolerance.
//
// Example:
//   std::vector<float> v = {1.0f, 1.0f};
//   std::vector<float> u = {1.0f + 31e-4, 1.0f + 31e-4};
//   EXPECT_THAT(absl::MakeConstSpan(v),
//   MeanSquaredError(absl::MakeConstSpan(u)));
template <typename Exp>
auto MeanSquaredErrorLt(const Exp& expected, double tol = 1e-5,
                        double* dump_mse = nullptr) {
  auto mse = [expected, dump_mse](const auto& actual) -> double {
    double err = 0.0;
    auto exp_begin = expected.cbegin();
    auto actual_begin = actual.cbegin();
    for (size_t i = 0; i < std::size(expected); ++i) {
      double actual_val = static_cast<double>(*actual_begin++);
      double expected_val = static_cast<double>(*exp_begin++);
      err += std::pow(actual_val - expected_val, 2);
    }
    const auto res = err / static_cast<double>(std::size(expected));
    if (dump_mse) {
      *dump_mse = res;
    }
    return res;
  };
  return ::testing::AllOf(::testing::SizeIs(::testing::Gt(0)),
                          ::testing::SizeIs(std::size(expected)),
                          ::testing::ResultOf(mse, ::testing::Le(tol)));
}

}  // namespace testing::litert

#endif  // ODML_LITERT_LITERT_TEST_MATCHERS_H_
