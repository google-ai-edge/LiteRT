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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_UTILS_MATCHERS_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_UTILS_MATCHERS_H_

#include <ostream>
#include <type_traits>
#include <utility>  // IWYU pragma: keep

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "tensor/utils/macros.h"

namespace litert::tensor {

// Checks that the result of `EXPR` (a `absl::StatusOr` object) is not an
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
#define LRT_TENSOR_ASSERT_OK_AND_ASSIGN(DECL, EXPR) \
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN_HELPER2(__LINE__, DECL, EXPR)

#define LRT_TENSOR_ASSERT_OK_AND_ASSIGN_HELPER1(LINE, DECL, EXPR) \
  auto&& litert_expected_value_or_error_##LINE = (EXPR);          \
  ASSERT_TRUE(litert_expected_value_or_error_##LINE.ok())         \
      << litert_expected_value_or_error_##LINE.status();          \
  _LRT_TENSOR_STRIP_PARENS(DECL) =                                \
      ::litert::tensor::ErrorStatusBuilder::ForwardWrappedValue(  \
          litert_expected_value_or_error_##LINE)

#define LRT_TENSOR_ASSERT_OK_AND_ASSIGN_HELPER2(LINE, DECL, EXPR) \
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN_HELPER1(LINE, DECL, EXPR)

}  // namespace litert::tensor

namespace testing::tensor {

class IsOkMatcher {
 public:
  template <class T>
  operator ::testing::Matcher<T>() const {  // NOLINT(*-explicit-constructor)
    return ::testing::Matcher<T>(new Impl<const T&>());
  }

  template <class V>
  class Impl : public ::testing::MatcherInterface<V> {
   public:
    using is_gtest_matcher = void;

    bool MatchAndExplain(
        V value, ::testing::MatchResultListener* listener) const override {
      return MatchAndExplainImpl(value, listener);
    }

    void DescribeTo(std::ostream* os) const override { *os << "is ok."; }

    void DescribeNegationTo(std::ostream* os) const override {
      *os << "is not ok.";
    }

   private:
    bool MatchAndExplainImpl(
        const absl::Status& value,
        ::testing::MatchResultListener* listener) const {
      if (!value.ok()) {
        *listener << value;
        return false;
      }
      return true;
    }

    template <class T>
    bool MatchAndExplainImpl(
        const absl::StatusOr<T>& value,
        ::testing::MatchResultListener* listener) const {
      if (!value.ok()) {
        *listener << value.status();
        return false;
      }
      return true;
    }
  };
};

inline IsOkMatcher IsOk() { return IsOkMatcher(); }

namespace detail {

template <class T>
T& ReadStatusValue(absl::StatusOr<T>& status_or) {
  return status_or.value();
}

template <class T>
const T& ReadStatusValue(const absl::StatusOr<T>& status_or) {
  return status_or.value();
}

struct NoValue {};
inline NoValue ReadStatusValue(...) { return {}; }

template <typename T, typename MatcherType>
bool MatchOkAndHolds(const T& arg, const MatcherType& matcher,
                     ::testing::MatchResultListener* listener) {
  if (!::testing::ExplainMatchResult(::testing::tensor::IsOk(), arg,
                                     listener)) {
    return false;
  }
  using ValueType = decltype(ReadStatusValue(arg));
  if constexpr (!std::is_same_v<ValueType, NoValue>) {
    return ::testing::ExplainMatchResult(matcher, ReadStatusValue(arg),
                                         listener);
  } else {
    return false;
  }
}

}  // namespace detail

MATCHER_P(IsOkAndHolds, matcher, "") {
  return detail::MatchOkAndHolds(arg, matcher, result_listener);
}

}  // namespace testing::tensor

namespace litert::tensor {

inline auto IsOk() { return ::testing::tensor::IsOk(); }

template <class Matcher>
auto IsOkAndHolds(Matcher&& matcher) {
  return ::testing::tensor::IsOkAndHolds(std::forward<Matcher>(matcher));
}

}  // namespace litert::tensor


#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_UTILS_MATCHERS_H_
