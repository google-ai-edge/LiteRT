// Copyright 2024 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_UTILS_MACROS_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_UTILS_MACROS_H_

#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "third_party/odml/litert/tensor/utils/source_location.h"

// Returns the result of `expr` if it represents an error status.
//
// LRT_TENSOR_RETURN_IF_ERROR(expr); LRT_TENSOR_RETURN_IF_ERROR(expr,
// return_value);
//
//
// - `return_value` An optional custom return value in case of error. When
// specified, an `ErrorStatusBuilder` variable named `_` holding the result of
// `expr` can be used to customize the error message.
//
// By default, the return value is an `ErrorStatusBuilder` constructed from the
// result of `expr`. The error message of this builder can be customized using
// its `Log*()` functions and the `<<` operator.
#define LRT_TENSOR_RETURN_IF_ERROR(...)           \
  LRT_TENSOR_RETURN_IF_ERROR_SELECT_OVERLOAD(     \
      (__VA_ARGS__, LRT_TENSOR_RETURN_IF_ERROR_2, \
       LRT_TENSOR_RETURN_IF_ERROR_1))(__VA_ARGS__)

// Evaluates an expression that should convert to a `absl::StatusOr` object.
//
// LRT_TENSOR_ASSIGN_OR_RETURN(decl, expr)
// LRT_TENSOR_ASSIGN_OR_RETURN(decl, expr, return_value)
//
// - If the object holds a value, it move-assigns the value to `decl`.
// - If the object holds an error, it returns the error, casting it to a
//   `LiteRtStatus` if required.
//
// @param return_value An optional custom return value in case of error. When
// specified, an `ErrorStatusBuilder` variable named `_` holding the result of
// `expr` can be used to customize the error message.
//
// @code
// LRT_TENSOR_ASSIGN_OR_RETURN(decl, expr, _ << "Failed while trying to ...");
// @endcode
#define LRT_TENSOR_ASSIGN_OR_RETURN(DECL, ...)                  \
  LRT_TENSOR_ASSIGN_OR_RETURN_SELECT_OVERLOAD(                  \
      (DECL, __VA_ARGS__, LRT_TENSOR_ASSIGN_OR_RETURN_HELPER_3, \
       LRT_TENSOR_ASSIGN_OR_RETURN_HELPER_2))(                  \
      _CONCAT_NAME(expected_value_or_error_, __LINE__), DECL, __VA_ARGS__)

// Works like `LRT_TENSOR_RETURN_IF_ERROR` but aborts the process on error.
#define LRT_TENSOR_ABORT_IF_ERROR(EXPR)                      \
  if (auto status = (EXPR);                                  \
      ::litert::tensor::ErrorStatusBuilder::IsError(status)) \
  ::litert::tensor::LogBeforeAbort(::litert::tensor::ErrorStatusBuilder(status))

// Works like `LRT_TENSOR_ASSIGN_OR` but aborts the process on error.
#define LRT_TENSOR_ASSIGN_OR_ABORT(DECL, ...)                  \
  LRT_TENSOR_ASSIGN_OR_ABORT_SELECT_OVERLOAD(                  \
      (DECL, __VA_ARGS__, LRT_TENSOR_ASSIGN_OR_ABORT_HELPER_3, \
       LRT_TENSOR_ASSIGN_OR_ABORT_HELPER_2))(                  \
      _CONCAT_NAME(expected_value_or_error_, __LINE__), DECL, __VA_ARGS__)

namespace litert::tensor {

// A helper class for building and handling error statuses.
//
// This class is meant to be used with the `LRT_TENSOR_RETURN_IF_ERROR` and
// `LRT_TENSOR_ASSIGN_OR_RETURN` macros.
//
// The error message can be extended with additional information using the `<<`
// operator.
class ErrorStatusBuilder {
 public:
  // @brief Specializes this class with an implicit conversion to
  // `absl::Status` and an `IsError()` member.
  template <class Error, class CRTP = void>
  struct ErrorConversion;

  template <class T>
  explicit ErrorStatusBuilder(T&& error,
                              source_location loc = source_location::current())
      : error_(AsError(std::forward<T>(error))), loc_(loc) {}

  // NOLINTBEGIN(*-explicit-constructor): This class transparently converts to
  // `absl::Status`.
  operator absl::Status() const noexcept {
    return absl::Status(error_.code(), LogMessage());
  }

  template <class T>
  operator absl::StatusOr<T>() const noexcept {
    return operator absl::Status();
  }

  template <class T, class A = decltype(T(absl::OkStatus()))>
  operator T() const noexcept {
    return T(static_cast<absl::Status>(*this));
  }
  // NOLINTEND(*-explicit-constructor)

  void Log() { ABSL_LOG(INFO) << LogMessage(); }

  template <class T>
  static constexpr bool IsError(T&& value) {
    return ErrorConversion<std::decay_t<T>>::IsError(std::forward<T>(value));
  }

  template <class T>
  static absl::Status AsError(T&& value) {
    return ErrorConversion<std::decay_t<T>>::AsError(std::forward<T>(value));
  }

  // @brief Appends data to the error message.
  template <class T>
  ErrorStatusBuilder& operator<<(T&& val) {
    if (!extra_log_) {
      extra_log_ = std::make_unique<std::stringstream>();
    }
    *extra_log_ << static_cast<T&&>(val);
    return *this;
  }

  template <class T>
  static T&& ForwardWrappedValue(absl::StatusOr<T>& e) {
    return std::move(e).value();
  }

  template <class T>
  static T& ForwardWrappedValue(absl::StatusOr<T&>& e) {
    return e.value();
  }

  template <class T>
  static T&& ForwardWrappedValue(T&& value) {
    return ErrorConversion<std::decay_t<T>>::Forward(std::forward<T>(value));
  }

 private:
  bool ShouldLog() const noexcept {
    return (!error_.message().empty() || extra_log_);
  }

  std::string LogMessage() const {
    std::stringstream sstr;
    const char* extra_log_sep = extra_log_ ? " " : "";
    sstr << error_.message() << "\n└[" << loc_.file_name() << ":" << loc_.line()
         << "]" << extra_log_sep << (extra_log_ ? extra_log_->str() : "");
    return sstr.str();
  }

  absl::Status error_;
  source_location loc_;
  std::unique_ptr<std::stringstream> extra_log_;
};

// NOLINTBEGIN(*-explicit-constructor)
template <>
struct ErrorStatusBuilder::ErrorConversion<bool> {
  static constexpr bool IsError(bool value) { return !value; };
  static absl::Status AsError(bool value) { return absl::UnknownError(""); }
};

template <class T>
struct ErrorStatusBuilder::ErrorConversion<T*>
    : ErrorStatusBuilder::ErrorConversion<bool> {};

template <class T>
struct ErrorStatusBuilder::ErrorConversion<
    T, std::enable_if_t<std::is_arithmetic_v<T>>>
    : ErrorStatusBuilder::ErrorConversion<bool> {};

template <>
struct ErrorStatusBuilder::ErrorConversion<absl::Status> {
  static bool IsError(const absl::Status& value) { return !value.ok(); };
  static absl::Status AsError(const absl::Status& value) { return value; }
};

template <class T>
struct ErrorStatusBuilder::ErrorConversion<absl::StatusOr<T>> {
  static bool IsError(const absl::StatusOr<T>& value) { return !value.ok(); };
  static absl::Status AsError(const absl::StatusOr<T>& value) {
    return value.status();
  }
  static absl::Status AsError(absl::StatusOr<T>&& value) {
    return std::move(value.status());
  }
};
// NOLINTEND(*-explicit-constructor)

class LogBeforeAbort {
 public:
  explicit LogBeforeAbort(ErrorStatusBuilder builder)
      : builder_(std::move(builder)) {}

  ~LogBeforeAbort() {
    // Cast to a LiteRtStatus to trigger the logging mechanism.
    builder_.Log();
    std::abort();
  }

  template <class T>
  LogBeforeAbort& operator<<(T&& val) {
    builder_ << val;
    return *this;
  }

 private:
  ErrorStatusBuilder builder_;
};

}  // namespace litert::tensor

///////////////// Implementation details start here. ///////////////////////

#define LRT_TENSOR_RETURN_IF_ERROR_SELECT_OVERLOAD_HELPER(_1, _2, OVERLOAD, \
                                                          ...)              \
  OVERLOAD

#define LRT_TENSOR_RETURN_IF_ERROR_SELECT_OVERLOAD(args) \
  LRT_TENSOR_RETURN_IF_ERROR_SELECT_OVERLOAD_HELPER args

#define LRT_TENSOR_RETURN_IF_ERROR_1(EXPR) LRT_TENSOR_RETURN_IF_ERROR_2(EXPR, _)

// NOLINTBEGIN(readability/braces)
#define LRT_TENSOR_RETURN_IF_ERROR_2(EXPR, RETURN_VALUE)                 \
  if (auto status = EXPR;                                                \
      ::litert::tensor::ErrorStatusBuilder::IsError(status))             \
    if (::litert::tensor::ErrorStatusBuilder _(std::move(status)); true) \
  return RETURN_VALUE
// NOLINTEND(readability/braces)

#define LRT_TENSOR_ASSIGN_OR_RETURN_SELECT_OVERLOAD_HELPER(_1, _2, _3,    \
                                                           OVERLOAD, ...) \
  OVERLOAD

#define LRT_TENSOR_ASSIGN_OR_RETURN_SELECT_OVERLOAD(args) \
  LRT_TENSOR_ASSIGN_OR_RETURN_SELECT_OVERLOAD_HELPER args

#define LRT_TENSOR_ASSIGN_OR_RETURN_HELPER_2(TMP_VAR, DECL, EXPR) \
  LRT_TENSOR_ASSIGN_OR_RETURN_HELPER_3(TMP_VAR, DECL, EXPR, _)

#define LRT_TENSOR_ASSIGN_OR_RETURN_HELPER_3(TMP_VAR, DECL, EXPR, \
                                             RETURN_VALUE)        \
  auto&& TMP_VAR = (EXPR);                                        \
  if (::litert::tensor::ErrorStatusBuilder::IsError(TMP_VAR)) {   \
    [[maybe_unused]] ::litert::tensor::ErrorStatusBuilder _(      \
        std::move(TMP_VAR));                                      \
    return RETURN_VALUE;                                          \
  }                                                               \
  _LRT_TENSOR_STRIP_PARENS(DECL) =                                \
      ::litert::tensor::ErrorStatusBuilder::ForwardWrappedValue(TMP_VAR)

#define LRT_TENSOR_ASSIGN_OR_ABORT_SELECT_OVERLOAD_HELPER(_1, _2, _3,    \
                                                          OVERLOAD, ...) \
  OVERLOAD

#define LRT_TENSOR_ASSIGN_OR_ABORT_SELECT_OVERLOAD(args) \
  LRT_TENSOR_ASSIGN_OR_ABORT_SELECT_OVERLOAD_HELPER args

#define LRT_TENSOR_ASSIGN_OR_ABORT_HELPER_2(TMP_VAR, DECL, EXPR) \
  LRT_TENSOR_ASSIGN_OR_ABORT_HELPER_3(TMP_VAR, DECL, EXPR, _)

#define LRT_TENSOR_ASSIGN_OR_ABORT_HELPER_3(TMP_VAR, DECL, EXPR,   \
                                            LOG_EXPRESSION)        \
  auto&& TMP_VAR = (EXPR);                                         \
  if (::litert::tensor::ErrorStatusBuilder::IsError(TMP_VAR)) {    \
    ::litert::tensor::ErrorStatusBuilder _(std::move(TMP_VAR));    \
    ::litert::tensor::LogBeforeAbort(std::move((LOG_EXPRESSION))); \
  }                                                                \
  _LRT_TENSOR_STRIP_PARENS(DECL) =                                 \
      ::litert::tensor::ErrorStatusBuilder::ForwardWrappedValue(TMP_VAR)

#define _CONCAT_NAME_IMPL(x, y) x##y

#define _CONCAT_NAME(x, y) _CONCAT_NAME_IMPL(x, y)

#define _RETURN_VAL(val) return val

// Removes outer parentheses from X if there are some.
//
// This is useful to allow macros parameters to have commas by putting them
// inside parentheses by stripping those when expanding the macro.
//
// For instance, WITHOUT USING THIS, the following is an error.
// ```
// LRT_TENSOR_ASSIGN_OR_RETURN(auto [a, b], SomeFunction());
//                                ^   ^
//          The above commas make it such that the macro has 3 arguments
// ```
// Using this, the following works:
// ```
// LRT_TENSOR_ASSIGN_OR_RETURN((auto [a, b]), SomeFunction());
//                         ^           ^
//          These surround a comma, preventing it to be used as the macro
//          argument separator. They are stripped internally by the macro.
//
// LRT_TENSOR_ASSIGN_OR_RETURN(auto a, SomeFunction());
//                         ^^^^^^
//         There is no parentheses surrounding the parameter and the macro still
//         works.
// ```
#ifndef _LRT_TENSOR_STRIP_PARENS
#define _LRT_TENSOR_STRIP_PARENS(X) _LRT_TENSOR_ESC(_LRT_TENSOR_ISH X)
#define _LRT_TENSOR_ISH(...) _LRT_TENSOR_ISH __VA_ARGS__
#define _LRT_TENSOR_ESC(...) _LRT_TENSOR_ESC_(__VA_ARGS__)
#define _LRT_TENSOR_ESC_(...) _LRT_TENSOR_VAN##__VA_ARGS__
#define _LRT_TENSOR_VAN_LRT_TENSOR_ISH
#endif

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_UTILS_MACROS_H_
