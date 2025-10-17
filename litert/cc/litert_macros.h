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

#ifndef ODML_LITERT_LITERT_CC_LITERT_MACROS_H_
#define ODML_LITERT_LITERT_CC_LITERT_MACROS_H_

#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_source_location.h"
#include "litert/cc/litert_expected.h"

// LITERT_RETURN_IF_ERROR(expr);
// LITERT_RETURN_IF_ERROR(expr, return_value);
//
// Returns the result of `expr` if it represents an LiteRT error status (either
// `litert::Expected` holding an error, a `LiteRtStatus` or a bool that
// evaluated to `false`).
//
// `return_value` may be specified to return a custom value in case of error.
//
// By when specifying `return_value`, an `ErrorStatusBuilder` variable called
// `_` holding the result of `expr` can be used to customize the error message.
//
// By default, the return value is an `ErrorStatusBuilder` built from using the
// result of `expr`. The error message of this builder can be customized by
// using its `*Log*()` functions and the << operator.
//
// ```cpp
// LITERT_RETURN_IF_ERROR(expr) << "Failed while trying to ...";
// ```
#define LITERT_RETURN_IF_ERROR(...)                                       \
  LITERT_RETURN_IF_ERROR_SELECT_OVERLOAD(                                 \
      (__VA_ARGS__, LITERT_RETURN_IF_ERROR_2, LITERT_RETURN_IF_ERROR_1))( \
      __VA_ARGS__)

// ASSIGN_OR_RETURN(decl, expr)
// ASSIGN_OR_RETURN(decl, expr, return_value)
//
// Evaluates `expr` that should convert to a `litert::Expected` object.
//
// - If the object holds a value, move-assigns the value to `decl`.
// - If the object holds an error, returns the error, casting it to a
// `LiteRtStatus` if required.
//
// `return_value` may be specified to return a custom value in case of error.
//
// By when specifying `return_value`, an `ErrorStatusBuilder` variable called
// `_` holding the result of `expr` can be used to customize the error message.
//
// ```cpp
// LITERT_ASSIGN_OR_RETURN(decl, expr, _ << "Failed while trying to ...");
// ```
#define LITERT_ASSIGN_OR_RETURN(DECL, ...)                                     \
  LITERT_ASSIGN_OR_RETURN_SELECT_OVERLOAD((DECL, __VA_ARGS__,                  \
                                           LITERT_ASSIGN_OR_RETURN_HELPER_3,   \
                                           LITERT_ASSIGN_OR_RETURN_HELPER_2))( \
      _CONCAT_NAME(expected_value_or_error_, __LINE__), DECL, __VA_ARGS__)

// Works as `LITERT_RETURN_IF_ERROR` but aborts the process in case of error.
#define LITERT_ABORT_IF_ERROR(EXPR)                                        \
  if (auto status = (EXPR); ::litert::ErrorStatusBuilder::IsError(status)) \
  ::litert::LogBeforeAbort(::litert::ErrorStatusBuilder(status))

// Works as `LITERT_ASSIGN_OR` but aborts the process in case of error.
#define LITERT_ASSIGN_OR_ABORT(DECL, ...)                                    \
  LITERT_ASSIGN_OR_ABORT_SELECT_OVERLOAD((DECL, __VA_ARGS__,                 \
                                          LITERT_ASSIGN_OR_ABORT_HELPER_3,   \
                                          LITERT_ASSIGN_OR_ABORT_HELPER_2))( \
      _CONCAT_NAME(expected_value_or_error_, __LINE__), DECL, __VA_ARGS__)

namespace litert {

// Converts implicitly to either `LiteRtStatus` or `litert::Expected` holding an
// error. This allows returning a status in functions using either of these as a
// return type in `LITERT_RETURN_IF_ERROR` and `LITERT_ASSIGN_OR_RETURN`.
//
// When a C++ error with a message is converted to a `LiteRtStatus`, the message
// is logged (as an error by default, use the `Log*()` functions to customize
// that).
//
// The error message may be completed with extra info by using the << operator.
class ErrorStatusBuilder {
 public:
  static ErrorStatusBuilder InvalidArgument(
      litert::SourceLocation loc = litert::SourceLocation::current()) {
    return ErrorStatusBuilder(kLiteRtStatusErrorInvalidArgument,
                              std::move(loc));
  }

  static ErrorStatusBuilder WrongVersion(
      litert::SourceLocation loc = litert::SourceLocation::current()) {
    return ErrorStatusBuilder(kLiteRtStatusErrorWrongVersion, std::move(loc));
  }

  explicit ErrorStatusBuilder(
      bool expr_result,
      litert::SourceLocation loc = litert::SourceLocation::current())
      : error_(kLiteRtStatusErrorUnknown), loc_(loc) {}

  template <class T>
  explicit ErrorStatusBuilder(
      const litert::Expected<T>& expected,
      litert::SourceLocation loc = litert::SourceLocation::current())
      : error_(expected.Error()), loc_(loc) {}

  template <class T>
  explicit ErrorStatusBuilder(
      litert::Expected<T>&& expected,
      litert::SourceLocation loc = litert::SourceLocation::current())
      : error_(std::move(expected.Error())), loc_(loc) {}

  explicit ErrorStatusBuilder(
      LiteRtStatus status,
      litert::SourceLocation loc = litert::SourceLocation::current())
      : error_(status), loc_(loc) {}

  explicit ErrorStatusBuilder(
      const litert::Unexpected& unexpected,
      litert::SourceLocation loc = litert::SourceLocation::current())
      : error_(unexpected.Error()), loc_(loc) {}

  explicit ErrorStatusBuilder(
      litert::Unexpected&& unexpected,
      litert::SourceLocation loc = litert::SourceLocation::current())
      : error_(std::move(unexpected.Error())), loc_(loc) {}

  explicit ErrorStatusBuilder(
      absl::Status&& status,
      litert::SourceLocation loc = litert::SourceLocation::current());

  template <class T>
  explicit ErrorStatusBuilder(
      absl::StatusOr<T>&& status,
      litert::SourceLocation loc = litert::SourceLocation::current())
      : ErrorStatusBuilder(std::move(status).status(), loc) {}

  // NOLINTBEGIN(*-explicit-constructor): This class transparently converts to
  // `LiteRtStatus` and `litert::Expected`.

  // Note: this conversion logs the error message if there is one unless NDEBUG
  // is set (generally in case of optimized builds).
  operator LiteRtStatus() const noexcept {
    PrintLog();
    return error_.Status();
  }

  template <class T>
  operator litert::Expected<T>() const noexcept {
    return litert::Unexpected(error_.Status(), LogMessage());
  }

  operator absl::Status() const noexcept { return ToAbslStatus(); }

  template <class T>
  operator absl::StatusOr<T>() const noexcept {
    return ToAbslStatus();
  }
  // NOLINTEND(*-explicit-constructor)

  static constexpr bool IsError(bool status) { return !status; }

  static constexpr bool IsError(LiteRtStatus status) {
    return status != kLiteRtStatusOk;
  }

  static constexpr bool IsError(const litert::Unexpected&) { return true; }

#if defined(LITERT_WINDOWS_OS)
  // absl::Status::ok() is not constexpr-compatible on Windows MSVC.
  static bool IsError(const absl::Status& s) { return !s.ok(); }

  // absl::Status::ok() is not constexpr-compatible on Windows MSVC.
  template <class T>
  static bool IsError(const absl::StatusOr<T>& s) {
    return !s.ok();
  }
#else
  static constexpr bool IsError(const absl::Status& s) { return !s.ok(); }

  template <class T>
  static constexpr bool IsError(const absl::StatusOr<T>& s) {
    return !s.ok();
  }
#endif

  template <class T>
  static constexpr bool IsError(const litert::Expected<T>& expected) {
    return !expected.HasValue();
  }

  void PrintLog() const noexcept {
#ifndef NDEBUG
    if (ShouldLog()) {
      auto logger = LiteRtGetDefaultLogger();
      LiteRtLogSeverity __min_severity__;
      if (LiteRtGetMinLoggerSeverity(logger, &__min_severity__) !=
          kLiteRtStatusOk) {
        __min_severity__ = kLiteRtLogSeverityVerbose;
      }
      if (log_level_ >= __min_severity__) {
        LiteRtLoggerLog(logger, log_level_, "%s", LogMessage().c_str());
      }
    }
#endif
  }

  // Appends data to the error message.
  template <class T>
  ErrorStatusBuilder& operator<<(T&& val) {
    if (!extra_log_) {
      extra_log_ = std::make_unique<std::stringstream>();
    }
    *extra_log_ << static_cast<T&&>(val);
    return *this;
  }

  // Sets the log level used when converting to a `LiteRtStatus`.
  ErrorStatusBuilder& Log(LiteRtLogSeverity log_level) noexcept {
    log_level_ = log_level;
    return *this;
  }

  // Sets the log level used when converting to a `LiteRtStatus` to `error`.
  ErrorStatusBuilder& LogVerbose() noexcept {
    return Log(kLiteRtLogSeverityVerbose);
  }

  // Sets the log level used when converting to a `LiteRtStatus` to `info`.
  ErrorStatusBuilder& LogInfo() noexcept { return Log(kLiteRtLogSeverityInfo); }

  // Sets the log level used when converting to a `LiteRtStatus` to `error`.
  ErrorStatusBuilder& LogWarning() noexcept {
    return Log(kLiteRtLogSeverityWarning);
  }

  // Sets the log level used when converting to a `LiteRtStatus` to `error`.
  ErrorStatusBuilder& LogError() noexcept {
    return Log(kLiteRtLogSeverityError);
  }

  // Prevent logging any message when converting to a `LiteRtStatus`.
  ErrorStatusBuilder& NoLog() noexcept { return Log(kLiteRtLogSeveritySilent); }

  template <class T>
  static T&& ForwardWrappedValue(Expected<T>& e) {
    return std::move(e.Value());
  }

  template <class T>
  static T& ForwardWrappedValue(Expected<T&>& e) {
    return e.Value();
  }

  template <class T>
  static T&& ForwardWrappedValue(absl::StatusOr<T>& e) {
    return std::move(e).value();
  }

  template <class T>
  static T& ForwardWrappedValue(absl::StatusOr<T&>& e) {
    return e.value();
  }

 private:
  bool ShouldLog() const noexcept {
    return log_level_ != kLiteRtLogSeveritySilent &&
           (!error_.Message().empty() || extra_log_);
  }

  absl::Status ToAbslStatus() const noexcept;
  std::string LogMessage() const;

  litert::Error error_;
  litert::SourceLocation loc_;
  std::unique_ptr<std::stringstream> extra_log_;
  LiteRtLogSeverity log_level_ = kLiteRtLogSeverityError;
};

class LogBeforeAbort {
 public:
  explicit LogBeforeAbort(ErrorStatusBuilder builder)
      : builder_(std::move(builder)) {}

  ~LogBeforeAbort() {
    // Cast to a LiteRtStatus to trigger the logging mechanism.
    [[maybe_unused]] LiteRtStatus s = static_cast<LiteRtStatus>(builder_);
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

}  // namespace litert

//////////// Implementation details start here. ///////////////////////

#define LITERT_RETURN_IF_ERROR_SELECT_OVERLOAD_HELPER(_1, _2, OVERLOAD, ...) \
  OVERLOAD

#define LITERT_RETURN_IF_ERROR_SELECT_OVERLOAD(args) \
  LITERT_RETURN_IF_ERROR_SELECT_OVERLOAD_HELPER args

#define LITERT_RETURN_IF_ERROR_1(EXPR) LITERT_RETURN_IF_ERROR_2(EXPR, _)

// NOLINTBEGIN(readability/braces)
#define LITERT_RETURN_IF_ERROR_2(EXPR, RETURN_VALUE)                     \
  if (auto status = EXPR; ::litert::ErrorStatusBuilder::IsError(status)) \
    if (::litert::ErrorStatusBuilder _(std::move(status)); true)         \
  return RETURN_VALUE
// NOLINTEND(readability/braces)

#define LITERT_ASSIGN_OR_RETURN_SELECT_OVERLOAD_HELPER(_1, _2, _3, OVERLOAD, \
                                                       ...)                  \
  OVERLOAD

#define LITERT_ASSIGN_OR_RETURN_SELECT_OVERLOAD(args) \
  LITERT_ASSIGN_OR_RETURN_SELECT_OVERLOAD_HELPER args

#define LITERT_ASSIGN_OR_RETURN_HELPER_2(TMP_VAR, DECL, EXPR) \
  LITERT_ASSIGN_OR_RETURN_HELPER_3(TMP_VAR, DECL, EXPR, _)

#define LITERT_ASSIGN_OR_RETURN_HELPER_3(TMP_VAR, DECL, EXPR, RETURN_VALUE) \
  auto&& TMP_VAR = (EXPR);                                                  \
  if (::litert::ErrorStatusBuilder::IsError(TMP_VAR)) {                     \
    [[maybe_unused]] ::litert::ErrorStatusBuilder _(std::move(TMP_VAR));    \
    return RETURN_VALUE;                                                    \
  }                                                                         \
  _LITERT_STRIP_PARENS(DECL) =                                              \
      ::litert::ErrorStatusBuilder::ForwardWrappedValue(TMP_VAR)

#define LITERT_ASSIGN_OR_ABORT_SELECT_OVERLOAD_HELPER(_1, _2, _3, OVERLOAD, \
                                                      ...)                  \
  OVERLOAD

#define LITERT_ASSIGN_OR_ABORT_SELECT_OVERLOAD(args) \
  LITERT_ASSIGN_OR_ABORT_SELECT_OVERLOAD_HELPER args

#define LITERT_ASSIGN_OR_ABORT_HELPER_2(TMP_VAR, DECL, EXPR) \
  LITERT_ASSIGN_OR_ABORT_HELPER_3(TMP_VAR, DECL, EXPR, _)

#define LITERT_ASSIGN_OR_ABORT_HELPER_3(TMP_VAR, DECL, EXPR, LOG_EXPRESSION) \
  auto&& TMP_VAR = (EXPR);                                                   \
  if (::litert::ErrorStatusBuilder::IsError(TMP_VAR)) {                      \
    ::litert::ErrorStatusBuilder _(std::move(TMP_VAR));                      \
    ::litert::LogBeforeAbort(std::move((LOG_EXPRESSION)));                   \
  }                                                                          \
  _LITERT_STRIP_PARENS(DECL) =                                               \
      ::litert::ErrorStatusBuilder::ForwardWrappedValue(TMP_VAR)

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
// LITERT_ASSIGN_OR_RETURN(auto [a, b], SomeFunction());
//                                ^   ^
//          The above commas make it such that the macro has 3 arguments
// ```
// Using this, the following works:
// ```
// LITERT_ASSIGN_OR_RETURN((auto [a, b]), SomeFunction());
//                         ^           ^
//          These surround a comma, preventing it to be used as the macro
//          argument separator. They are stripped internally by the macro.
//
// LITERT_ASSIGN_OR_RETURN(auto a, SomeFunction());
//                         ^^^^^^
//         There is no parentheses surrounding the parameter and the macro still
//         works.
// ```
#ifndef _LITERT_STRIP_PARENS
#define _LITERT_STRIP_PARENS(X) _LITERT_ESC(_LITERT_ISH X)
#define _LITERT_ISH(...) _LITERT_ISH __VA_ARGS__
#define _LITERT_ESC(...) _LITERT_ESC_(__VA_ARGS__)
#define _LITERT_ESC_(...) _LITERT_VAN##__VA_ARGS__
#define _LITERT_VAN_LITERT_ISH
#endif

#define LITERT_CHECK_STATUS_HAS_CODE(expr, code) ABSL_CHECK(expr == code);

#define LITERT_CHECK_STATUS_OK(expr) \
  LITERT_CHECK_STATUS_HAS_CODE(expr, kLiteRtStatusOk);

#define LITERT_ENSURE(cond, status, msg) \
  if (!(cond)) {                         \
    LITERT_LOG(LITERT_ERROR, "%s", msg); \
    return status;                       \
  }

#define LITERT_ENSURE_SUPPORTED(cond, msg) \
  LITERT_ENSURE(cond, kLiteRtStatusErrorUnsupported, msg);

#define LITERT_RETURN_IF_ERROR_OR_NOT_MATCHED(expr)                          \
  if (LiteRtStatus status = expr;                                            \
      (status != kLiteRtStatusOk && status != kLiteRtStatusLegalizeNoMatch)) \
    return status;

#define LITERT_STACK_ARRAY(ty, var, size, init) \
  ty* var = (ty*)alloca(sizeof(ty) * size);     \
  for (ty* e = var; e < var + size; ++e) {      \
    *e = init;                                  \
  }

#endif  // ODML_LITERT_LITERT_CC_LITERT_MACROS_H_
