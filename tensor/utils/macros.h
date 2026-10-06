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

#include <type_traits>
#include <utility>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_builder.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/source_location.h"  // from @com_google_absl

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

namespace internal {

template <class T, class = void>
struct IsComplete : std::false_type {};

template <class T>
struct IsComplete<T, std::void_t<decltype(sizeof(T))>> : std::true_type {};

template <class T>
struct IsStatusOr : std::false_type {};

template <class T>
struct IsStatusOr<absl::StatusOr<T>> : std::true_type {};

// Detects `Conversion::AsError(E, absl::SourceLocation)`.
template <class Conversion, class E, class = void>
struct HasLocationAwareAsError : std::false_type {};

template <class Conversion, class E>
struct HasLocationAwareAsError<
    Conversion, E,
    std::void_t<decltype(Conversion::AsError(
        std::declval<E>(), std::declval<absl::SourceLocation>()))>>
    : std::true_type {};

// Detects `R Conversion::FromError(const absl::Status&, absl::SourceLocation)`.
template <class Conversion, class R, class = void>
struct HasFromErrorImpl : std::false_type {};

template <class Conversion, class R>
struct HasFromErrorImpl<
    Conversion, R,
    std::enable_if_t<std::is_same_v<decltype(Conversion::FromError(
                                        std::declval<const absl::Status&>(),
                                        std::declval<absl::SourceLocation>())),
                                    R>>> : std::true_type {};

// `Conversion` may be an `ErrorConversion` that was never specialized.
template <class Conversion, class R>
struct HasFromError : std::conjunction<IsComplete<Conversion>,
                                       HasFromErrorImpl<Conversion, R>> {};

}  // namespace internal

// An `absl::StatusBuilder` that can also be created from, and converted to,
// error types other than `absl::Status`.
//
// This class is meant to be used with the `LRT_TENSOR_RETURN_IF_ERROR` and
// `LRT_TENSOR_ASSIGN_OR_RETURN` macros. Each macro call adds its location to
// the status source location trace. All `absl::StatusBuilder` functions can be
// used to customize the returned error.
//
// Support for an error type `E` is added by specializing
// `ErrorStatusBuilder::ErrorConversion<E>`:
//
// - To use an `E` value as a macro expression:
//   - `static bool IsError(const E&)`.
//   - `static absl::Status AsError(E)`, or
//     `static absl::Status AsError(E, absl::SourceLocation)` to create the
//     status at the macro call location.
//   - Optionally, `static E& Forward(E&)` when `E` holds the value that
//     `LRT_TENSOR_ASSIGN_OR_RETURN` assigns.
// - To return an `E` using the macros' default return value:
//   - `static E FromError(const absl::Status&, absl::SourceLocation)`.
//   This isn't needed for types that can be constructed from an `absl::Status`.
class ErrorStatusBuilder : public absl::StatusBuilder {
 public:
  template <class Error, class CRTP = void>
  struct ErrorConversion;

  template <class T, class = std::enable_if_t<!std::is_base_of_v<
                         absl::StatusBuilder, std::decay_t<T>>>>
  explicit ErrorStatusBuilder(
      T&& error, absl::SourceLocation loc = absl::SourceLocation::current())
      : absl::StatusBuilder(AsError(std::forward<T>(error), loc), loc) {}

  // Takes over the state of `builder`.
  //
  // `absl::StatusBuilder` functions return an `absl::StatusBuilder`. The macros
  // use this to turn it back into an `ErrorStatusBuilder`, which keeps the
  // conversions below available.
  explicit ErrorStatusBuilder(absl::StatusBuilder&& builder)
      : absl::StatusBuilder(std::move(builder)) {}

  // Converts to return types other than `absl::Status` and `absl::StatusOr`,
  // which `absl::StatusBuilder` already handles.
  //
  // Uses `ErrorConversion<T>::FromError` when it exists, otherwise constructs a
  // `T` from the status (and the builder location if `T` accepts it).
  template <
      class T,
      class = std::enable_if_t<
          !std::is_same_v<T, absl::Status> && !internal::IsStatusOr<T>::value &&
          !std::is_base_of_v<T, ErrorStatusBuilder> &&
          (internal::HasFromError<ErrorConversion<T>, T>::value ||
           std::is_constructible_v<T, absl::Status, absl::SourceLocation> ||
           std::is_constructible_v<T, absl::Status>)>>
  operator T() && {  // NOLINT(google-explicit-constructor)
    const absl::SourceLocation loc = source_location();
    absl::Status status = static_cast<absl::StatusBuilder&&>(*this);
    if constexpr (internal::HasFromError<ErrorConversion<T>, T>::value) {
      return ErrorConversion<T>::FromError(status, loc);
    } else if constexpr (std::is_constructible_v<T, absl::Status,
                                                 absl::SourceLocation>) {
      return T(std::move(status), loc);
    } else {
      return T(std::move(status));
    }
  }

  template <class T>
  static constexpr bool IsError(T&& value) {
    return ErrorConversion<std::decay_t<T>>::IsError(std::forward<T>(value));
  }

  // Converts `value` to an `absl::Status`.
  //
  // `loc` is forwarded to `ErrorConversion` specializations that create the
  // status at the macro call location.
  template <class T>
  static absl::Status AsError(
      T&& value, absl::SourceLocation loc = absl::SourceLocation::current()) {
    using Conversion = ErrorConversion<std::decay_t<T>>;
    if constexpr (internal::HasLocationAwareAsError<Conversion, T&&>::value) {
      return Conversion::AsError(std::forward<T>(value), loc);
    } else {
      return Conversion::AsError(std::forward<T>(value));
    }
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
};

// NOLINTBEGIN(*-explicit-constructor)
template <>
struct ErrorStatusBuilder::ErrorConversion<bool> {
  static constexpr bool IsError(bool value) { return !value; };
  // absl only records source locations for statuses with a message.
  static absl::Status AsError(bool /*value*/, absl::SourceLocation loc) {
    return absl::UnknownError("Check failed", loc);
  }
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

namespace internal {

// Turns `absl::StatusBuilder` objects back into an `ErrorStatusBuilder`.
//
// `absl::StatusBuilder` functions return an `absl::StatusBuilder&`, which only
// converts to `absl::Status` and `absl::StatusOr`. The macros return
// `ReturnValue() = return_value` to keep the conversions to other types
// available.
//
// Values that aren't an `absl::StatusBuilder` are returned unchanged.
struct ReturnValue {
  template <class T>
  decltype(auto) operator=(T&& value) && {  // NOLINT(*-assign-operator*)
    if constexpr (std::is_base_of_v<absl::StatusBuilder, std::decay_t<T>>) {
      // We forcefully move. `value` is either a temporary or the macros' `_`
      // variable.
      // NOLINTNEXTLINE(bugprone-move-forwarding-reference)
      return ErrorStatusBuilder(absl::StatusBuilder(std::move(value)));
    } else {
      return std::forward<T>(value);
    }
  }
};

}  // namespace internal

class LogBeforeAbort {
 public:
  explicit LogBeforeAbort(absl::StatusBuilder builder)
      : builder_(std::move(builder)) {}

  ~LogBeforeAbort() {
    ABSL_LOG(FATAL) << absl::Status(builder_).ToString(
        absl::StatusToStringMode::kWithEverything);
  }

  template <class T>
  LogBeforeAbort& operator<<(T&& val) {
    builder_ << val;
    return *this;
  }

 private:
  absl::StatusBuilder builder_;
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
  return ::litert::tensor::internal::ReturnValue() = RETURN_VALUE
// NOLINTEND(readability/braces)

#define LRT_TENSOR_ASSIGN_OR_RETURN_SELECT_OVERLOAD_HELPER(_1, _2, _3,    \
                                                           OVERLOAD, ...) \
  OVERLOAD

#define LRT_TENSOR_ASSIGN_OR_RETURN_SELECT_OVERLOAD(args) \
  LRT_TENSOR_ASSIGN_OR_RETURN_SELECT_OVERLOAD_HELPER args

#define LRT_TENSOR_ASSIGN_OR_RETURN_HELPER_2(TMP_VAR, DECL, EXPR) \
  LRT_TENSOR_ASSIGN_OR_RETURN_HELPER_3(TMP_VAR, DECL, EXPR, _)

#define LRT_TENSOR_ASSIGN_OR_RETURN_HELPER_3(TMP_VAR, DECL, EXPR,      \
                                             RETURN_VALUE)             \
  auto&& TMP_VAR = (EXPR);                                             \
  if (::litert::tensor::ErrorStatusBuilder::IsError(TMP_VAR)) {        \
    [[maybe_unused]] ::litert::tensor::ErrorStatusBuilder _(           \
        std::move(TMP_VAR));                                           \
    return ::litert::tensor::internal::ReturnValue() = (RETURN_VALUE); \
  }                                                                    \
  _LRT_TENSOR_STRIP_PARENS(DECL) =                                     \
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
