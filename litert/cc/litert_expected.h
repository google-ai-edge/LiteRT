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

#ifndef ODML_LITERT_LITERT_CC_LITERT_EXPECTED_H_
#define ODML_LITERT_LITERT_CC_LITERT_EXPECTED_H_

#include <functional>
#include <initializer_list>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/internal/litert_detail.h"

/// @file
/// @brief Defines an `Expected` class for handling return values that may be
/// an error.
///
/// This implementation is similar to `absl::StatusOr` or `std::expected`
/// (C++23) but is better integrated with `LiteRtStatus` as the canonical status
/// code.

namespace litert {

/// @brief A C++ wrapper for a `LiteRtStatus` code, providing a status and an
/// error message.
class Error {
 public:
  /// @brief Constructs an `Error` from a status and an optional error message.
  /// @note `::litert::Status::kOk` should not be passed.
  explicit Error(Status status, std::string message = "")
      : status_(static_cast<LiteRtStatus>(status)),
        message_(std::move(message)) {
    ABSL_DCHECK(status != Status::kOk);
  }

  [[deprecated("Use the constructor that takes ::litert::Status instead.")]]
  explicit Error(LiteRtStatus status, std::string message = "")
      : status_(status), message_(std::move(message)) {
    ABSL_DCHECK(status != kLiteRtStatusOk);
  }

  /// @brief Gets the status.
  /// @todo Rename to `Status()` after the deprecated function is removed.
  constexpr Status StatusCC() const {
    return static_cast<enum Status>(status_);
  }

  [[deprecated("Use StatusCC() instead.")]]
  constexpr LiteRtStatus Status() const { return status_; }

  /// @brief Gets the error message. Returns an empty string if none was
  /// attached.
  const std::string& Message() const { return message_; }

  friend std::ostream& operator<<(std::ostream& stream, const Error& error) {
    stream << LiteRtGetStatusString(error.Status());
    if (!error.Message().empty()) {
      stream << ": " << error.Message();
    }
    return stream;
  }

  template <class Sink>
  friend void AbslStringify(Sink& sink, const Error& error) {
    absl::Format(&sink, "%s", LiteRtGetStatusString(error.Status()));
    if (!error.Message().empty()) {
      absl::Format(&sink, ": %v", error.Message());
    }
  }

 private:
  LiteRtStatus status_;
  std::string message_;
};

/// @brief A utility for generic return values that represents a failure.
class Unexpected {
 public:
  template <class... Args>
  constexpr explicit Unexpected(Args&&... args)
      : error_(std::forward<Args>(args)...) {}

  /// @brief Allows for implicit conversion from a convertible `Error` value
  /// in-place.
  // NOLINTNEXTLINE(*-explicit-constructor)
  Unexpected(class Error&& e) : error_(std::move(e)) {}

  Unexpected(Unexpected&& other) = default;
  Unexpected(const Unexpected& other) = default;
  Unexpected& operator=(Unexpected&& other) = default;
  Unexpected& operator=(const Unexpected& other) = default;

  constexpr const class Error& Error() const& noexcept { return error_; }
  constexpr class Error& Error() & noexcept { return error_; }
  constexpr const class Error&& Error() const&& noexcept {
    return std::move(error_);
  }
  constexpr class Error&& Error() && noexcept { return std::move(error_); }

  template <class Sink>
  friend void AbslStringify(Sink& sink, const Unexpected& unexpected) {
    AbslStringify(sink, unexpected.Error());
  }

 private:
  class Error error_;
};

/// @brief A utility for generic return values that may represent a failure.
///
/// `Expected` stores and owns the lifetime of either an `Unexpected` object or
/// a value of type `T`. `T` can be any primitive or non-primitive type.
///
/// No dynamic allocations occur during initialization, so the underlying `T` is
/// only movable (as opposed to being releasable). Arguments should be
/// constructed in-place when initializing the `Expected` object if possible.
///
/// `Unexpected&&` and `T&&` can be implicitly cast to an `Expected`. For
/// example:
/// @code
/// Expected<Foo> Bar() {
///   bool success = ...
///   if (!success) {
///     return Unexpected(kLiteRtStatus, "Bad Baz");
///   }
///   return Foo();
/// }
/// @endcode
template <class T>
class Expected {
 public:
  using StorageType =
      std::conditional_t<std::is_reference_v<T>,
                         std::reference_wrapper<std::remove_reference_t<T>>, T>;

  /// @brief The following type definitions are in snake_case to match standard
  /// member types.
  using value_type = std::decay_t<T>;
  using pointer = std::remove_reference_t<T>*;
  using const_pointer = const value_type*;
  using reference = std::remove_reference_t<T>&;
  using const_reference = const value_type&;

  /// @brief Constructs `T` from an initializer list in-place.
  template <class U>
  Expected(std::initializer_list<U> il) : has_value_(true), value_(il) {}

  /// @brief Constructs `T` from forwarded arguments in-place.
  template <class... Args>
  explicit Expected(Args&&... args)
      : has_value_(true), value_(std::forward<Args>(args)...) {}

  // NOLINTBEGIN(*-explicit-constructor)

  /// @brief Allows for implicit conversion from a convertible `T` value
  /// in-place.
  Expected(reference t) : has_value_(true), value_(t) {}
  /// @brief Copy-constructs from a constant reference.
  ///
  /// This is disabled if `T` is a constant reference.
  template <
      class U,
      class = std::enable_if_t<std::is_same_v<std::decay_t<U>, value_type>>,
      class = std::enable_if<!std::is_same_v<const U&, reference>>>
  Expected(const U& t) : has_value_(true), value_(t) {}
  Expected(value_type&& t) : has_value_(true), value_(std::move(t)) {}

  /// @brief Constructs from an `Unexpected` object in-place.
  ///
  /// Allows for implicit conversion from `Error`.
  Expected(const Unexpected& err) : has_value_(false), unexpected_(err) {}
  Expected(Unexpected&& err) : has_value_(false), unexpected_(std::move(err)) {}
  Expected(const class Error& e) : has_value_(false), unexpected_(e) {}

  // NOLINTEND(*-explicit-constructor)

  // Copy/move

  Expected(Expected&& other) : has_value_(other.HasValue()) {
    if (HasValue()) {
      ConstructAt(std::addressof(value_), std::move(other.value_));
    } else {
      ConstructAt(std::addressof(unexpected_), std::move(other.unexpected_));
    }
  }

  Expected(const Expected& other) : has_value_(other.has_value_) {
    if (HasValue()) {
      ConstructAt(std::addressof(value_), other.value_);
      value_ = other.value_;
    } else {
      ConstructAt(std::addressof(unexpected_), other.unexpected_);
    }
  }

  Expected& operator=(Expected&& other) {
    if (this != &other) {
      if (HasValue()) {
        if (other.HasValue()) {
          value_ = std::move(other.value_);
        } else {
          value_.~StorageType();
          ConstructAt(std::addressof(unexpected_),
                      std::move(other.unexpected_));
        }
      } else {
        if (other.HasValue()) {
          unexpected_.~Unexpected();
          ConstructAt(std::addressof(value_), std::move(other.value_));
        } else {
          unexpected_ = std::move(other.unexpected_);
        }
      }
      has_value_ = other.has_value_;
    }
    return *this;
  }

  Expected& operator=(const Expected& other) {
    if (this != &other) {
      if (HasValue()) {
        if (other.HasValue()) {
          value_ = other.value_;
        } else {
          value_.~StorageType();
          ConstructAt(std::addressof(unexpected_), other.unexpected_);
        }
      } else {
        if (other.HasValue()) {
          unexpected_.~Unexpected();
          ConstructAt(std::addressof(value_), other.value_);
        } else {
          unexpected_ = other.unexpected_;
        }
      }
      has_value_ = other.has_value_;
    }
    return *this;
  }

  ~Expected() {
    if (has_value_ && std::is_destructible<T>()) {
      value_.~StorageType();
    } else {
      unexpected_.~Unexpected();
    }
  }

  /// @brief Observers for the `T` value. The program exits if it doesn't have
  /// one.
  const_reference Value() const& {
    CheckVal();
    if constexpr (std::is_reference_v<T>) {
      return value_.get();
    } else {
      return value_;
    }
  }

  reference Value() & {
    CheckVal();
    if constexpr (std::is_reference_v<T>) {
      return value_.get();
    } else {
      return value_;
    }
  }

  /// @brief Deleted: an `Expected` should always be checked before accessing
  /// its value.
  reference& Value() const&& = delete;

  /// @brief Deleted: an `Expected` should always be checked before accessing
  /// its value.
  reference& Value() && = delete;

  const_pointer operator->() const& {
    CheckVal();
    if constexpr (std::is_reference_v<T>) {
      return &(value_.get());
    } else {
      return &value_;
    }
  }

  pointer operator->() & {
    CheckVal();
    if constexpr (std::is_reference_v<T>) {
      return &(value_.get());
    } else {
      return &value_;
    }
  }

  /// @brief Deleted: an `Expected` should always be checked before accessing
  /// its value.
  const_pointer operator->() const&& = delete;

  /// @brief Deleted: an `Expected` should always be checked before accessing
  /// its value.
  pointer operator->() && = delete;

  const_reference operator*() const& { return Value(); }

  reference operator*() & { return Value(); }

  /// @brief Deleted: an `Expected` should always be checked before accessing
  /// its value.
  reference& operator*() const&& = delete;

  /// @brief Deleted: an `Expected` should always be checked before accessing
  /// its value.
  reference& operator*() && = delete;

  /// @brief Observer for `Unexpected`. The program exits if it doesn't have
  /// one.
  const class Error& Error() const& {
    CheckNoVal();
    return unexpected_.Error();
  }

  class Error& Error() & {
    CheckNoVal();
    return unexpected_.Error();
  }

  /// @brief Deleted: an `Expected` should always be checked before accessing
  /// its error.
  const class Error&& Error() const&& = delete;

  /// @brief Deleted: an `Expected` should always be checked before accessing
  /// its error.
  class Error&& Error() && = delete;

  /// @brief Checks if this `Expected` contains a `T` value.
  ///
  /// If not, it contains an `Unexpected`.
  bool HasValue() const { return has_value_; }

  /// @brief Converts to `bool` for `HasValue`.
  explicit operator bool() const { return HasValue(); }

 private:
  bool has_value_;
  union {
    StorageType value_;
    Unexpected unexpected_;
  };
  void CheckNoVal() const { ABSL_CHECK(!HasValue()); }
  void CheckVal() const { ABSL_CHECK(HasValue()); }
};

template <class T>
Expected(const T&) -> Expected<T>;

template <class T>
Expected(T&&) -> Expected<T>;

namespace internal {
template <class T>
struct CanBeAbslFormated {
  template <class U>
  static constexpr auto Check(int)
      -> decltype(absl::StrCat(std::declval<U>()), true) {
    return true;
  }
  template <class U>
  static constexpr bool Check(...) {
    return false;
  }
  enum { value = Check<T>(0) };
};
}  // namespace internal

template <class Sink, class T>
void AbslStringify(Sink& sink, const Expected<T>& expected) {
  if (!expected.HasValue()) {
    absl::Format(&sink, "%v", expected.Error());
  } else {
    if constexpr (std::is_same_v<T, void>) {
      sink.Append("void expected value");
    } else {
      if constexpr (internal::CanBeAbslFormated<T>::value) {
        absl::Format(&sink, "%v", expected.Value());
      } else {
        absl::Format(&sink, "unformattable expected value");
      }
    }
  }
}

/// @brief A specialization of `Expected` for `void`.
///
/// This specialization is used to simplify returning a valid value (e.g.,
/// `return {};`).
template <>
class Expected<void> {
 public:
  /// @brief Implicit construction is used to simplify returning a valid value
  /// (e.g., `return {};`).
  Expected() : unexpected_(std::nullopt) {}

  // NOLINTBEGIN(*-explicit-constructor)

  /// @brief Constructs from an `Unexpected` object in-place.
  Expected(const Unexpected& err) : unexpected_(err) {}
  Expected(Unexpected&& err) : unexpected_(std::move(err)) {}

  /// @brief Allows for implicit conversion from `Error`.
  Expected(const Error& e) : unexpected_(e) {}

  // NOLINTEND(*-explicit-constructor)

  /// @brief Observer for `Unexpected`. The program exits if it doesn't have
  /// one.
  const class Error& Error() const& {
    CheckNoVal();
    return unexpected_->Error();
  }

  class Error& Error() & {
    CheckNoVal();
    return unexpected_->Error();
  }

  const class Error&& Error() const&& {
    CheckNoVal();
    return std::move(unexpected_->Error());
  }

  class Error&& Error() && {
    CheckNoVal();
    return std::move(unexpected_->Error());
  }

  /// @brief Checks if this `Expected` contains a `T` value.
  ///
  /// If not, it contains an `Unexpected`.
  bool HasValue() const { return !unexpected_.has_value(); }

  /// @brief Converts to `bool` for `HasValue`.
  explicit operator bool() const { return HasValue(); }

 private:
  std::optional<Unexpected> unexpected_;
  void CheckNoVal() const { ABSL_CHECK(!HasValue()); }
  void CheckVal() const { ABSL_CHECK(HasValue()); }
};

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_LITERT_EXPECTED_H_
