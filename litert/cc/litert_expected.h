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
#include "litert/cc/litert_detail.h"

namespace litert {

// An "Expected" incapsulates the result of some routine which may have an
// unexpected result. Unexpected results in this context are a standard
// LiteRtStatus plus extra usability data such as error messages. This is
// similar to an absl::StatusOr or std::expected (C++23) but better integrated
// with LiteRtStatus as the canonical status code.

// C++ wrapper around LiteRtStatus code. Provides a status as well
// as an error message.
class Error {
 public:
  // Construct Unexpected from status and optional error message.
  //
  // NOTE: kLiteRtStatusOk should not be passed to Unexpected.
  explicit Error(LiteRtStatus status, std::string message = "")
      : status_(status), message_(std::move(message)) {
    ABSL_DCHECK(status != kLiteRtStatusOk);
  }

  // Get the status.
  constexpr LiteRtStatus Status() const { return status_; }

  // Get the error message, empty string if none was attached.
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

class Unexpected {
 public:
  template <class... Args>
  constexpr explicit Unexpected(Args&&... args)
      : error_(std::forward<Args>(args)...) {}

  // Allow for implicit conversion from convertible Error value inplace.
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

// Utility for generic return values that may be a statused failure. Expected
// stores and owns the lifetime of either an Unexpected, or a T. T may be any
// type, primitive or non-primitive.
//
// No dynamic allocations occur during initialization, so the underlying T is
// only movable (as opposed to something like "release"). Arguments should be
// constructed in place at the time of initializing the expected if possible.
//
// Unexpected&& and T&& may be implicitly casted to an Expected. For example,
//
// Expected<Foo> Bar() {
//   bool success = ...
//   if (!success) {
//     return Unexpected(kLiteRtStatus, "Bad Baz");
//   }
//   return Foo();
// }
template <class T>
class Expected {
 public:
  using StorageType =
      std::conditional_t<std::is_reference_v<T>,
                         std::reference_wrapper<std::remove_reference_t<T>>, T>;

  // The following type defs are snake case to match the standard member types.

  using value_type = std::decay_t<T>;
  using pointer = std::remove_reference_t<T>*;
  using const_pointer = const value_type*;
  using reference = std::remove_reference_t<T>&;
  using const_reference = const value_type&;

  // Construct Expected with T inplace.

  // Construct T from initializer list inplace.
  template <class U>
  Expected(std::initializer_list<U> il) : has_value_(true), value_(il) {}

  // Construct T from forwarded args inplace.
  template <class... Args>
  explicit Expected(Args&&... args)
      : has_value_(true), value_(std::forward<Args>(args)...) {}

  // NOLINTBEGIN(*-explicit-constructor)

  // Allow for implicit conversion from convertible T value inplace.
  Expected(reference t) : has_value_(true), value_(t) {}
  // Copy constructs from a constant reference.
  //
  // Disabled if `T` is a constant reference.
  template <
      class U,
      class = std::enable_if_t<std::is_same_v<std::decay_t<U>, value_type>>,
      class = std::enable_if<!std::is_same_v<const U&, reference>>>
  Expected(const U& t) : has_value_(true), value_(t) {}
  Expected(value_type&& t) : has_value_(true), value_(std::move(t)) {}

  // Construct from Unexpected inplace.

  // Allow for implicit conversion from Error.
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

  // Observers for T value, program exits if it doesn't have one.
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

  // Deleted: an Expected should always be checked before accessing its value.
  reference& Value() const&& = delete;

  // Deleted: an Expected should always be checked before accessing its value.
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

  // Deleted: an Expected should always be checked before accessing its value.
  const_pointer operator->() const&& = delete;

  // Deleted: an Expected should always be checked before accessing its value.
  pointer operator->() && = delete;

  const_reference operator*() const& { return Value(); }

  reference operator*() & { return Value(); }

  // Deleted: an Expected should always be checked before accessing its value.
  reference& operator*() const&& = delete;

  // Deleted: an Expected should always be checked before accessing its value.
  reference& operator*() && = delete;

  // Observer for Unexpected, program exits if it doesn't have one.
  const class Error& Error() const& {
    CheckNoVal();
    return unexpected_.Error();
  }

  class Error& Error() & {
    CheckNoVal();
    return unexpected_.Error();
  }

  // Deleted: an Expected should always be checked before accessing its error.
  const class Error&& Error() const&& = delete;

  // Deleted: an Expected should always be checked before accessing its error.
  class Error&& Error() && = delete;

  // Does this expected contain a T Value. It contains an unexpected if not.
  bool HasValue() const { return has_value_; }

  // Convert to bool for HasValue.
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

template <>
class Expected<void> {
 public:
  // Implicit construction is used to simplify returning a valid value, e.g., in
  // "return {};"
  Expected() : unexpected_(std::nullopt) {}

  // NOLINTBEGIN(*-explicit-constructor)

  // Construct from Unexpected inplace.
  Expected(const Unexpected& err) : unexpected_(err) {}
  Expected(Unexpected&& err) : unexpected_(std::move(err)) {}

  // Allow for implicit conversion from Error.
  Expected(const Error& e) : unexpected_(e) {}

  // NOLINTEND(*-explicit-constructor)

  // Observer for Unexpected, program exits if it doesn't have one.
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

  // Does this expected contain a T Value. It contains an unexpected if not.
  bool HasValue() const { return !unexpected_.has_value(); }

  // Convert to bool for HasValue.
  explicit operator bool() const { return HasValue(); }

 private:
  std::optional<Unexpected> unexpected_;
  void CheckNoVal() const { ABSL_CHECK(!HasValue()); }
  void CheckVal() const { ABSL_CHECK(HasValue()); }
};

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_LITERT_EXPECTED_H_
