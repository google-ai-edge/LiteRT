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

#ifndef ODML_LITERT_LITERT_CC_LITERT_ANY_H_
#define ODML_LITERT_LITERT_CC_LITERT_ANY_H_

#include <any>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <variant>

#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

/// @file
/// @brief Defines an RTTI-free replacement for `std::any` using `std::variant`
///        and provides conversion utilities between `LiteRtAny` and `std::any`.

namespace litert {
/// @brief An RTTI-free replacement for `std::any`, implemented using
/// `std::variant`.
using LiteRtVariant =
    std::variant<std::monostate,     // Empty/None type
                 bool,               // kLiteRtAnyTypeBool
                 int8_t,             // kLiteRtAnyTypeInt (small)
                 int16_t,            // kLiteRtAnyTypeInt (medium)
                 int32_t,            // kLiteRtAnyTypeInt (large)
                 int64_t,            // kLiteRtAnyTypeInt (full)
                 float,              // kLiteRtAnyTypeReal (single)
                 double,             // kLiteRtAnyTypeReal (double)
                 const char*,        // kLiteRtAnyTypeString
                 absl::string_view,  // kLiteRtAnyTypeString (alternative)
                 const void*,        // kLiteRtAnyTypeVoidPtr
                 void*               // kLiteRtAnyTypeVoidPtr (non-const)
                 >;

using any = LiteRtVariant;

inline LiteRtVariant ToStdAny(LiteRtAny litert_any) {
  switch (litert_any.type) {
    case kLiteRtAnyTypeNone:
      return std::monostate{};
    case kLiteRtAnyTypeBool:
      return litert_any.bool_value;
    case kLiteRtAnyTypeInt:
      return litert_any.int_value;
    case kLiteRtAnyTypeReal:
      return litert_any.real_value;
    case kLiteRtAnyTypeString:
      return litert_any.str_value;
    case kLiteRtAnyTypeVoidPtr:
      return const_cast<void*>(litert_any.ptr_value);
    default:
      return std::monostate{};
  }
}

inline Expected<LiteRtAny> ToLiteRtAny(const LiteRtVariant& var) {
  return std::visit(
      [](auto&& arg) -> Expected<LiteRtAny> {
        using T = std::decay_t<decltype(arg)>;
        LiteRtAny result;

        if constexpr (std::is_same_v<T, std::monostate>) {
          result.type = kLiteRtAnyTypeNone;
          return result;
        } else if constexpr (std::is_same_v<T, bool>) {
          result.type = kLiteRtAnyTypeBool;
          result.bool_value = arg;
          return result;
        } else if constexpr (std::is_same_v<T, int8_t> ||
                             std::is_same_v<T, int16_t> ||
                             std::is_same_v<T, int32_t> ||
                             std::is_same_v<T, int64_t>) {
          result.type = kLiteRtAnyTypeInt;
          result.int_value = static_cast<int64_t>(arg);
          return result;
        } else if constexpr (std::is_same_v<T, float> ||
                             std::is_same_v<T, double>) {
          result.type = kLiteRtAnyTypeReal;
          result.real_value = static_cast<double>(arg);
          return result;
        } else if constexpr (std::is_same_v<T, const char*>) {
          result.type = kLiteRtAnyTypeString;
          result.str_value = arg;
          return result;
        } else if constexpr (std::is_same_v<T, absl::string_view>) {
          result.type = kLiteRtAnyTypeString;
          result.str_value = arg.data();
          return result;
        } else if constexpr (std::is_same_v<T, const void*>) {
          result.type = kLiteRtAnyTypeVoidPtr;
          result.ptr_value = arg;
          return result;
        } else if constexpr (std::is_same_v<T, void*>) {
          result.type = kLiteRtAnyTypeVoidPtr;
          result.ptr_value = arg;
          return result;
        } else {
          return Error(kLiteRtStatusErrorInvalidArgument,
                       "Invalid type for ToLiteRtAny");
        }
      },
      var);
}

namespace internal {

inline Expected<void> CheckType(const LiteRtAny& any,
                                const LiteRtAnyType type) {
  if (any.type != type) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 absl::StrFormat("Wrong LiteRtAny type. Expected %s, got %s.",
                                 LiteRtAnyTypeToString(type),
                                 LiteRtAnyTypeToString(any.type)));
  }
  return {};
}

template <class T>
Expected<T> GetInt(const LiteRtAny& any) {
  LITERT_RETURN_IF_ERROR(CheckType(any, kLiteRtAnyTypeInt));
  if (any.int_value > std::numeric_limits<T>::max() ||
      any.int_value < std::numeric_limits<T>::lowest()) {
    return Error(
        kLiteRtStatusErrorInvalidArgument,
        absl::StrFormat("LiteRtAny integer is out of range. %v <= %v <= %v",
                        std::numeric_limits<T>::lowest(), any.int_value,
                        std::numeric_limits<T>::max()));
  }
  return static_cast<T>(any.int_value);
}

template <class T>
Expected<T> GetReal(const LiteRtAny& any) {
  LITERT_RETURN_IF_ERROR(CheckType(any, kLiteRtAnyTypeReal));
  if (any.real_value > std::numeric_limits<T>::max() ||
      any.real_value < std::numeric_limits<T>::lowest()) {
    return Error(
        kLiteRtStatusErrorInvalidArgument,
        absl::StrFormat("LiteRtAny real is out of range. %v <= %v <= %v",
                        std::numeric_limits<T>::lowest(), any.real_value,
                        std::numeric_limits<T>::max()));
  }
  return static_cast<T>(any.real_value);
}
}  // namespace internal

/// @brief Extracts a value from a `LiteRtAny` object with type checking.
template <class T>
inline Expected<T> Get(const LiteRtAny& any);

template <>
inline Expected<bool> Get(const LiteRtAny& any) {
  LITERT_RETURN_IF_ERROR(internal::CheckType(any, kLiteRtAnyTypeBool));
  return any.bool_value;
}

template <>
inline Expected<int8_t> Get(const LiteRtAny& any) {
  return internal::GetInt<int8_t>(any);
}

template <>
inline Expected<int16_t> Get(const LiteRtAny& any) {
  return internal::GetInt<int16_t>(any);
}

template <>
inline Expected<int32_t> Get(const LiteRtAny& any) {
  return internal::GetInt<int32_t>(any);
}

template <>
inline Expected<int64_t> Get(const LiteRtAny& any) {
  return internal::GetInt<int64_t>(any);
}

template <>
inline Expected<float> Get(const LiteRtAny& any) {
  return internal::GetReal<float>(any);
}

template <>
inline Expected<double> Get(const LiteRtAny& any) {
  return internal::GetReal<double>(any);
}

template <>
inline Expected<std::string> Get(const LiteRtAny& any) {
  LITERT_RETURN_IF_ERROR(internal::CheckType(any, kLiteRtAnyTypeString));
  return std::string(any.str_value);
}

template <>
inline Expected<absl::string_view> Get(const LiteRtAny& any) {
  LITERT_RETURN_IF_ERROR(internal::CheckType(any, kLiteRtAnyTypeString));
  return absl::string_view(any.str_value);
}

template <>
inline Expected<const void*> Get(const LiteRtAny& any) {
  LITERT_RETURN_IF_ERROR(internal::CheckType(any, kLiteRtAnyTypeVoidPtr));
  return any.ptr_value;
}

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_LITERT_ANY_H_
