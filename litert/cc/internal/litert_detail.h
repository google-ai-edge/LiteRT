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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_LITERT_DETAIL_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_LITERT_DETAIL_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"

/// @file
/// @brief Provides a collection of miscellaneous compile-time and runtime
///        utilities.

namespace litert {

template <typename T>
using RemoveCvRefT = typename std::remove_cv_t<std::remove_reference_t<T>>;

template <typename Cond, typename T, typename... Rest>
struct SelectHelper {
  using type =
      typename std::conditional_t<Cond::value, T,
                                  typename SelectHelper<Rest...>::type>;
};

template <typename Cond, typename T>
struct SelectHelper<Cond, T> {
  using type = typename std::conditional_t<Cond::value, T, std::monostate>;
};

/// @brief A type trait for conditional type selection.
///
/// This works like a sequence of `if-else if-...-else`, where `std::monostate`
/// is the default type if no condition is met.
///
/// Example:
/// @code
///   using TestVal = int;
///   using Test = SelectT<std::is_floating_point<TestVal>, double,
///                       std::is_integral<TestVal>, long,
///                       std::is_class<TestVal>, std::string>;
///   // Test will be `long`.
///   static_assert(std::is_same_v<Test, long>);
/// @endcode
template <typename... Branches>
struct Select {
  using type = typename SelectHelper<Branches...>::type;
};

template <typename... Branches>
using SelectT = typename Select<Branches...>::type;

template <typename... Ts>
struct TypeList {
  static constexpr size_t kSize = sizeof...(Ts);
};

template <template <typename...> typename C, typename... Lists, typename... Ts,
          typename... SoFar, typename Functor>
void ExpandProductHelper(Functor& f, TypeList<Ts...>,
                         TypeList<SoFar...> sofar) {
  ((ExpandProductHelper<C, Lists...>(f, TypeList<SoFar..., Ts>())), ...);
}

template <template <typename...> typename C, typename HeadList,
          typename... Lists, typename... SoFar, typename Functor>
void ExpandProductHelper(Functor& f, TypeList<SoFar...> sofar) {
  ExpandProductHelper<C, Lists...>(f, HeadList(), sofar);
}

template <template <typename...> typename C, typename... SoFar,
          typename Functor>
void ExpandProductHelper(Functor& f, TypeList<SoFar...> sofar) {
  f.template operator()<C<SoFar...>>();
}

/// @brief A utility for specializing a template against the Cartesian product
///        of given type lists.
///
/// The number of lists passed must match the number of template parameters of
/// the template `C`. `C` cannot have non-type template parameters, but this can
/// be achieved using `std::integral_constant`.
///
/// The `Functor` is a callback that is templated on each member of the
/// Cartesian product. It must support a zero-arity `operator()` with a single
/// template parameter.
///
/// Example:
/// @code
/// template <typename LeftType, typename RightType>
/// struct MyStruct {
///   using L = LeftType;
///   using R = RightType;
/// };
///
/// struct MyFunctor {
///   template <typename T>
///   void operator()() {
///     std::cerr << typeid(T::L).name() << ", " << typeid(T::R).name() << "\n";
///   }
/// };
///
/// ExpandProduct<MyStruct, TypeList<double, float>, TypeList<int, char, char>>
///   (MyFunctor());
///
/// // Output:
/// // d, i
/// // d, c
/// // d, c
/// // f, i
/// // f, c
/// // f, c
/// @endcode
template <template <typename...> typename C, typename... Lists,
          typename Functor>
void ExpandProduct(Functor& f) {
  ExpandProductHelper<C, Lists...>(f, TypeList<>());
}

/// @brief In-place initializes an array from a vector.
///
/// This is required if `T` does not have a default constructor.
template <typename T, size_t... Is>
auto VecToArrayHelper(std::vector<T>&& v, std::index_sequence<Is...>)
    -> std::array<T, sizeof...(Is)> {
  if (v.size() < sizeof...(Is)) {
    LITERT_ABORT;
  }
  return std::array<T, sizeof...(Is)>{std::move(v[Is])...};
}

template <size_t N, typename T>
std::array<T, N> VecToArray(std::vector<T>&& v) {
  return VecToArrayHelper(std::move(v), std::make_index_sequence<N>());
}

/// @brief Constructs an object in-place.
/// @see `std::construct_at` from C++20.
template <class T, class... Args>
T* ConstructAt(T* p, Args&&... args) {
  return ::new (static_cast<void*>(p)) T(std::forward<Args>(args)...);
}

/// @brief Checks if a binary predicate holds for all elements of two zipped
///        iterables of the same size.
template <typename LeftVals, typename RightVals = LeftVals>
bool AllZip(const LeftVals& lhs, const RightVals& rhs,
            std::function<bool(const typename LeftVals::value_type&,
                               const typename RightVals::value_type&)>
                bin_pred) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (auto i = 0; i < lhs.size(); ++i) {
    if (!bin_pred(lhs.at(i), rhs.at(i))) {
      return false;
    }
  }
  return true;
}

/// @brief Checks if a binary predicate holds for any element of two zipped
///        iterables of the same size.
template <typename LeftVals, typename RightVals = LeftVals>
bool AnyZip(const LeftVals& lhs, const RightVals& rhs,
            std::function<bool(const typename LeftVals::value_type&,
                               const typename RightVals::value_type&)>
                bin_pred) {
  auto neg = [&](const auto& l, const auto& r) { return !bin_pred(l, r); };
  return !(AllZip(lhs, rhs, neg));
}

/// @brief Checks if an element exists in a range.
template <class It, class T>
bool Contains(It begin, It end, const T& val) {
  return std::find(begin, end, val) != end;
}

/// @brief Checks if an element satisfying a predicate exists in a range.
template <class It, class UPred>
bool ContainsIf(It begin, It end, UPred u_pred) {
  return std::find_if(begin, end, u_pred) != end;
}

/// @brief Finds the index of a given element if it is present.
template <class T, class It>
std::optional<size_t> FindInd(It begin, It end, T val) {
  auto it = std::find(begin, end, val);
  return (it == end) ? std::nullopt : std::make_optional(it - begin);
}

/// @brief Computes the average of a container.
template <typename It>
auto Avg(It begin, It end) -> RemoveCvRefT<decltype(*std::declval<It>())> {
  using T = decltype(Avg(begin, end));
  const auto size = std::distance(begin, end);
  if (size == 0) {
    return std::numeric_limits<T>::max();
  }
  return std::accumulate(begin, end, T{}) / static_cast<T>(size);
}

/// @brief Checks if a string starts with a given prefix.
/// @note `std::string::ends_with` is not available until C++20, and
/// `absl::StartsWith` is not portable.
inline bool StartsWith(absl::string_view str, absl::string_view prefix) {
  return str.size() >= prefix.size() &&
         std::equal(prefix.begin(), prefix.end(), str.begin());
}
/// @brief Checks if a string ends with a given suffix.
/// @note `std::string::ends_with` is not available until C++20, and
/// `absl::EndsWith` is not portable.
inline bool EndsWith(absl::string_view str, absl::string_view suffix) {
  return str.size() >= suffix.size() &&
         std::equal(suffix.rbegin(), suffix.rend(), str.rbegin());
}

/// @brief Compile-time strings.
template <size_t Len>
using StrLiteral = const char (&)[Len];

template <size_t Len>
using CtStrData = std::array<char, Len>;

template <size_t Len>
class CtStr {
 public:
  using Data = CtStrData<Len>;

  constexpr CtStr() = default;
  constexpr explicit CtStr(Data&& data) : data_(std::move(data)) {}

  template <size_t N, class I = std::make_index_sequence<N - 1>>
  constexpr explicit CtStr(StrLiteral<N> lit) : CtStr(lit, I{}) {}

  constexpr absl::string_view Str() const {
    return absl::string_view(data_.data(), data_.size());
  }

  //  private:
  template <size_t N, size_t... I>
  constexpr CtStr(StrLiteral<N> lit, std::index_sequence<I...>)
      : data_(Data({lit[I]...})) {}

  Data data_ = {};
};
template <size_t N>
CtStr(CtStrData<N>&&) -> CtStr<N>;
template <size_t N, class I = std::make_index_sequence<N - 1>>
CtStr(StrLiteral<N>) -> CtStr<N - 1>;

/// @brief Concatenates compile-time strings.
template <size_t... Ns>
constexpr auto CtStrConcat(StrLiteral<Ns>... strs) {
  using Out = CtStr<(Ns + ...) - sizeof...(Ns)>;
  typename Out::Data data = {};
  auto cur = data.begin();
  (
      [&]() {
        for (auto i = 0; i < Ns - 1; ++i) {
          *cur++ = strs[i];
        }
      }(),
      ...);
  return CtStr(std::move(data));
}

/// @brief A constexpr-friendly ceiling function.
template <typename T>
constexpr T Ceil(T val, T divisor) {
  return (val + divisor - 1) / divisor;
}

namespace internal {

/// @brief Calls a function `get` and asserts that its return value is equal to
///        the expected value.
template <class F, class Expected, typename... Args>
inline void AssertEq(F get, Expected expected, Args&&... args) {
  auto status = get(std::forward<Args>(args)...);
  ABSL_CHECK_EQ(status, expected);
}

/// @brief Calls a function `get` and asserts that it returns `true`.
template <class F, typename... Args>
inline void AssertTrue(F get, Args&&... args) {
  AssertEq(get, true, std::forward<Args>(args)...);
}

/// @brief Calls a function `get` and asserts that it returns an OK
/// `LiteRtStatus`.
template <class F, typename... Args>
inline void AssertOk(F get, Args&&... args) {
  AssertEq(get, kLiteRtStatusOk, std::forward<Args>(args)...);
}

}  // namespace internal
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_LITERT_DETAIL_H_
