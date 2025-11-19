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

// Use std::conditional to support sequence of if, if-else..., else
// std::monostate
//
// Example:
//   using TestVal = int;
//   using Test = SelectT<std::is_floating_point<TestVal>, double,
//      std::is_integral<TestVal>, long, std::is_class<TestVal>, std::string>;
//   static_assert(std::is_same_v<Test, long>);
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

// Utillity for specializing a template against the cartesian product of given
// type lists.
//
// The number of lists passed must match the number of template parameters of
// the template being specialized, which is C. This template cannot have
// non-type template parameters, but that behavior can be achieved with the
// standard `std::integral_constant` template.
//    The Functor is used to perform a call back templatated on each of the
// members of the cartesian product. It must support a zero-arity void
// () operatory with a single template parameter.
//
// EXAMPLE:
//
// template <typename LeftType, typename RightType>
// struct MyStruct {
//   using L = LeftType;
//   using R = RightType;
// };
//
// struct MyFunctor {
//   template <typename T>
//   void operator()() {
//     std::cerr << typeid(T::L).name() << ", " << typeid(T::R).name() << "\n";
//   }
// };
//
//  ExpandProduct<MyStruct, TypeList<double, float>, TypeList<int, char,
//    char>(MyFunctor());
// /*
//  d, i
//  d, c
//  d, c
//  f, i
//  f, c
//  f, c
// */
template <template <typename...> typename C, typename... Lists,
          typename Functor>
void ExpandProduct(Functor& f) {
  ExpandProductHelper<C, Lists...>(f, TypeList<>());
}

// Inplace initialize an array from vector. This is required if T offers
// no default constructor.
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

// See "std::construct_at" from C++20.
template <class T, class... Args>
T* ConstructAt(T* p, Args&&... args) {
  return ::new (static_cast<void*>(p)) T(std::forward<Args>(args)...);
}

// Reduce all over zipped iters of same size.
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

// Reduce any over zipped iters of same size.
template <typename LeftVals, typename RightVals = LeftVals>
bool AnyZip(const LeftVals& lhs, const RightVals& rhs,
            std::function<bool(const typename LeftVals::value_type&,
                               const typename RightVals::value_type&)>
                bin_pred) {
  auto neg = [&](const auto& l, const auto& r) { return !bin_pred(l, r); };
  return !(AllZip(lhs, rhs, neg));
}

// Does element exist in range.
template <class It, class T>
bool Contains(It begin, It end, const T& val) {
  return std::find(begin, end, val) != end;
}

// Does element exist in range satisfying pred.
template <class It, class UPred>
bool ContainsIf(It begin, It end, UPred u_pred) {
  return std::find_if(begin, end, u_pred) != end;
}

// Get the ind of the given element if it is present.
template <class T, class It>
std::optional<size_t> FindInd(It begin, It end, T val) {
  auto it = std::find(begin, end, val);
  return (it == end) ? std::nullopt : std::make_optional(it - begin);
}

// Average the container.
template <typename It>
auto Avg(It begin, It end) -> RemoveCvRefT<decltype(*std::declval<It>())> {
  using T = decltype(Avg(begin, end));
  const auto size = std::distance(begin, end);
  if (size == 0) {
    return std::numeric_limits<T>::max();
  }
  return std::accumulate(begin, end, T{}) / static_cast<T>(size);
}

// Check string prefix/suffix. Std ends with not supported until
// C++20 & absl::StartsWith/EndsWith not portable.
inline bool StartsWith(absl::string_view str, absl::string_view prefix) {
  return str.size() >= prefix.size() &&
         std::equal(prefix.begin(), prefix.end(), str.begin());
}
inline bool EndsWith(absl::string_view str, absl::string_view suffix) {
  return str.size() >= suffix.size() &&
         std::equal(suffix.rbegin(), suffix.rend(), str.rbegin());
}

// Compile time strings.

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

// Concat compile time strings.

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

// Constexpr friendly ceiling.
template <typename T>
constexpr T Ceil(T val, T divisor) {
  return (val + divisor - 1) / divisor;
}

namespace internal {

// Call function "get" and assert it returns value equal to given expected
// value.
template <class F, class Expected, typename... Args>
inline void AssertEq(F get, Expected expected, Args&&... args) {
  auto status = get(std::forward<Args>(args)...);
  ABSL_CHECK_EQ(status, expected);
}

// Call function "get" and assert it returns true.
template <class F, typename... Args>
inline void AssertTrue(F get, Args&&... args) {
  AssertEq(get, true, std::forward<Args>(args)...);
}

// Call function "get" and assert it returns an OK LiteRtStatus.
template <class F, typename... Args>
inline void AssertOk(F get, Args&&... args) {
  AssertEq(get, kLiteRtStatusOk, std::forward<Args>(args)...);
}

}  // namespace internal
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_LITERT_DETAIL_H_
