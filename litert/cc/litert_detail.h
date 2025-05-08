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

#ifndef ODML_LITERT_LITERT_CC_LITERT_DETAIL_H_
#define ODML_LITERT_LITERT_CC_LITERT_DETAIL_H_

#include <array>
#include <cstddef>
#include <functional>
#include <optional>
#include <utility>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"

namespace litert {

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

 private:
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

#endif  // ODML_LITERT_LITERT_CC_LITERT_DETAIL_H_
