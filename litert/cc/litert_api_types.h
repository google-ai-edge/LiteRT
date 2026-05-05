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

#ifndef ODML_LITERT_LITERT_CC_LITERT_API_TYPES_H_
#define ODML_LITERT_LITERT_CC_LITERT_API_TYPES_H_

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef LITERT_NO_ABSL
#include <span>
#else
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/inlined_vector.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#endif  // LITERT_NO_ABSL

namespace litert {

#ifdef LITERT_NO_ABSL
using StringView = std::string_view;

template <typename T>
using Span = std::span<T>;

template <typename K, typename V, typename Hash = std::hash<K>,
          typename Eq = std::equal_to<K>,
          typename Alloc = std::allocator<std::pair<const K, V>>>
using FlatHashMap = std::unordered_map<K, V, Hash, Eq, Alloc>;

template <typename Signature>
using AnyInvocable = std::function<Signature>;

template <typename T, std::size_t N>
using SmallVector = std::vector<T>;
#else
using StringView = absl::string_view;

template <typename T>
using Span = absl::Span<T>;

template <typename K, typename V, typename... Args>
using FlatHashMap = absl::flat_hash_map<K, V, Args...>;

template <typename Signature>
using AnyInvocable = absl::AnyInvocable<Signature>;

template <typename T, std::size_t N>
using SmallVector = absl::InlinedVector<T, N>;
#endif  // LITERT_NO_ABSL

namespace internal {

[[noreturn]] inline void CheckFailed(const char* expression, const char* file,
                                     int line) {
  std::cerr << "LiteRT check failed: " << expression << " at " << file << ":"
            << line << '\n';
  std::abort();
}

inline void Check(bool condition, const char* expression, const char* file,
                  int line) {
  if (!condition) {
    CheckFailed(expression, file, line);
  }
}

template <typename T>
T* DieIfNull(T* ptr, const char* expression, const char* file, int line) {
  if (ptr == nullptr) {
    CheckFailed(expression, file, line);
  }
  return ptr;
}

template <typename T>
constexpr Span<T> MakeSpan(T* data, std::size_t size) {
  return Span<T>(data, size);
}

template <typename T>
constexpr Span<const T> MakeConstSpan(const T* data, std::size_t size) {
  return Span<const T>(data, size);
}

template <typename T, std::size_t N>
constexpr Span<const T> MakeConstSpan(const T (&data)[N]) {
  return Span<const T>(data, N);
}

template <typename Container>
constexpr auto MakeSpan(Container& container)
    -> Span<typename Container::value_type> {
  return Span<typename Container::value_type>(container.data(),
                                              container.size());
}

template <typename Container>
constexpr auto MakeConstSpan(const Container& container)
    -> Span<const typename Container::value_type> {
  return Span<const typename Container::value_type>(container.data(),
                                                    container.size());
}

inline constexpr StringView NullSafeStringView(const char* str) {
  return str == nullptr ? StringView() : StringView(str);
}

template <typename CleanupFn>
class Cleanup {
 public:
  explicit Cleanup(CleanupFn cleanup_fn)
      : cleanup_fn_(std::move(cleanup_fn)), active_(true) {}
  Cleanup(Cleanup&& other)
      : cleanup_fn_(std::move(other.cleanup_fn_)), active_(other.active_) {
    other.active_ = false;
  }
  Cleanup(const Cleanup&) = delete;
  Cleanup& operator=(const Cleanup&) = delete;
  Cleanup& operator=(Cleanup&&) = delete;
  ~Cleanup() {
    if (active_) {
      cleanup_fn_();
    }
  }

 private:
  CleanupFn cleanup_fn_;
  bool active_;
};

template <typename CleanupFn>
Cleanup<std::decay_t<CleanupFn>> MakeCleanup(CleanupFn&& cleanup_fn) {
  return Cleanup<std::decay_t<CleanupFn>>(
      std::forward<CleanupFn>(cleanup_fn));
}

}  // namespace internal

}  // namespace litert

#define LITERT_INTERNAL_CHECK(expr) \
  ::litert::internal::Check(static_cast<bool>(expr), #expr, __FILE__, __LINE__)

#define LITERT_INTERNAL_CHECK_EQ(lhs, rhs) \
  LITERT_INTERNAL_CHECK((lhs) == (rhs))

#ifndef NDEBUG
#define LITERT_INTERNAL_DCHECK(expr) LITERT_INTERNAL_CHECK(expr)
#else
#define LITERT_INTERNAL_DCHECK(expr) \
  do {                               \
  } while (false)
#endif

#define LITERT_INTERNAL_DIE_IF_NULL(expr) \
  ::litert::internal::DieIfNull((expr), #expr, __FILE__, __LINE__)

#endif  // ODML_LITERT_LITERT_CC_LITERT_API_TYPES_H_
