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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_API_TYPES_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_API_TYPES_H_

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <type_traits>
#include <utility>

#ifdef LITERT_NO_ABSL
#include <functional>
#include <span>  // NOLINT
#include <string_view>
#include <unordered_map>
#include <vector>
#else  // LITERT_NO_ABSL
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/inlined_vector.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#endif  // LITERT_NO_ABSL

namespace litert {

// Minimum C++ standard by API mode:
// - Default mode uses Abseil-backed vocabulary types and is compatible with the
//   current LiteRT C++17 build configuration.
// - LITERT_NO_ABSL mode maps to standard library type definitions. The
//   strongest requirement here is std::span, which requires C++20. Other
//   std-backed aliases below require C++17 or older.
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
#else   // LITERT_NO_ABSL
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

// Checks an internal invariant. If condition is false, this function always
// calls CheckFailed(), which logs the failed expression and terminates the
// process with std::abort(). It does not become a no-op in release builds.
// Build-variant behavior is provided by the LITERT_INTERNAL_DCHECK macro below,
// which calls this function only when NDEBUG is not defined.
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
constexpr Span<T> MakeLiteRtSpan(T* data, std::size_t size) {
  return Span<T>(data, size);
}

template <typename T>
constexpr Span<const T> MakeLiteRtConstSpan(const T* data, std::size_t size) {
  return Span<const T>(data, size);
}

template <typename T, std::size_t N>
constexpr Span<const T> MakeLiteRtConstSpan(const T (&data)[N]) {
  return Span<const T>(data, N);
}

template <typename Container>
constexpr auto MakeLiteRtSpan(Container& container)
    -> Span<typename Container::value_type> {
  return Span<typename Container::value_type>(container.data(),
                                              container.size());
}

template <typename Container>
constexpr auto MakeLiteRtConstSpan(const Container& container)
    -> Span<const typename Container::value_type> {
  return Span<const typename Container::value_type>(container.data(),
                                                    container.size());
}

constexpr StringView NullSafeStringView(const char* str) {
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
  return Cleanup<std::decay_t<CleanupFn>>(std::forward<CleanupFn>(cleanup_fn));
}

}  // namespace internal

}  // namespace litert

#define LITERT_INTERNAL_CHECK(expr) \
  ::litert::internal::Check(static_cast<bool>(expr), #expr, __FILE__, __LINE__)

#define LITERT_INTERNAL_CHECK_EQ(lhs, rhs) LITERT_INTERNAL_CHECK((lhs) == (rhs))

#ifndef NDEBUG
#define LITERT_INTERNAL_DCHECK(expr) LITERT_INTERNAL_CHECK(expr)
#else
#define LITERT_INTERNAL_DCHECK(expr) \
  do {                               \
  } while (false)
#endif

#define LITERT_INTERNAL_DIE_IF_NULL(expr) \
  ::litert::internal::DieIfNull((expr), #expr, __FILE__, __LINE__)

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_API_TYPES_H_
