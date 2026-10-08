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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_UTILS_SOURCE_LOCATION_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_UTILS_SOURCE_LOCATION_H_

#ifdef __cpp_lib_source_location

#include <source_location>  // NOLINT(build/c++20): needed for OSS.

namespace litert::tensor {

using source_location = std::source_location;

}  // namespace litert::tensor

#else

#include <cstdint>

namespace litert::tensor {

#if defined(__has_builtin)
#define LITERT_HAS_BUILTIN(x) __has_builtin(x)
#else
#define LITERT_HAS_BUILTIN(x) 0
#endif

#if (LITERT_HAS_BUILTIN(__builtin_FILE) &&  \
     LITERT_HAS_BUILTIN(__builtin_LINE)) || \
    (defined(__GNUC__) && (__GNUC__ >= 7))
#define LITERT_INTERNAL_BUILTIN_FILE __builtin_FILE()
#define LITERT_INTERNAL_BUILTIN_LINE __builtin_LINE()
#else
#define LITERT_INTERNAL_BUILTIN_FILE "unknown"
#define LITERT_INTERNAL_BUILTIN_LINE 0
#endif

/// @brief Stores a file name and a line number.
///
/// This class mimics a subset of `source_location` and is intended to be
/// replaced by it when the project updates to C++20.
class source_location {
  // We have this to prevent `current()` parameters from being modified.
  struct PrivateTag {};

 public:
  constexpr source_location() noexcept = default;
  source_location(const source_location&) noexcept = default;
  source_location(source_location&&) noexcept = default;
  source_location& operator=(const source_location&) noexcept = default;
  source_location& operator=(source_location&&) noexcept = default;

  /// @brief Creates a `SourceLocation` with the line and file corresponding to
  /// the call site.
  static constexpr source_location current(
      PrivateTag = PrivateTag{},
      const char* file = LITERT_INTERNAL_BUILTIN_FILE,
      uint32_t line = LITERT_INTERNAL_BUILTIN_LINE) {
    return source_location{file, line};
  }

  constexpr const char* file_name() const { return file_; }
  constexpr uint32_t line() const { return line_; }

 private:
  /// @brief Constructs a `SourceLocation` object.
  ///
  /// @note This is private, as `source_location` does not provide a way
  /// to manually construct a source location.
  constexpr source_location(const char* file, uint32_t line)
      : file_(file), line_(line) {}

  const char* file_ = "unknown";
  uint32_t line_ = 0;
};

}  // namespace litert::tensor

#endif

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_UTILS_SOURCE_LOCATION_H_
