// Copyright 2025 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_LITERT_SOURCE_LOCATION_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_LITERT_SOURCE_LOCATION_H_

#include <cstdint>

/// @file
/// @brief Defines a class for representing source code locations.

namespace litert {

#if defined(__has_builtin)
#define LITERT_HAS_BUILTIN(x) __has_builtin(x)
#else
#define LITERT_HAS_BUILTIN(x) 0
#endif

#if LITERT_HAS_BUILTIN(__builtin_FILE) && LITERT_HAS_BUILTIN(__builtin_LINE)
#define LITERT_INTERNAL_BUILTIN_FILE __builtin_FILE()
#define LITERT_INTERNAL_BUILTIN_LINE __builtin_LINE()
#else
#define LITERT_INTERNAL_BUILTIN_FILE "unknown"
#define LITERT_INTERNAL_BUILTIN_LINE 0
#endif

/// @brief Stores a file name and a line number.
///
/// This class mimics a subset of `std::source_location` and is intended to be
/// replaced by it when the project updates to C++20.
class SourceLocation {
  // We have this to prevent `current()` parameters from begin modified.
  struct PrivateTag {};

 public:
  /// @brief Creates a `SourceLocation` with the line and file corresponding to
  /// the call site.
  static constexpr SourceLocation current(
      PrivateTag = PrivateTag{},
      const char* file = LITERT_INTERNAL_BUILTIN_FILE,
      uint32_t line = LITERT_INTERNAL_BUILTIN_LINE) {
    return SourceLocation{file, line};
  }

  constexpr const char* file_name() const { return file_; }
  constexpr uint32_t line() const { return line_; }

 private:
  /// @brief Constructs a `SourceLocation` object.
  ///
  /// @note This is private, as `std::source_location` does not provide a way
  /// to manually construct a source location.
  constexpr SourceLocation(const char* file, uint32_t line)
      : file_(file), line_(line) {}

  const char* file_;
  uint32_t line_;
};

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_LITERT_SOURCE_LOCATION_H_
