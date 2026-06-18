/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_INTERNAL_TYPE_ID_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_INTERNAL_TYPE_ID_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::tensor::internal {

namespace type_id {

// Computes the FNV-1a 64-bit hash function.
constexpr uint64_t ConstexprHash(absl::string_view str) {
  uint64_t hash = 14695981039346656037ull;
  for (const char c : str) {
    hash ^= static_cast<uint64_t>(c);
    hash *= 1099511628211ull;
  }
  return hash;
}

constexpr absl::string_view::size_type find(absl::string_view str,
                                            absl::string_view substr) {
  const absl::string_view::size_type len = substr.length();
  for (absl::string_view::size_type pos = 0; pos < str.length() - len; ++pos) {
    if (str.substr(pos, len) == substr) {
      return pos;
    }
  }
  return absl::string_view::npos;
}

template <size_t N>
constexpr absl::string_view::size_type rfind(
    absl::string_view str, std::array<absl::string_view, N> substrs) {
  for (absl::string_view substr : substrs) {
    const absl::string_view::size_type len = substr.length();
    for (absl::string_view::size_type pos = str.length() - len; pos > 0;
         --pos) {
      if (str.substr(pos, len) == substr) {
        return pos;
      }
    }
    if (str.substr(0, len) == substr) {
      return 0;
    }
  }
  return absl::string_view::npos;
}

// Extracts the type name from the pretty function.
template <class T>
constexpr absl::string_view ExtractTypeName() {
#if defined(__clang__)
  constexpr absl::string_view name = __PRETTY_FUNCTION__;
  constexpr absl::string_view prefix = "[T = ";
  constexpr std::array<absl::string_view, 1> suffixes = {"]"};
#elif defined(__GNUC__)
  constexpr absl::string_view name = __PRETTY_FUNCTION__;
  constexpr absl::string_view prefix = "[with T = ";
  constexpr std::array<absl::string_view, 2> suffixes = {";", "]"};
#elif defined(_MSC_VER)
  constexpr absl::string_view name = __FUNCSIG__;
  constexpr absl::string_view prefix = "ExtractTypeName<";
  constexpr std::array<absl::string_view, 1> suffixes = {">(void)"};
#else
#error "Unsupported compiler: function signature macro not available."
#endif
  constexpr absl::string_view::size_type start =
      find(name, prefix) + prefix.size();
  constexpr absl::string_view::size_type finish = rfind(name, suffixes);
  constexpr absl::string_view raw_name = name.substr(start, finish - start);
  return raw_name;
}

// Implements `strlen` on an array.
template <size_t N>
constexpr size_t DataLen(const std::array<char, N>& arr) {
  for (size_t i = 0; i < N; ++i) {
    if (arr[i] == 0) {
      return i;
    }
  }
  return N;
}

// Returns true if `c` is a character that is part of an identifier.
constexpr bool IsIdentifierChar(const char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
         (c >= '0' && c <= '9') || c == '_';
}

// Tries to normalize the type name across compilers.
//
// - Removes spaces.
// - Removes 'struct', 'class', 'union', 'enum' keywords.
template <size_t N>
constexpr auto FilterName(absl::string_view str) {
  std::array<char, N + 1> data{};
  bool previous_is_id_char = false;
  constexpr std::array<absl::string_view, 4> keywords = {"struct ", "class ",
                                                         "union ", "enum "};
  for (int i = 0, j = 0; i < str.size(); ++i) {
    if (!previous_is_id_char) {
      for (absl::string_view kw : keywords) {
        if (str.substr(i, kw.size()) == kw) {
          i += kw.size();
        }
      }
    }
    if (str[i] == ' ') {
      previous_is_id_char = false;
      continue;
    }
    previous_is_id_char = IsIdentifierChar(str[i]);
    data[j++] = str[i];
  }
  return data;
}

template <size_t Size, size_t N>
constexpr auto SubArray(const std::array<char, N>& arr) {
  std::array<char, Size + 1> res{};
  for (size_t i = 0; i < Size; ++i) {
    res[i] = arr[i];
  }
  return res;
}

template <class T>
constexpr auto ComputeName() {
  constexpr absl::string_view name = ExtractTypeName<T>();
  constexpr auto data = FilterName<name.size()>(name);
  return SubArray<DataLen(data)>(data);
}

template <class T>
static constexpr auto kTypeName = ComputeName<T>();

}  // namespace type_id

// Generate type identifiers.
//
// This class implements a type identification that we use to explicitely
// opt-into to get the concrete class of some of the type erased Tensor API
// types.
//
// > Warning: This tries its best to be stable across compilers but this still
// > relies on tricks to extract the class name from the compiler. It's enough
// > for what we want to do but has severe limitations.
// >
// > A non-exhaustive list of these:
//
// > - Template type with default template arguments will not generate stable
// >   identifiers across compilers.
// > - Explicitly specialized template classes will not generate stable
// >   identifiers across compilers.
class TypeId {
 public:
  template <class T>
  static constexpr absl::string_view Name() {
    return absl::string_view(type_id::kTypeName<T>.data(),
                             type_id::kTypeName<T>.size() - 1);
  }

  template <typename T>
  static constexpr TypeId Get() {
    return GetExact<std::decay_t<T>>();
  }

  template <typename T>
  static constexpr TypeId GetExact() {
    constexpr TypeId id(type_id::ConstexprHash(Name<T>()));
    return id;
  }

  constexpr bool operator==(const TypeId& other) const {
    return id_ == other.id_;
  }
  constexpr bool operator!=(const TypeId& other) const {
    return id_ != other.id_;
  }

  template <class Sink>
  friend void AbslStringify(Sink& sink, TypeId id) {
    sink.Append(std::to_string(id.id_));
  }

 private:
  explicit constexpr TypeId(uint64_t id) : id_(id) {}
  uint64_t id_;
};

}  // namespace litert::tensor::internal

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_INTERNAL_TYPE_ID_H_
