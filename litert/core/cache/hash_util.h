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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CORE_CACHE_HASH_UTIL_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CORE_CACHE_HASH_UTIL_H_

#include <cstdint>
#include <functional>

namespace litert {

inline void HashCombine(uint64_t& seed) {}  // NOLINT

template <typename T, typename... Rest>
inline void HashCombine(uint64_t& seed, const T& v, Rest... rest) {
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  HashCombine(seed, rest...);
}

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CORE_CACHE_HASH_UTIL_H_
