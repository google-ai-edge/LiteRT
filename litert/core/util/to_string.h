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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_UTIL_TO_STRING_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_UTIL_TO_STRING_H_

#include <sstream>
#include <string>
#include <vector>

namespace litert::internal {

template <typename T>
std::string ToString(const T& t) {
  std::ostringstream os;
  os << t;
  return os.str();
}

template <typename Iter>
std::string ToString(Iter begin, Iter end) {
  std::ostringstream os;
  os << "{";
  for (auto i = begin; i != end; ++i) {
    os << ToString(*i) << ", ";
  }
  os << "}";
  return os.str();
}

template <typename T>
std::string ToString(const std::vector<T>& v) {
  return ToString(v.begin(), v.end());
}

}  // namespace litert::internal

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_UTIL_TO_STRING_H_
