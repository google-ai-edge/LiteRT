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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_FLAGS_FLAG_TYPES_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_FLAGS_FLAG_TYPES_H_

#include <map>
#include <string>
#include <vector>

#include "absl/flags/marshalling.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

// Int list list flag implementation (this is not included in absl::flags).

namespace litert::tools {

template <typename T = int>
struct IntList {
  std::vector<T> elements;
};

template <typename T = int>
std::string AbslUnparseFlag(const IntList<T>& list) {
  return absl::StrJoin(list.elements, ",", [](std::string* out, T element) {
    out->append(absl::UnparseFlag(element));
  });
}

template <typename T = int>
bool AbslParseFlag(absl::string_view text, IntList<T>* list,
                   std::string* error) {
  list->elements.clear();
  if (text.empty()) {
    return true;
  }
  for (const auto& part : absl::StrSplit(text, ',')) {
    T element;
    if (!absl::ParseFlag(part, &element, error)) {
      return false;
    }
    list->elements.push_back(element);
  }
  return true;
}

struct IntListMap {
  std::map<int, std::vector<int>> elements;
};

std::string AbslUnparseFlag(const IntListMap& list);

bool AbslParseFlag(absl::string_view text, IntListMap* list, std::string* error);

}  // namespace litert::tools

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_FLAGS_FLAG_TYPES_H_
