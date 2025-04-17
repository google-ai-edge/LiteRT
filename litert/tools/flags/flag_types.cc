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

#include "litert/tools/flags/flag_types.h"

#include <string>
#include <vector>

#include "absl/flags/marshalling.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl

namespace litert::tools {

std::string AbslUnparseFlag(const IntList& list) {
  return absl::StrJoin(list.elements, ",", [](std::string* out, int element) {
    out->append(absl::UnparseFlag(element));
  });
}

bool AbslParseFlag(absl::string_view text, IntList* list, std::string* error) {
  list->elements.clear();
  if (text.empty()) {
    return true;
  }
  for (const auto& part : absl::StrSplit(text, ',')) {
    int element;
    if (!absl::ParseFlag(part, &element, error)) {
      return false;
    }
    list->elements.push_back(element);
  }
  return true;
}

}  // namespace litert::tools
