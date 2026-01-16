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

#include <map>
#include <string>
#include <vector>

#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"

namespace litert::tools {

std::string AbslUnparseFlag(const IntListMap& list) {
  std::vector<std::string> parts;
  for (const auto& [index, elements] : list.elements) {
    if (elements.empty()) {
      continue;
    }
    std::string part = absl::StrCat(index, "|");
    absl::StrAppend(&part, absl::StrJoin(elements, ","));
    parts.push_back(part);
  }
  return absl::StrJoin(parts, ";");
}

bool AbslParseFlag(absl::string_view text, IntListMap* list,
                   std::string* error) {
  // text will have format like 0|1,2,3;1|2-5;2|1,2,6-10
  list->elements.clear();
  if (text.empty()) {
    return true;
  }
  for (const auto& part : absl::StrSplit(text, ';')) {
    std::vector<std::string> chunks = absl::StrSplit(part, '|');
    if (chunks.size() != 2) {
      *error = absl::StrCat("Invalid format: ", part);
      return false;
    }
    int index;
    if (!absl::SimpleAtoi(chunks[0], &index)) {
      *error = absl::StrCat("Invalid index: ", chunks[0]);
      return false;
    }

    if (index < 0) {
      *error = absl::StrCat("Index must be non-negative: ", index);
      return false;
    }

    for (const auto& value_part : absl::StrSplit(chunks[1], ',')) {
      std::vector<std::string> range_chunks = absl::StrSplit(value_part, '-');
      if (range_chunks.size() == 1) {
        int value;
        if (!absl::SimpleAtoi(range_chunks[0], &value)) {
          *error = absl::StrCat("Invalid value: ", range_chunks[0]);
          return false;
        }
        list->elements[index].push_back(value);
      } else if (range_chunks.size() == 2) {
        int start, end;
        if (!absl::SimpleAtoi(range_chunks[0], &start) ||
            !absl::SimpleAtoi(range_chunks[1], &end)) {
          *error = absl::StrCat("Invalid range: ", value_part);
          return false;
        }
        if (start > end) {
          *error = absl::StrCat("Invalid range (start > end): ", value_part);
          return false;
        }
        for (int i = start; i <= end; ++i) {
          list->elements[index].push_back(i);
        }
      } else {
        *error = absl::StrCat("Invalid value part: ", value_part);
        return false;
      }
    }
  }
  return true;
}

}  // namespace litert::tools
