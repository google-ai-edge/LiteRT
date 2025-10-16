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

#include "litert/vendors/cc/namespace_heuristics.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace litert {
namespace {

// Calculates the Levenshtein edit distance between two strings with an optional
// threshold for early exit. Returns -1 if the distance exceeds the threshold.
int EditDistance(absl::string_view str1, absl::string_view str2,
                 const int threshold = std::numeric_limits<int>::max()) {
  absl::string_view shorter_str = (str1.length() < str2.length()) ? str1 : str2;
  absl::string_view longer_str = (str1.length() < str2.length()) ? str2 : str1;
  const int short_len = shorter_str.length();
  const int long_len = longer_str.length();

  std::vector<int> prev_row(short_len + 1);
  std::vector<int> curr_row(short_len + 1);

  // Initialize the first row with values from 0 to short_len.
  for (int j = 0; j <= short_len; ++j) {
    prev_row[j] = j;
  }

  for (int i = 1; i <= long_len; ++i) {
    curr_row[0] = i;  // Represents deletion of all characters in the prefix.
    int min_in_row = i;

    for (int j = 1; j <= short_len; ++j) {
      // If characters are the same, cost is 0, otherwise 1.
      const int cost = (longer_str[i - 1] == shorter_str[j - 1]) ? 0 : 1;
      // The edit distance is the minimum of:
      // Deletion: prev_row[j] + 1
      // Insertion: curr_row[j-1] + 1
      // Substitution: prev_row[j-1] + cost
      curr_row[j] = std::min(
          {prev_row[j] + 1, curr_row[j - 1] + 1, prev_row[j - 1] + cost});
      min_in_row = std::min(min_in_row, curr_row[j]);
    }

    // Swaps rows for the next iteration.
    prev_row.swap(curr_row);

    // Early exits if the minimum distance in the current row already exceeds
    // the threshold.
    if (min_in_row > threshold) {
      return -1;
    }
  }

  return prev_row[short_len];
}

// Preprocesses a candidate name for matching. It extracts the final path
// component (after the last '/'), removes non-alphabetic characters, and
// converts the result to lowercase.
std::string PreprocessCandidateName(absl::string_view name) {
  // Finds the start of the last path component.
  const size_t last_slash_pos = name.find_last_of('/');
  if (last_slash_pos != absl::string_view::npos) {
    name.remove_prefix(last_slash_pos + 1);
  }

  std::string result;
  result.reserve(name.length());

  // Builds the result string by appending only desired characters.
  for (const char c : name) {
    if (absl::ascii_isalpha(c)) {
      result += absl::ascii_tolower(c);
    }
  }
  return result;
}

}  // namespace

std::string TfliteNodeNamespaceHeuristic(
    absl::string_view op_name, absl::Span<const std::string> candidate_names) {
  if (candidate_names.empty()) {
    return "";
  }
  if (candidate_names.size() == 1) {
    return candidate_names.front();
  }

  const std::string processed_op_name =
      absl::StrReplaceAll(op_name, {{"_", ""}});

  std::string best_match = candidate_names.front();
  int min_distance = std::numeric_limits<int>::max();

  // A threshold to prune the search space. If the edit distance is
  // significantly larger than the op name length, the candidate is likely
  // irrelevant. The multiplier is a heuristic.
  constexpr int kDistanceThresholdMultiplier = 3;
  const int max_distance_threshold =
      kDistanceThresholdMultiplier * processed_op_name.length();

  // Iterates backwards, as later names in the candidate list might be more
  // relevant in some contexts (eg. more specific tensor names).
  for (auto it = candidate_names.rbegin(); it != candidate_names.rend(); ++it) {
    absl::string_view name = *it;
    const std::string processed_candidate = PreprocessCandidateName(name);

    if (processed_candidate.empty()) {
      continue;
    }

    // This allows for aggressive early termination.
    const int current_threshold =
        std::min(min_distance, max_distance_threshold);

    if (current_threshold == 0) break;

    const int cur_distance =
        EditDistance(processed_op_name, processed_candidate, current_threshold);

    // If EditDistance returned -1, it means the threshold was exceeded, so
    // this candidate is not better than our current best. We can skip it.
    if (cur_distance == -1) {
      continue;
    }

    if (cur_distance < min_distance) {
      min_distance = cur_distance;
      best_match = std::string(name);
    }
  }
  return best_match;
}

}  // namespace litert
