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

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_DISPATCH_ACTIVE_FUNCTIONS_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_DISPATCH_ACTIVE_FUNCTIONS_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl

namespace litert::qnn {

// Returns the subset of `binary_graphs` (the graphs stored in one QNN context
// binary) that should be enabled when creating a QNN context from that binary:
// the graphs listed in `active_functions` plus `function_name`, which is the
// graph that is being requested right now. An empty result means that all the
// graphs should be enabled, i.e. no selection is possible or needed.
inline std::vector<std::string> SelectGraphsToEnable(
    absl::Span<const std::string> binary_graphs,
    const absl::flat_hash_set<std::string>& active_functions,
    absl::string_view function_name) {
  if (active_functions.empty()) {
    return {};
  }
  std::vector<std::string> enabled;
  for (const std::string& graph : binary_graphs) {
    if (graph == function_name || active_functions.contains(graph)) {
      enabled.push_back(graph);
    }
  }
  if (enabled.empty() || enabled.size() == binary_graphs.size()) {
    return {};
  }
  return enabled;
}

}  // namespace litert::qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_DISPATCH_ACTIVE_FUNCTIONS_H_
