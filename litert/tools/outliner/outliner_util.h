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

#ifndef ODML_LITERT_LITERT_TOOLS_OUTLINER_OUTLINER_UTIL_H_
#define ODML_LITERT_LITERT_TOOLS_OUTLINER_OUTLINER_UTIL_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "litert/core/model/model.h"

namespace litert::tools {

struct OutlinerOptions {
  std::vector<std::string> start_tensors;
  std::vector<std::string> end_tensors;
  std::string composite_name;
  absl::flat_hash_map<std::string, std::string> attributes;
};

// Identifies and outlines a subgraph into a StableHLO Composite op.
absl::Status OutlineSubgraph(LiteRtModelT& model, size_t subgraph_index,
                             const OutlinerOptions& options);

}  // namespace litert::tools

#endif  // ODML_LITERT_LITERT_TOOLS_OUTLINER_OUTLINER_UTIL_H_
