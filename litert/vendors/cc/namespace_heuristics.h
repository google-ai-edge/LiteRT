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

#ifndef ODML_LITERT_LITERT_VENDORS_CC_NAMESPACE_HEURISTICS_H_
#define ODML_LITERT_LITERT_VENDORS_CC_NAMESPACE_HEURISTICS_H_

#include <string>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_op_code.h"

namespace litert {

// Obtains the best matching namespace for the TFLite node based on the provided
// op name (eg. "conv_2d") and candidate names.
//
// The candidate names are obtained from the tensor names. The node namespace is
// obtained by the following steps:
// 1. If there are no candidate names, returns an empty string.
// 2. If there is only one candidate name, returns the candidate name.
// 3. If there are multiple candidate names, iterates backwards and returns the
// candidate name with the minimum edit distance to the op name.
// 4. If there are no candidate names that are close enough to the op name,
// returns the first candidate name.
std::string TfliteNodeNamespaceHeuristic(
    absl::string_view op_name, absl::Span<const std::string> candidate_names);

// Returns the TFLite operator name for the given LiteRtOpCode.
absl::string_view GetTfliteOpName(LiteRtOpCode op_code);

}  // namespace litert

#endif  // ODML_LITERT_LITERT_VENDORS_CC_NAMESPACE_HEURISTICS_H_
