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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_FLAGS_APPLY_PLUGIN_FLAGS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_FLAGS_APPLY_PLUGIN_FLAGS_H_

#include <cstddef>
#include <string>
#include <vector>

#include "absl/flags/declare.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/options/litert_compiler_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/options/litert_compiler_options.h"
#include "litert/tools/flags/flag_types.h"

ABSL_DECLARE_FLAG(std::string, cmd);

ABSL_DECLARE_FLAG(::litert::tools::IntList, subgraphs);

// Compiler plugin partition strategy flag.
ABSL_DECLARE_FLAG(LiteRtCompilerOptionsPartitionStrategy, partition_strategy);
bool AbslParseFlag(absl::string_view text,
                   LiteRtCompilerOptionsPartitionStrategy* partition_strategy,
                   std::string* error);
std::string AbslUnparseFlag(
    LiteRtCompilerOptionsPartitionStrategy partition_strategy);

ABSL_DECLARE_FLAG(size_t, compiler_options_max_partitions);

// Input shape overrides for compiler options.
// The three shape-specification modes are mutually exclusive:
//   --input_shape: Positional shapes in subgraph input index order (0, 1, ...),
//     formatted as "<d0>:<d1>:...". Must be passed once per subgraph input.
//   --input_shape_by_tensor: Shapes matched by internal subgraph tensor name,
//     formatted as "<tensor_name>@<d0>:<d1>:...".
//   --input_shape_by_sig: Shapes matched by SignatureDef input argument alias,
//     formatted as "<sig_input_name>@<d0>:<d1>:...".
//   --input_shape_sig_key: Target signature key selecting which signature the
//     above shape overrides apply to (defaults to "", the first signature).
ABSL_DECLARE_FLAG(std::vector<std::string>, input_shape);
ABSL_DECLARE_FLAG(std::vector<std::string>, input_shape_by_tensor);
ABSL_DECLARE_FLAG(std::vector<std::string>, input_shape_by_sig);
ABSL_DECLARE_FLAG(std::string, input_shape_sig_key);

namespace litert {

Expected<void> UpdateCompilerOptionsFromFlags(CompilerOptions& options);

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_FLAGS_APPLY_PLUGIN_FLAGS_H_
