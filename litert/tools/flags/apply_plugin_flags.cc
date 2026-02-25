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

#include "litert/tools/flags/apply_plugin_flags.h"

#include <cstdint>
#include <string>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/options/litert_compiler_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/options/litert_compiler_options.h"
#include "litert/tools/flags/flag_types.h"

ABSL_FLAG(std::string, cmd, "partition",
          "Routine to run (apply, partition, compile, info, noop).");

ABSL_FLAG(::litert::tools::IntList<int>, subgraphs,
          ::litert::tools::IntList<int>{},
          "If provides, only the subgraphs with the given indices "
          "are applied with the plugin.");

ABSL_FLAG(LiteRtCompilerOptionsPartitionStrategy, partition_strategy,
          kLiteRtCompilerOptionsPartitionStrategyDefault,
          "Partition strategy for the compiler.");

ABSL_FLAG(::litert::tools::IntListMap, skip_delegation_ops_by_subgraph,
          ::litert::tools::IntListMap{},
          "Operator ids to skip delegation to any vendors. "
          "A map of subgraph index to a comma-separated list of op ids. "
          "Format: index|id1,id2;index|id3-id5");

// NOLINTBEGIN(*alien-types*)
// TODO: Move absl parse/unparse function to same file as enum types if
// it becomes an issue.

bool AbslParseFlag(absl::string_view text,
                   LiteRtCompilerOptionsPartitionStrategy* partition_strategy,
                   std::string* error) {
  if (text == "default") {
    *partition_strategy = kLiteRtCompilerOptionsPartitionStrategyDefault;
    return true;
  }
  if (text == "weakly_connected") {
    *partition_strategy =
        kLiteRtCompilerOptionsPartitionStrategyWeaklyConnected;
    return true;
  }
  *error = "Unknown partition strategy";
  return false;
}

std::string AbslUnparseFlag(
    LiteRtCompilerOptionsPartitionStrategy partition_strategy) {
  switch (partition_strategy) {
    case kLiteRtCompilerOptionsPartitionStrategyDefault:
      return "default";
    case kLiteRtCompilerOptionsPartitionStrategyWeaklyConnected:
      return "weakly_connected";
  }
}
// NOLINTEND(*alien-types*)

namespace litert {

Expected<void> UpdateCompilerOptionsFromFlags(CompilerOptions& options) {
  LITERT_RETURN_IF_ERROR(
      options.SetPartitionStrategy(absl::GetFlag(FLAGS_partition_strategy)));

  // Parse skip delegation ids.
  for (const auto& [subgraph_index, op_ids] :
       absl::GetFlag(FLAGS_skip_delegation_ops_by_subgraph).elements) {
    // Cast int to uint32_t.
    std::vector<std::uint32_t> uint_op_ids;
    uint_op_ids.reserve(op_ids.size());
    for (const auto& op_id : op_ids) {
      uint_op_ids.push_back(static_cast<std::uint32_t>(op_id));
    }
    LITERT_RETURN_IF_ERROR(
        options.AddSkipDelegationOps(subgraph_index, uint_op_ids));
  }
  return {};
}
}  // namespace litert
