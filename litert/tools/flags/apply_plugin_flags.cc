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

#include <string>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/options/litert_compiler_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/options/compiler_options.h"
#include "litert/tools/flags/flag_types.h"

ABSL_FLAG(std::string, cmd, "partition",
          "Routine to run (apply, partition, compile, info, noop).");

ABSL_FLAG(::litert::tools::IntList, subgraphs, ::litert::tools::IntList{},
          "If provides, only the subgraphs with the given indices "
          "are applied with the plugin.");

ABSL_FLAG(LiteRtCompilerOptionsPartitionStrategy, partition_strategy,
          kLiteRtCompilerOptionsPartitionStrategyDefault,
          "Partition strategy for the compiler.");

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

Expected<CompilerOptions> CompilerOptionsFromFlags() {
  LITERT_ASSIGN_OR_RETURN(auto options, CompilerOptions::Create());

  const auto partition_strategy = absl::GetFlag(FLAGS_partition_strategy);
  LITERT_RETURN_IF_ERROR(options.SetPartitionStrategy(partition_strategy));

  return options;
}
}  // namespace litert
