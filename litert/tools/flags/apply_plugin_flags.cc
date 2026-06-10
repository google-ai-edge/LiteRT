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
#include "litert/cc/options/litert_compiler_options.h"
#include "litert/tools/flags/flag_types.h"

ABSL_FLAG(std::string, cmd, "partition",
          "Routine to run (apply, partition, compile, info, noop).");

ABSL_FLAG(::litert::tools::IntList, subgraphs, ::litert::tools::IntList{},
          "If provides, only the subgraphs with the given indices "
          "are applied with the plugin.");

ABSL_FLAG(LiteRtCompilerOptionsPartitionStrategy, partition_strategy,
          kLiteRtCompilerOptionsPartitionStrategyDefault,
          "Partition strategy for the compiler. One of: default, "
          "weakly_connected, transformer_block, transformer_layer_cut.");

ABSL_FLAG(std::string, transformer_layer_cuts, "",
          "Per-signature layer-cut spec for the transformer_layer_cut strategy. "
          "Format: ';'-separated 'signature=cuts' groups, each cuts a "
          "','-separated list of layer indices; a bare list (or empty key) is "
          "the default for all signatures. Examples: "
          "--transformer_layer_cuts=16  or  "
          "--transformer_layer_cuts=\"prefill_128=16,32;decode=8\". Only used "
          "with --partition_strategy=transformer_layer_cut.");

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
  if (text == "transformer_block") {
    *partition_strategy =
        kLiteRtCompilerOptionsPartitionStrategyTransformerBlock;
    return true;
  }
  if (text == "transformer_layer_cut") {
    *partition_strategy =
        kLiteRtCompilerOptionsPartitionStrategyTransformerLayerCut;
    return true;
  }
  *error =
      "Unknown partition strategy (expected: default, weakly_connected, "
      "transformer_block, transformer_layer_cut)";
  return false;
}

std::string AbslUnparseFlag(
    LiteRtCompilerOptionsPartitionStrategy partition_strategy) {
  switch (partition_strategy) {
    case kLiteRtCompilerOptionsPartitionStrategyDefault:
      return "default";
    case kLiteRtCompilerOptionsPartitionStrategyWeaklyConnected:
      return "weakly_connected";
    case kLiteRtCompilerOptionsPartitionStrategyTransformerBlock:
      return "transformer_block";
    case kLiteRtCompilerOptionsPartitionStrategyTransformerLayerCut:
      return "transformer_layer_cut";
  }
  return "default";
}
// NOLINTEND(*alien-types*)

namespace litert {

Expected<void> UpdateCompilerOptionsFromFlags(CompilerOptions& options) {
  LITERT_RETURN_IF_ERROR(
      options.SetPartitionStrategy(absl::GetFlag(FLAGS_partition_strategy)));

  const auto cuts_spec = absl::GetFlag(FLAGS_transformer_layer_cuts);
  if (!cuts_spec.empty()) {
    LITERT_RETURN_IF_ERROR(options.SetTransformerLayerCuts(cuts_spec));
  }

  return {};
}
}  // namespace litert
