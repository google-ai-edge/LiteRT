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

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
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
          "Partition strategy for the compiler.");

ABSL_FLAG(size_t, compiler_options_max_partitions, 0,
          "Maximum number of partitions allowed. 0 means unlimited.");

ABSL_FLAG(std::vector<std::string>, input, {},
          "Input shapes, e.g. --input=1:224:224:3 --input=1:10.");
ABSL_FLAG(
    std::string, signature, "",
    "Signature key to use. Defaults to the first signature if not provided.");
ABSL_FLAG(std::vector<std::string>, input_name, {},
          "Input shapes by tensor name, e.g. --input_name=arg0@1:224:224:3.");
ABSL_FLAG(std::vector<std::string>, signature_name, {},
          "Input shapes by signature name, e.g. "
          "--signature_name=image@1:224:224:3.");

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
namespace {

// Parses "<d0>:<d1>:..." into a list of dims. "-1" keeps a dim dynamic.
Expected<std::vector<int32_t>> ParseShapeSpec(absl::string_view shape_spec) {
  if (shape_spec.empty()) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Shape specification cannot be empty");
  }
  std::vector<int32_t> dims;
  for (absl::string_view dim_str : absl::StrSplit(shape_spec, ':')) {
    int32_t dim = 0;
    if (!absl::SimpleAtoi(dim_str, &dim)) {
      return Unexpected(
          kLiteRtStatusErrorInvalidArgument,
          absl::StrCat("Invalid shape specification: ", shape_spec));
    }
    dims.push_back(dim);
  }
  return dims;
}

// Parses "<name>@<d0>:<d1>:..." into (name, dims).
Expected<std::pair<std::string, std::vector<int32_t>>> ParseNamedShapeSpec(
    absl::string_view spec) {
  const std::pair<absl::string_view, absl::string_view> parts =
      absl::StrSplit(spec, absl::MaxSplits('@', 1));
  if (parts.first.empty() || parts.second.empty()) {
    return Unexpected(
        kLiteRtStatusErrorInvalidArgument,
        absl::StrCat("Expected <name>@<d0>:<d1>:..., got: ", spec));
  }
  LITERT_ASSIGN_OR_RETURN(auto dims, ParseShapeSpec(parts.second));
  return std::make_pair(std::string(parts.first), std::move(dims));
}

}  // namespace

Expected<void> UpdateCompilerOptionsFromFlags(CompilerOptions& options) {
  LITERT_RETURN_IF_ERROR(
      options.SetPartitionStrategy(absl::GetFlag(FLAGS_partition_strategy)));
  LITERT_RETURN_IF_ERROR(options.SetMaxPartitions(static_cast<size_t>(
      absl::GetFlag(FLAGS_compiler_options_max_partitions))));

  const std::string signature_key = absl::GetFlag(FLAGS_signature);

  for (const std::string& spec : absl::GetFlag(FLAGS_input)) {
    LITERT_ASSIGN_OR_RETURN(auto dims, ParseShapeSpec(spec));
    LITERT_RETURN_IF_ERROR(options.AddPositionalInputShape(
        signature_key, absl::MakeConstSpan(dims)));
  }

  for (const std::string& spec : absl::GetFlag(FLAGS_input_name)) {
    LITERT_ASSIGN_OR_RETURN(auto named, ParseNamedShapeSpec(spec));
    LITERT_RETURN_IF_ERROR(options.AddTensorInputShape(
        signature_key, named.first, absl::MakeConstSpan(named.second)));
  }

  for (const std::string& spec : absl::GetFlag(FLAGS_signature_name)) {
    LITERT_ASSIGN_OR_RETURN(auto named, ParseNamedShapeSpec(spec));
    LITERT_RETURN_IF_ERROR(options.AddSignatureInputShape(
        signature_key, named.first, absl::MakeConstSpan(named.second)));
  }

  return {};
}

}  // namespace litert
