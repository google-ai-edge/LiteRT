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

#include "litert/c/options/litert_compiler_options.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_options_helper.h"
#include "litert/c/litert_common.h"

struct LrtCompilerOptions {
  std::optional<LiteRtCompilerOptionsPartitionStrategy> partition_strategy;
  std::optional<bool> dummy_option;
  std::optional<size_t> max_partitions;
  std::vector<std::string> positional_input_shapes;
  std::vector<std::string> tensor_input_shapes;
  std::vector<std::string> signature_input_shapes;
};

namespace {

std::string FormatDims(const int32_t* dims, size_t rank) {
  std::string out;
  for (size_t i = 0; i < rank; ++i) {
    if (i > 0) absl::StrAppend(&out, ":");
    absl::StrAppend(&out, dims[i]);
  }
  return out;
}

void WriteTomlStringArray(std::string& toml_str, absl::string_view key,
                          const std::vector<std::string>& values) {
  if (values.empty()) return;
  absl::StrAppend(&toml_str, key, " = [");
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) absl::StrAppend(&toml_str, ", ");
    absl::StrAppend(&toml_str, "\"", values[i], "\"");
  }
  absl::StrAppend(&toml_str, "]\n");
}

bool HasInvalidTomlChars(absl::string_view s) {
  return s.find_first_of("\"\'\\\n\r") != absl::string_view::npos;
}

}  // namespace

LiteRtStatus LrtCreateCompilerOptions(LrtCompilerOptions** options) {
  if (!options) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *options = new LrtCompilerOptions();
  if (!*options) {
    return kLiteRtStatusErrorMemoryAllocationFailure;
  }
  return kLiteRtStatusOk;
}

void LrtDestroyCompilerOptions(LrtCompilerOptions* options) {
  if (options) {
    delete options;
  }
}

LiteRtStatus LrtGetOpaqueCompilerOptionsData(const LrtCompilerOptions* options,
                                             const char** identifier,
                                             void** payload,
                                             void (**payload_deleter)(void*)) {
  if (!options || !identifier || !payload || !payload_deleter) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  std::string toml_str;
  if (options->partition_strategy.has_value()) {
    absl::StrAppend(&toml_str, "partition_strategy = ",
                    static_cast<int>(options->partition_strategy.value()),
                    "\n");
  }
  if (options->dummy_option.has_value()) {
    absl::StrAppend(&toml_str, "dummy_option = ",
                    options->dummy_option.value() ? "true" : "false", "\n");
  }
  if (options->max_partitions.has_value()) {
    absl::StrAppend(&toml_str,
                    "max_partitions = ", options->max_partitions.value(), "\n");
  }
  WriteTomlStringArray(toml_str, "positional_input_shapes",
                       options->positional_input_shapes);
  WriteTomlStringArray(toml_str, "tensor_input_shapes",
                       options->tensor_input_shapes);
  WriteTomlStringArray(toml_str, "signature_input_shapes",
                       options->signature_input_shapes);

  *identifier = LrtGetCompilerOptionsIdentifier();
  litert::internal::MakeCStringPayload(toml_str, payload, payload_deleter);

  return kLiteRtStatusOk;
}

const char* LrtGetCompilerOptionsIdentifier() {
  return "compiler_options_string";
}

LiteRtStatus LrtSetCompilerOptionsPartitionStrategy(
    LrtCompilerOptions* options,
    LiteRtCompilerOptionsPartitionStrategy partition_strategy) {
  if (!options) return kLiteRtStatusErrorInvalidArgument;
  options->partition_strategy = partition_strategy;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetCompilerOptionsPartitionStrategy(
    const LrtCompilerOptions* options,
    LiteRtCompilerOptionsPartitionStrategy* partition_strategy) {
  if (!options || !partition_strategy) return kLiteRtStatusErrorInvalidArgument;
  if (!options->partition_strategy.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *partition_strategy = options->partition_strategy.value();
  return kLiteRtStatusOk;
}

LiteRtStatus LrtSetCompilerOptionsDummyOption(LrtCompilerOptions* options,
                                              bool dummy_option) {
  if (!options) return kLiteRtStatusErrorInvalidArgument;
  options->dummy_option = dummy_option;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetCompilerOptionsDummyOption(const LrtCompilerOptions* options,
                                              bool* dummy_option) {
  if (!options || !dummy_option) return kLiteRtStatusErrorInvalidArgument;
  if (!options->dummy_option.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *dummy_option = options->dummy_option.value();
  return kLiteRtStatusOk;
}

LiteRtStatus LrtSetCompilerOptionsMaxPartitions(LrtCompilerOptions* options,
                                                size_t max_partitions) {
  if (!options) return kLiteRtStatusErrorInvalidArgument;
  options->max_partitions = max_partitions;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetCompilerOptionsMaxPartitions(
    const LrtCompilerOptions* options, size_t* max_partitions) {
  if (!options || !max_partitions) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!options->max_partitions.has_value()) {
    return kLiteRtStatusErrorNotFound;
  }
  *max_partitions = options->max_partitions.value();
  return kLiteRtStatusOk;
}

LiteRtStatus LrtAddCompilerOptionsPositionalInputShape(
    LrtCompilerOptions* options, const char* signature_key, const int32_t* dims,
    size_t rank) {
  if (!options || (rank > 0 && !dims)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const absl::string_view sig = signature_key ? signature_key : "";
  if (HasInvalidTomlChars(sig) || absl::StrContains(sig, '@')) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->positional_input_shapes.push_back(
      absl::StrCat(sig, "@", FormatDims(dims, rank)));
  return kLiteRtStatusOk;
}

LiteRtStatus LrtAddCompilerOptionsTensorInputShape(LrtCompilerOptions* options,
                                                   const char* signature_key,
                                                   const char* tensor_name,
                                                   const int32_t* dims,
                                                   size_t rank) {
  if (!options || !tensor_name || tensor_name[0] == '\0' ||
      (rank > 0 && !dims)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const absl::string_view sig = signature_key ? signature_key : "";
  const absl::string_view name(tensor_name);
  if (HasInvalidTomlChars(sig) || absl::StrContains(sig, '@') ||
      HasInvalidTomlChars(name)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->tensor_input_shapes.push_back(
      absl::StrCat(sig, "@", name, "@", FormatDims(dims, rank)));
  return kLiteRtStatusOk;
}

LiteRtStatus LrtAddCompilerOptionsSignatureInputShape(
    LrtCompilerOptions* options, const char* signature_key,
    const char* input_name, const int32_t* dims, size_t rank) {
  if (!options || !input_name || input_name[0] == '\0' || (rank > 0 && !dims)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const absl::string_view sig = signature_key ? signature_key : "";
  const absl::string_view name(input_name);
  if (HasInvalidTomlChars(sig) || absl::StrContains(sig, '@') ||
      HasInvalidTomlChars(name)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->signature_input_shapes.push_back(
      absl::StrCat(sig, "@", name, "@", FormatDims(dims, rank)));
  return kLiteRtStatusOk;
}
