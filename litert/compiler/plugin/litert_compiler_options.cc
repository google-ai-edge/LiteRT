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

#include "litert/compiler/plugin/litert_compiler_options.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/options/litert_compiler_options.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/litert_toml_parser.h"

namespace litert {
namespace internal {
namespace {

LiteRtStatus ParseDimsSpec(absl::string_view shape_spec,
                           std::vector<int32_t>* dims) {
  if (shape_spec.empty()) {
    return kLiteRtStatusOk;
  }
  for (absl::string_view dim_str : absl::StrSplit(shape_spec, ':')) {
    int32_t dim = 0;
    if (!absl::SimpleAtoi(dim_str, &dim)) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    dims->push_back(dim);
  }
  return kLiteRtStatusOk;
}

LiteRtStatus ParsePositionalShapeEntry(
    absl::string_view raw, LiteRtCompilerOptionsInputShapeEntry* entry) {
  // Format: "<signature_key>@<d0>:<d1>:..." (or "<d0>:<d1>:...")
  // An empty `<d0>:<d1>:...` after '@' denotes a 0-D scalar shape `[]`.
  if (raw.empty()) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const size_t last_at = raw.rfind('@');
  absl::string_view shape_spec;
  if (last_at != absl::string_view::npos) {
    entry->signature_key = std::string(raw.substr(0, last_at));
    shape_spec = raw.substr(last_at + 1);
  } else {
    shape_spec = raw;
  }
  return ParseDimsSpec(shape_spec, &entry->dims);
}

LiteRtStatus ParseNamedShapeEntry(absl::string_view raw,
                                  LiteRtCompilerOptionsInputShapeEntry* entry) {
  // Format: "<signature_key>@<name>@<d0>:<d1>:..." (or "<name>@<d0>:<d1>:...")
  // Split `signature_key` on the first '@' and `shape_spec` on the last '@' so
  // `name` in the middle may contain '@' characters (e.g. "func@0/arg0").
  const size_t first_at = raw.find('@');
  const size_t last_at = raw.rfind('@');
  if (first_at == absl::string_view::npos) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  absl::string_view shape_spec;
  if (first_at != last_at) {
    entry->signature_key = std::string(raw.substr(0, first_at));
    entry->name = std::string(raw.substr(first_at + 1, last_at - first_at - 1));
    shape_spec = raw.substr(last_at + 1);
  } else {
    entry->name = std::string(raw.substr(0, first_at));
    shape_spec = raw.substr(first_at + 1);
  }
  if (entry->name.empty()) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  return ParseDimsSpec(shape_spec, &entry->dims);
}

LiteRtStatus ParseShapeSpecializationEntry(
    absl::string_view raw,
    LiteRtCompilerOptionsShapeSpecializationEntry* entry) {
  // Format: "<source_key>@<new_key>@<input_name>@<d0>:<d1>:...". Signature
  // keys may not contain '@'; `input_name` may, so split the first two '@'
  // from the left and the shape spec on the last '@'.
  const size_t first_at = raw.find('@');
  if (first_at == absl::string_view::npos) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const size_t second_at = raw.find('@', first_at + 1);
  const size_t last_at = raw.rfind('@');
  if (second_at == absl::string_view::npos || last_at <= second_at) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  entry->source_signature_key = std::string(raw.substr(0, first_at));
  entry->new_signature_key =
      std::string(raw.substr(first_at + 1, second_at - first_at - 1));
  entry->input_name =
      std::string(raw.substr(second_at + 1, last_at - second_at - 1));
  if (entry->new_signature_key.empty() || entry->input_name.empty()) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(ParseDimsSpec(raw.substr(last_at + 1), &entry->dims));
  for (int32_t d : entry->dims) {
    if (d <= 0) return kLiteRtStatusErrorInvalidArgument;
  }
  return kLiteRtStatusOk;
}

}  // namespace

LiteRtStatus ParseLiteRtCompilerOptions(const void* data, size_t size,
                                        LiteRtCompilerOptionsT* options) {
  return ParseToml(
      absl::string_view(static_cast<const char*>(data), size),
      [options](absl::string_view key,
                absl::string_view value) -> LiteRtStatus {
        if (key == "partition_strategy") {
          LITERT_ASSIGN_OR_RETURN(auto strategy, ParseTomlInt(value));
          options->partition_strategy =
              static_cast<LiteRtCompilerOptionsPartitionStrategy>(strategy);
        } else if (key == "dummy_option") {
          LITERT_ASSIGN_OR_RETURN(options->dummy_option, ParseTomlBool(value));
        } else if (key == "max_partitions") {
          LITERT_ASSIGN_OR_RETURN(auto max_partitions, ParseTomlInt(value));
          options->max_partitions = static_cast<size_t>(max_partitions);
        } else if (key == "positional_input_shapes") {
          LITERT_ASSIGN_OR_RETURN(auto arr, ParseTomlStringArray(value));
          for (const auto& item : arr) {
            LiteRtCompilerOptionsInputShapeEntry entry;
            LITERT_RETURN_IF_ERROR(ParsePositionalShapeEntry(item, &entry));
            options->positional_input_shapes.push_back(std::move(entry));
          }
        } else if (key == "tensor_input_shapes") {
          LITERT_ASSIGN_OR_RETURN(auto arr, ParseTomlStringArray(value));
          for (const auto& item : arr) {
            LiteRtCompilerOptionsInputShapeEntry entry;
            LITERT_RETURN_IF_ERROR(ParseNamedShapeEntry(item, &entry));
            options->tensor_input_shapes.push_back(std::move(entry));
          }
        } else if (key == "signature_input_shapes") {
          LITERT_ASSIGN_OR_RETURN(auto arr, ParseTomlStringArray(value));
          for (const auto& item : arr) {
            LiteRtCompilerOptionsInputShapeEntry entry;
            LITERT_RETURN_IF_ERROR(ParseNamedShapeEntry(item, &entry));
            options->signature_input_shapes.push_back(std::move(entry));
          }
        } else if (key == "shape_specializations") {
          LITERT_ASSIGN_OR_RETURN(auto arr, ParseTomlStringArray(value));
          for (const auto& item : arr) {
            LiteRtCompilerOptionsShapeSpecializationEntry entry;
            LITERT_RETURN_IF_ERROR(ParseShapeSpecializationEntry(item, &entry));
            options->shape_specializations.push_back(std::move(entry));
          }
        }
        return kLiteRtStatusOk;
      });
}

}  // namespace internal
}  // namespace litert
