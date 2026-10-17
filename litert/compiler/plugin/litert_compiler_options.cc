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
    return kLiteRtStatusErrorInvalidArgument;
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
  std::vector<absl::string_view> parts =
      absl::StrSplit(raw, absl::MaxSplits('@', 1));
  absl::string_view shape_spec;
  if (parts.size() == 2) {
    entry->signature_key = std::string(parts[0]);
    shape_spec = parts[1];
  } else {
    shape_spec = parts[0];
  }
  return ParseDimsSpec(shape_spec, &entry->dims);
}

LiteRtStatus ParseNamedShapeEntry(absl::string_view raw,
                                  LiteRtCompilerOptionsInputShapeEntry* entry) {
  // Format: "<signature_key>@<name>@<d0>:<d1>:..." (or "<name>@<d0>:<d1>:...")
  std::vector<absl::string_view> parts =
      absl::StrSplit(raw, absl::MaxSplits('@', 2));
  absl::string_view shape_spec;
  if (parts.size() == 3) {
    entry->signature_key = std::string(parts[0]);
    entry->name = std::string(parts[1]);
    shape_spec = parts[2];
  } else if (parts.size() == 2) {
    entry->name = std::string(parts[0]);
    shape_spec = parts[1];
  } else {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (entry->name.empty()) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  return ParseDimsSpec(shape_spec, &entry->dims);
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
        }
        return kLiteRtStatusOk;
      });
}

}  // namespace internal
}  // namespace litert
