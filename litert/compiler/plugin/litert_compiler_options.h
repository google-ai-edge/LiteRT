
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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_COMPILER_PLUGIN_LITERT_COMPILER_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_COMPILER_PLUGIN_LITERT_COMPILER_OPTIONS_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/c/options/litert_compiler_options.h"

struct LiteRtCompilerOptionsInputShapeEntry {
  std::string signature_key;
  std::string name;  // Empty for positional_input_shapes.
  std::vector<int32_t> dims;
};

// One input of an "Add" shape specialization: `source_signature_key` is
// instantiated again as static signature `new_signature_key` with signature
// input `input_name` fixed to `dims`. Entries sharing `new_signature_key`
// describe one specialization.
struct LiteRtCompilerOptionsShapeSpecializationEntry {
  std::string source_signature_key;
  std::string new_signature_key;
  std::string input_name;
  std::vector<int32_t> dims;
};

// Internal LiteRt compiler options struct.
struct LiteRtCompilerOptionsT {
  LiteRtCompilerOptionsPartitionStrategy partition_strategy =
      kLiteRtCompilerOptionsPartitionStrategyDefault;

  bool dummy_option = false;

  size_t max_partitions = 0;

  std::vector<LiteRtCompilerOptionsInputShapeEntry> positional_input_shapes;
  std::vector<LiteRtCompilerOptionsInputShapeEntry> tensor_input_shapes;
  std::vector<LiteRtCompilerOptionsInputShapeEntry> signature_input_shapes;

  // "Add" mode shape specializations (new static signatures cloned from a
  // source signature). In-place refinement of a source is expressed with the
  // *_input_shapes lists above.
  std::vector<LiteRtCompilerOptionsShapeSpecializationEntry>
      shape_specializations;

  bool HasInputShapes() const {
    return !positional_input_shapes.empty() || !tensor_input_shapes.empty() ||
           !signature_input_shapes.empty();
  }

  bool HasShapeSpecializations() const {
    return !shape_specializations.empty();
  }

  static const char* Identifier() { return "compiler_options_string"; }
};

namespace litert {
namespace internal {

LiteRtStatus ParseLiteRtCompilerOptions(const void* data, size_t size,
                                        LiteRtCompilerOptionsT* options);

}  // namespace internal
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_COMPILER_PLUGIN_LITERT_COMPILER_OPTIONS_H_
