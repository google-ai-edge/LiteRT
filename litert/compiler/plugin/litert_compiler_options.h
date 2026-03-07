
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

#include "litert/c/litert_common.h"
#include "litert/c/options/litert_compiler_options.h"

// Internal LiteRt compiler options struct.
struct LiteRtCompilerOptionsT {
  LiteRtCompilerOptionsPartitionStrategy partition_strategy =
      kLiteRtCompilerOptionsPartitionStrategyDefault;

  bool dummy_option = false;

  static const char* Identifier() { return "compiler_options_string"; }
};

namespace litert {
namespace internal {

LiteRtStatus ParseLiteRtCompilerOptions(const void* data, size_t size,
                                        LiteRtCompilerOptionsT* options);

}  // namespace internal
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_COMPILER_PLUGIN_LITERT_COMPILER_OPTIONS_H_
