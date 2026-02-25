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

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/options/litert_compiler_options.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/litert_toml_parser.h"

namespace litert {
namespace internal {

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
        }
        return kLiteRtStatusOk;
      });
}

}  // namespace internal
}  // namespace litert
