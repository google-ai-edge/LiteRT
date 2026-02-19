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

#include "litert/runtime/litert_runtime_options.h"

#include <cstddef>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/litert_toml_parser.h"

namespace litert {
namespace internal {

LiteRtStatus ParseLiteRtRuntimeOptions(const void* data, size_t size,
                                       LiteRtRuntimeOptionsT* options) {
  if (!data || size == 0 || !options) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  absl::string_view payload(static_cast<const char*>(data), size);

  return litert::internal::ParseToml(
      payload,
      [options](absl::string_view key,
                absl::string_view value) -> LiteRtStatus {
        if (key == "enable_profiling") {
          LITERT_ASSIGN_OR_RETURN(options->enable_profiling,
                                  litert::internal::ParseTomlBool(value));
        } else if (key == "compress_quantization_zero_points") {
          LITERT_ASSIGN_OR_RETURN(options->compress_quantization_zero_points,
                                  litert::internal::ParseTomlBool(value));
        } else if (key == "error_reporter_mode") {
          LITERT_ASSIGN_OR_RETURN(auto val,
                                  litert::internal::ParseTomlInt(value));
          options->error_reporter_mode =
              static_cast<LiteRtErrorReporterMode>(val);
        }
        return kLiteRtStatusOk;
      });
}

}  // namespace internal
}  // namespace litert
