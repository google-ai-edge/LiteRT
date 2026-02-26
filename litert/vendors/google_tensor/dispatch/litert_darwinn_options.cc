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

#include "litert/vendors/google_tensor/dispatch/litert_darwinn_options.h"

#include <cstddef>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/litert_toml_parser.h"

namespace litert {
namespace internal {

LiteRtStatus ParseLiteRtDarwinnRuntimeOptions(
    const void* data, size_t size, LiteRtDarwinnRuntimeOptionsT* options) {
  if (!data || size == 0 || !options) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  absl::string_view payload(static_cast<const char*>(data), size);

  return litert::internal::ParseToml(
      payload,
      [options](absl::string_view key,
                absl::string_view value) -> LiteRtStatus {
        if (key == "inference_power_state") {
          LITERT_ASSIGN_OR_RETURN(auto val,
                                  litert::internal::ParseTomlInt(value));
          options->inference_power_state = val;
        } else if (key == "inference_memory_power_state") {
          LITERT_ASSIGN_OR_RETURN(auto val,
                                  litert::internal::ParseTomlInt(value));
          options->inference_memory_power_state = val;
        } else if (key == "inference_priority") {
          LITERT_ASSIGN_OR_RETURN(auto val,
                                  litert::internal::ParseTomlInt(value));
          options->inference_priority = val;
        } else if (key == "atomic_inference") {
          LITERT_ASSIGN_OR_RETURN(options->atomic_inference,
                                  litert::internal::ParseTomlBool(value));
        } else if (key == "prefer_coherent") {
          LITERT_ASSIGN_OR_RETURN(options->prefer_coherent,
                                  litert::internal::ParseTomlBool(value));
        }
        return kLiteRtStatusOk;
      });
}

}  // namespace internal
}  // namespace litert
