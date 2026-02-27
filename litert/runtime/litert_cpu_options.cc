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

#include "litert/runtime/litert_cpu_options.h"

#include <cstddef>
#include <string>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/litert_toml_parser.h"

namespace litert {
namespace internal {

LiteRtStatus ParseLiteRtCpuOptions(const void* data, size_t size,
                                   LiteRtCpuOptionsT* options) {
  if (!data || size == 0 || !options) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  absl::string_view payload(static_cast<const char*>(data), size);

  return litert::internal::ParseToml(
      payload,
      [options](absl::string_view key,
                absl::string_view value) -> LiteRtStatus {
        if (key == "num_threads") {
          LITERT_ASSIGN_OR_RETURN(options->xnn.num_threads,
                                  litert::internal::ParseTomlInt(value));
        } else if (key == "flags") {
          LITERT_ASSIGN_OR_RETURN(auto val,
                                  litert::internal::ParseTomlInt(value));
          options->xnn.flags = val;
        } else if (key == "weight_cache_file_path") {
          absl::string_view path = value;
          if (path.size() >= 2 && path.front() == '"' && path.back() == '"') {
            path = path.substr(1, path.size() - 2);
          }
          options->weight_cache_file_path_buffer = std::string(path);
          options->xnn.weight_cache_file_path =
              options->weight_cache_file_path_buffer.c_str();

        } else if (key == "weight_cache_file_descriptor") {
          LITERT_ASSIGN_OR_RETURN(options->xnn.weight_cache_file_descriptor,
                                  litert::internal::ParseTomlInt(value));
        }
        return kLiteRtStatusOk;
      });
}

}  // namespace internal
}  // namespace litert
