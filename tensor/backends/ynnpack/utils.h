/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_YNNPACK_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_YNNPACK_UTILS_H_

// This header provides functions that allow using the
// `LRT_TENSOR_RETURN_IF_ERROR` macro with YNNPACK functions that return a
// `ynn_status` code. IWUY doesn't recognize this and treats the header as
// unused.

// IWYU pragma: always_keep

#include <string>

#include "ynnpack/include/ynnpack.h"  // from @XNNPACK
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "tensor/utils/macros.h"

namespace litert::tensor {

inline absl::Status YnnStatusToAbsl(enum ynn_status status,
                                    absl::string_view label) {
  if (status == ynn_status_success) {
    return absl::OkStatus();
  }
  std::string message = absl::StrCat("ynn_status=", static_cast<int>(status));
  if (!label.empty()) {
    absl::StrAppend(&message, ";", label);
  }
  switch (status) {
    case ynn_status_invalid_parameter:
      return absl::InvalidArgumentError(message);
    case ynn_status_unsupported_parameter:
    case ynn_status_deprecated:
      return absl::UnimplementedError(message);
    default:
      return absl::InternalError(message);
  }
}

template <>
struct ErrorStatusBuilder::ErrorConversion<ynn_status> {
  static constexpr bool IsError(ynn_status value) {
    return value != ynn_status_success;
  }
  static absl::Status AsError(ynn_status value) {
    return YnnStatusToAbsl(value, "");
  }
};

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_YNNPACK_UTILS_H_
