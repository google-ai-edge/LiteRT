/* Copyright 2025 Google LLC.

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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_XNNPACK_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_XNNPACK_UTILS_H_

#include "xnnpack.h"  // from @XNNPACK
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "tensor/utils/macros.h"

namespace litert::tensor {

template <>
struct ErrorStatusBuilder::ErrorConversion<xnn_status> {
  static constexpr bool IsError(xnn_status value) {
    return value != xnn_status_success;
  }
  static absl::Status AsError(xnn_status value) {
    return absl::UnknownError(
        absl::StrCat("xnn_status=", static_cast<int>(value)));
  }
};

inline absl::Status XnnStatusToAbsl(enum xnn_status status,
                                    absl::string_view label) {
  if (status == xnn_status_success) {
    return absl::OkStatus();
  }
  return absl::UnknownError(
      absl::StrCat("xnn_status=", static_cast<int>(status), ";", label));
}

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_XNNPACK_UTILS_H_
