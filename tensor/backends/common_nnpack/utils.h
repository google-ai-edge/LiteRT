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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_COMMON_NNPACK_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_COMMON_NNPACK_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/arithmetic_graph.h"
#include "tensor/internal/graph.h"

namespace litert::tensor {

struct NnpackPadding {
  size_t top = 0;
  size_t right = 0;
  size_t bottom = 0;
  size_t left = 0;
};

struct NnpackTransposeConvPadding : public NnpackPadding {
  size_t adj_h = 0;
  size_t adj_w = 0;
};

struct NnpackActivationBounds {
  float min;
  float max;
};

NnpackActivationBounds GetActivationBounds(FusedActivation activation);

absl::StatusOr<NnpackPadding> ComputePadding(
    Padding padding, size_t input_height, size_t input_width,
    size_t kernel_height, size_t kernel_width, size_t stride_h, size_t stride_w,
    size_t dilation_h, size_t dilation_w);

absl::StatusOr<NnpackTransposeConvPadding> ComputeTransposeConvPadding(
    Padding padding, size_t input_height, size_t input_width,
    size_t kernel_height, size_t kernel_width, size_t stride_h, size_t stride_w,
    size_t output_height, size_t output_width);

absl::StatusOr<std::vector<size_t>> ToNnpackDims(
    absl::Span<const int32_t> shape);

template <typename T>
absl::Status ValidateTensorType(absl::string_view op_name,
                                const graph::TensorInformation& tensor_info,
                                absl::Span<const T> allowed_types) {
  for (const auto& type : allowed_types) {
    if (tensor_info.type == type) {
      return absl::OkStatus();
    }
  }
  return absl::InvalidArgumentError(absl::StrFormat(
      "%s: unsupported tensor type: %d. Allowed types: [%s]", op_name,
      static_cast<int>(tensor_info.type), absl::StrJoin(allowed_types, ", ")));
}

template <typename T>
absl::Status ValidateTensorType(absl::string_view op_name,
                                const graph::TensorInformation& tensor_info,
                                std::initializer_list<T> allowed_types) {
  return ValidateTensorType(
      op_name, tensor_info,
      absl::Span<const T>(allowed_types.begin(), allowed_types.size()));
}

absl::Status ValidateFp32OrQuantizedConstantWeights(
    absl::string_view op_name, const graph::TensorInformation& weights_info,
    bool is_external);

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_COMMON_NNPACK_UTILS_H_
