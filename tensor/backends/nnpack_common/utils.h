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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_NNPACK_COMMON_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_NNPACK_COMMON_UTILS_H_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/arithmetic_graph.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/utils/macros.h"

namespace litert::tensor {

inline constexpr float kInf = std::numeric_limits<float>::infinity();

struct ActivationBounds {
  float output_min;
  float output_max;
};

absl::StatusOr<ActivationBounds> GetActivationBounds(FusedActivation activation,
                                                     absl::string_view op_name);

struct BinaryIOIds {
  uint32_t lhs;
  uint32_t rhs;
  uint32_t output;
};

struct UnaryIOIds {
  uint32_t input;
  uint32_t output;
};

struct PaddingValues {
  uint32_t top;
  uint32_t right;
  uint32_t bottom;
  uint32_t left;
};

PaddingValues ComputePadding(Padding padding, int input_h, int input_w,
                             int filter_h, int filter_w, int stride_h,
                             int stride_w, int dilation_h, int dilation_w);

struct TransposeConvPaddingValues {
  uint32_t top;
  uint32_t right;
  uint32_t bottom;
  uint32_t left;
  uint32_t adj_h;
  uint32_t adj_w;
};

absl::StatusOr<TransposeConvPaddingValues> ComputeTransposeConvPadding(
    Padding padding, int input_h, int input_w, int filter_h, int filter_w,
    int stride_h, int stride_w, int output_h, int output_w);

absl::StatusOr<std::vector<size_t>> ToNnpackDims(
    absl::Span<const int32_t> shape);

template <Type... Types>
absl::Status ValidateTensorType(const graph::Tensor& tensor,
                                absl::string_view op_name) {
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& info, graph::GetInfo(tensor));
  if (!((info.type == Types) || ...)) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s only supports %v tensors. Got type id %v.", op_name,
                        absl::StrJoin({Types...}, ", "), info.type));
  }
  return absl::OkStatus();
}

absl::Status ValidateFp32OrQuantizedConstantWeights(const graph::Tensor& tensor,
                                                    absl::string_view op_name);

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_NNPACK_COMMON_UTILS_H_
