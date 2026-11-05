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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_XNNPACK_DEQUANTIZE_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_XNNPACK_DEQUANTIZE_H_

#include <cstddef>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/internal/graph.h"

namespace litert::tensor {

// Dequantizes the constant tensor `raw_data`, described by `info`, to float32
// on the host.
//
// - `info.type` must be a signed 2, 4 or 8-bit integer type.
// - `info.quantization` must be a `PerChannelAffineQuantization` with one
//   scale per index of its quantized dimension.
// - There may be no zero point, a single zero point shared by all the scales,
//   or one per scale.
//
// Returns Unimplemented for other tensors.
absl::StatusOr<std::vector<float>> DequantizeConstantTensor(
    const graph::TensorInformation& info, absl::Span<const std::byte> raw_data);

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_XNNPACK_DEQUANTIZE_H_
