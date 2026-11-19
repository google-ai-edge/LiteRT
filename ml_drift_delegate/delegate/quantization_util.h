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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_QUANTIZATION_UTIL_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_QUANTIZATION_UTIL_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tflite/core/c/common.h"

namespace ml_drift {
namespace delegate {

// Validates a quantized (`kTfLiteInt8` or `kTfLiteUInt8`) tensor's type,
// dimensions, and backing storage, and returns its required dequantized `float`
// buffer size in bytes (`element_count * sizeof(float)`).
//
// Returns:
// - `UnimplementedError` if `quantized.type` is neither `kTfLiteInt8` nor
//   `kTfLiteUInt8`.
// - `InvalidArgumentError` if `quantized.dims` is null, has a negative axis or
//   size, overflows `int` element count, or overflows `size_t` byte size.
// - `FailedPreconditionError` if `element_count > 0` and either
//   `quantized.bytes < element_count` or `quantized.data.raw == nullptr`.
// - `0` when `element_count == 0` (even if `quantized.data.raw == nullptr`).
absl::StatusOr<size_t> ValidatedDequantizedBytes(const TfLiteTensor& quantized);

// Dequantizes `quantized` (`kTfLiteInt8` or `kTfLiteUInt8`) into `dst`.
//
// When `element_count == 0`, returns `OkStatus()` without dereferencing data
// pointers. When `element_count > 0`, requires `dst.size() >= element_count`
// and `dst.data() != nullptr` (returns `InvalidArgumentError` otherwise), in
// addition to all invariants enforced by `ValidatedDequantizedBytes`.
absl::Status DequantizeTo(const TfLiteTensor& quantized, absl::Span<float> dst);

// Quantizes `src` into `quantized` (`kTfLiteInt8` or `kTfLiteUInt8`).
//
// When `element_count == 0`, returns `OkStatus()` without dereferencing data
// pointers. When `element_count > 0`, requires `src.size() >= element_count`
// and `src.data() != nullptr` (returns `InvalidArgumentError` otherwise), in
// addition to all invariants enforced by `ValidatedDequantizedBytes`.
absl::Status QuantizeFrom(absl::Span<const float> src, TfLiteTensor& quantized);

// Dequantizes input tensors pre-inference, leaving float tensors intact.
// `input_indices` contains dequantized (`kTfLiteFloat32`) tensor indices used
// as inputs to the GPU delegate. `quant_conversion_map` contains a
// bidirectional mapping between each dequantized tensor index and its original
// quantized tensor index.
absl::Status DequantizeInputs(
    TfLiteContext* context, const std::vector<int64_t>& input_indices,
    const absl::flat_hash_map<int, int>& quant_conversion_map);

// Quantizes output tensors post-inference, leaving float tensors intact.
// `output_indices` contains dequantized (`kTfLiteFloat32`) tensor indices
// produced by the GPU delegate. `quant_conversion_map` contains a bidirectional
// mapping between each dequantized tensor index and its original quantized
// tensor index.
absl::Status QuantizeOutputs(
    TfLiteContext* context, const std::vector<int64_t>& output_indices,
    const absl::flat_hash_map<int, int>& quant_conversion_map);

}  // namespace delegate
}  // namespace ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_QUANTIZATION_UTIL_H_
