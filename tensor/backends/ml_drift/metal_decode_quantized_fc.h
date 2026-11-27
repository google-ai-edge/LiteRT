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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_ML_DRIFT_METAL_DECODE_QUANTIZED_FC_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_ML_DRIFT_METAL_DECODE_QUANTIZED_FC_H_

#include <optional>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift

namespace litert::tensor {

// Attempts to lower a single-token (`b == h == w == 1`) quantized
// `FullyConnected` op (linear `kOSpatialIOGroupI4O4` INT2 / INT4 / INT8
// weights) using an Apple Metal decode-optimized kernel with:
// - hoisted per-channel / per-block scale and zero-point application (avoiding
//   per-K-step scale/bias FMAs inside the inner loop),
// - Apple FP16-pipe `floor` unpack for INT4 / INT2 without inner-loop power-of-
//   2 rescaling (folding the exact factor into the hoisted output scale) and
//   zero-mean `char4` (`^ 0x80808080u`) unpack for INT8,
// - vectorized `uint4` multi-slice loads for wide INT2 / INT4 layers,
// - Apple M-series lockstep workgroup sync on wide `lm_head` outputs, and
// - full 32-lane coalesced `wg_x >= 32` workgroups on mobile Apple GPUs with
//   single-barrier threadgroup reduction for `wg_y <= 8`.
//
// Enabled by default on Apple Metal GPUs (`LRT_METAL_DECODE_FC=0` disables;
// `LRT_METAL_DECODE_FC=1` forces it on any ML Drift backend for testing).
// Returns `std::nullopt` when ineligible so callers fall back cleanly.
std::optional<::ml_drift::GpuModelBuilder::TensorHandle>
TryAddMetalDecodeQuantizedFullyConnected(
    ::ml_drift::GpuModelBuilder* builder,
    const ::ml_drift::GpuModelBuilder::TensorHandle& src,
    const ::ml_drift::GpuModelBuilder::Weights& weights,
    const ::ml_drift::GpuModelBuilder::TensorHandle* biases,
    absl::string_view op_name);

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_ML_DRIFT_METAL_DECODE_QUANTIZED_FC_H_
