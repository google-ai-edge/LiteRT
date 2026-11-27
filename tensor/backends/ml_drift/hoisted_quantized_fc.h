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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_ML_DRIFT_HOISTED_QUANTIZED_FC_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_ML_DRIFT_HOISTED_QUANTIZED_FC_H_

#include <optional>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift

namespace litert::tensor {

// Adds a hoisted-scale FP32-accumulated quantized `FullyConnected` operation
// (`INT2`, `INT4`, or `INT8`, per-channel or blockwise, linear buffer or 2D
// texture weights, `seq_len == 1` decode or `seq_len > 1` tiled prefill) when
// either:
//   - `force_for_f32_scale` is true (FP16 graph whose `weights_scale` was
//     promoted to FP32 because it contains FP16-subnormal values `< 2^-13`,
//     which GPUs like Adreno 750 flush to zero under FP16 FTZ), OR
//   - OpenCL single-token decode (`b == h == w == 1`) in an FP16 graph is
//     explicitly opted in via `LRT_OPENCL_HOISTED_FC=1` (off by default so
//     normal-scale FP16 FCs retain ML Drift's FP16 inner loop).
//
// Hoisting scale and zero-point out of the inner K-loop removes per-step
// dequantization FMAs, keeps FP16-subnormal scales exact in FP32, and avoids
// cumulative FP16 rounding drift over long reductions while preserving ML
// Drift's elementwise epilogue fusion contract (`res_value` -> `dst_tensor`).
// Returns `std::nullopt` if the op is not eligible or disabled.
std::optional<::ml_drift::GpuModelBuilder::TensorHandle>
TryAddHoistedQuantizedFullyConnected(
    ::ml_drift::GpuModelBuilder* builder,
    const ::ml_drift::GpuModelBuilder::TensorHandle& src,
    const ::ml_drift::GpuModelBuilder::Weights& weights,
    const ::ml_drift::GpuModelBuilder::TensorHandle* biases,
    absl::string_view op_name, bool force_for_f32_scale = false);

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_ML_DRIFT_HOISTED_QUANTIZED_FC_H_
