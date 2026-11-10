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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_TILED_QUANTIZED_FULLY_CONNECTED_KERNEL_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_TILED_QUANTIZED_FULLY_CONNECTED_KERNEL_H_

#include <memory>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/kernels/fully_connected.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift

namespace litert::ml_drift {

// Creates a spatially tiled quantized (INT2/INT4/INT8) FullyConnected GPU
// operation that amortizes weight loads across `tile_w` spatial tokens per
// thread and dispatches spatial tiles of the same output-slice block back to
// back for L2 weight reuse.
std::unique_ptr<::ml_drift::GPUOperation> CreateTiledQuantizedFullyConnected(
    const ::ml_drift::GpuInfo& gpu_info,
    ::ml_drift::CalculationsPrecision precision,
    const ::ml_drift::TensorDescriptor& src,
    const ::ml_drift::TensorDescriptor& dst,
    const ::ml_drift::ExternalWeights& weights,
    const ::ml_drift::TensorDescriptor* bias,
    const ::ml_drift::BHWC& dst_shape);

// Adds a quantized (INT2/INT4/INT8) FullyConnected operation to `builder`,
// selecting between the 1x1 decode kernel, `CreateTiledQuantizedFullyConnected`
// for short/medium spatial batches, and ML Drift's weight-conversion +
// `conv_wave_memory` path for large prefills.
//
// TODO: b/577601974 - Also use this for the FULLY_CONNECTED ops of converted
// LiteRT models, which ML Drift core lowers today
// (`GpuModelBuilder::FullyConnectedInt4ExternalWeights` /
// `FullyConnectedInt8ExternalWeights`): select it in the LiteRT ML Drift
// delegate (litert/ml_drift) rather than in ML Drift core, after an A/B on
// Adreno, Mali and Apple GPUs.
::ml_drift::GpuModelBuilder::TensorHandle AddQuantizedFullyConnected(
    ::ml_drift::GpuModelBuilder* builder,
    const ::ml_drift::GpuModelBuilder::TensorHandle& src,
    const ::ml_drift::GpuModelBuilder::Weights& weights,
    const ::ml_drift::GpuModelBuilder::TensorHandle* biases,
    absl::string_view op_name);

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_TILED_QUANTIZED_FULLY_CONNECTED_KERNEL_H_
