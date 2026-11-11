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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_SYMMETRIC_QUANTIZED_FULLY_CONNECTED_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_SYMMETRIC_QUANTIZED_FULLY_CONNECTED_H_

#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift

namespace litert::ml_drift {

// Builds an int2 / int4 / int8 FullyConnected with external weights and no
// zero point, which is how symmetric blockwise weights are bound: the inputs
// are (src, weights, scale) or (src, weights, bias, scale), and the kernel gets
// `zero_point = nullptr`. A const bias in the op attributes is applied to the
// 3-input form. ML Drift's selector expects a zero-point input for these ops,
// so the LiteRT op selectors try this first.
//
// Returns an `InvalidArgumentError` if a quantized FullyConnected op has
// missing / mismatched attributes or has a zero point with only 3 inputs, and
// an `UnimplementedError` for any other op (including a 4+-input quantized
// FullyConnected with a zero point, or 1-input inline weights), so the caller
// can fall back to ML Drift's selector.
absl::Status CreateSymmetricQuantizedFullyConnectedFromNode(
    const ::ml_drift::GpuInfo& gpu_info,
    const ::ml_drift::CreateGpuModelInfo& create_info,
    const std::vector<::ml_drift::Value*>& inputs,
    const std::vector<::ml_drift::Value*>& outputs,
    const ::ml_drift::Node& node, ::ml_drift::GpuModelBuilder* model_builder);

absl::Status CreateSymmetricQuantizedFullyConnectedFromIrOp(
    const ::ml_drift::GpuInfo& gpu_info,
    const ::ml_drift::CreateGpuModelInfo& create_info,
    const std::vector<const ::ml_drift::ir::IrTensor*>& inputs,
    const std::vector<const ::ml_drift::ir::IrTensor*>& outputs,
    const ::ml_drift::ir::IrOp& op, ::ml_drift::GpuModelBuilder* model_builder);

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_SYMMETRIC_QUANTIZED_FULLY_CONNECTED_H_
