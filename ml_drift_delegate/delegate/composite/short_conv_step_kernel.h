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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_SHORT_CONV_STEP_KERNEL_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_SHORT_CONV_STEP_KERNEL_H_

#include <memory>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift

namespace litert::ml_drift {

absl::Status CreateShortConvStepFromNode(
    const std::vector<::ml_drift::Value*>& inputs,
    const std::vector<::ml_drift::Value*>& outputs,
    const ::ml_drift::Node& node, ::ml_drift::GpuModelBuilder* model_builder);

absl::Status CreateShortConvStepFromIrOp(
    const std::vector<const ::ml_drift::ir::IrTensor*>& inputs,
    const std::vector<const ::ml_drift::ir::IrTensor*>& outputs,
    const ::ml_drift::ir::IrOp& node,
    ::ml_drift::GpuModelBuilder* model_builder);

// Exposed for testing in short_conv_step_kernel_test.
std::unique_ptr<::ml_drift::GPUOperation> CreateFusedShortConvStep(
    const ::ml_drift::TensorDescriptor& in_proj_desc,
    const ::ml_drift::TensorDescriptor& conv_state_desc,
    const ::ml_drift::TensorDescriptor& conv_weight_desc,
    const ::ml_drift::TensorDescriptor* conv_bias_desc,
    const ::ml_drift::TensorDescriptor& dst_desc,
    const ::ml_drift::TensorDescriptor& next_state_desc,
    int num_slices, int hidden_size, int conv_L_cache);

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_SHORT_CONV_STEP_KERNEL_H_
