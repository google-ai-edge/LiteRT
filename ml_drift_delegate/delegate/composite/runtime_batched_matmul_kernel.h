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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_RUNTIME_BATCHED_MATMUL_KERNEL_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_RUNTIME_BATCHED_MATMUL_KERNEL_H_

#include <vector>

#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift

namespace litert::ml_drift {

// Operation to add the runtime aware batched matmul kernel. Uses the values set
// in RuntimeCheck to early return from the computation. See
// ConvRuntimeCheckDesc in common/task/gpu_operation.h for more details.
// The case where external weights is set, uses the FullyConnected kernel,
// otherwise default to BMM.
// Int8 support is also added for the FC case, expects scale and channel_count
// to be set in the attributes.
absl::Status CreateRuntimeBatchedMatMulFromNode(
    const std::vector<::ml_drift::Value*>& inputs,
    const std::vector<::ml_drift::Value*>& outputs,
    const ::ml_drift::Node& node,
    ::ml_drift::GpuModelBuilder* model_builder);

absl::Status CreateRuntimeBatchedMatMulFromIrOp(
    const std::vector<const ::ml_drift::ir::IrTensor*>& inputs,
    const std::vector<const ::ml_drift::ir::IrTensor*>& outputs,
    const ::ml_drift::ir::IrOp& node,
    ::ml_drift::GpuModelBuilder* model_builder);

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_RUNTIME_BATCHED_MATMUL_KERNEL_H_
