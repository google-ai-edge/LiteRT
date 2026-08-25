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

#ifndef ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_EXPERTS_REMAP_BUILDER_H_
#define ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_EXPERTS_REMAP_BUILDER_H_

#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift

namespace litert::ml_drift {

std::vector<::ml_drift::GpuModelBuilder::TensorHandle> CreateExpertsRemap(
    ::ml_drift::GpuModelBuilder& builder,
    const ::ml_drift::GpuModelBuilder::TensorHandle& indices, int num_experts);

::ml_drift::GpuModelBuilder::TensorHandle ExpertsRemapTo(
    ::ml_drift::GpuModelBuilder& builder,
    const ::ml_drift::GpuModelBuilder::TensorHandle& src,
    const ::ml_drift::GpuModelBuilder::TensorHandle& experts_remap,
    int num_active_experts);

::ml_drift::GpuModelBuilder::TensorHandle ExpertsRemapFrom(
    ::ml_drift::GpuModelBuilder& builder,
    const ::ml_drift::GpuModelBuilder::TensorHandle& src,
    const ::ml_drift::GpuModelBuilder::TensorHandle& experts_remap,
    int num_active_experts);

absl::StatusOr<::ml_drift::GpuModelBuilder::TensorHandle>
MakeConvWithPackedGroups(
    ::ml_drift::GpuModelBuilder& builder,
    const ::ml_drift::GpuModelBuilder::TensorHandle& src,
    const ::ml_drift::GpuModelBuilder::TensorHandle& params,
    const ::ml_drift::GpuModelBuilder::Weights& weights,
    int num_active_experts);

}  // namespace litert::ml_drift

#endif  // ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_EXPERTS_REMAP_BUILDER_H_
