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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_SDPA_TRANSPOSED_KERNEL_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_SDPA_TRANSPOSED_KERNEL_H_

#include <cstdint>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/sdpa_transposed_parser.h"
#include "ml_drift_delegate/delegate/composite/sdpa_transposed_support.h"  // IWYU pragma: export

namespace litert::ml_drift {

// `SupportsFusedSdpaKernels` and the other device and shape checks of the
// fused kernels are declared in `sdpa_transposed_support.h`.

// Builds `odml.sdpa_transposed` as a fused Flash-Attention kernel when one
// applies, otherwise as the decomposed BatchedMatMul + Softmax graph.
// `max_decomposed_score_bytes` is the score tensor budget of the decomposed
// prefill graph on backends without the MMA prefill kernel (see
// `UsesWorkGroupFlashPrefill`); tests lower it to run small prefill shapes on
// the work-group Flash kernel.
// `max_decomposed_score_chunk_bytes` bounds the score tensor that the
// decomposed graph materializes at once (see
// `kMaxDecomposedSdpaScoreChunkBytes`); on OpenCL and WebGPU, scores within
// `max_decomposed_score_bytes` are not split (see
// `DecomposedSdpaScoreChunkBudget`). Tests lower both to split small shapes
// into chunks.
absl::Status BuildSdpaTransposedGpuGraph(
    const std::vector<uint32_t>& input_ids, uint32_t output_id,
    const SdpaTransposedAttributes& attr,
    ::ml_drift::GpuModelBuilder* model_builder,
    uint64_t max_decomposed_score_bytes = kMaxDecomposedSdpaScoreBytes,
    uint64_t max_decomposed_score_chunk_bytes =
        kMaxDecomposedSdpaScoreChunkBytes);

absl::Status CreateSdpaTransposedFromNode(
    const std::vector<::ml_drift::Value*>& inputs,
    const std::vector<::ml_drift::Value*>& outputs,
    const ::ml_drift::Node& node, ::ml_drift::GpuModelBuilder* model_builder);

absl::Status CreateSdpaTransposedFromIrOp(
    const std::vector<const ::ml_drift::ir::IrTensor*>& inputs,
    const std::vector<const ::ml_drift::ir::IrTensor*>& outputs,
    const ::ml_drift::ir::IrOp& node,
    ::ml_drift::GpuModelBuilder* model_builder);

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_SDPA_TRANSPOSED_KERNEL_H_
