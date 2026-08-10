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

#include "ml_drift_delegate/delegate/composite/sdpa_transposed_kernel.h"

#include <any>
#include <cstdint>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/sdpa_transposed_parser.h"

namespace litert::ml_drift {

absl::Status BuildSdpaTransposedGpuGraph(
    const std::vector<uint32_t>& input_ids, uint32_t output_id,
    const SdpaTransposedAttributes& attr,
    ::ml_drift::GpuModelBuilder* model_builder) {
  if (input_ids.size() < 3) {
    return absl::InvalidArgumentError(
        "SDPA transposed expects at least 3 inputs (Q, K, V).");
  }

  ABSL_ASSIGN_OR_RETURN(auto q, model_builder->GetTensor(input_ids[0]));
  ABSL_ASSIGN_OR_RETURN(auto k, model_builder->GetTensor(input_ids[1]));
  ABSL_ASSIGN_OR_RETURN(auto v, model_builder->GetTensor(input_ids[2]));

  ::ml_drift::GpuModelBuilder::TensorHandle mask;
  bool has_mask = false;
  if (input_ids.size() > 3) {
    ABSL_ASSIGN_OR_RETURN(mask, model_builder->GetTensor(input_ids[3]));
    has_mask = true;
  }

  ::ml_drift::GpuModelBuilder::TensorHandle param_tensor;
  ::ml_drift::GpuModelBuilder::TensorHandle* param_tensor_ptr = nullptr;
  if (input_ids.size() > 4) {
    ABSL_ASSIGN_OR_RETURN(param_tensor, model_builder->GetTensor(input_ids[4]));
    param_tensor_ptr = &param_tensor;
  }

  ::ml_drift::WeightsDescription bmm1_desc = attr.bmm1_weights.desc;
  bmm1_desc.type = q.tensor_desc.GetDataType();
  const ::ml_drift::GpuModelBuilder::Weights bmm1_external_weights =
      ::ml_drift::CreateExternalWeights(k, bmm1_desc,
                                        attr.bmm1_weights.weights_shape);
  ::ml_drift::ConvRuntimeCheckDesc bmm1_runtime_check = {
      .dst_end_ch_index = attr.runtime_check.src_end_ch_index,
  };

  ABSL_ASSIGN_OR_RETURN(
      auto logits, model_builder->FullyConnectedExternalWeights(
                  q, bmm1_external_weights, /*biases=*/nullptr,
                  /*src_exp=*/nullptr, bmm1_runtime_check, param_tensor_ptr));

  if (has_mask) {
    if (mask.tensor_desc.GetDataType() == ::ml_drift::DataType::BOOL) {
      ::ml_drift::Tensor<::ml_drift::StrongShape<::ml_drift::Layout::BHWC>,
                         ::ml_drift::DataType::FLOAT32>
          fill_tensor;
      fill_tensor.shape = ::ml_drift::BHWC(1, 1, 1, 1);
      // Use a large negative value to simulate -inf. std::limit<float>::min()
      // causes regression.
      fill_tensor.data = {-10000.0f};
      auto neg_val = model_builder->AddConstantTensor(
          fill_tensor, logits.tensor_desc.GetDataType());
      logits = model_builder->SelectV2(mask, logits, neg_val);
    } else {
      logits = model_builder->Add(logits, mask);
    }
  }

  ::ml_drift::SoftmaxRuntimeCheckDesc softmax_runtime_check = {
      .end_ch_index = attr.runtime_check.src_end_ch_index,
  };
  auto sfmx_partial = model_builder->SoftmaxReduce(
      logits, softmax_runtime_check, param_tensor_ptr);

  ::ml_drift::WeightsDescription bmm2_desc = attr.bmm2_weights.desc;
  bmm2_desc.type = logits.tensor_desc.GetDataType();
  const ::ml_drift::GpuModelBuilder::Weights bmm2_external_weights =
      ::ml_drift::CreateExternalWeights(v, bmm2_desc,
                                        attr.bmm2_weights.weights_shape);
  ::ml_drift::ConvRuntimeCheckDesc bmm2_runtime_check = {
      .src_end_ch_index = attr.runtime_check.src_end_ch_index,
  };

  ABSL_ASSIGN_OR_RETURN(
      auto output, model_builder->FullyConnectedExternalWeights(
                  logits, bmm2_external_weights, /*biases=*/nullptr,
                  &sfmx_partial, bmm2_runtime_check, param_tensor_ptr));

  return model_builder->UpdateOutputTensor(output, output_id);
}

absl::Status CreateSdpaTransposedFromNode(
    const std::vector<::ml_drift::Value*>& inputs,
    const std::vector<::ml_drift::Value*>& outputs,
    const ::ml_drift::Node& node, ::ml_drift::GpuModelBuilder* model_builder) {
  const SdpaTransposedAttributes& attr =
      std::any_cast<const SdpaTransposedAttributes&>(node.operation.attributes);
  std::vector<uint32_t> input_ids;
  input_ids.reserve(inputs.size());
  for (const auto* input : inputs) input_ids.push_back(input->id);
  return BuildSdpaTransposedGpuGraph(input_ids, outputs[0]->id, attr,
                                     model_builder);
}

absl::Status CreateSdpaTransposedFromIrOp(
    const std::vector<const ::ml_drift::ir::IrTensor*>& inputs,
    const std::vector<const ::ml_drift::ir::IrTensor*>& outputs,
    const ::ml_drift::ir::IrOp& node,
    ::ml_drift::GpuModelBuilder* model_builder) {
  const SdpaTransposedAttributes& attr =
      std::any_cast<const SdpaTransposedAttributes&>(node.attr);
  std::vector<uint32_t> input_ids;
  input_ids.reserve(inputs.size());
  for (const auto* input : inputs) input_ids.push_back(input->id);
  return BuildSdpaTransposedGpuGraph(input_ids, outputs[0]->id, attr,
                                     model_builder);
}

}  // namespace litert::ml_drift
