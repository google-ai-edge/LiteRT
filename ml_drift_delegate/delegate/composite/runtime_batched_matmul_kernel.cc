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

#include "ml_drift_delegate/delegate/composite/runtime_batched_matmul_kernel.h"

#include <any>
#include <cstdint>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/runtime_batched_matmul_parser.h"

namespace litert::ml_drift {

absl::Status BuildRuntimeBatchedMatMulGpuGraph(
    const std::vector<uint32_t>& input_ids, uint32_t output_id,
    RuntimeBatchedMatMulAttributes attr,
    ::ml_drift::GpuModelBuilder* model_builder) {
  if (!attr.runtime_check.src_end_ch_index.has_value() &&
      !attr.runtime_check.dst_end_ch_index.has_value()) {
    return absl::InvalidArgumentError(
        "Runtime Batched MatMul requires runtime check information.");
  }
  auto& runtime_check_attr = attr.runtime_check;
  ::ml_drift::ConvRuntimeCheckDesc runtime_check = {
      .src_end_ch_index = runtime_check_attr.src_end_ch_index,
      .dst_end_ch_index = runtime_check_attr.dst_end_ch_index,
  };

  // FC case
  if (attr.external_weights.has_value()) {
    if (input_ids.size() < 2 || input_ids.size() > 5) {
      return absl::InvalidArgumentError(
          "Expected 2, 3, 4, or 5 inputs for FullyConnectedExternalWeights.");
    }

    ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(input_ids[0]));
    ABSL_ASSIGN_OR_RETURN(auto weights, model_builder->GetTensor(input_ids[1]));
    ::ml_drift::GpuModelBuilder::TensorHandle bias;
    ::ml_drift::GpuModelBuilder::TensorHandle* bias_ptr = nullptr;
    ::ml_drift::GpuModelBuilder::TensorHandle src_exp;
    ::ml_drift::GpuModelBuilder::TensorHandle* src_exp_ptr = nullptr;
    ::ml_drift::GpuModelBuilder::TensorHandle runtime_check_tensor;
    ::ml_drift::GpuModelBuilder::TensorHandle* runtime_check_tensor_ptr =
        nullptr;
    if (runtime_check_attr.src_end_ch_index.has_value() ||
        runtime_check_attr.dst_end_ch_index.has_value()) {
      if (input_ids.size() < 3) {
        return absl::InvalidArgumentError("Missing runtime check tensor.");
      }
      if (input_ids.size() > 3) {
        ABSL_ASSIGN_OR_RETURN(bias, model_builder->GetTensor(input_ids[2]));
        bias_ptr = &bias;
      }
      if (input_ids.size() > 4) {
        ABSL_ASSIGN_OR_RETURN(src_exp, model_builder->GetTensor(input_ids[3]));
        src_exp_ptr = &src_exp;
      }
      ABSL_ASSIGN_OR_RETURN(
          runtime_check_tensor,
          model_builder->GetTensor(input_ids[input_ids.size() - 1]));
      runtime_check_tensor_ptr = &runtime_check_tensor;
    } else {
      if (input_ids.size() > 4) {
        return absl::InvalidArgumentError("Unexpected runtime check tensor.");
      }
      if (input_ids.size() > 2) {
        ABSL_ASSIGN_OR_RETURN(bias, model_builder->GetTensor(input_ids[2]));
        bias_ptr = &bias;
      }
      if (input_ids.size() > 3) {
        ABSL_ASSIGN_OR_RETURN(src_exp, model_builder->GetTensor(input_ids[3]));
        src_exp_ptr = &src_exp;
      }
    }

    const ::ml_drift::OHWI& weights_shape =
        attr.external_weights->weights_shape;
    ::ml_drift::WeightsDescription& weights_desc =
        attr.external_weights->desc;
    if (weights_desc.type != ::ml_drift::DataType::UINT8) {
      weights_desc.type = src.tensor_desc.GetDataType();
    }

    if (weights_desc.type == ::ml_drift::DataType::UINT8) {
      if (!attr.scale.has_value()) {
        return absl::InvalidArgumentError(
            "Runtime Batched MatMul requires channel_count and scale for "
            "quantized case.");
      }
      ::ml_drift::GpuModelBuilder::TensorHandle scale_tensor =
          model_builder->AddConstantTensor(attr.scale.value(),
                                           src.tensor_desc.GetDataType());
      const ::ml_drift::GpuModelBuilder::Weights external_weights =
          CreateExternalWeights(
              weights, weights_desc, weights_shape,
              ::ml_drift::OHWI(attr.scale.value().shape.v, 1, 1, 1),
              &scale_tensor);
      auto output = model_builder->FullyConnectedInt8ExternalWeights(
          src, external_weights, bias_ptr, src_exp_ptr, runtime_check,
          runtime_check_tensor_ptr);
      return model_builder->UpdateOutputTensor(output, output_id);
    } else {
      const ::ml_drift::GpuModelBuilder::Weights external_weights =
          CreateExternalWeights(weights, weights_desc, weights_shape);
      ABSL_ASSIGN_OR_RETURN(auto output,
                            model_builder->FullyConnectedExternalWeights(
                                src, external_weights, bias_ptr, src_exp_ptr,
                                runtime_check, runtime_check_tensor_ptr));
      return model_builder->UpdateOutputTensor(output, output_id);
    }
  }
  // bmm case
  ::ml_drift::BatchedMatMulAttributes bmm_attr;
  if (!attr.transpose_left.has_value() && !attr.transpose_right.has_value()) {
    return absl::InvalidArgumentError(
        "Runtime BatchedMatMul requires transpose_left or transpose_right for "
        "BMM shape case.");
  }
  bmm_attr.transpose_left = attr.transpose_left.value();
  bmm_attr.transpose_right = attr.transpose_right.value();
  ABSL_ASSIGN_OR_RETURN(auto left, model_builder->GetTensor(input_ids[0]));
  ABSL_ASSIGN_OR_RETURN(auto right, model_builder->GetTensor(input_ids[1]));
  ABSL_ASSIGN_OR_RETURN(auto runtime_check_tensor,
                        model_builder->GetTensor(input_ids[2]));

  ABSL_ASSIGN_OR_RETURN(
      auto output,
      model_builder->BatchedMatMul(left, right, bmm_attr, /*src_exp=*/nullptr,
                                   runtime_check, &runtime_check_tensor));
  return model_builder->UpdateOutputTensor(output, output_id);
}

absl::Status CreateRuntimeBatchedMatMulFromNode(
    const std::vector<::ml_drift::Value*>& inputs,
    const std::vector<::ml_drift::Value*>& outputs,
    const ::ml_drift::Node& node, ::ml_drift::GpuModelBuilder* model_builder) {
  const RuntimeBatchedMatMulAttributes& attr =
      std::any_cast<const RuntimeBatchedMatMulAttributes&>(
          node.operation.attributes);
  std::vector<uint32_t> input_ids;
  input_ids.reserve(inputs.size());
  for (const auto* input : inputs) input_ids.push_back(input->id);
  return BuildRuntimeBatchedMatMulGpuGraph(input_ids, outputs[0]->id, attr,
                                           model_builder);
}

absl::Status CreateRuntimeBatchedMatMulFromIrOp(
    const std::vector<const ::ml_drift::ir::IrTensor*>& inputs,
    const std::vector<const ::ml_drift::ir::IrTensor*>& outputs,
    const ::ml_drift::ir::IrOp& node,
    ::ml_drift::GpuModelBuilder* model_builder) {
  const RuntimeBatchedMatMulAttributes& attr =
      std::any_cast<const RuntimeBatchedMatMulAttributes&>(node.attr);
  std::vector<uint32_t> input_ids;
  input_ids.reserve(inputs.size());
  for (const auto* input : inputs) input_ids.push_back(input->id);
  return BuildRuntimeBatchedMatMulGpuGraph(input_ids, outputs[0]->id, attr,
                                           model_builder);
}

}  // namespace litert::ml_drift
