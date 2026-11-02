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

#include "ml_drift_delegate/delegate/composite/moe_experts_kernel.h"

#include <any>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder_moe_util.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/ir/moe_experts_parser.h"
#include "ml_drift_delegate/delegate/composite/moe_experts_parser.h"

namespace litert::ml_drift {
namespace {

// Returns true if `weights` were already repacked into the GPU weights layout
// (e.g. by SharedMemoryManager for shared constant tensors), in which case
// WeightsConversion must be skipped.
bool IsPrepackedWeights(
    const ::ml_drift::GpuModelBuilder::TensorHandle& weights) {
  const ::ml_drift::DataType type = weights.tensor_desc.GetDataType();
  return type == ::ml_drift::DataType::kUint32 ||
         type == ::ml_drift::DataType::kUint16 ||
         type == ::ml_drift::DataType::kUint8 ||
         type == ::ml_drift::DataType::kUint4 ||
         type == ::ml_drift::DataType::kUint2;
}

::ml_drift::GpuModelBuilder::Weights BuildExpertWeights(
    ::ml_drift::GpuModelBuilder* model_builder,
    const ::ml_drift::GpuModelBuilder::TensorHandle& src,
    const ::ml_drift::GpuModelBuilder::TensorHandle& weights,
    const MoeScaleTensor* weight_scale,
    const ::ml_drift::GpuModelBuilder::TensorHandle* shared_weight_scale,
    int input_channels, int output_channels, int num_experts,
    MoeExpertsAttributes::WeightType weight_type) {
  const ::ml_drift::OHWI weights_shape(output_channels, num_experts, 1,
                                       input_channels);
  ::ml_drift::WeightsDescription weights_desc =
      weight_type == MoeExpertsAttributes::WeightType::kInt4
          ? model_builder->GetFullyConnectedInt4WeightsDesc(weights_shape)
          : weight_type == MoeExpertsAttributes::WeightType::kInt8
                ? model_builder->GetFullyConnectedInt8WeightsDesc(weights_shape)
                : model_builder->GetFullyConnectedWeightsDesc(
                      src.tensor_desc.GetDataType(), weights_shape);

  ::ml_drift::GpuModelBuilder::TensorHandle scale_handle;
  ::ml_drift::GpuModelBuilder::TensorHandle* scale_handle_ptr = nullptr;
  if (shared_weight_scale != nullptr) {
    // The scale tensor is shared across subgraphs: SharedMemoryManager already
    // converted it to the GPU scale layout, so it is used as is instead of
    // uploading a private copy per subgraph.
    scale_handle = *shared_weight_scale;
    scale_handle_ptr = &scale_handle;
  } else if (weight_scale != nullptr && !weight_scale->empty()) {
    auto scale_desc = ::ml_drift::ScaleOrZeroPointToTensorDesc(
        model_builder->gpu_info(), *weight_scale,
        src.tensor_desc.GetDataType());
    scale_handle = model_builder->AddConstantTensor(std::move(scale_desc));
    scale_handle_ptr = &scale_handle;
  }

  ::ml_drift::GpuModelBuilder::Weights result;
  result.shape = weights_shape;
  result.desc = weights_desc;
  if (scale_handle_ptr != nullptr) {
    result.scale = *scale_handle_ptr;
    // The scale tensor carries its own layout: [out_channels, experts, 1,
    // blocks_per_row]. Taking it verbatim means a per-output-channel scale
    // (blocks_per_row == 1) and a blockwise scale are handled by the same
    // code path; the kernel indexes it by expert and block already.
    result.scale_zp_shape = weight_scale->shape;
  }
  if (IsPrepackedWeights(weights)) {
    result.weights = weights;
  } else {
    std::vector<::ml_drift::GpuModelBuilder::TensorHandle> converted_weights =
        model_builder->WeightsConversion(weights, ::ml_drift::Layout::kOHWI,
                                         result.desc, result.shape,
                                         scale_handle_ptr,
                                         /*weights_zero_point=*/nullptr);
    result.weights = converted_weights[0];
  }
  return result;
}

// `*_shared_scale` point at the block-scale tensors shared across subgraphs
// (already in the GPU scale layout) and are null when the scales come from the
// op attributes instead.
absl::Status BuildMoeExpertsGpuGraph(
    ::ml_drift::GpuModelBuilder* model_builder,
    const ::ml_drift::GpuModelBuilder::TensorHandle& src,
    const ::ml_drift::GpuModelBuilder::TensorHandle& top_weights,
    const ::ml_drift::GpuModelBuilder::TensorHandle& top_indices,
    const ::ml_drift::GpuModelBuilder::TensorHandle& gate_weight,
    const ::ml_drift::GpuModelBuilder::TensorHandle& ff1_weight,
    const ::ml_drift::GpuModelBuilder::TensorHandle& linear_weight,
    const ::ml_drift::GpuModelBuilder::TensorHandle& per_expert_scale,
    const MoeScaleTensor* gate_scale_ptr, const MoeScaleTensor* ff1_scale_ptr,
    const MoeScaleTensor* linear_scale_ptr,
    const ::ml_drift::GpuModelBuilder::TensorHandle* gate_shared_scale,
    const ::ml_drift::GpuModelBuilder::TensorHandle* ff1_shared_scale,
    const ::ml_drift::GpuModelBuilder::TensorHandle* linear_shared_scale,
    int model_dim, int hidden_dim, int num_experts, int num_active_experts,
    MoeExpertsAttributes::WeightType weight_type, ::ml_drift::ValueId output_id,
    const ::ml_drift::BHWC& output_shape) {
  const ::ml_drift::BHWC src_shape = src.tensor_desc.GetBHWCShape();
  const int sequence_size = src_shape.w;
  const bool use_packed_groups =
      sequence_size * num_active_experts > num_experts;

  auto expert_src = src;
  auto expert_params = top_indices;
  ::ml_drift::GpuModelBuilder::TensorHandle experts_packed_remap;

  if (use_packed_groups) {
    auto vals = ::ml_drift::CreateExpertsRemap(*model_builder, top_indices,
                                               num_experts);
    experts_packed_remap = vals[2];
    expert_params = vals[1];  // experts count and offsets
    expert_src = ::ml_drift::ExpertsRemapTo(
        *model_builder, src, experts_packed_remap, num_active_experts);
  } else if (sequence_size != 1) {
    auto t =
        model_builder->Tile(src, ::ml_drift::Axis::kHeight, num_active_experts);
    t = model_builder->Transpose(t, ::ml_drift::BHWC(0, 2, 1, 3));
    auto packed_shape = t.tensor_desc.GetBHWCShape();
    packed_shape.h = packed_shape.w * packed_shape.h;
    packed_shape.w = 1;
    expert_src = model_builder->Reshape(t, packed_shape);
    expert_params = model_builder->Reshape(
        top_indices,
        ::ml_drift::BHWC(1, 1, 1, sequence_size * num_active_experts));
  }

  auto run_expert_projection =
      [&](const ::ml_drift::GpuModelBuilder::TensorHandle& input,
          const ::ml_drift::GpuModelBuilder::TensorHandle& weights_handle,
          const MoeScaleTensor* scale_ptr,
          const ::ml_drift::GpuModelBuilder::TensorHandle* shared_scale,
          int in_channels, int out_channels)
      -> absl::StatusOr<::ml_drift::GpuModelBuilder::TensorHandle> {
    auto w = BuildExpertWeights(model_builder, input, weights_handle, scale_ptr,
                                shared_scale, in_channels, out_channels,
                                num_experts, weight_type);
    if (use_packed_groups) {
      return ::ml_drift::MakeConvWithPackedGroups(
          *model_builder, input, expert_params, w, num_active_experts);
    } else {
      return ::ml_drift::MakeConvWithBatchIds(*model_builder, input,
                                              expert_params, w);
    }
  };

  ABSL_ASSIGN_OR_RETURN(
      auto gate,
      run_expert_projection(expert_src, gate_weight, gate_scale_ptr,
                            gate_shared_scale, model_dim, hidden_dim));
  gate = model_builder->MakeGeluTanh(gate);

  ABSL_ASSIGN_OR_RETURN(
      auto ff1, run_expert_projection(expert_src, ff1_weight, ff1_scale_ptr,
                                      ff1_shared_scale, model_dim, hidden_dim));
  auto hidden = model_builder->Multiplication(ff1, gate);

  ABSL_ASSIGN_OR_RETURN(
      auto expert_outputs,
      run_expert_projection(hidden, linear_weight, linear_scale_ptr,
                            linear_shared_scale, hidden_dim, model_dim));

  if (use_packed_groups) {
    expert_outputs =
        ::ml_drift::ExpertsRemapFrom(*model_builder, expert_outputs,
                                     experts_packed_remap, num_active_experts);
    expert_outputs =
        model_builder->Transpose(expert_outputs, ::ml_drift::BHWC(0, 2, 1, 3));
  } else {
    expert_outputs = model_builder->Reshape(
        expert_outputs,
        ::ml_drift::BHWC(1, sequence_size, num_active_experts, model_dim));
  }

  ABSL_ASSIGN_OR_RETURN(
      auto combined,
      ::ml_drift::ScaleWithBatchIds(*model_builder, expert_outputs, top_indices,
                                    top_weights, per_expert_scale));
  if (combined.tensor_desc.GetBHWCShape() != output_shape) {
    combined = model_builder->Reshape(combined, output_shape);
  }
  return model_builder->UpdateOutputTensor(combined, output_id);
}

}  // namespace

absl::Status CreateMoeExpertsFromNode(
    const ::ml_drift::CreateGpuModelInfo& create_info,
    const std::vector<::ml_drift::Value*>& inputs,
    const std::vector<::ml_drift::Value*>& outputs,
    const ::ml_drift::Node& node, ::ml_drift::GpuModelBuilder* model_builder) {
  const MoeExpertsAttributes& attr =
      std::any_cast<const MoeExpertsAttributes&>(node.operation.attributes);
  const int expected_inputs =
      attr.weight_type == MoeExpertsAttributes::WeightType::kFp32 ? 7 : 10;
  if (inputs.size() != expected_inputs || outputs.size() != 1) {
    return absl::InvalidArgumentError(
        "MoE experts operation received an unexpected input/output count.");
  }

  ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]->id));
  ABSL_ASSIGN_OR_RETURN(auto top_weights,
                        model_builder->GetTensor(inputs[1]->id));
  ABSL_ASSIGN_OR_RETURN(auto top_indices,
                        model_builder->GetTensor(inputs[2]->id));

  ::ml_drift::GpuModelBuilder::TensorHandle gate_weight;
  ::ml_drift::GpuModelBuilder::TensorHandle ff1_weight;
  ::ml_drift::GpuModelBuilder::TensorHandle linear_weight;
  ::ml_drift::GpuModelBuilder::TensorHandle per_expert_scale;
  const MoeScaleTensor* gate_scale_ptr = nullptr;
  const MoeScaleTensor* ff1_scale_ptr = nullptr;
  const MoeScaleTensor* linear_scale_ptr = nullptr;
  ::ml_drift::GpuModelBuilder::TensorHandle gate_shared_scale;
  ::ml_drift::GpuModelBuilder::TensorHandle ff1_shared_scale;
  ::ml_drift::GpuModelBuilder::TensorHandle linear_shared_scale;
  const ::ml_drift::GpuModelBuilder::TensorHandle* gate_shared_scale_ptr =
      nullptr;
  const ::ml_drift::GpuModelBuilder::TensorHandle* ff1_shared_scale_ptr =
      nullptr;
  const ::ml_drift::GpuModelBuilder::TensorHandle* linear_shared_scale_ptr =
      nullptr;

  if (attr.weight_type == MoeExpertsAttributes::WeightType::kFp32) {
    ABSL_ASSIGN_OR_RETURN(gate_weight, model_builder->GetTensor(inputs[3]->id));
    ABSL_ASSIGN_OR_RETURN(ff1_weight, model_builder->GetTensor(inputs[4]->id));
    ABSL_ASSIGN_OR_RETURN(linear_weight,
                          model_builder->GetTensor(inputs[5]->id));
    ABSL_ASSIGN_OR_RETURN(per_expert_scale,
                          model_builder->GetTensor(inputs[6]->id));
  } else {
    ABSL_ASSIGN_OR_RETURN(gate_weight, model_builder->GetTensor(inputs[3]->id));
    ABSL_ASSIGN_OR_RETURN(ff1_weight, model_builder->GetTensor(inputs[5]->id));
    ABSL_ASSIGN_OR_RETURN(linear_weight,
                          model_builder->GetTensor(inputs[7]->id));
    ABSL_ASSIGN_OR_RETURN(per_expert_scale,
                          model_builder->GetTensor(inputs[9]->id));
    if (!attr.ff_gate_scale.has_value() || !attr.ff1_scale.has_value() ||
        !attr.linear_scale.has_value()) {
      return absl::InvalidArgumentError(
          "MoE int8 expert weights require per-expert scale tensors.");
    }
    gate_scale_ptr = &attr.ff_gate_scale.value();
    ff1_scale_ptr = &attr.ff1_scale.value();
    linear_scale_ptr = &attr.linear_scale.value();
    // A shape-only attribute means the parser registered the scale tensor as
    // a shared constant: SharedMemoryManager converted it to the GPU scale
    // layout once and exposes it on the corresponding scale input.
    if (gate_scale_ptr->empty()) {
      ABSL_ASSIGN_OR_RETURN(gate_shared_scale,
                            model_builder->GetTensor(inputs[4]->id));
      gate_shared_scale_ptr = &gate_shared_scale;
    }
    if (ff1_scale_ptr->empty()) {
      ABSL_ASSIGN_OR_RETURN(ff1_shared_scale,
                            model_builder->GetTensor(inputs[6]->id));
      ff1_shared_scale_ptr = &ff1_shared_scale;
    }
    if (linear_scale_ptr->empty()) {
      ABSL_ASSIGN_OR_RETURN(linear_shared_scale,
                            model_builder->GetTensor(inputs[8]->id));
      linear_shared_scale_ptr = &linear_shared_scale;
    }
  }

  return BuildMoeExpertsGpuGraph(
      model_builder, src, top_weights, top_indices, gate_weight, ff1_weight,
      linear_weight, per_expert_scale, gate_scale_ptr, ff1_scale_ptr,
      linear_scale_ptr, gate_shared_scale_ptr, ff1_shared_scale_ptr,
      linear_shared_scale_ptr, attr.model_dim, attr.hidden_dim,
      attr.num_experts, attr.num_active_experts, attr.weight_type,
      outputs[0]->id, outputs[0]->tensor.shape);
}

absl::Status CreateMoeExpertsFromIrOp(
    const ::ml_drift::CreateGpuModelInfo& create_info,
    const std::vector<const ::ml_drift::ir::IrTensor*>& inputs,
    const std::vector<const ::ml_drift::ir::IrTensor*>& outputs,
    const ::ml_drift::ir::IrOp& node,
    ::ml_drift::GpuModelBuilder* model_builder) {
  const ir::MoeExpertsAttributes& attr =
      std::any_cast<const ir::MoeExpertsAttributes&>(node.attr);
  if (inputs.size() != 7 || outputs.size() != 1) {
    return absl::InvalidArgumentError(
        "MoE experts operation received an unexpected input/output count.");
  }

  ABSL_ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]->id));
  ABSL_ASSIGN_OR_RETURN(auto top_weights,
                        model_builder->GetTensor(inputs[1]->id));
  ABSL_ASSIGN_OR_RETURN(auto top_indices,
                        model_builder->GetTensor(inputs[2]->id));

  ::ml_drift::GpuModelBuilder::TensorHandle gate_weight;
  ::ml_drift::GpuModelBuilder::TensorHandle ff1_weight;
  ::ml_drift::GpuModelBuilder::TensorHandle linear_weight;
  ::ml_drift::GpuModelBuilder::TensorHandle per_expert_scale;
  const MoeScaleTensor* gate_scale_ptr = nullptr;
  const MoeScaleTensor* ff1_scale_ptr = nullptr;
  const MoeScaleTensor* linear_scale_ptr = nullptr;

  ABSL_ASSIGN_OR_RETURN(gate_weight, model_builder->GetTensor(inputs[3]->id));
  ABSL_ASSIGN_OR_RETURN(ff1_weight, model_builder->GetTensor(inputs[4]->id));
  ABSL_ASSIGN_OR_RETURN(linear_weight, model_builder->GetTensor(inputs[5]->id));
  ABSL_ASSIGN_OR_RETURN(per_expert_scale,
                        model_builder->GetTensor(inputs[6]->id));

  if (attr.weight_type == ir::MoeExpertsAttributes::WeightType::kInt8 ||
      attr.weight_type == ir::MoeExpertsAttributes::WeightType::kInt4) {
    if (!attr.ff_gate_scale.has_value() || !attr.ff1_scale.has_value() ||
        !attr.linear_scale.has_value()) {
      return absl::InvalidArgumentError(
          "MoE quantized expert weights require per-expert scale tensors.");
    }
    gate_scale_ptr = &attr.ff_gate_scale.value();
    ff1_scale_ptr = &attr.ff1_scale.value();
    linear_scale_ptr = &attr.linear_scale.value();
  }

  auto legacy_weight_type =
      attr.weight_type == ir::MoeExpertsAttributes::WeightType::kInt8
          ? MoeExpertsAttributes::WeightType::kInt8
          : (attr.weight_type == ir::MoeExpertsAttributes::WeightType::kInt4
                 ? MoeExpertsAttributes::WeightType::kInt4
                 : MoeExpertsAttributes::WeightType::kFp32);

  return BuildMoeExpertsGpuGraph(
      model_builder, src, top_weights, top_indices, gate_weight, ff1_weight,
      linear_weight, per_expert_scale, gate_scale_ptr, ff1_scale_ptr,
      linear_scale_ptr, /*gate_shared_scale=*/nullptr,
      /*ff1_shared_scale=*/nullptr, /*linear_shared_scale=*/nullptr,
      attr.model_dim, attr.hidden_dim, attr.num_experts,
      attr.num_active_experts, legacy_weight_type, outputs[0]->id,
      outputs[0]->desc.GetBHWCShape());
}

}  // namespace litert::ml_drift
