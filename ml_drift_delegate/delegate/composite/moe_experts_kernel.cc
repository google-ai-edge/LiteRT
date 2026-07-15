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
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/kernels/fully_connected.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/ir/moe_experts_parser.h"
#include "ml_drift_delegate/delegate/composite/moe_experts_parser.h"

namespace litert::ml_drift {
namespace {

class ScaleWithBatchIdsOp : public ::ml_drift::GPUOperation {
 public:
  ScaleWithBatchIdsOp() = default;
  ::ml_drift::int3 GetGridSize() const override {
    return ::ml_drift::int3(dst_[0]->Width() * dst_[0]->Batch(),
                            dst_[0]->Height(), dst_[0]->Slices());
  }

  ScaleWithBatchIdsOp(ScaleWithBatchIdsOp&& operation) = default;
  ScaleWithBatchIdsOp& operator=(ScaleWithBatchIdsOp&& operation) = default;
  ScaleWithBatchIdsOp(const ScaleWithBatchIdsOp&) = delete;
  ScaleWithBatchIdsOp& operator=(const ScaleWithBatchIdsOp&) = delete;
};

std::unique_ptr<::ml_drift::GPUOperation> CreateScaleWithBatchIds(
    const ::ml_drift::TensorDescriptor& input,
    const ::ml_drift::TensorDescriptor& scales,
    const ::ml_drift::TensorDescriptor& batch_ids,
    const ::ml_drift::TensorDescriptor& dst) {
  ScaleWithBatchIdsOp op;
  op.AddSrcTensor("input", input);
  op.AddSrcTensor("scales", scales);
  op.AddSrcTensor("batch_ids", batch_ids);
  op.AddDstTensor("dst", dst);
  op.code_ = R"(
MAIN_FUNCTION($0) {
  int linear_id = ucl::GetGlobalId<0>();
  int x = linear_id / args.dst.Batch();
  int b = linear_id % args.dst.Batch();
  int y = ucl::GetGlobalId<1>();
  int s = ucl::GetGlobalId<2>();
  if (x >= args.dst.Width() || y >= args.dst.Height() ||
      s >= args.dst.Slices()) {
    return;
  }
  args.input::type in_value = args.input.Read(x, y, s, b);
  int expert_id;
  args.batch_ids.ReadPerChannel<int>(expert_id, 0, 0, y, 0);
  args.scales::scalar_type scale_value;
  args.scales.ReadPerChannel(scale_value, 0, 0, expert_id, 0);
  args.dst.Write(ucl::Convert<args.dst::type>(scale_value * in_value), x, y, s,
                 b);
}
)";
  return std::make_unique<ScaleWithBatchIdsOp>(std::move(op));
}

absl::StatusOr<::ml_drift::GpuModelBuilder::TensorHandle>
CreateDispatchTokenIndices(::ml_drift::GpuModelBuilder* model_builder,
                           int sequence_size, int num_active_experts) {
  const int num_dispatches = sequence_size * num_active_experts;
  ::ml_drift::TensorInt32 token_indices;
  token_indices.shape = ::ml_drift::BHWC(1, 1, 1, num_dispatches);
  token_indices.data.resize(num_dispatches);
  for (int token = 0; token < sequence_size; ++token) {
    for (int route = 0; route < num_active_experts; ++route) {
      token_indices.data[token * num_active_experts + route] = token;
    }
  }
  ::ml_drift::TensorDescriptor token_indices_desc(
      ::ml_drift::DataType::INT32, ::ml_drift::TensorStorageType::BUFFER,
      ::ml_drift::Layout::HWC);
  token_indices_desc.UploadData(token_indices);
  return model_builder->AddConstantTensor(std::move(token_indices_desc));
}

absl::StatusOr<::ml_drift::GpuModelBuilder::TensorHandle> ExpertFullyConnected(
    ::ml_drift::GpuModelBuilder* model_builder,
    const ::ml_drift::CreateGpuModelInfo& create_info,
    const ::ml_drift::GpuModelBuilder::TensorHandle& src,
    const ::ml_drift::GpuModelBuilder::TensorHandle& batch_ids,
    const ::ml_drift::GpuModelBuilder::TensorHandle& weights,
    const MoeScaleTensor* weight_scale, int input_channels, int output_channels,
    int num_experts, int num_dispatches,
    MoeExpertsAttributes::WeightType weight_type, const std::string& name) {
  const ::ml_drift::OHWI weights_shape(output_channels, num_experts, 1,
                                       input_channels);
  ::ml_drift::WeightsDescription weights_desc =
      weight_type == MoeExpertsAttributes::WeightType::kInt8
          ? model_builder->GetFullyConnectedInt8WeightsDesc(weights_shape)
          : model_builder->GetFullyConnectedWeightsDesc(
                src.tensor_desc.GetDataType(), weights_shape);

  ::ml_drift::GpuModelBuilder::TensorHandle scale_handle;
  ::ml_drift::GpuModelBuilder::TensorHandle* scale_handle_ptr = nullptr;
  if (weight_scale != nullptr) {
    auto scale_desc = ::ml_drift::ScaleOrZeroPointToTensorDesc(
        model_builder->gpu_info(), *weight_scale,
        src.tensor_desc.GetDataType());
    scale_handle = model_builder->AddConstantTensor(std::move(scale_desc));
    scale_handle_ptr = &scale_handle;
  }

  std::vector<::ml_drift::GpuModelBuilder::TensorHandle> converted_weights =
      model_builder->WeightsConversion(weights, ::ml_drift::Layout::OHWI,
                                       weights_desc, weights_shape,
                                       scale_handle_ptr,
                                       /*weights_zero_point=*/nullptr);

  ::ml_drift::GpuModelBuilder::TensorHandle dst = model_builder->AddTensor(
      1, num_dispatches, 1, output_channels, src.tensor_desc.GetDataType());
  const ::ml_drift::BHWC dst_shape = dst.tensor_desc.GetBHWCShape();
  ::ml_drift::ExternalWeights external_weights;
  external_weights.desc = weights_desc;
  external_weights.shape = weights_shape;
  if (scale_handle_ptr != nullptr) {
    external_weights.scale = &scale_handle_ptr->tensor_desc;
  }

  ASSIGN_OR_RETURN(auto operation,
                   ::ml_drift::CreateFullyConnectedWeightsBatchIds(
                       model_builder->gpu_info(), create_info.precision,
                       src.tensor_desc, batch_ids.tensor_desc, dst.tensor_desc,
                       external_weights, /*bias=*/nullptr, &dst_shape));
  operation.flops_ = dst_shape.DimensionsProduct() * input_channels * 2;

  std::vector<::ml_drift::GpuModelBuilder::TensorHandle> srcs = {src,
                                                                 batch_ids};
  srcs.insert(srcs.end(), converted_weights.begin(), converted_weights.end());
  if (scale_handle_ptr != nullptr) {
    srcs.push_back(*scale_handle_ptr);
  }

  model_builder->AddGpuOperation(
      srcs, {dst},
      std::make_unique<::ml_drift::FullyConnected>(std::move(operation)), name);
  return dst;
}

absl::StatusOr<::ml_drift::GpuModelBuilder::TensorHandle> ScaleWithBatchIds(
    ::ml_drift::GpuModelBuilder* model_builder,
    const ::ml_drift::GpuModelBuilder::TensorHandle& input,
    const ::ml_drift::GpuModelBuilder::TensorHandle& scales,
    const ::ml_drift::GpuModelBuilder::TensorHandle& batch_ids) {
  if (input.tensor_desc.GetBHWCShape().h !=
      batch_ids.tensor_desc.GetBHWCShape().c) {
    return absl::InvalidArgumentError(
        "MoE ScaleWithBatchIds requires input.h == batch_ids.c.");
  }
  ::ml_drift::GpuModelBuilder::TensorHandle dst = model_builder->AddTensor(
      input.tensor_desc.GetBHWCShape(), input.tensor_desc.GetDataType());
  model_builder->AddGpuOperation(
      {input, scales, batch_ids}, {dst},
      CreateScaleWithBatchIds(input.tensor_desc, scales.tensor_desc,
                              batch_ids.tensor_desc, dst.tensor_desc),
      "moe_scale_with_batch_ids");
  return dst;
}

absl::Status BuildMoeExpertsGpuGraph(
    ::ml_drift::GpuModelBuilder* model_builder,
    const ::ml_drift::CreateGpuModelInfo& create_info,
    const ::ml_drift::GpuModelBuilder::TensorHandle& src,
    const ::ml_drift::GpuModelBuilder::TensorHandle& top_weights,
    const ::ml_drift::GpuModelBuilder::TensorHandle& top_indices,
    const ::ml_drift::GpuModelBuilder::TensorHandle& gate_weight,
    const ::ml_drift::GpuModelBuilder::TensorHandle& ff1_weight,
    const ::ml_drift::GpuModelBuilder::TensorHandle& linear_weight,
    const ::ml_drift::GpuModelBuilder::TensorHandle& per_expert_scale,
    const MoeScaleTensor* gate_scale_ptr, const MoeScaleTensor* ff1_scale_ptr,
    const MoeScaleTensor* linear_scale_ptr, int model_dim, int hidden_dim,
    int num_experts, int num_active_experts,
    MoeExpertsAttributes::WeightType weight_type, ::ml_drift::ValueId output_id,
    const ::ml_drift::BHWC& output_shape) {
  const ::ml_drift::BHWC src_shape = src.tensor_desc.GetBHWCShape();
  const int sequence_size = src_shape.w;
  const int num_dispatches = sequence_size * num_active_experts;
  auto src_tokens = model_builder->Reshape(
      src, ::ml_drift::BHWC(1, sequence_size, 1, model_dim));
  auto flat_top_indices = model_builder->Reshape(
      top_indices, ::ml_drift::BHWC(1, 1, 1, num_dispatches));
  ASSIGN_OR_RETURN(auto token_indices,
                   CreateDispatchTokenIndices(model_builder, sequence_size,
                                              num_active_experts));
  auto dispatched_src = model_builder->Gather(src_tokens, token_indices,
                                              ::ml_drift::Axis::HEIGHT);

  ASSIGN_OR_RETURN(auto gate, ExpertFullyConnected(
                                  model_builder, create_info, dispatched_src,
                                  flat_top_indices, gate_weight, gate_scale_ptr,
                                  model_dim, hidden_dim, num_experts,
                                  num_dispatches, weight_type, "moe_ff_gate"));
  gate = model_builder->MakeGelu(gate);

  ASSIGN_OR_RETURN(auto ff1, ExpertFullyConnected(
                                 model_builder, create_info, dispatched_src,
                                 flat_top_indices, ff1_weight, ff1_scale_ptr,
                                 model_dim, hidden_dim, num_experts,
                                 num_dispatches, weight_type, "moe_ff1"));
  auto hidden = model_builder->Multiplication(gate, ff1);

  ASSIGN_OR_RETURN(auto expert_outputs,
                   ExpertFullyConnected(
                       model_builder, create_info, hidden, flat_top_indices,
                       linear_weight, linear_scale_ptr, hidden_dim, model_dim,
                       num_experts, num_dispatches, weight_type, "moe_linear"));
  ASSIGN_OR_RETURN(expert_outputs,
                   ScaleWithBatchIds(model_builder, expert_outputs,
                                     per_expert_scale, flat_top_indices));

  auto reshaped_top_weights = model_builder->Reshape(
      top_weights, ::ml_drift::BHWC(1, sequence_size, 1, num_active_experts));
  auto reshaped_expert_outputs = model_builder->Reshape(
      expert_outputs,
      ::ml_drift::BHWC(1, sequence_size, num_active_experts, model_dim));
  ASSIGN_OR_RETURN(auto combined,
                   model_builder->BatchedMatMul(reshaped_top_weights,
                                                reshaped_expert_outputs));
  combined = model_builder->Reshape(combined, output_shape);
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

  ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]->id));
  ASSIGN_OR_RETURN(auto top_weights, model_builder->GetTensor(inputs[1]->id));
  ASSIGN_OR_RETURN(auto top_indices, model_builder->GetTensor(inputs[2]->id));

  ::ml_drift::GpuModelBuilder::TensorHandle gate_weight;
  ::ml_drift::GpuModelBuilder::TensorHandle ff1_weight;
  ::ml_drift::GpuModelBuilder::TensorHandle linear_weight;
  ::ml_drift::GpuModelBuilder::TensorHandle per_expert_scale;
  const MoeScaleTensor* gate_scale_ptr = nullptr;
  const MoeScaleTensor* ff1_scale_ptr = nullptr;
  const MoeScaleTensor* linear_scale_ptr = nullptr;

  if (attr.weight_type == MoeExpertsAttributes::WeightType::kFp32) {
    ASSIGN_OR_RETURN(gate_weight, model_builder->GetTensor(inputs[3]->id));
    ASSIGN_OR_RETURN(ff1_weight, model_builder->GetTensor(inputs[4]->id));
    ASSIGN_OR_RETURN(linear_weight, model_builder->GetTensor(inputs[5]->id));
    ASSIGN_OR_RETURN(per_expert_scale, model_builder->GetTensor(inputs[6]->id));
  } else {
    ASSIGN_OR_RETURN(gate_weight, model_builder->GetTensor(inputs[3]->id));
    ASSIGN_OR_RETURN(ff1_weight, model_builder->GetTensor(inputs[5]->id));
    ASSIGN_OR_RETURN(linear_weight, model_builder->GetTensor(inputs[7]->id));
    ASSIGN_OR_RETURN(per_expert_scale, model_builder->GetTensor(inputs[9]->id));
    if (!attr.ff_gate_scale.has_value() || !attr.ff1_scale.has_value() ||
        !attr.linear_scale.has_value()) {
      return absl::InvalidArgumentError(
          "MoE int8 expert weights require per-expert scale tensors.");
    }
    gate_scale_ptr = &attr.ff_gate_scale.value();
    ff1_scale_ptr = &attr.ff1_scale.value();
    linear_scale_ptr = &attr.linear_scale.value();
  }

  return BuildMoeExpertsGpuGraph(
      model_builder, create_info, src, top_weights, top_indices, gate_weight,
      ff1_weight, linear_weight, per_expert_scale, gate_scale_ptr,
      ff1_scale_ptr, linear_scale_ptr, attr.model_dim, attr.hidden_dim,
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

  ASSIGN_OR_RETURN(auto src, model_builder->GetTensor(inputs[0]->id));
  ASSIGN_OR_RETURN(auto top_weights, model_builder->GetTensor(inputs[1]->id));
  ASSIGN_OR_RETURN(auto top_indices, model_builder->GetTensor(inputs[2]->id));

  ::ml_drift::GpuModelBuilder::TensorHandle gate_weight;
  ::ml_drift::GpuModelBuilder::TensorHandle ff1_weight;
  ::ml_drift::GpuModelBuilder::TensorHandle linear_weight;
  ::ml_drift::GpuModelBuilder::TensorHandle per_expert_scale;
  const MoeScaleTensor* gate_scale_ptr = nullptr;
  const MoeScaleTensor* ff1_scale_ptr = nullptr;
  const MoeScaleTensor* linear_scale_ptr = nullptr;

  ASSIGN_OR_RETURN(gate_weight, model_builder->GetTensor(inputs[3]->id));
  ASSIGN_OR_RETURN(ff1_weight, model_builder->GetTensor(inputs[4]->id));
  ASSIGN_OR_RETURN(linear_weight, model_builder->GetTensor(inputs[5]->id));
  ASSIGN_OR_RETURN(per_expert_scale, model_builder->GetTensor(inputs[6]->id));

  if (attr.weight_type == ir::MoeExpertsAttributes::WeightType::kInt8) {
    if (!attr.ff_gate_scale.has_value() || !attr.ff1_scale.has_value() ||
        !attr.linear_scale.has_value()) {
      return absl::InvalidArgumentError(
          "MoE int8 expert weights require per-expert scale tensors.");
    }
    gate_scale_ptr = &attr.ff_gate_scale.value();
    ff1_scale_ptr = &attr.ff1_scale.value();
    linear_scale_ptr = &attr.linear_scale.value();
  }

  auto legacy_weight_type =
      attr.weight_type == ir::MoeExpertsAttributes::WeightType::kInt8
          ? MoeExpertsAttributes::WeightType::kInt8
          : MoeExpertsAttributes::WeightType::kFp32;

  return BuildMoeExpertsGpuGraph(
      model_builder, create_info, src, top_weights, top_indices, gate_weight,
      ff1_weight, linear_weight, per_expert_scale, gate_scale_ptr,
      ff1_scale_ptr, linear_scale_ptr, attr.model_dim, attr.hidden_dim,
      attr.num_experts, attr.num_active_experts, legacy_weight_type,
      outputs[0]->id, outputs[0]->desc.GetBHWCShape());
}

}  // namespace litert::ml_drift
