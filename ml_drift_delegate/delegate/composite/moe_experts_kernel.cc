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
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder_moe_util.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/kernels/elementwise.h"  // from @ml_drift
#include "ml_drift/common/kernels/special/experts_remap.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/buffer_desc.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "ml_drift/common/util.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/ir/moe_experts_parser.h"
#include "ml_drift_delegate/delegate/composite/moe_experts_parser.h"

namespace litert::ml_drift {
namespace {

bool CanUseAppleMppDhwbcc4(const ::ml_drift::GpuModelBuilder& model_builder,
                           ::ml_drift::DataType src_dtype, int num_dispatches,
                           int num_experts, int in_channels) {
  const ::ml_drift::GpuInfo& gpu_info = model_builder.gpu_info();
  const int average_task_size =
      ::ml_drift::DivideRoundUp(num_dispatches, num_experts);
  const int in_slices = ::ml_drift::DivideRoundUp(in_channels, 4);
  return average_task_size >= 32 && in_slices % 8 == 0 &&
         model_builder.GetConvPrecision(src_dtype) ==
             ::ml_drift::CalculationsPrecision::kF16 &&
         gpu_info.IsApiMetal() && gpu_info.IsApple() &&
         gpu_info.apple_info.gpu_family >=
             ::ml_drift::AppleInfo::Family::kApple10 &&
         gpu_info.metal_info.IsMslVersionEqualOrHigher(4, 0) &&
         !gpu_info.metal_info.has_bugs_with_mpp;
}

::ml_drift::GpuModelBuilder::TensorHandle ConvertToDhwbcc4(
    ::ml_drift::GpuModelBuilder* model_builder,
    const ::ml_drift::GpuModelBuilder::TensorHandle& src) {
  ::ml_drift::TensorDescriptor tensor_desc{
      src.tensor_desc.GetDataType(), ::ml_drift::TensorStorageType::kBuffer,
      src.tensor_desc.GetLayout(),
      ::ml_drift::TensorDescriptor::PhysicalLayout1D::kDHWBCC4};
  tensor_desc.SetBHWCShape(src.tensor_desc.GetBHWCShape());
  ::ml_drift::GpuModelBuilder::TensorHandle new_tensor =
      model_builder->AddTensor(tensor_desc);
  model_builder->AddGpuOperation(
      {src}, {new_tensor},
      std::make_unique<::ml_drift::GPUOperation>(
          ::ml_drift::CreateElementwiseOneInput(
              model_builder->gpu_info(), src.tensor_desc,
              new_tensor.tensor_desc, ::ml_drift::OperationType::kCopy)),
      "to_dhwbcc4");
  return new_tensor;
}

// Writes `src` into `dst` in `kDHWBCC4` layout with slices along X so each
// SIMDgroup shares one `packed_idx` lookup and writes contiguous slices.
std::unique_ptr<::ml_drift::GPUOperation> CreateExpertsRemapToDhwbcc4Op(
    const ::ml_drift::TensorDescriptor& src,
    const ::ml_drift::TensorDescriptor& experts_packed_map,
    const ::ml_drift::TensorDescriptor& dst) {
  ::ml_drift::GPUOperation op;
  op.AddSrcTensor("src", src);
  op.AddSrcTensor("experts_packed_map", experts_packed_map);
  op.AddDstTensor("dst", dst);
  const ::ml_drift::BHWC dst_shape = dst.GetBHWCShape();
  op.SetGridSize(::ml_drift::int3(::ml_drift::DivideRoundUp(dst_shape.c, 4),
                                  dst_shape.w, 1));
  op.work_group_size_ = ::ml_drift::int3(32, 1, 1);
  op.code_ = R"(
MAIN_FUNCTION($0) {
  int s = ucl::GetGlobalId<0>();
  int packed_idx = ucl::GetGlobalId<1>();
  if (s >= args.dst.Slices() || packed_idx >= args.dst.Width()) {
    return;
  }
  int4 coords = args.experts_packed_map.Read(packed_idx, 0, 0);
  int seq_id = coords.y;
  int expert_id = min(coords.x, args.src.Height() - 1);
  args.src::type val = args.src.Read(seq_id, expert_id, s);
  args.dst.Write(val, packed_idx, 0, s);
}
)";
  return std::make_unique<::ml_drift::GPUOperation>(std::move(op));
}

// Inverts `experts_packed_map` (`packed_idx -> (ae_id, seq_id)`) into a flat
// int32 buffer `inv_remap[seq_id * num_active_experts + ae_id] = packed_idx`.
std::unique_ptr<::ml_drift::GPUOperation> CreateBuildInverseExpertsRemapOp(
    const ::ml_drift::TensorDescriptor& experts_packed_map, int num_dispatches,
    int num_active_experts) {
  ::ml_drift::GPUOperation op;
  op.AddSrcTensor("experts_packed_map", experts_packed_map);
  ::ml_drift::BufferDescriptor inv_remap_desc;
  inv_remap_desc.element_type = ::ml_drift::DataType::kInt32;
  inv_remap_desc.element_size = 1;
  op.AddDstBuffer("inv_remap", inv_remap_desc);
  op.args_.AddInt("num_dispatches", num_dispatches);
  op.args_.AddInt("num_active_experts", num_active_experts);
  op.SetGridSize(::ml_drift::int3(num_dispatches, 1, 1));
  op.work_group_size_ = ::ml_drift::int3(32, 1, 1);
  op.code_ = R"(
MAIN_FUNCTION($0) {
  int packed_idx = ucl::GetGlobalId<0>();
  if (packed_idx >= args.num_dispatches) {
    return;
  }
  int4 coords = args.experts_packed_map.Read(packed_idx, 0, 0);
  int ae_id = coords.x;
  int seq_id = coords.y;
  args.inv_remap.Write(packed_idx, seq_id * args.num_active_experts + ae_id);
}
)";
  return std::make_unique<::ml_drift::GPUOperation>(std::move(op));
}

// Gathers packed expert outputs directly via `inv_remap` and computes the
// weighted reduction per token without materializing `[1, AE, S, D]` or
// running a standalone `[1, AE, S, D] -> [1, S, AE, D]` transpose.
std::unique_ptr<::ml_drift::GPUOperation> CreateScaleWithBatchIdsFromPacked(
    const ::ml_drift::TensorDescriptor& input,
    const ::ml_drift::TensorDescriptor& active_ids,
    const ::ml_drift::TensorDescriptor& active_scales,
    const ::ml_drift::TensorDescriptor& scales,
    const ::ml_drift::TensorDescriptor& dst, int num_active_experts) {
  ::ml_drift::GPUOperation op;
  op.AddSrcTensor("input", input);
  ::ml_drift::BufferDescriptor inv_remap_desc;
  inv_remap_desc.element_type = ::ml_drift::DataType::kInt32;
  inv_remap_desc.element_size = 1;
  op.AddSrcBuffer("inv_remap", inv_remap_desc);
  op.AddSrcTensor("active_ids", active_ids);
  op.AddSrcTensor("active_scales", active_scales);
  op.AddSrcTensor("scales", scales);
  op.AddDstTensor("dst", dst);
  op.args_.AddInt("num_active_experts", num_active_experts);
  const ::ml_drift::BHWC dst_shape = dst.GetBHWCShape();
  op.SetGridSize(::ml_drift::int3(::ml_drift::DivideRoundUp(dst_shape.c, 4),
                                  dst_shape.w, 1));
  op.work_group_size_ = ::ml_drift::int3(32, 1, 1);
  op.code_ = R"(
MAIN_FUNCTION($0) {
  int s = ucl::GetGlobalId<0>();
  int x = ucl::GetGlobalId<1>();
  if (s >= args.dst.Slices() || x >= args.dst.Width()) {
    return;
  }
  float4 sum = ucl::Init<float4>(0.0f);
  for (int ae_id = 0; ae_id < args.num_active_experts; ++ae_id) {
    int packed_idx = args.inv_remap.Read(x * args.num_active_experts + ae_id);
    int expert_id;
    args.active_ids.ReadPerChannel<int>(expert_id, x, 0, ae_id, 0);
    float scale_value;
    args.scales.ReadPerChannel<float>(scale_value, 0, 0, expert_id, 0);
    float4 in_value = ucl::Convert<float4>(args.input.Read(packed_idx, 0, s, 0));
    float active_scale;
    args.active_scales.ReadPerChannel<float>(active_scale, x, 0, ae_id, 0);
    sum += in_value * (scale_value * active_scale);
  }
  args.dst::type res_value = ucl::Convert<args.dst::type>(sum);
  args.dst.Write(res_value, x, 0, s, 0);
}
)";
  return std::make_unique<::ml_drift::GPUOperation>(std::move(op));
}

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
    const ::ml_drift::TensorDescriptor& active_ids,
    const ::ml_drift::TensorDescriptor& active_scales,
    const ::ml_drift::TensorDescriptor& scales,
    const ::ml_drift::TensorDescriptor& dst) {
  ScaleWithBatchIdsOp op;
  op.AddSrcTensor("input", input);
  op.AddSrcTensor("active_ids", active_ids);
  op.AddSrcTensor("active_scales", active_scales);
  op.AddSrcTensor("scales", scales);
  op.AddDstTensor("dst", dst);
  op.tensor_to_grid_ = ::ml_drift::TensorToGrid::kWBToX_HDToY_SToZ;
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
  float4 sum = ucl::Init<float4>(0.0f);
  for (int ae_id = 0; ae_id < args.input.Width(); ++ae_id) {
    int expert_id;
    args.active_ids.ReadPerChannel<int>(expert_id, x, 0, ae_id, b);
    float scale_value;
    args.scales.ReadPerChannel<float>(scale_value, 0, 0, expert_id, 0);
    float4 in_value = ucl::Convert<float4>(args.input.Read(ae_id, x, s, b));
    float active_scale;
    args.active_scales.ReadPerChannel<float>(active_scale, x, 0, ae_id, b);
    sum += in_value * (scale_value * active_scale);
  }
  args.dst::type res_value = ucl::Convert<args.dst::type>(sum);
  args.dst.Write(res_value, x, y, s, b);
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
      ::ml_drift::DataType::kInt32, ::ml_drift::TensorStorageType::kBuffer,
      ::ml_drift::Layout::kHWC);
  token_indices_desc.UploadData(token_indices);
  return model_builder->AddConstantTensor(std::move(token_indices_desc));
}

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

absl::StatusOr<::ml_drift::GpuModelBuilder::TensorHandle> ScaleWithBatchIds(
    ::ml_drift::GpuModelBuilder* model_builder,
    const ::ml_drift::GpuModelBuilder::TensorHandle& input,
    const ::ml_drift::GpuModelBuilder::TensorHandle& active_ids,
    const ::ml_drift::GpuModelBuilder::TensorHandle& active_scales,
    const ::ml_drift::GpuModelBuilder::TensorHandle& scales) {
  const ::ml_drift::BHWC input_shape = input.tensor_desc.GetBHWCShape();
  const ::ml_drift::BHWC ids_shape = active_ids.tensor_desc.GetBHWCShape();
  const ::ml_drift::BHWC weights_shape =
      active_scales.tensor_desc.GetBHWCShape();
  if (input_shape.h != ids_shape.w || input_shape.w != ids_shape.c ||
      ids_shape != weights_shape) {
    return absl::InvalidArgumentError(
        "MoE ScaleWithBatchIds requires input [B, S, AE, D] and active_ids / "
        "active_scales [B, 1, S, AE].");
  }
  const ::ml_drift::BHWC dst_shape(input_shape.b, 1, input_shape.h,
                                   input_shape.c);
  ::ml_drift::GpuModelBuilder::TensorHandle dst =
      model_builder->AddTensor(dst_shape, input.tensor_desc.GetDataType());
  model_builder->AddGpuOperation(
      {input, active_ids, active_scales, scales}, {dst},
      CreateScaleWithBatchIds(input.tensor_desc, active_ids.tensor_desc,
                              active_scales.tensor_desc, scales.tensor_desc,
                              dst.tensor_desc),
      "moe_scale_with_batch_ids");
  return dst;
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
    MoeExpertsAttributes::WeightType weight_type,
    MoeExpertsAttributes::Activation activation, ::ml_drift::ValueId output_id,
    const ::ml_drift::BHWC& output_shape) {
  const ::ml_drift::BHWC src_shape = src.tensor_desc.GetBHWCShape();
  const int sequence_size = src_shape.w;
  const int num_dispatches = sequence_size * num_active_experts;
  const bool use_packed_groups =
      sequence_size * num_active_experts > num_experts;

  ::ml_drift::GpuModelBuilder::TensorHandle expert_src;
  ::ml_drift::GpuModelBuilder::TensorHandle expert_params;
  ::ml_drift::GpuModelBuilder::TensorHandle experts_packed_remap;

  const bool use_apple_mpp_in_dhwbcc4 =
      use_packed_groups && src_shape.b == 1 &&
      CanUseAppleMppDhwbcc4(*model_builder, src.tensor_desc.GetDataType(),
                            num_dispatches, num_experts, model_dim);
  const bool use_apple_mpp_hidden_dhwbcc4 =
      use_packed_groups && src_shape.b == 1 &&
      CanUseAppleMppDhwbcc4(*model_builder, src.tensor_desc.GetDataType(),
                            num_dispatches, num_experts, hidden_dim);

  auto maybe_reshape = [&](const ::ml_drift::GpuModelBuilder::TensorHandle& h,
                           const ::ml_drift::BHWC& target) {
    if (h.tensor_desc.GetBHWCShape() == target) {
      return h;
    }
    return model_builder->Reshape(h, target);
  };

  if (use_packed_groups) {
    auto vals = ::ml_drift::CreateExpertsRemap(*model_builder, top_indices,
                                               num_experts);
    experts_packed_remap = vals[2];
    expert_params = vals[1];  // experts count and offsets
    if (use_apple_mpp_in_dhwbcc4) {
      ::ml_drift::TensorDescriptor remapped_desc{
          src.tensor_desc.GetDataType(), ::ml_drift::TensorStorageType::kBuffer,
          ::ml_drift::Layout::kHWC,
          ::ml_drift::TensorDescriptor::PhysicalLayout1D::kDHWBCC4};
      remapped_desc.SetBHWCShape(
          ::ml_drift::BHWC(1, 1, num_dispatches, model_dim));
      expert_src = model_builder->AddTensor(remapped_desc);
      model_builder->AddGpuOperation(
          {src, experts_packed_remap}, {expert_src},
          CreateExpertsRemapToDhwbcc4Op(src.tensor_desc,
                                        experts_packed_remap.tensor_desc,
                                        expert_src.tensor_desc),
          "experts_remap_to_dhwbcc4");
    } else {
      expert_src = ::ml_drift::ExpertsRemapTo(
          *model_builder, src, experts_packed_remap, num_active_experts);
    }
  } else {
    expert_params =
        maybe_reshape(top_indices, ::ml_drift::BHWC(1, 1, 1, num_dispatches));
    if (sequence_size == 1) {
      expert_src = maybe_reshape(src, ::ml_drift::BHWC(1, 1, 1, model_dim));
    } else {
      auto src_tokens =
          maybe_reshape(src, ::ml_drift::BHWC(1, sequence_size, 1, model_dim));
      ABSL_ASSIGN_OR_RETURN(
          auto token_indices,
          CreateDispatchTokenIndices(model_builder, sequence_size,
                                     num_active_experts));
      expert_src = model_builder->Gather(src_tokens, token_indices,
                                         ::ml_drift::Axis::kHeight);
    }
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
  if (activation == MoeExpertsAttributes::Activation::kSilu) {
    gate = model_builder->SiLU(gate);
  } else {
    gate = model_builder->MakeGeluTanh(gate);
  }

  ABSL_ASSIGN_OR_RETURN(
      auto ff1, run_expert_projection(expert_src, ff1_weight, ff1_scale_ptr,
                                      ff1_shared_scale, model_dim, hidden_dim));
  auto hidden = model_builder->Multiplication(ff1, gate);
  if (use_apple_mpp_hidden_dhwbcc4) {
    hidden = ConvertToDhwbcc4(model_builder, hidden);
  }

  ABSL_ASSIGN_OR_RETURN(
      auto expert_outputs,
      run_expert_projection(hidden, linear_weight, linear_scale_ptr,
                            linear_shared_scale, hidden_dim, model_dim));

  auto active_ids = maybe_reshape(
      top_indices, ::ml_drift::BHWC(1, 1, sequence_size, num_active_experts));
  auto active_scales = maybe_reshape(
      top_weights, ::ml_drift::BHWC(1, 1, sequence_size, num_active_experts));
  auto expert_scales =
      maybe_reshape(per_expert_scale, ::ml_drift::BHWC(1, 1, 1, num_experts));

  ::ml_drift::GpuModelBuilder::TensorHandle combined;
  if (use_packed_groups && src_shape.b == 1) {
    // Convert packed outputs to kDHWBCC4 (links into `linear`'s epilogue) so
    // the fused gather+scale reduction reads contiguous slices per token.
    expert_outputs = ConvertToDhwbcc4(model_builder, expert_outputs);

    ::ml_drift::TensorDescriptor inv_remap_desc(
        ::ml_drift::DataType::kInt32, ::ml_drift::TensorStorageType::kBuffer,
        ::ml_drift::Layout::kHWC);
    inv_remap_desc.SetBHWCShape(::ml_drift::BHWC(1, 1, 1, num_dispatches));
    auto inv_remap = model_builder->AddTensor(inv_remap_desc);
    model_builder->AddGpuOperation(
        {experts_packed_remap}, {inv_remap},
        CreateBuildInverseExpertsRemapOp(experts_packed_remap.tensor_desc,
                                         num_dispatches, num_active_experts),
        "build_inverse_experts_remap");

    const ::ml_drift::BHWC dst_shape(1, 1, sequence_size, model_dim);
    combined = model_builder->AddTensor(
        dst_shape, expert_outputs.tensor_desc.GetDataType());
    model_builder->AddGpuOperation(
        {expert_outputs, inv_remap, active_ids, active_scales, expert_scales},
        {combined},
        CreateScaleWithBatchIdsFromPacked(
            expert_outputs.tensor_desc, active_ids.tensor_desc,
            active_scales.tensor_desc, expert_scales.tensor_desc,
            combined.tensor_desc, num_active_experts),
        "moe_scale_with_batch_ids_from_packed");
  } else {
    if (use_packed_groups) {
      const ::ml_drift::BHWC shape = expert_outputs.tensor_desc.GetBHWCShape();
      auto remapped_dst = model_builder->AddTensor(
          ::ml_drift::BHWC(shape.b, num_active_experts,
                           shape.w / num_active_experts, shape.c),
          expert_outputs.tensor_desc.GetDataType());
      auto remap_from = ::ml_drift::CreateExpertsRemapFromOp(
          expert_outputs.tensor_desc, experts_packed_remap.tensor_desc,
          remapped_dst.tensor_desc);
      remap_from->tensor_to_grid_ = ::ml_drift::TensorToGrid::kCustom;
      model_builder->AddGpuOperation({expert_outputs, experts_packed_remap},
                                     {remapped_dst}, std::move(remap_from),
                                     "experts_remap_from");
      expert_outputs =
          model_builder->Transpose(remapped_dst, ::ml_drift::BHWC(0, 2, 1, 3));
    } else {
      expert_outputs = maybe_reshape(
          expert_outputs,
          ::ml_drift::BHWC(1, sequence_size, num_active_experts, model_dim));
    }

    ABSL_ASSIGN_OR_RETURN(
        combined, ScaleWithBatchIds(model_builder, expert_outputs, active_ids,
                                    active_scales, expert_scales));
  }
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
      attr.activation, outputs[0]->id, outputs[0]->tensor.shape);
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
  auto legacy_activation =
      attr.activation == ir::MoeExpertsAttributes::Activation::kSilu
          ? MoeExpertsAttributes::Activation::kSilu
          : MoeExpertsAttributes::Activation::kGeluTanh;

  return BuildMoeExpertsGpuGraph(
      model_builder, src, top_weights, top_indices, gate_weight, ff1_weight,
      linear_weight, per_expert_scale, gate_scale_ptr, ff1_scale_ptr,
      linear_scale_ptr, /*gate_shared_scale=*/nullptr,
      /*ff1_shared_scale=*/nullptr, /*linear_shared_scale=*/nullptr,
      attr.model_dim, attr.hidden_dim, attr.num_experts,
      attr.num_active_experts, legacy_weight_type, legacy_activation,
      outputs[0]->id, outputs[0]->desc.GetBHWCShape());
}

}  // namespace litert::ml_drift
