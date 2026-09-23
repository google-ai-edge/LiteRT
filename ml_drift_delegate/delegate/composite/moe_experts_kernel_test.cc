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
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/strings/match.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/ir/moe_experts_parser.h"
#include "ml_drift_delegate/delegate/composite/moe_experts_parser.h"

namespace litert::ml_drift {
namespace {

MoeScaleTensor MakeMoeScale(int out_channels, int num_experts, int num_blocks) {
  MoeScaleTensor scale;
  scale.shape = ::ml_drift::OHWI(out_channels, num_experts, 1, num_blocks);
  scale.data.assign(out_channels * num_experts * num_blocks, 0.25f);
  return scale;
}

void BuildAndVerifyMoeGpuModel(int sequence_size, int num_experts,
                               int num_active_experts, int model_dim,
                               int hidden_dim,
                               ir::MoeExpertsAttributes::WeightType weight_type,
                               int num_blocks, ::ml_drift::GpuModel* gpu_model,
                               const ::ml_drift::GpuInfo& gpu_info) {
  ::ml_drift::CreateGpuModelInfo create_info;
  create_info.precision = ::ml_drift::CalculationsPrecision::F32;
  ::ml_drift::GpuModelBuilder model_builder(gpu_info, {});

  auto src =
      model_builder.AddTensor(::ml_drift::BHWC(1, 1, sequence_size, model_dim),
                              ::ml_drift::DataType::FLOAT32);
  auto top_weights = model_builder.AddTensor(
      ::ml_drift::BHWC(1, 1, sequence_size, num_active_experts),
      ::ml_drift::DataType::FLOAT32);
  auto top_indices = model_builder.AddTensor(
      ::ml_drift::BHWC(1, 1, sequence_size, num_active_experts),
      ::ml_drift::DataType::INT32);

  ::ml_drift::DataType weight_data_type = ::ml_drift::DataType::FLOAT32;
  if (weight_type == ir::MoeExpertsAttributes::WeightType::kInt8) {
    weight_data_type = ::ml_drift::DataType::INT8;
  } else if (weight_type == ir::MoeExpertsAttributes::WeightType::kInt4) {
    weight_data_type = ::ml_drift::DataType::INT4;
  }

  auto add_weight_tensor = [&](int out_ch, int in_ch) {
    if (weight_type == ir::MoeExpertsAttributes::WeightType::kInt4) {
      ::ml_drift::TensorDescriptor desc(::ml_drift::DataType::INT4,
                                        ::ml_drift::TensorStorageType::BUFFER,
                                        ::ml_drift::Layout::LINEAR);
      desc.SetBHWCShape(::ml_drift::BHWC(out_ch, num_experts, 1, in_ch));
      return model_builder.AddTensor(desc);
    }
    return model_builder.AddTensor(
        ::ml_drift::BHWC(out_ch, num_experts, 1, in_ch), weight_data_type);
  };

  auto gate_weight = add_weight_tensor(hidden_dim, model_dim);
  auto ff1_weight = add_weight_tensor(hidden_dim, model_dim);
  auto linear_weight = add_weight_tensor(model_dim, hidden_dim);
  auto per_expert_scale = model_builder.AddTensor(
      ::ml_drift::BHWC(1, 1, 1, num_experts), ::ml_drift::DataType::FLOAT32);
  auto dst =
      model_builder.AddTensor(::ml_drift::BHWC(1, 1, sequence_size, model_dim),
                              ::ml_drift::DataType::FLOAT32);

  ir::MoeExpertsAttributes attr;
  attr.num_experts = num_experts;
  attr.num_active_experts = num_active_experts;
  attr.model_dim = model_dim;
  attr.hidden_dim = hidden_dim;
  attr.weight_type = weight_type;
  if (weight_type != ir::MoeExpertsAttributes::WeightType::kFp32) {
    attr.ff_gate_scale = MakeMoeScale(hidden_dim, num_experts, num_blocks);
    attr.ff1_scale = MakeMoeScale(hidden_dim, num_experts, num_blocks);
    attr.linear_scale = MakeMoeScale(model_dim, num_experts, num_blocks);
  }

  auto make_ir_tensor =
      [](const ::ml_drift::GpuModelBuilder::TensorHandle& handle) {
        ::ml_drift::ir::IrTensor tensor(handle.id);
        tensor.desc = handle.tensor_desc;
        return tensor;
      };
  ::ml_drift::ir::IrTensor t_src = make_ir_tensor(src);
  ::ml_drift::ir::IrTensor t_top_weights = make_ir_tensor(top_weights);
  ::ml_drift::ir::IrTensor t_top_indices = make_ir_tensor(top_indices);
  ::ml_drift::ir::IrTensor t_gate = make_ir_tensor(gate_weight);
  ::ml_drift::ir::IrTensor t_ff1 = make_ir_tensor(ff1_weight);
  ::ml_drift::ir::IrTensor t_linear = make_ir_tensor(linear_weight);
  ::ml_drift::ir::IrTensor t_expert_scale = make_ir_tensor(per_expert_scale);
  ::ml_drift::ir::IrTensor t_dst = make_ir_tensor(dst);

  ::ml_drift::ir::IrOp node(0);
  node.attr = std::move(attr);

  ASSERT_OK(
      CreateMoeExpertsFromIrOp(create_info,
                               {&t_src, &t_top_weights, &t_top_indices, &t_gate,
                                &t_ff1, &t_linear, &t_expert_scale},
                               {&t_dst}, node, &model_builder));

  ASSERT_OK(model_builder.GetGpuModel(
      {src.id, top_weights.id, top_indices.id, gate_weight.id, ff1_weight.id,
       linear_weight.id, per_expert_scale.id},
      {dst.id}, gpu_model));

  for (auto& gpu_node : gpu_model->nodes) {
    ASSERT_NE(gpu_node.gpu_operation, nullptr);
    EXPECT_OK(gpu_node.gpu_operation->AssembleCode(gpu_info))
        << "Failed to assemble shader code for node: " << gpu_node.name;
  }
}

bool HasNodeWithSubstring(const ::ml_drift::GpuModel& gpu_model,
                          const std::string& substring) {
  for (const auto& node : gpu_model.nodes) {
    if (absl::StrContains(node.name, substring)) {
      return true;
    }
  }
  return false;
}

::ml_drift::GpuInfo MakeTestGpuInfo() {
  ::ml_drift::GpuInfo gpu_info;
  gpu_info.gpu_api = ::ml_drift::GpuApi::kOpenCl;
  gpu_info.opencl_info.max_work_group_size_x = 256;
  gpu_info.opencl_info.max_work_group_size_y = 256;
  gpu_info.opencl_info.max_work_group_size_z = 64;
  gpu_info.opencl_info.max_work_group_total_size = 256;
  return gpu_info;
}

TEST(MoeExpertsKernelTest, DecodeSingleTokenSkipsGatherAndFusesScaleReduction) {
  const ::ml_drift::GpuInfo gpu_info = MakeTestGpuInfo();
  ::ml_drift::GpuModel gpu_model;

  // sequence_size = 1, num_active_experts = 2, num_experts = 8 (1 * 2 <= 8).
  BuildAndVerifyMoeGpuModel(
      /*sequence_size=*/1, /*num_experts=*/8, /*num_active_experts=*/2,
      /*model_dim=*/16, /*hidden_dim=*/32,
      ir::MoeExpertsAttributes::WeightType::kFp32, /*num_blocks=*/1, &gpu_model,
      gpu_info);

  EXPECT_FALSE(HasNodeWithSubstring(gpu_model, "gather"));
  EXPECT_FALSE(HasNodeWithSubstring(gpu_model, "batched_matmul"));
  EXPECT_TRUE(HasNodeWithSubstring(gpu_model, "moe_scale_with_batch_ids"));
}

TEST(MoeExpertsKernelTest,
     MultiTokenNonPackedUsesGatherAndFusedScaleReduction) {
  const ::ml_drift::GpuInfo gpu_info = MakeTestGpuInfo();
  ::ml_drift::GpuModel gpu_model;

  // sequence_size = 2, num_active_experts = 2, num_experts = 8 (2 * 2 <= 8).
  BuildAndVerifyMoeGpuModel(
      /*sequence_size=*/2, /*num_experts=*/8, /*num_active_experts=*/2,
      /*model_dim=*/16, /*hidden_dim=*/32,
      ir::MoeExpertsAttributes::WeightType::kFp32, /*num_blocks=*/1, &gpu_model,
      gpu_info);

  EXPECT_TRUE(HasNodeWithSubstring(gpu_model, "gather"));
  EXPECT_FALSE(HasNodeWithSubstring(gpu_model, "batched_matmul"));
  EXPECT_TRUE(HasNodeWithSubstring(gpu_model, "moe_scale_with_batch_ids"));
}

TEST(MoeExpertsKernelTest,
     PrefillPackedGroupsUsesExpertsRemapAndFusedScaleReduction) {
  const ::ml_drift::GpuInfo gpu_info = MakeTestGpuInfo();
  ::ml_drift::GpuModel gpu_model;

  // sequence_size = 8, num_active_experts = 2, num_experts = 4 (8 * 2 > 4).
  BuildAndVerifyMoeGpuModel(
      /*sequence_size=*/8, /*num_experts=*/4, /*num_active_experts=*/2,
      /*model_dim=*/16, /*hidden_dim=*/32,
      ir::MoeExpertsAttributes::WeightType::kFp32, /*num_blocks=*/1, &gpu_model,
      gpu_info);

  EXPECT_TRUE(HasNodeWithSubstring(gpu_model, "experts_remap"));
  EXPECT_FALSE(HasNodeWithSubstring(gpu_model, "gather"));
  EXPECT_FALSE(HasNodeWithSubstring(gpu_model, "batched_matmul"));
  EXPECT_TRUE(HasNodeWithSubstring(gpu_model, "moe_scale_with_batch_ids"));
}

TEST(MoeExpertsKernelTest,
     QuantizedPerChannelAndBlockwiseScalesAssembleForDecodeAndPrefill) {
  const ::ml_drift::GpuInfo gpu_info = MakeTestGpuInfo();

  // 1. Int8 per-channel (num_blocks = 1) decode.
  {
    ::ml_drift::GpuModel gpu_model;
    BuildAndVerifyMoeGpuModel(
        /*sequence_size=*/1, /*num_experts=*/8, /*num_active_experts=*/2,
        /*model_dim=*/16, /*hidden_dim=*/32,
        ir::MoeExpertsAttributes::WeightType::kInt8, /*num_blocks=*/1,
        &gpu_model, gpu_info);
    EXPECT_FALSE(HasNodeWithSubstring(gpu_model, "gather"));
    EXPECT_TRUE(HasNodeWithSubstring(gpu_model, "moe_scale_with_batch_ids"));
  }

  // 2. Int8 blockwise (num_blocks = 2) decode.
  {
    ::ml_drift::GpuModel gpu_model;
    BuildAndVerifyMoeGpuModel(
        /*sequence_size=*/1, /*num_experts=*/8, /*num_active_experts=*/2,
        /*model_dim=*/16, /*hidden_dim=*/32,
        ir::MoeExpertsAttributes::WeightType::kInt8, /*num_blocks=*/2,
        &gpu_model, gpu_info);
    EXPECT_FALSE(HasNodeWithSubstring(gpu_model, "gather"));
    EXPECT_TRUE(HasNodeWithSubstring(gpu_model, "moe_scale_with_batch_ids"));
  }

  // 3. Int4 blockwise (num_blocks = 2) prefill (packed groups).
  {
    ::ml_drift::GpuModel gpu_model;
    BuildAndVerifyMoeGpuModel(
        /*sequence_size=*/8, /*num_experts=*/4, /*num_active_experts=*/2,
        /*model_dim=*/16, /*hidden_dim=*/32,
        ir::MoeExpertsAttributes::WeightType::kInt4, /*num_blocks=*/2,
        &gpu_model, gpu_info);
    EXPECT_TRUE(HasNodeWithSubstring(gpu_model, "experts_remap"));
    EXPECT_TRUE(HasNodeWithSubstring(gpu_model, "moe_scale_with_batch_ids"));
  }
}

}  // namespace
}  // namespace litert::ml_drift
