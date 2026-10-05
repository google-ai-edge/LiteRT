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

#include "ml_drift_delegate/delegate/gpu_backend_vulkan.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/model_hints.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/syrtis/testing/vulkan_test.h"  // from @ml_drift
#include "ml_drift/syrtis/vulkan_spatial_tensor.h"  // from @ml_drift
#include "litert/c/internal/litert_runtime_context.h"
#include "ml_drift_delegate/delegate/gpu_backend.h"
#include "ml_drift_delegate/delegate/shared_vulkan_env.h"

namespace litert::ml_drift {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;

class GpuBackendVulkanTest : public ::ml_drift::syrtis::VulkanOperationTest {
 protected:
  void SetUp() override {
    ::ml_drift::syrtis::VulkanOperationTest::SetUp();
    shared_vulkan_env_.vulkan_env() = std::move(*exec_env_.GetEnv());
  }

  // Builds a multi-node graph with external mutable input and output tensors.
  std::pair<::ml_drift::ValueId, ::ml_drift::ValueId> BuildMultiNodeModel(
      ::ml_drift::CreateGpuModelInfo& create_info,
      ::ml_drift::GpuModel& gpu_model) {
    create_info.precision = ::ml_drift::CalculationsPrecision::kF32;
    create_info.storage_type = ::ml_drift::TensorStorageType::kBuffer;
    create_info.hints.Add(::ml_drift::ModelHints::kFastTuning);

    ::ml_drift::GpuModelBuilder builder(
        shared_vulkan_env_.vulkan_env().GetInfo(), create_info.hints,
        create_info.precision, create_info.storage_type);
    auto in = builder.AddTensor(::ml_drift::BHWC(1, 1, 1, 4),
                                ::ml_drift::DataType::kFloat32);
    auto cur = in;
    for (int i = 0; i < 8; ++i) {
      cur = builder.Multiplication(cur, 2.0f);
      cur = builder.Add(cur, 1.0f);
    }
    auto out = cur;

    ABSL_CHECK_OK(builder.GetGpuModel(std::vector<uint32_t>{in.id},
                                      std::vector<uint32_t>{out.id},
                                      &gpu_model));
    create_info.external_mutable_tensors[in.id] = gpu_model.tensors[in.id];
    create_info.external_mutable_tensors[out.id] = gpu_model.tensors[out.id];
    return {in.id, out.id};
  }

  SharedVulkanEnv shared_vulkan_env_;
  LiteRtRuntimeContext runtime_context_{};
};

TEST_F(GpuBackendVulkanTest,
       BindAndGetSpatialTensorSerializedWithBackgroundCommandBufferPrep) {
  GpuBackendVulkan backend(&shared_vulkan_env_, &runtime_context_);
  backend.set_num_steps_of_command_buffer_preparations(2);

  ::ml_drift::CreateGpuModelInfo create_info;
  ::ml_drift::GpuModel gpu_model;
  auto [in_id, out_id] = BuildMultiNodeModel(create_info, gpu_model);
  const ::ml_drift::TensorDescriptor in_desc = gpu_model.tensors.at(in_id);
  const ::ml_drift::TensorDescriptor out_desc = gpu_model.tensors.at(out_id);

  ::ml_drift::syrtis::VulkanSpatialTensor in_tensor_a;
  ::ml_drift::syrtis::VulkanSpatialTensor out_tensor_a;
  ::ml_drift::syrtis::VulkanSpatialTensor in_tensor_b;
  ::ml_drift::syrtis::VulkanSpatialTensor out_tensor_b;
  ASSERT_THAT(::ml_drift::syrtis::CreateTensor(
                  in_desc, &shared_vulkan_env_.vulkan_env(), &in_tensor_a),
              IsOk());
  ASSERT_THAT(::ml_drift::syrtis::CreateTensor(
                  out_desc, &shared_vulkan_env_.vulkan_env(), &out_tensor_a),
              IsOk());
  ASSERT_THAT(::ml_drift::syrtis::CreateTensor(
                  in_desc, &shared_vulkan_env_.vulkan_env(), &in_tensor_b),
              IsOk());
  ASSERT_THAT(::ml_drift::syrtis::CreateTensor(
                  out_desc, &shared_vulkan_env_.vulkan_env(), &out_tensor_b),
              IsOk());

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GpuInferenceContext> ctx,
      backend.CreateInferenceContext(create_info, gpu_model,
                                     /*serialized_model=*/nullptr,
                                     /*may_share_memory_manager=*/true));

  // Repeatedly bind external tensors, query them, and dispatch. Because
  // num_steps_of_command_buffer_preparations > 1, Dispatch() spawns a
  // background thread that calls AddToCommandBuffer() while the next
  // iteration's BindSpatialTensor() and GetSpatialTensor() run on the calling
  // thread. Note that this test only verifies that there is no crash, not
  // output correctness.
  for (int step = 0; step < 20; ++step) {
    auto* in_ptr = (step % 2 == 0) ? &in_tensor_a : &in_tensor_b;
    auto* out_ptr = (step % 2 == 0) ? &out_tensor_a : &out_tensor_b;

    ASSERT_THAT(ctx->BindSpatialTensor(in_id, in_ptr), IsOk());
    ASSERT_THAT(ctx->BindSpatialTensor(out_id, out_ptr), IsOk());
    EXPECT_THAT(ctx->GetSpatialTensor(in_id), IsOkAndHolds(in_ptr));
    EXPECT_THAT(ctx->GetSpatialTensor(out_id), IsOkAndHolds(out_ptr));
    ASSERT_THAT(ctx->Dispatch(), IsOk());
  }
  ASSERT_THAT(backend.WaitForCompletion(), IsOk());
}

}  // namespace
}  // namespace litert::ml_drift
