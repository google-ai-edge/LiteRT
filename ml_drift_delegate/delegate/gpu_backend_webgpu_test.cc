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

#include "ml_drift_delegate/delegate/gpu_backend_webgpu.h"

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
#include "ml_drift/webgpu/execution_environment.h"  // from @ml_drift
#include "ml_drift/webgpu/spatial_tensor.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/gpu_backend.h"
#include "ml_drift_delegate/delegate/shared_webgpu_env.h"

namespace litert::ml_drift {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;

class GpuBackendWebGpuTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_THAT(backend_.wgpu_env().Initialize(), IsOk());
  }

  // Builds a multi-node graph with external mutable input and output tensors.
  std::pair<::ml_drift::ValueId, ::ml_drift::ValueId> BuildMultiNodeModel(
      ::ml_drift::CreateGpuModelInfo& create_info,
      ::ml_drift::GpuModel& gpu_model) {
    create_info.precision = ::ml_drift::CalculationsPrecision::kF32;
    create_info.storage_type = ::ml_drift::TensorStorageType::kBuffer;
    create_info.hints.Add(::ml_drift::ModelHints::kFastTuning);

    ::ml_drift::GpuModelBuilder builder(
        backend_.wgpu_env().GetInfo(), create_info.hints, create_info.precision,
        create_info.storage_type);
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

  GpuBackendWebGpu backend_;
};

TEST_F(GpuBackendWebGpuTest,
       BindAndGetSpatialTensorSerializedWithBackgroundCommandBufferPrep) {
  backend_.set_num_steps_of_command_buffer_preparations(2);
  const auto& env = backend_.wgpu_env();

  ::ml_drift::CreateGpuModelInfo create_info;
  ::ml_drift::GpuModel gpu_model;
  auto [in_id, out_id] = BuildMultiNodeModel(create_info, gpu_model);
  const ::ml_drift::TensorDescriptor in_desc = gpu_model.tensors.at(in_id);
  const ::ml_drift::TensorDescriptor out_desc = gpu_model.tensors.at(out_id);

  ::ml_drift::webgpu::SpatialTensor in_tensor_a;
  ::ml_drift::webgpu::SpatialTensor out_tensor_a;
  ::ml_drift::webgpu::SpatialTensor in_tensor_b;
  ::ml_drift::webgpu::SpatialTensor out_tensor_b;
  ASSERT_THAT(CreateTensor(env.device(), in_desc, &in_tensor_a), IsOk());
  ASSERT_THAT(CreateTensor(env.device(), out_desc, &out_tensor_a), IsOk());
  ASSERT_THAT(CreateTensor(env.device(), in_desc, &in_tensor_b), IsOk());
  ASSERT_THAT(CreateTensor(env.device(), out_desc, &out_tensor_b), IsOk());

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GpuInferenceContext> ctx,
      backend_.CreateInferenceContext(create_info, gpu_model,
                                      /*serialized_model=*/nullptr,
                                      /*may_share_memory_manager=*/true));

  // Repeatedly bind external tensors, query them, and dispatch. Because
  // num_steps_of_command_buffer_preparations > 1, Dispatch() spawns a
  // background thread that calls CreateCommandBuffers() while the next
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
  ASSERT_THAT(backend_.WaitForCompletion(), IsOk());
}

TEST_F(GpuBackendWebGpuTest,
       MultipleContextsSharingMemoryManagerSerializeAccess) {
  backend_.set_num_steps_of_command_buffer_preparations(2);
  const auto& env = backend_.wgpu_env();

  ::ml_drift::CreateGpuModelInfo create_info_1;
  ::ml_drift::GpuModel gpu_model_1;
  auto [in_id_1, out_id_1] = BuildMultiNodeModel(create_info_1, gpu_model_1);
  const ::ml_drift::TensorDescriptor in_desc_1 =
      gpu_model_1.tensors.at(in_id_1);
  const ::ml_drift::TensorDescriptor out_desc_1 =
      gpu_model_1.tensors.at(out_id_1);

  ::ml_drift::CreateGpuModelInfo create_info_2;
  ::ml_drift::GpuModel gpu_model_2;
  auto [in_id_2, out_id_2] = BuildMultiNodeModel(create_info_2, gpu_model_2);
  const ::ml_drift::TensorDescriptor in_desc_2 =
      gpu_model_2.tensors.at(in_id_2);
  const ::ml_drift::TensorDescriptor out_desc_2 =
      gpu_model_2.tensors.at(out_id_2);

  ::ml_drift::webgpu::SpatialTensor in_tensor_1;
  ::ml_drift::webgpu::SpatialTensor out_tensor_1;
  ::ml_drift::webgpu::SpatialTensor in_tensor_2;
  ::ml_drift::webgpu::SpatialTensor out_tensor_2;
  ASSERT_THAT(CreateTensor(env.device(), in_desc_1, &in_tensor_1), IsOk());
  ASSERT_THAT(CreateTensor(env.device(), out_desc_1, &out_tensor_1), IsOk());
  ASSERT_THAT(CreateTensor(env.device(), in_desc_2, &in_tensor_2), IsOk());
  ASSERT_THAT(CreateTensor(env.device(), out_desc_2, &out_tensor_2), IsOk());

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GpuInferenceContext> ctx_1,
      backend_.CreateInferenceContext(create_info_1, gpu_model_1,
                                      /*serialized_model=*/nullptr,
                                      /*may_share_memory_manager=*/true));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GpuInferenceContext> ctx_2,
      backend_.CreateInferenceContext(create_info_2, gpu_model_2,
                                      /*serialized_model=*/nullptr,
                                      /*may_share_memory_manager=*/true));

  // When ctx_1->Dispatch() returns, ctx_1's background thread is preparing
  // command buffers and reading from the shared MemoryManager while ctx_2 binds
  // its external tensors on the shared MemoryManager and dispatches.
  for (int step = 0; step < 20; ++step) {
    ASSERT_THAT(ctx_1->BindSpatialTensor(in_id_1, &in_tensor_1), IsOk());
    ASSERT_THAT(ctx_1->BindSpatialTensor(out_id_1, &out_tensor_1), IsOk());
    ASSERT_THAT(ctx_1->Dispatch(), IsOk());

    ASSERT_THAT(ctx_2->BindSpatialTensor(in_id_2, &in_tensor_2), IsOk());
    ASSERT_THAT(ctx_2->BindSpatialTensor(out_id_2, &out_tensor_2), IsOk());
    EXPECT_THAT(ctx_2->GetSpatialTensor(in_id_2), IsOkAndHolds(&in_tensor_2));
    ASSERT_THAT(ctx_2->Dispatch(), IsOk());
  }
  // Destroy ctx_2 while it is the environment's last context and verify ctx_1
  // can still bind and dispatch cleanly, i.e. the last context does not dangle.
  ctx_2.reset();
  EXPECT_EQ(SharedWebGpuEnv::Get().GetLastContext(&env), nullptr);
  ASSERT_THAT(ctx_1->BindSpatialTensor(in_id_1, &in_tensor_1), IsOk());
  ASSERT_THAT(ctx_1->BindSpatialTensor(out_id_1, &out_tensor_1), IsOk());
  ASSERT_THAT(ctx_1->Dispatch(), IsOk());
  ASSERT_THAT(backend_.WaitForCompletion(), IsOk());
}

TEST_F(GpuBackendWebGpuTest,
       MultipleBackendsSharingExecutionEnvironmentShareLastContext) {
  backend_.set_num_steps_of_command_buffer_preparations(2);
  const auto& env = backend_.wgpu_env();

  ::ml_drift::CreateGpuModelInfo main_create_info;
  ::ml_drift::GpuModel main_gpu_model;
  auto [main_in_id, main_out_id] =
      BuildMultiNodeModel(main_create_info, main_gpu_model);
  const ::ml_drift::TensorDescriptor main_in_desc =
      main_gpu_model.tensors.at(main_in_id);
  const ::ml_drift::TensorDescriptor main_out_desc =
      main_gpu_model.tensors.at(main_out_id);

  ::ml_drift::webgpu::SpatialTensor main_in_tensor;
  ::ml_drift::webgpu::SpatialTensor main_out_tensor;
  ASSERT_THAT(CreateTensor(env.device(), main_in_desc, &main_in_tensor),
              IsOk());
  ASSERT_THAT(CreateTensor(env.device(), main_out_desc, &main_out_tensor),
              IsOk());

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GpuInferenceContext> main_ctx,
      backend_.CreateInferenceContext(main_create_info, main_gpu_model,
                                      /*serialized_model=*/nullptr,
                                      /*may_share_memory_manager=*/true));

  // Simulate an auxiliary runner (such as LiteRtLogitMaskRunner) that creates
  // its own GpuBackendWebGpu on the same ExecutionEnvironment* with
  // num_steps_of_command_buffer_preparations == 0 and dispatches in between
  // main_ctx's dispatches, while main_ctx's background command-buffer
  // preparation thread is in flight. The last context is tracked per
  // environment, so a backend wrapping a different environment must not
  // observe it.
  GpuBackendWebGpu other_env_backend;
  SharedWebGpuEnv& shared_env = SharedWebGpuEnv::Get();
  for (int round = 0; round < 5; ++round) {
    ASSERT_THAT(main_ctx->BindSpatialTensor(main_in_id, &main_in_tensor),
                IsOk());
    ASSERT_THAT(main_ctx->BindSpatialTensor(main_out_id, &main_out_tensor),
                IsOk());
    ASSERT_THAT(main_ctx->Dispatch(), IsOk());
    EXPECT_EQ(shared_env.GetLastContext(&env), main_ctx.get());
    EXPECT_EQ(shared_env.GetLastContext(&other_env_backend.wgpu_env()),
              nullptr);

    GpuBackendWebGpu aux_backend(&backend_.wgpu_env());
    ::ml_drift::CreateGpuModelInfo aux_create_info;
    ::ml_drift::GpuModel aux_gpu_model;
    auto [aux_in_id, aux_out_id] =
        BuildMultiNodeModel(aux_create_info, aux_gpu_model);
    const ::ml_drift::TensorDescriptor aux_in_desc =
        aux_gpu_model.tensors.at(aux_in_id);
    const ::ml_drift::TensorDescriptor aux_out_desc =
        aux_gpu_model.tensors.at(aux_out_id);

    ::ml_drift::webgpu::SpatialTensor aux_in_tensor;
    ::ml_drift::webgpu::SpatialTensor aux_out_tensor;
    ASSERT_THAT(CreateTensor(env.device(), aux_in_desc, &aux_in_tensor),
                IsOk());
    ASSERT_THAT(CreateTensor(env.device(), aux_out_desc, &aux_out_tensor),
                IsOk());

    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<GpuInferenceContext> aux_ctx,
        aux_backend.CreateInferenceContext(aux_create_info, aux_gpu_model,
                                           /*serialized_model=*/nullptr,
                                           /*may_share_memory_manager=*/true));

    for (int step = 0; step < 4; ++step) {
      ASSERT_THAT(main_ctx->BindSpatialTensor(main_in_id, &main_in_tensor),
                  IsOk());
      ASSERT_THAT(main_ctx->BindSpatialTensor(main_out_id, &main_out_tensor),
                  IsOk());
      ASSERT_THAT(main_ctx->Dispatch(), IsOk());

      ASSERT_THAT(aux_ctx->BindSpatialTensor(aux_in_id, &aux_in_tensor),
                  IsOk());
      ASSERT_THAT(aux_ctx->BindSpatialTensor(aux_out_id, &aux_out_tensor),
                  IsOk());
      ASSERT_THAT(aux_ctx->Dispatch(), IsOk());
    }

    // Dispatch main_ctx once more so its background thread is active when
    // aux_backend releases tensors and is destroyed.
    ASSERT_THAT(main_ctx->BindSpatialTensor(main_in_id, &main_in_tensor),
                IsOk());
    ASSERT_THAT(main_ctx->BindSpatialTensor(main_out_id, &main_out_tensor),
                IsOk());
    ASSERT_THAT(main_ctx->Dispatch(), IsOk());
    ASSERT_THAT(aux_backend.ReleaseSpatialTensorMemory(&aux_in_tensor), IsOk());
    ASSERT_THAT(aux_backend.ReleaseSpatialTensorMemory(&aux_out_tensor),
                IsOk());
  }

  ASSERT_THAT(backend_.WaitForCompletion(), IsOk());
}

}  // namespace
}  // namespace litert::ml_drift
