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
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/model_hints.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift/webgpu/execution_environment.h"  // from @ml_drift
#include "ml_drift/webgpu/command_buffer_capture.h"  // from @ml_drift
#include "ml_drift/webgpu/spatial_tensor.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/gpu_backend.h"
#include "ml_drift_delegate/delegate/shared_webgpu_env.h"

namespace litert::ml_drift {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;

// One read/write binding is required by WebGPU for an in-place update.
class ProfilingCounter : public ::ml_drift::GPUOperation {
 public:
  std::vector<::ml_drift::int3> GetPossibleKernelWorkGroups(
      ::ml_drift::TuningType, const ::ml_drift::GpuInfo&,
      const ::ml_drift::KernelInfo&) const override {
    // Tuning uses aliased temporary tensors; this counter needs one workgroup.
    return {::ml_drift::int3(1, 1, 1)};
  }
  ProfilingCounter(const ::ml_drift::TensorDescriptor& input,
                   const ::ml_drift::TensorDescriptor& state) {
    AddSrcTensor("input", input);
    AddDstTensor("state", state);
    args_.AddObjectRef("state", ::ml_drift::AccessType::kReadWrite,
                      std::make_unique<::ml_drift::TensorDescriptor>(state));
    tensor_to_grid_ = ::ml_drift::TensorToGrid::kWBToX_HDToY_SToZ;
    work_group_size_ = ::ml_drift::int3(1, 1, 1);
    code_ = R"(MAIN_FUNCTION($0) {
      if (ucl::GetGlobalId<0>() != 0 || ucl::GetGlobalId<1>() != 0 || ucl::GetGlobalId<2>() != 0) return;
      float4 value = args.state.Read(0, 0, 0);
      args.state.Write(value + ucl::Init<float4>(1.0f), 0, 0, 0);
    })";
  }
};

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

  std::pair<::ml_drift::ValueId, ::ml_drift::ValueId> BuildCounterModel(
      ::ml_drift::CreateGpuModelInfo& info, ::ml_drift::GpuModel& model) {
    info.precision = ::ml_drift::CalculationsPrecision::kF32;
    info.storage_type = ::ml_drift::TensorStorageType::kBuffer;
    ::ml_drift::GpuModelBuilder builder(backend_.wgpu_env().GetInfo(),
        info.hints, info.precision, info.storage_type);
    auto in = builder.AddTensor(::ml_drift::BHWC(1, 1, 1, 4),
                                ::ml_drift::DataType::kFloat32);
    auto out = builder.AddTensor(::ml_drift::BHWC(1, 1, 1, 4),
                                 ::ml_drift::DataType::kFloat32);
    builder.AddGpuOperation({in}, out,
        std::make_unique<ProfilingCounter>(in.tensor_desc, out.tensor_desc),
        "capture_counter");
    ABSL_CHECK_OK(builder.GetGpuModel(std::vector<uint32_t>{in.id},
                                      std::vector<uint32_t>{out.id}, &model));
    info.external_mutable_tensors[in.id] = model.tensors.at(in.id);
    info.external_mutable_tensors[out.id] = model.tensors.at(out.id);
    return {in.id, out.id};
  }
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

  // Initialize input tensors with distinct values so each ping-pong slot
  // produces a different output (8 steps of x * 2 + 1 => x * 256 + 255):
  //   in_tensor_a = 0.0 -> 255.0 (step 0, uncached), updated to 2.0 -> 767.0
  //                        before cached command buffers are used in step >= 2
  //   in_tensor_b = 1.0 -> 511.0
  ::ml_drift::TensorFloat32 in_cpu_a;
  in_cpu_a.shape = ::ml_drift::BHWC(1, 1, 1, 4);
  in_cpu_a.data = {0.0f, 0.0f, 0.0f, 0.0f};
  ::ml_drift::TensorDescriptor in_desc_a = in_desc;
  in_desc_a.UploadData(in_cpu_a);

  ::ml_drift::TensorFloat32 in_cpu_b;
  in_cpu_b.shape = ::ml_drift::BHWC(1, 1, 1, 4);
  in_cpu_b.data = {1.0f, 1.0f, 1.0f, 1.0f};
  ::ml_drift::TensorDescriptor in_desc_b = in_desc;
  in_desc_b.UploadData(in_cpu_b);

  ::ml_drift::webgpu::SpatialTensor in_tensor_a;
  ::ml_drift::webgpu::SpatialTensor out_tensor_a;
  ::ml_drift::webgpu::SpatialTensor in_tensor_b;
  ::ml_drift::webgpu::SpatialTensor out_tensor_b;
  ASSERT_THAT(in_tensor_a.CreateFromDescriptor(env.device(), in_desc_a),
              IsOk());
  ASSERT_THAT(CreateTensor(env.device(), out_desc, &out_tensor_a), IsOk());
  ASSERT_THAT(in_tensor_b.CreateFromDescriptor(env.device(), in_desc_b),
              IsOk());
  ASSERT_THAT(CreateTensor(env.device(), out_desc, &out_tensor_b), IsOk());

  auto ctx_result = backend_.CreateInferenceContext(
      create_info, gpu_model, nullptr, true);
  ASSERT_THAT(ctx_result, IsOk());
  auto ctx = std::move(*ctx_result);
  ASSERT_THAT(ctx->SetCommandBufferHint(4), IsOk());

  // Repeatedly bind external tensors, query them, and dispatch. Because
  // num_steps_of_command_buffer_preparations > 1, Dispatch() spawns a
  // background thread that calls CreateCommandBuffers() while the next
  // iteration's BindSpatialTensor() and GetSpatialTensor() run on the calling
  // thread.
  for (int step = 0; step < 20; ++step) {
    if (step == 2) {
      // Update in_tensor_a before cached command buffer slot 0 is replayed so
      // we can distinguish slot 0 re-executing on in_tensor_a -> out_tensor_a
      // (output 767.0) from a stale output of step 0 (255.0) if slot 0 had
      // erroneously recorded step 1's bindings (in_tensor_b -> out_tensor_b).
      ASSERT_THAT(backend_.WaitForCompletion(), IsOk());
      in_cpu_a.data = {2.0f, 2.0f, 2.0f, 2.0f};
      in_desc_a.UploadData(in_cpu_a);
      ASSERT_THAT(in_tensor_a.UploadDescriptorData(env.queue(), in_desc_a),
                  IsOk());
    }
    auto* in_ptr = (step % 2 == 0) ? &in_tensor_a : &in_tensor_b;
    auto* out_ptr = (step % 2 == 0) ? &out_tensor_a : &out_tensor_b;

    ASSERT_THAT(ctx->BindSpatialTensor(in_id, in_ptr), IsOk());
    ASSERT_THAT(ctx->BindSpatialTensor(out_id, out_ptr), IsOk());
    EXPECT_THAT(ctx->GetSpatialTensor(in_id), IsOkAndHolds(in_ptr));
    EXPECT_THAT(ctx->GetSpatialTensor(out_id), IsOkAndHolds(out_ptr));
    ASSERT_THAT(ctx->Dispatch(), IsOk());

    if (step == 0) {
      EXPECT_THAT(ctx->SetCommandBufferHint(8),
                  StatusIs(absl::StatusCode::kFailedPrecondition));
    }
    if (step == 2) {
      // Verify immediately after step 2 (before any subsequent odd step) that
      // cached slot 0 recorded step 0's tensor_a bindings rather than step 1's
      // tensor_b bindings.
      ASSERT_THAT(backend_.WaitForCompletion(), IsOk());
      ::ml_drift::TensorDescriptor step2_out_desc_a = out_desc;
      ASSERT_THAT(out_tensor_a.ToDescriptor(env.device(), &step2_out_desc_a),
                  IsOk());
      ::ml_drift::TensorFloat32 step2_out_cpu_a;
      step2_out_desc_a.DownloadData(&step2_out_cpu_a);
      EXPECT_THAT(step2_out_cpu_a.data,
                  ElementsAre(767.0f, 767.0f, 767.0f, 767.0f));
    }
  }
  ASSERT_THAT(backend_.WaitForCompletion(), IsOk());

  ::ml_drift::TensorDescriptor out_desc_a = out_desc;
  ASSERT_THAT(out_tensor_a.ToDescriptor(env.device(), &out_desc_a), IsOk());
  ::ml_drift::TensorFloat32 out_cpu_a;
  out_desc_a.DownloadData(&out_cpu_a);
  EXPECT_THAT(out_cpu_a.data, ElementsAre(767.0f, 767.0f, 767.0f, 767.0f));

  ::ml_drift::TensorDescriptor out_desc_b = out_desc;
  ASSERT_THAT(out_tensor_b.ToDescriptor(env.device(), &out_desc_b), IsOk());
  ::ml_drift::TensorFloat32 out_cpu_b;
  out_desc_b.DownloadData(&out_cpu_b);
  EXPECT_THAT(out_cpu_b.data, ElementsAre(511.0f, 511.0f, 511.0f, 511.0f));
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

  auto ctx_1_result = backend_.CreateInferenceContext(
      create_info_1, gpu_model_1, nullptr, true);
  ASSERT_THAT(ctx_1_result, IsOk());
  auto ctx_1 = std::move(*ctx_1_result);
  auto ctx_2_result = backend_.CreateInferenceContext(
      create_info_2, gpu_model_2, nullptr, true);
  ASSERT_THAT(ctx_2_result, IsOk());
  auto ctx_2 = std::move(*ctx_2_result);

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

  absl::StatusOr<std::unique_ptr<GpuInferenceContext>> main_ctx_result =
      backend_.CreateInferenceContext(main_create_info, main_gpu_model,
                                      /*serialized_model=*/nullptr,
                                      /*may_share_memory_manager=*/true);
  ASSERT_THAT(main_ctx_result, IsOk());
  std::unique_ptr<GpuInferenceContext> main_ctx = std::move(*main_ctx_result);

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

    absl::StatusOr<std::unique_ptr<GpuInferenceContext>> aux_ctx_result =
        aux_backend.CreateInferenceContext(aux_create_info, aux_gpu_model,
                                           /*serialized_model=*/nullptr,
                                           /*may_share_memory_manager=*/true);
    ASSERT_THAT(aux_ctx_result, IsOk());
    std::unique_ptr<GpuInferenceContext> aux_ctx = std::move(*aux_ctx_result);

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

TEST_F(GpuBackendWebGpuTest, ProfiledDispatchExecutesStateUpdateOnce) {
  backend_.set_num_steps_of_command_buffer_preparations(2);
  const auto& env = backend_.wgpu_env();
  if (!env.GetInfo().webgpu_info.supports_timestamp_query) GTEST_SKIP();
  ::ml_drift::CreateGpuModelInfo info;
  info.precision = ::ml_drift::CalculationsPrecision::kF32;
  info.storage_type = ::ml_drift::TensorStorageType::kBuffer;
  ::ml_drift::GpuModelBuilder builder(env.GetInfo(), info.hints, info.precision,
                                    info.storage_type);
  auto in = builder.AddTensor(::ml_drift::BHWC(1, 1, 1, 4),
                              ::ml_drift::DataType::kFloat32);
  auto out = builder.AddTensor(::ml_drift::BHWC(1, 1, 1, 4),
                               ::ml_drift::DataType::kFloat32);
  builder.AddGpuOperation({in}, out,
      std::make_unique<ProfilingCounter>(in.tensor_desc, out.tensor_desc),
      "profiling_counter");
  ::ml_drift::GpuModel model;
  ASSERT_THAT(builder.GetGpuModel(std::vector<uint32_t>{in.id},
                                  std::vector<uint32_t>{out.id}, &model), IsOk());
  info.external_mutable_tensors[in.id] = model.tensors.at(in.id);
  info.external_mutable_tensors[out.id] = model.tensors.at(out.id);
  ::ml_drift::webgpu::SpatialTensor state;
  ::ml_drift::webgpu::SpatialTensor input_tensor;
  ASSERT_THAT(CreateTensor(env.device(), model.tensors.at(in.id), &input_tensor), IsOk());
  ASSERT_THAT(CreateTensor(env.device(), model.tensors.at(out.id), &state), IsOk());
  auto ctx_result = backend_.CreateInferenceContext(info, model, nullptr, true);
  ASSERT_THAT(ctx_result, IsOk());
  auto ctx = std::move(*ctx_result);
  ASSERT_EQ(env.GetDeviceUnhandledErrorsTotalCount(), 0);
  ASSERT_THAT(ctx->BindSpatialTensor(in.id, &input_tensor), IsOk());
  ASSERT_THAT(ctx->BindSpatialTensor(out.id, &state), IsOk());
  float values[4] = {};
  ASSERT_THAT(state.WriteData(env.queue(), values), IsOk());
  // Exercise ordinary -> profiled -> ordinary transitions with background
  // command preparation enabled. An extra Dispatch after profiling is visible.
  ASSERT_THAT(ctx->Dispatch(), IsOk());
  ASSERT_THAT(backend_.WaitForCompletion(), IsOk());
  ::ml_drift::ProfilingInfo profile;
  ASSERT_THAT(ctx->DispatchWithProfiling(profile), IsOk());
  ASSERT_EQ(profile.dispatches.size(), 1);
  EXPECT_FALSE(profile.dispatches[0].label.empty());
  ::ml_drift::TensorDescriptor desc;
  ASSERT_THAT(state.ToDescriptor(env.device(), &desc), IsOk());
  ::ml_drift::TensorFloat32 actual;
  desc.DownloadData(&actual);
  for (float value : actual.data) EXPECT_EQ(value, 2);
  ASSERT_THAT(ctx->Dispatch(), IsOk());
  ASSERT_THAT(backend_.WaitForCompletion(), IsOk());
  ASSERT_THAT(state.ToDescriptor(env.device(), &desc), IsOk());
  desc.DownloadData(&actual);
  for (float value : actual.data) EXPECT_EQ(value, 3);
  ctx.reset();
  EXPECT_EQ(env.GetDeviceUnhandledErrorsTotalCount(), 0);
}

TEST_F(GpuBackendWebGpuTest, CoarseCapturePreservesStateAndFuturePreparation) {
  const auto& env = backend_.wgpu_env();
  if (!env.GetInfo().webgpu_info.supports_timestamp_query) GTEST_SKIP();
  for (int future_steps : {0, 2}) {
    backend_.set_num_steps_of_command_buffer_preparations(future_steps);
    auto pool_result = ::ml_drift::webgpu::CommandBufferCapturePool::Create(
        env.device(), 16, 4);
    ASSERT_THAT(pool_result, IsOk());
    auto capture = std::make_shared<WebGpuCommandCapture>();
    capture->pool = std::move(*pool_result);
    std::set<uint64_t> submitted, preparations;
    std::set<std::thread::id> preparation_threads;
    std::mutex mutex;
    int cached = 0;
    capture->event = [&](char phase, const char* name, uint64_t id, uint64_t) {
      if (phase == 'B' && std::string(name) == "prepare_command_buffers") {
        std::lock_guard<std::mutex> lock(mutex);
        EXPECT_TRUE(preparations.insert(id).second);
        preparation_threads.insert(std::this_thread::get_id());
      }
    };
    capture->submitted = [&](uint64_t id, uint64_t context, uint64_t prep,
                              bool from_cache, bool captured) {
      std::lock_guard<std::mutex> lock(mutex);
      EXPECT_TRUE(captured);
      EXPECT_NE(context, 0);
      EXPECT_TRUE(preparations.count(prep));
      EXPECT_TRUE(submitted.insert(id).second);
      if (from_cache) ++cached;
    };
    backend_.set_command_capture(capture);
    ::ml_drift::CreateGpuModelInfo info;
    ::ml_drift::GpuModel model;
    auto [in, out] = BuildCounterModel(info, model);
    ::ml_drift::webgpu::SpatialTensor input, state;
    ASSERT_THAT(CreateTensor(env.device(), model.tensors.at(in), &input), IsOk());
    ASSERT_THAT(CreateTensor(env.device(), model.tensors.at(out), &state), IsOk());
    auto result = backend_.CreateInferenceContext(info, model, nullptr, false);
    ASSERT_THAT(result, IsOk());
    auto ctx = std::move(*result);
    ASSERT_THAT(ctx->BindSpatialTensor(in, &input), IsOk());
    ASSERT_THAT(ctx->BindSpatialTensor(out, &state), IsOk());
    float zeros[4] = {};
    ASSERT_THAT(state.WriteData(env.queue(), zeros), IsOk());
    for (int i = 0; i < 8; ++i) ASSERT_THAT(ctx->Dispatch(), IsOk());
    // Cancels the last speculative preparation without executing it.
    ctx.reset();
    ASSERT_THAT(backend_.WaitForCompletion(), IsOk());
    ASSERT_THAT(capture->pool->Drain(absl::Seconds(10)), IsOk());
    auto results = capture->pool->CollectReady();
    ASSERT_EQ(results.size(), 8);
    for (const auto& row : results) {
      EXPECT_THAT(row.status, IsOk());
      EXPECT_EQ(submitted.erase(row.invocation_id), 1);
      ASSERT_EQ(row.passes.size(), 1);
      EXPECT_EQ(row.passes[0].first_node, 0);
      EXPECT_EQ(row.passes[0].node_count, 1);
      EXPECT_EQ(row.passes[0].enabled_nodes, 1);
      EXPECT_LE(row.passes[0].begin_ns, row.passes[0].end_ns);
    }
    EXPECT_TRUE(submitted.empty());
    EXPECT_EQ(capture->pool->dropped_captures(), 0);
    EXPECT_EQ(cached > 0, future_steps > 0);
    EXPECT_EQ(preparation_threads.size() > 1, future_steps > 0);
    ::ml_drift::TensorDescriptor desc;
    ASSERT_THAT(state.ToDescriptor(env.device(), &desc), IsOk());
    ::ml_drift::TensorFloat32 actual;
    desc.DownloadData(&actual);
    for (float value : actual.data) EXPECT_EQ(value, 8);
    backend_.set_command_capture(nullptr);
  }
  EXPECT_EQ(env.GetDeviceUnhandledErrorsTotalCount(), 0);
}

TEST_F(GpuBackendWebGpuTest, ExhaustedCapturePoolDoesNotDropInference) {
  const auto& env = backend_.wgpu_env();
  if (!env.GetInfo().webgpu_info.supports_timestamp_query) GTEST_SKIP();
  backend_.set_num_steps_of_command_buffer_preparations(2);
  auto pool = ::ml_drift::webgpu::CommandBufferCapturePool::Create(
      env.device(), 1, 4);
  ASSERT_THAT(pool, IsOk());
  auto capture = std::make_shared<WebGpuCommandCapture>();
  capture->pool = std::move(*pool);
  backend_.set_command_capture(capture);
  ::ml_drift::CreateGpuModelInfo info;
  ::ml_drift::GpuModel model;
  auto [in, out] = BuildCounterModel(info, model);
  ::ml_drift::webgpu::SpatialTensor input, state;
  ASSERT_THAT(CreateTensor(env.device(), model.tensors.at(in), &input), IsOk());
  ASSERT_THAT(CreateTensor(env.device(), model.tensors.at(out), &state), IsOk());
  auto result = backend_.CreateInferenceContext(info, model, nullptr, false);
  ASSERT_THAT(result, IsOk());
  auto ctx = std::move(*result);
  ASSERT_THAT(ctx->BindSpatialTensor(in, &input), IsOk());
  ASSERT_THAT(ctx->BindSpatialTensor(out, &state), IsOk());
  float zeros[4] = {};
  ASSERT_THAT(state.WriteData(env.queue(), zeros), IsOk());
  for (int i = 0; i < 8; ++i) ASSERT_THAT(ctx->Dispatch(), IsOk());
  // Switching to detailed profiling must cancel cached capture leases safely.
  ::ml_drift::ProfilingInfo detailed;
  ASSERT_THAT(ctx->DispatchWithProfiling(detailed), IsOk());
  ASSERT_THAT(ctx->Dispatch(), IsOk());
  ctx.reset();
  ASSERT_THAT(backend_.WaitForCompletion(), IsOk());
  ASSERT_THAT(capture->pool->Drain(absl::Seconds(10)), IsOk());
  auto results = capture->pool->CollectReady();
  ASSERT_EQ(results.size(), 1);
  EXPECT_THAT(results[0].status, IsOk());
  EXPECT_GT(capture->pool->dropped_captures(), 0);
  EXPECT_NE(capture->pool->TryAcquire(), nullptr);
  ::ml_drift::TensorDescriptor desc;
  ASSERT_THAT(state.ToDescriptor(env.device(), &desc), IsOk());
  ::ml_drift::TensorFloat32 actual;
  desc.DownloadData(&actual);
  for (float value : actual.data) EXPECT_EQ(value, 10);
  EXPECT_EQ(env.GetDeviceUnhandledErrorsTotalCount(), 0);
}

TEST_F(GpuBackendWebGpuTest, CoarseCaptureCancelsSpeculationAcrossContexts) {
  const auto& env = backend_.wgpu_env();
  if (!env.GetInfo().webgpu_info.supports_timestamp_query) GTEST_SKIP();
  backend_.set_num_steps_of_command_buffer_preparations(2);
  auto pool = ::ml_drift::webgpu::CommandBufferCapturePool::Create(
      env.device(), 16, 4);
  ASSERT_THAT(pool, IsOk());
  auto capture = std::make_shared<WebGpuCommandCapture>();
  capture->pool = std::move(*pool);
  std::set<uint64_t> contexts, invocations, preparations;
  capture->submitted = [&](uint64_t id, uint64_t context, uint64_t prep,
                            bool, bool captured) {
    EXPECT_TRUE(captured);
    contexts.insert(context);
    EXPECT_TRUE(invocations.insert(id).second);
    EXPECT_TRUE(preparations.insert(prep).second);
  };
  backend_.set_command_capture(capture);
  ::ml_drift::CreateGpuModelInfo info;
  ::ml_drift::GpuModel model;
  auto [in, out] = BuildCounterModel(info, model);
  ::ml_drift::webgpu::SpatialTensor input, state_a, state_b;
  ASSERT_THAT(CreateTensor(env.device(), model.tensors.at(in), &input), IsOk());
  ASSERT_THAT(CreateTensor(env.device(), model.tensors.at(out), &state_a), IsOk());
  ASSERT_THAT(CreateTensor(env.device(), model.tensors.at(out), &state_b), IsOk());
  auto result_a = backend_.CreateInferenceContext(info, model, nullptr, false);
  ASSERT_THAT(result_a, IsOk());
  auto a = std::move(*result_a);
  // InitFromGpuModel consumes operations. Build an independent second graph.
  ::ml_drift::CreateGpuModelInfo info_b;
  ::ml_drift::GpuModel model_b;
  auto [in_b, out_b] = BuildCounterModel(info_b, model_b);
  auto result_b = backend_.CreateInferenceContext(info_b, model_b, nullptr, false);
  ASSERT_THAT(result_b, IsOk());
  auto b = std::move(*result_b);
  ASSERT_THAT(a->BindSpatialTensor(in, &input), IsOk());
  ASSERT_THAT(a->BindSpatialTensor(out, &state_a), IsOk());
  ASSERT_THAT(b->BindSpatialTensor(in_b, &input), IsOk());
  ASSERT_THAT(b->BindSpatialTensor(out_b, &state_b), IsOk());
  float zeros[4] = {};
  ASSERT_THAT(state_a.WriteData(env.queue(), zeros), IsOk());
  ASSERT_THAT(state_b.WriteData(env.queue(), zeros), IsOk());
  for (auto* ctx : {a.get(), b.get(), a.get()}) {
    for (int i = 0; i < 4; ++i) ASSERT_THAT(ctx->Dispatch(), IsOk());
  }
  a.reset();
  b.reset();
  ASSERT_THAT(backend_.WaitForCompletion(), IsOk());
  ASSERT_THAT(capture->pool->Drain(absl::Seconds(10)), IsOk());
  auto results = capture->pool->CollectReady();
  EXPECT_EQ(results.size(), 12);
  EXPECT_EQ(contexts.size(), 2);
  for (const auto& row : results) {
    EXPECT_THAT(row.status, IsOk());
    EXPECT_EQ(invocations.erase(row.invocation_id), 1);
  }
  EXPECT_TRUE(invocations.empty());
  EXPECT_EQ(capture->pool->dropped_captures(), 0);
  for (const auto& [state, expected] :
       {std::pair{&state_a, 8.0f}, std::pair{&state_b, 4.0f}}) {
    ::ml_drift::TensorDescriptor desc;
    ASSERT_THAT(state->ToDescriptor(env.device(), &desc), IsOk());
    ::ml_drift::TensorFloat32 actual;
    desc.DownloadData(&actual);
    for (float value : actual.data) EXPECT_EQ(value, expected);
  }
  // All pending and unused speculative leases were released.
  std::vector<std::unique_ptr<::ml_drift::webgpu::CommandBufferCapture>> leases;
  for (int i = 0; i < 16; ++i) {
    leases.push_back(capture->pool->TryAcquire());
    EXPECT_NE(leases.back(), nullptr);
  }
  EXPECT_EQ(env.GetDeviceUnhandledErrorsTotalCount(), 0);
}

}  // namespace
}  // namespace litert::ml_drift
