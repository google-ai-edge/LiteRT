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

#include "ml_drift_delegate/delegate/delegate_vulkan.h"

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/debugging/leak_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "ml_drift/syrtis/environment.h"  // from @ml_drift
#include "ml_drift/syrtis/vulkan_wrapper.h"  // from @ml_drift
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "ml_drift_delegate/delegate/delegate_options.h"
#include "ml_drift_delegate/delegate/precision.h"
#include "ml_drift_delegate/delegate/shared_vulkan_env.h"
#include "tflite/schema/schema_generated.h"
#include "tflite/version.h"

namespace litert::ml_drift {
namespace {

using ::absl_testing::IsOk;
using ::testing::FloatNear;
using ::testing::Pointwise;

std::vector<uint8_t> BuildAddModelBuffer() {
  flatbuffers::FlatBufferBuilder builder;

  auto op_code = tflite::CreateOperatorCode(
      builder, /*deprecated_builtin_code=*/tflite::BuiltinOperator_ADD,
      /*custom_code=*/0, /*version=*/1,
      /*builtin_code=*/tflite::BuiltinOperator_ADD);
  auto opcodes_vec = builder.CreateVector(
      std::vector<flatbuffers::Offset<tflite::OperatorCode>>{op_code});

  auto empty_buffer = tflite::CreateBuffer(builder);
  auto buffers_vec = builder.CreateVector(
      std::vector<flatbuffers::Offset<tflite::Buffer>>{empty_buffer});

  const std::array<int32_t, 4> shape = {1, 2, 2, 1};
  auto make_tensor = [&](const char* name) {
    return tflite::CreateTensor(
        builder, builder.CreateVector(shape.data(), shape.size()),
        tflite::TensorType_FLOAT32, /*buffer=*/0, builder.CreateString(name));
  };
  std::vector<flatbuffers::Offset<tflite::Tensor>> tensors = {
      make_tensor("lhs"),
      make_tensor("rhs"),
      make_tensor("out"),
  };

  auto add_opts =
      tflite::CreateAddOptions(builder, tflite::ActivationFunctionType_NONE)
          .Union();
  auto add_op =
      tflite::CreateOperator(builder, /*opcode_index=*/0,
                             builder.CreateVector(std::vector<int32_t>{0, 1}),
                             builder.CreateVector(std::vector<int32_t>{2}),
                             tflite::BuiltinOptions_AddOptions, add_opts);

  auto subgraph = tflite::CreateSubGraph(
      builder, builder.CreateVector(tensors),
      builder.CreateVector(std::vector<int32_t>{0, 1}),
      builder.CreateVector(std::vector<int32_t>{2}),
      builder.CreateVector(
          std::vector<flatbuffers::Offset<tflite::Operator>>{add_op}),
      builder.CreateString("main"));
  auto subgraphs_vec = builder.CreateVector(
      std::vector<flatbuffers::Offset<tflite::SubGraph>>{subgraph});

  auto model = tflite::CreateModel(
      builder, TFLITE_SCHEMA_VERSION, opcodes_vec, subgraphs_vec,
      builder.CreateString("Add test"), buffers_vec);
  builder.Finish(model, tflite::ModelIdentifier());
  return std::vector<uint8_t>(builder.GetBufferPointer(),
                              builder.GetBufferPointer() + builder.GetSize());
}

SharedVulkanEnv* GetEnvironmentSharedVulkanEnv(LiteRtEnvironment env) {
  LiteRtEnvironmentOptions env_options = nullptr;
  if (LiteRtGetEnvironmentOptions(env, &env_options) != kLiteRtStatusOk) {
    return nullptr;
  }
  LiteRtAny vulkan_env_any;
  if (LiteRtGetEnvironmentOptionsValue(env_options,
                                       kLiteRtEnvOptionTagVulkanEnvironment,
                                       &vulkan_env_any) != kLiteRtStatusOk) {
    return nullptr;
  }
  return reinterpret_cast<SharedVulkanEnv*>(vulkan_env_any.int_value);
}

TEST(DelegateVulkanTest,
     ExternalVulkanEnvironmentImplicitQueueFamilyIndexAndReuse) {
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN/TSAN";
#endif
  absl::LeakCheckDisabler disable_leak_check;

  ASSERT_EQ(::ml_drift::syrtis::InitVulkan(), 1);
  ::ml_drift::syrtis::Environment vk_env;
  ASSERT_THAT(::ml_drift::syrtis::CreateEnvironment(&vk_env), IsOk());
  VkQueue vk_queue = VK_NULL_HANDLE;
  ASSERT_THAT(vk_env.GetQueue().WithQueue([&](VkQueue q) {
    vk_queue = q;
    return absl::OkStatus();
  }),
              IsOk());

  const LiteRtAny phys_dev_any = {
      .type = kLiteRtAnyTypeInt,
      .int_value = reinterpret_cast<int64_t>(vk_env.GetPhysicalDevice()),
  };
  const LiteRtAny dev_any = {
      .type = kLiteRtAnyTypeInt,
      .int_value = reinterpret_cast<int64_t>(vk_env.GetDevice()),
  };
  const LiteRtAny queue_any = {
      .type = kLiteRtAnyTypeInt,
      .int_value = reinterpret_cast<int64_t>(vk_queue),
  };
  const std::array<LiteRtEnvOption, 3> external_options = {
      LiteRtEnvOption{.tag = kLiteRtEnvOptionTagVulkanPhysicalDevice,
                      .value = phys_dev_any},
      LiteRtEnvOption{.tag = kLiteRtEnvOptionTagVulkanDevice, .value = dev_any},
      LiteRtEnvOption{.tag = kLiteRtEnvOptionTagVulkanQueue,
                      .value = queue_any},
  };

  LiteRtEnvironment environment = nullptr;
  ASSERT_EQ(LiteRtCreateEnvironment(external_options.size(),
                                    external_options.data(), &environment),
            kLiteRtStatusOk);
  absl::Cleanup env_cleanup = [&]() {
    if (environment) LiteRtDestroyEnvironment(environment);
  };

  auto options1 = MlDriftVulkanDelegateDefaultOptionsPtr();
  options1->runtime_context = LrtGetRuntimeContext();
  auto delegate1 =
      CreateMlDriftVulkanDelegate(std::move(options1), environment);
  ASSERT_NE(delegate1, nullptr);

  SharedVulkanEnv* shared_env1 = GetEnvironmentSharedVulkanEnv(environment);
  ASSERT_NE(shared_env1, nullptr);
  EXPECT_EQ(shared_env1->vulkan_env().GetPhysicalDevice(),
            vk_env.GetPhysicalDevice());
  EXPECT_EQ(shared_env1->vulkan_env().GetDevice(), vk_env.GetDevice());
  EXPECT_NE(shared_env1->command_pool(), VK_NULL_HANDLE);
  EXPECT_EQ(shared_env1->vulkan_env().GetQueue().QueueFamilyIndex(),
            vk_env.GetQueue().QueueFamilyIndex());

  VkQueue observed_queue = VK_NULL_HANDLE;
  EXPECT_THAT(shared_env1->vulkan_env().GetQueue().WithQueue([&](VkQueue q) {
    observed_queue = q;
    return absl::OkStatus();
  }),
              IsOk());
  EXPECT_EQ(observed_queue, vk_queue);

  // Creating a second delegate on the same LiteRtEnvironment reuses the
  // already registered SharedVulkanEnv.
  auto options2 = MlDriftVulkanDelegateDefaultOptionsPtr();
  options2->runtime_context = LrtGetRuntimeContext();
  options2->precision = MlDriftDelegatePrecision::kFp16;
  options2->use_f32_accum_for_fp16 = true;
  auto delegate2 =
      CreateMlDriftVulkanDelegate(std::move(options2), environment);
  ASSERT_NE(delegate2, nullptr);
  EXPECT_EQ(GetEnvironmentSharedVulkanEnv(environment), shared_env1);
}

TEST(DelegateVulkanTest,
     ExternalVulkanEnvironmentExplicitInstanceAndQueueFamilyIndex) {
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN/TSAN";
#endif
  absl::LeakCheckDisabler disable_leak_check;

  ASSERT_EQ(::ml_drift::syrtis::InitVulkan(), 1);
  ::ml_drift::syrtis::Environment vk_env;
  ASSERT_THAT(::ml_drift::syrtis::CreateEnvironment(&vk_env), IsOk());
  VkQueue vk_queue = VK_NULL_HANDLE;
  ASSERT_THAT(vk_env.GetQueue().WithQueue([&](VkQueue q) {
    vk_queue = q;
    return absl::OkStatus();
  }),
              IsOk());

  const VkApplicationInfo app_info = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "DelegateVulkanTest",
      .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
      .pEngineName = "",
      .engineVersion = VK_MAKE_VERSION(1, 0, 0),
      .apiVersion = VK_MAKE_VERSION(1, 1, 0),
  };
  const VkInstanceCreateInfo instance_create_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app_info,
  };
  VkInstance vk_instance = VK_NULL_HANDLE;
  ASSERT_EQ(::ml_drift::syrtis::vkCreateInstance(&instance_create_info, nullptr,
                                                 &vk_instance),
            VK_SUCCESS);
  absl::Cleanup instance_cleanup = [&]() {
    if (vk_instance != VK_NULL_HANDLE) {
      ::ml_drift::syrtis::vkDestroyInstance(vk_instance, nullptr);
    }
  };

  const uint32_t expected_queue_family_index =
      vk_env.GetQueue().QueueFamilyIndex();
  const std::array<Environment::Option, 5> external_options = {
      Environment::Option{Environment::OptionTag::VulkanInstance,
                          reinterpret_cast<int64_t>(vk_instance)},
      Environment::Option{
          Environment::OptionTag::VulkanPhysicalDevice,
          reinterpret_cast<int64_t>(vk_env.GetPhysicalDevice())},
      Environment::Option{Environment::OptionTag::VulkanDevice,
                          reinterpret_cast<int64_t>(vk_env.GetDevice())},
      Environment::Option{Environment::OptionTag::VulkanQueue,
                          reinterpret_cast<int64_t>(vk_queue)},
      Environment::Option{Environment::OptionTag::VulkanQueueFamilyIndex,
                          static_cast<int64_t>(expected_queue_family_index)},
  };
  auto env = litert::Environment::Create(external_options);
  ASSERT_TRUE(env);

  const std::vector<uint8_t> model_buf = BuildAddModelBuffer();
  auto litert_options = litert::Options::Create();
  ASSERT_TRUE(litert_options);
  ASSERT_TRUE(
      litert_options->SetHardwareAccelerators(litert::HwAccelerators::kGpu));
  auto gpu_options = litert_options->GetGpuOptions();
  ASSERT_TRUE(gpu_options);
  ASSERT_TRUE(gpu_options->SetPrecision(litert::GpuOptions::Precision::kFp32));
  ASSERT_TRUE(gpu_options->SetBufferStorageType(
      litert::GpuOptions::BufferStorageType::kBuffer));

  auto compiled_model = litert::CompiledModel::Create(
      *env, litert::BufferRef<uint8_t>(model_buf.data(), model_buf.size()),
      *litert_options);
  ASSERT_TRUE(compiled_model) << compiled_model.Error().Message();

  SharedVulkanEnv* shared_env = GetEnvironmentSharedVulkanEnv(env->Get());
  ASSERT_NE(shared_env, nullptr);
  EXPECT_EQ(shared_env->vulkan_env().GetQueue().QueueFamilyIndex(),
            expected_queue_family_index);

  auto input_buffers = compiled_model->CreateInputBuffers();
  ASSERT_TRUE(input_buffers);
  ASSERT_EQ(input_buffers->size(), 2);
  const std::array<float, 4> lhs = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::array<float, 4> rhs = {10.0f, 20.0f, 30.0f, 40.0f};
  ASSERT_TRUE((*input_buffers)[0].Write<float>(absl::MakeConstSpan(lhs)));
  ASSERT_TRUE((*input_buffers)[1].Write<float>(absl::MakeConstSpan(rhs)));

  auto output_buffers = compiled_model->CreateOutputBuffers();
  ASSERT_TRUE(output_buffers);
  ASSERT_EQ(output_buffers->size(), 1);

  ASSERT_TRUE(compiled_model->Run(*input_buffers, *output_buffers));

  std::array<float, 4> actual_out = {0.0f, 0.0f, 0.0f, 0.0f};
  ASSERT_TRUE((*output_buffers)[0].Read<float>(absl::MakeSpan(actual_out)));
  const std::array<float, 4> expected_out = {11.0f, 22.0f, 33.0f, 44.0f};
  EXPECT_THAT(actual_out, Pointwise(FloatNear(1e-5f), expected_out));
}

TEST(DelegateVulkanTest, InvalidOptionsAndNullExternalHandlesReturnNull) {
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN/TSAN";
#endif
  absl::LeakCheckDisabler disable_leak_check;

  const LiteRtAny zero_any = {.type = kLiteRtAnyTypeInt, .int_value = 0};
  const std::array<LiteRtEnvOption, 3> null_external_options = {
      LiteRtEnvOption{.tag = kLiteRtEnvOptionTagVulkanPhysicalDevice,
                      .value = zero_any},
      LiteRtEnvOption{.tag = kLiteRtEnvOptionTagVulkanDevice,
                      .value = zero_any},
      LiteRtEnvOption{.tag = kLiteRtEnvOptionTagVulkanQueue, .value = zero_any},
  };
  LiteRtEnvironment environment = nullptr;
  ASSERT_EQ(LiteRtCreateEnvironment(null_external_options.size(),
                                    null_external_options.data(), &environment),
            kLiteRtStatusOk);
  absl::Cleanup env_cleanup = [&]() {
    if (environment) LiteRtDestroyEnvironment(environment);
  };

  // Null options pointer should fail.
  EXPECT_EQ(CreateMlDriftVulkanDelegate(nullptr, environment), nullptr);

  // Missing runtime_context should fail.
  auto options_no_ctx = MlDriftVulkanDelegateDefaultOptionsPtr();
  options_no_ctx->runtime_context = nullptr;
  EXPECT_EQ(CreateMlDriftVulkanDelegate(std::move(options_no_ctx), environment),
            nullptr);

  // Null external Vulkan handles should fail.
  auto options_null_vk = MlDriftVulkanDelegateDefaultOptionsPtr();
  options_null_vk->runtime_context = LrtGetRuntimeContext();
  options_null_vk->litert_benchmark_mode = true;
  options_null_vk->litert_external_tensors_mode = true;
  options_null_vk->serialization_dir = "/tmp";
  options_null_vk->model_token = "test_token";
  EXPECT_EQ(
      CreateMlDriftVulkanDelegate(std::move(options_null_vk), environment),
      nullptr);
}

TEST(DelegateVulkanTest, DefaultSharedVulkanEnvironmentSingletonReuse) {
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN/TSAN";
#endif
  absl::LeakCheckDisabler disable_leak_check;

  LiteRtEnvironment env1 = nullptr;
  ASSERT_EQ(LiteRtCreateEnvironment(0, nullptr, &env1), kLiteRtStatusOk);
  absl::Cleanup env1_cleanup = [&]() {
    if (env1) LiteRtDestroyEnvironment(env1);
  };

  LiteRtEnvironment env2 = nullptr;
  ASSERT_EQ(LiteRtCreateEnvironment(0, nullptr, &env2), kLiteRtStatusOk);
  absl::Cleanup env2_cleanup = [&]() {
    if (env2) LiteRtDestroyEnvironment(env2);
  };

  auto options1 = MlDriftVulkanDelegateDefaultOptionsPtr();
  options1->runtime_context = LrtGetRuntimeContext();
  auto delegate1 = CreateMlDriftVulkanDelegate(std::move(options1), env1);
  ASSERT_NE(delegate1, nullptr);

  auto options2 = MlDriftVulkanDelegateDefaultOptionsPtr();
  options2->runtime_context = LrtGetRuntimeContext();
  options2->precision = MlDriftDelegatePrecision::kFp32;
  auto delegate2 = CreateMlDriftVulkanDelegate(std::move(options2), env2);
  ASSERT_NE(delegate2, nullptr);

  SharedVulkanEnv* shared_env1 = GetEnvironmentSharedVulkanEnv(env1);
  SharedVulkanEnv* shared_env2 = GetEnvironmentSharedVulkanEnv(env2);
  ASSERT_NE(shared_env1, nullptr);
  EXPECT_EQ(shared_env1, shared_env2);
}

}  // namespace
}  // namespace litert::ml_drift
