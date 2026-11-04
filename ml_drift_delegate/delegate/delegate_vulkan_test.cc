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
#include <cstring>
#include <memory>
#include <utility>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/syrtis/environment.h"  // from @ml_drift
#include "ml_drift/syrtis/vulkan_wrapper.h"  // from @ml_drift
#include "third_party/odml/infra/ml_drift_delegate/testdata/simple_add.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/runtime/external_litert_buffer_context.h"
#include "litert/runtime/tensor_identifier.h"
#include "litert/runtime/tfl_utils.h"
#include "ml_drift_delegate/delegate/delegate_options.h"
#include "ml_drift_delegate/delegate/delegate_types.h"
#include "ml_drift_delegate/delegate/precision.h"
#include "ml_drift_delegate/delegate/shared_vulkan_env.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"
#include "tflite/interpreter_builder.h"
#include "tflite/kernels/register.h"
#include "tflite/model_builder.h"

namespace litert::ml_drift {
namespace {

using ::absl_testing::IsOk;
using ::testing::FloatNear;
using ::testing::Pointwise;

void NoOpDestroyCallback(void*) {}

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

TEST(DelegateVulkanTest, InvalidOptionsReturnNull) {
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN/TSAN";
#endif
  LiteRtEnvironment environment = nullptr;
  ASSERT_EQ(LiteRtCreateEnvironment(0, nullptr, &environment), kLiteRtStatusOk);
  absl::Cleanup env_cleanup = [&]() {
    if (environment) LiteRtDestroyEnvironment(environment);
  };

  EXPECT_EQ(CreateMlDriftVulkanDelegate(nullptr, environment), nullptr);

  MlDriftDelegateOptionsPtr options_no_runtime_context =
      MlDriftVulkanDelegateDefaultOptionsPtr();
  options_no_runtime_context->runtime_context = nullptr;
  EXPECT_EQ(CreateMlDriftVulkanDelegate(std::move(options_no_runtime_context),
                                        environment),
            nullptr);
}

TEST(DelegateVulkanTest, ReusesExistingSharedVulkanEnvAndRunsInference) {
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN/TSAN";
#endif
  ASSERT_EQ(::ml_drift::syrtis::InitVulkan(), 1);
  ::ml_drift::syrtis::Environment vk_env;
  ASSERT_THAT(::ml_drift::syrtis::CreateEnvironment(&vk_env), IsOk());
  VkQueue vk_queue = VK_NULL_HANDLE;
  ASSERT_THAT(vk_env.GetQueue().WithQueue([&](VkQueue queue) {
    vk_queue = queue;
    return absl::OkStatus();
  }),
              IsOk());

  std::unique_ptr<SharedVulkanEnv> external_shared_env =
      SharedVulkanEnv::CreateFromExternalHandles(
          VK_NULL_HANDLE, vk_env.GetPhysicalDevice(), vk_env.GetDevice(),
          vk_queue, /*queue_family_index=*/-1);
  ASSERT_NE(external_shared_env, nullptr);

  const std::array<LiteRtEnvOption, 2> env_options = {
      LiteRtEnvOption{
          .tag = kLiteRtEnvOptionTagVulkanEnvironment,
          .value = {.type = kLiteRtAnyTypeInt,
                    .int_value =
                        reinterpret_cast<int64_t>(external_shared_env.get())},
      },
      LiteRtEnvOption{
          .tag = kLiteRtEnvOptionTagCallbackOnGpuEnvDestroy,
          .value = {.type = kLiteRtAnyTypeVoidPtr,
                    .ptr_value =
                        reinterpret_cast<const void*>(NoOpDestroyCallback)},
      },
  };
  LiteRtEnvironment env = nullptr;
  ASSERT_EQ(
      LiteRtCreateEnvironment(env_options.size(), env_options.data(), &env),
      kLiteRtStatusOk);
  absl::Cleanup env_cleanup = [&]() {
    if (env) LiteRtDestroyEnvironment(env);
  };

  const FileToc* file_ptr = simple_add_create();
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromBuffer(file_ptr->data, file_ptr->size);
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  ASSERT_EQ(
      tflite::InterpreterBuilder(
          *model, tflite::ops::builtin::BuiltinOpResolver())(&interpreter),
      kTfLiteOk);
  ASSERT_NE(interpreter, nullptr);

  auto get_tensor_id = [&interpreter](const TfLiteOpaqueTensor* target_tensor)
      -> litert::internal::TfLiteTensorIdentifier {
    litert::Expected<litert::internal::TfLiteTensorIdentifier> tensor_id =
        litert::internal::GetTensorIdentifier(
            *interpreter, reinterpret_cast<const TfLiteTensor*>(target_tensor));
    if (!tensor_id) {
      return {-1, -1};
    }
    return *tensor_id;
  };
  LiteRtExternalLiteRtBufferContextT buffer_context(env, get_tensor_id);
  interpreter->SetExternalContext(kTfLiteLiteRtBufferContext, &buffer_context);

  MlDriftDelegateOptionsPtr options = MlDriftVulkanDelegateDefaultOptionsPtr();
  options->runtime_context = LrtGetRuntimeContext();
  options->precision = MlDriftDelegatePrecision::kFp32;
  TfLiteDelegatePtr delegate =
      CreateMlDriftVulkanDelegate(std::move(options), env);
  ASSERT_NE(delegate, nullptr);
  EXPECT_EQ(GetEnvironmentSharedVulkanEnv(env), external_shared_env.get());

  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(std::move(delegate)),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  float* input0 = interpreter->typed_input_tensor<float>(0);
  float* input1 = interpreter->typed_input_tensor<float>(1);
  ASSERT_NE(input0, nullptr);
  ASSERT_NE(input1, nullptr);
  const std::array<float, 10> lhs = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const std::array<float, 10> rhs = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
  std::memcpy(input0, lhs.data(), sizeof(lhs));
  std::memcpy(input1, rhs.data(), sizeof(rhs));

  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);

  const float* output = interpreter->typed_output_tensor<float>(0);
  ASSERT_NE(output, nullptr);
  const std::array<float, 10> expected = {11, 22, 33, 44, 55,
                                          66, 77, 88, 99, 110};
  EXPECT_THAT(absl::MakeConstSpan(output, expected.size()),
              Pointwise(FloatNear(1e-5f), expected));
}

TEST(DelegateVulkanTest, DefaultSharedVulkanEnvironmentSingletonReuse) {
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN/TSAN";
#endif
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

  MlDriftDelegateOptionsPtr options1 = MlDriftVulkanDelegateDefaultOptionsPtr();
  options1->runtime_context = LrtGetRuntimeContext();
  TfLiteDelegatePtr delegate1 =
      CreateMlDriftVulkanDelegate(std::move(options1), env1);
  ASSERT_NE(delegate1, nullptr);

  MlDriftDelegateOptionsPtr options2 = MlDriftVulkanDelegateDefaultOptionsPtr();
  options2->runtime_context = LrtGetRuntimeContext();
  options2->precision = MlDriftDelegatePrecision::kFp16;
  options2->use_f32_accum_for_fp16 = true;
  TfLiteDelegatePtr delegate2 =
      CreateMlDriftVulkanDelegate(std::move(options2), env2);
  ASSERT_NE(delegate2, nullptr);

  SharedVulkanEnv* shared_env1 = GetEnvironmentSharedVulkanEnv(env1);
  SharedVulkanEnv* shared_env2 = GetEnvironmentSharedVulkanEnv(env2);
  ASSERT_NE(shared_env1, nullptr);
  EXPECT_EQ(shared_env1, shared_env2);
}

}  // namespace
}  // namespace litert::ml_drift
