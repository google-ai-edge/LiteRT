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

#include "ml_drift_delegate/delegate/buffer_handler_vulkan.h"

#include <array>
#include <cstdint>
#include <memory>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/syrtis/environment.h"  // from @ml_drift
#include "ml_drift/syrtis/vulkan_wrapper.h"  // from @ml_drift
#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_tensor_buffer.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "ml_drift_delegate/delegate/shared_vulkan_env.h"

namespace litert::ml_drift {
namespace {

using ::absl_testing::IsOk;
using ::testing::Each;
using ::testing::FloatEq;
using ::testing::FloatNear;
using ::testing::Pointwise;

LiteRtRankedTensorType Make1DFloatTensorType(int32_t dim0) {
  LiteRtRankedTensorType tensor_type = {};
  tensor_type.element_type = kLiteRtElementTypeFloat32;
  tensor_type.layout.rank = 1;
  tensor_type.layout.dimensions[0] = dim0;
  tensor_type.layout.has_strides = false;
  return tensor_type;
}

class BufferHandlerVulkanTest : public ::testing::Test {
 protected:
  void SetUp() override {
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
    GTEST_SKIP() << "GPU tests are not supported in MSAN/TSAN";
#endif
    ASSERT_EQ(::ml_drift::syrtis::InitVulkan(), 1);
    ASSERT_THAT(::ml_drift::syrtis::CreateEnvironment(&vk_env_), IsOk());
    VkQueue vk_queue = VK_NULL_HANDLE;
    ASSERT_THAT(vk_env_.GetQueue().WithQueue([&](VkQueue q) {
      vk_queue = q;
      return absl::OkStatus();
    }),
                IsOk());
    shared_env_ = SharedVulkanEnv::CreateFromExternalHandles(
        VK_NULL_HANDLE, vk_env_.GetPhysicalDevice(), vk_env_.GetDevice(),
        vk_queue, /*queue_family_index=*/-1);
    ASSERT_NE(shared_env_, nullptr);
  }

  ::ml_drift::syrtis::Environment vk_env_;
  std::unique_ptr<SharedVulkanEnv> shared_env_;
};

TEST_F(BufferHandlerVulkanTest, CreateAndDestroyVulkanMemory) {
  const LiteRtRankedTensorType tensor_type = Make1DFloatTensorType(4);
  const int bytes = 4 * sizeof(float);
  HwMemoryInfoPtr memory_info = nullptr;

  ASSERT_EQ(LiteRtCreateVulkanMemory(shared_env_.get(), /*queue_id=*/nullptr,
                                     &tensor_type,
                                     kLiteRtTensorBufferTypeVulkanBuffer, bytes,
                                     bytes, &memory_info),
            kLiteRtStatusOk);
  ASSERT_NE(memory_info, nullptr);
  EXPECT_NE(memory_info->memory_handle, nullptr);
  EXPECT_NE(memory_info->raw_handle, nullptr);

  EXPECT_EQ(LiteRtDestroyVulkanMemory(memory_info), kLiteRtStatusOk);
}

TEST_F(BufferHandlerVulkanTest, CreateAndDestroyRejectsInvalidArguments) {
  const LiteRtRankedTensorType tensor_type = Make1DFloatTensorType(4);
  const int bytes = 4 * sizeof(float);
  HwMemoryInfoPtr memory_info = nullptr;

  EXPECT_EQ(
      LiteRtCreateVulkanMemory(
          /*device_id=*/nullptr, /*queue_id=*/nullptr, &tensor_type,
          kLiteRtTensorBufferTypeVulkanBuffer, bytes, bytes, &memory_info),
      kLiteRtStatusErrorInvalidArgument);

  EXPECT_EQ(LiteRtCreateVulkanMemory(
                shared_env_.get(), /*queue_id=*/nullptr, &tensor_type,
                kLiteRtTensorBufferTypeVulkanBuffer, bytes,
                /*packed_bytes=*/bytes * 100, &memory_info),
            kLiteRtStatusErrorInvalidArgument);

  EXPECT_EQ(LiteRtDestroyVulkanMemory(nullptr),
            kLiteRtStatusErrorInvalidArgument);
}

TEST_F(BufferHandlerVulkanTest, LockAndUnlockVulkanBufferPackedRoundTrip) {
  const LiteRtRankedTensorType tensor_type = Make1DFloatTensorType(4);
  const int bytes = 4 * sizeof(float);
  HwMemoryInfoPtr memory_info = nullptr;

  ASSERT_EQ(LiteRtCreateVulkanMemory(shared_env_.get(), /*queue_id=*/nullptr,
                                     &tensor_type,
                                     kLiteRtTensorBufferTypeVulkanBufferPacked,
                                     bytes, bytes, &memory_info),
            kLiteRtStatusOk);
  absl::Cleanup cleanup = [&]() { LiteRtDestroyVulkanMemory(memory_info); };

  void* host_memory = nullptr;
  ASSERT_EQ(LiteRtLockVulkanMemory(
                memory_info, kLiteRtTensorBufferLockModeWrite, &host_memory),
            kLiteRtStatusOk);
  ASSERT_NE(host_memory, nullptr);

  // Locking again while locked must fail.
  void* duplicate_lock = nullptr;
  EXPECT_EQ(LiteRtLockVulkanMemory(memory_info, kLiteRtTensorBufferLockModeRead,
                                   &duplicate_lock),
            kLiteRtStatusErrorRuntimeFailure);

  const std::array<float, 4> expected = {1.0f, 2.0f, 3.0f, 4.0f};
  float* float_memory = static_cast<float*>(host_memory);
  for (int i = 0; i < expected.size(); ++i) {
    float_memory[i] = expected[i];
  }

  ASSERT_EQ(LiteRtUnlockVulkanMemory(memory_info), kLiteRtStatusOk);
  ASSERT_THAT(shared_env_->SubmitCommandBuffer(/*wait_for_completion=*/true),
              IsOk());
  // Unlocking again while unlocked must fail.
  EXPECT_EQ(LiteRtUnlockVulkanMemory(memory_info),
            kLiteRtStatusErrorRuntimeFailure);

  host_memory = nullptr;
  ASSERT_EQ(LiteRtLockVulkanMemory(memory_info, kLiteRtTensorBufferLockModeRead,
                                   &host_memory),
            kLiteRtStatusOk);
  ASSERT_NE(host_memory, nullptr);
  EXPECT_THAT(absl::MakeConstSpan(static_cast<const float*>(host_memory),
                                  expected.size()),
              Pointwise(FloatNear(1e-5f), expected));
  ASSERT_EQ(LiteRtUnlockVulkanMemory(memory_info), kLiteRtStatusOk);
  ASSERT_THAT(shared_env_->SubmitCommandBuffer(/*wait_for_completion=*/true),
              IsOk());
}

TEST_F(BufferHandlerVulkanTest, LockAndUnlockVulkanBufferRoundTrip) {
  const LiteRtRankedTensorType tensor_type = Make1DFloatTensorType(4);
  const int bytes = 4 * sizeof(float);
  HwMemoryInfoPtr memory_info = nullptr;

  ASSERT_EQ(LiteRtCreateVulkanMemory(shared_env_.get(), /*queue_id=*/nullptr,
                                     &tensor_type,
                                     kLiteRtTensorBufferTypeVulkanBuffer, bytes,
                                     bytes, &memory_info),
            kLiteRtStatusOk);
  absl::Cleanup cleanup = [&]() { LiteRtDestroyVulkanMemory(memory_info); };

  void* host_memory = nullptr;
  ASSERT_EQ(LiteRtLockVulkanMemory(
                memory_info, kLiteRtTensorBufferLockModeWrite, &host_memory),
            kLiteRtStatusOk);
  ASSERT_NE(host_memory, nullptr);

  const std::array<float, 4> expected = {10.0f, 20.0f, 30.0f, 40.0f};
  float* float_memory = static_cast<float*>(host_memory);
  for (int i = 0; i < expected.size(); ++i) {
    float_memory[i] = expected[i];
  }
  ASSERT_EQ(LiteRtUnlockVulkanMemory(memory_info), kLiteRtStatusOk);
  ASSERT_THAT(shared_env_->SubmitCommandBuffer(/*wait_for_completion=*/true),
              IsOk());

  host_memory = nullptr;
  ASSERT_EQ(LiteRtLockVulkanMemory(memory_info, kLiteRtTensorBufferLockModeRead,
                                   &host_memory),
            kLiteRtStatusOk);
  ASSERT_NE(host_memory, nullptr);
  EXPECT_THAT(absl::MakeConstSpan(static_cast<const float*>(host_memory),
                                  expected.size()),
              Pointwise(FloatNear(1e-5f), expected));
  ASSERT_EQ(LiteRtUnlockVulkanMemory(memory_info), kLiteRtStatusOk);
  ASSERT_THAT(shared_env_->SubmitCommandBuffer(/*wait_for_completion=*/true),
              IsOk());
}

TEST_F(BufferHandlerVulkanTest, ClearVulkanMemoryZerosBufferContents) {
  const LiteRtRankedTensorType tensor_type = Make1DFloatTensorType(4);
  const int bytes = 4 * sizeof(float);
  HwMemoryInfoPtr memory_info = nullptr;

  ASSERT_EQ(LiteRtCreateVulkanMemory(shared_env_.get(), /*queue_id=*/nullptr,
                                     &tensor_type,
                                     kLiteRtTensorBufferTypeVulkanBufferPacked,
                                     bytes, bytes, &memory_info),
            kLiteRtStatusOk);
  absl::Cleanup cleanup = [&]() { LiteRtDestroyVulkanMemory(memory_info); };

  void* host_memory = nullptr;
  ASSERT_EQ(LiteRtLockVulkanMemory(
                memory_info, kLiteRtTensorBufferLockModeWrite, &host_memory),
            kLiteRtStatusOk);
  float* float_memory = static_cast<float*>(host_memory);
  for (int i = 0; i < 4; ++i) {
    float_memory[i] = static_cast<float>(i + 1);
  }
  ASSERT_EQ(LiteRtUnlockVulkanMemory(memory_info), kLiteRtStatusOk);
  ASSERT_THAT(shared_env_->SubmitCommandBuffer(/*wait_for_completion=*/true),
              IsOk());

  ASSERT_EQ(LiteRtClearVulkanMemory(memory_info), kLiteRtStatusOk);
  ASSERT_THAT(shared_env_->SubmitCommandBuffer(/*wait_for_completion=*/true),
              IsOk());

  host_memory = nullptr;
  ASSERT_EQ(LiteRtLockVulkanMemory(memory_info, kLiteRtTensorBufferLockModeRead,
                                   &host_memory),
            kLiteRtStatusOk);
  EXPECT_THAT(absl::MakeConstSpan(static_cast<const float*>(host_memory), 4),
              Each(FloatEq(0.0f)));
  ASSERT_EQ(LiteRtUnlockVulkanMemory(memory_info), kLiteRtStatusOk);
}

}  // namespace
}  // namespace litert::ml_drift
