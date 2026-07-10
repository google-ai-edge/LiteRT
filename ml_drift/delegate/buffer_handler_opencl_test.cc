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

#include "third_party/odml/litert/ml_drift/delegate/buffer_handler_opencl.h"

#include <array>
#include <cstddef>
#include <cstdint>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/cl/buffer.h"  // from @ml_drift
#include "ml_drift/cl/cl_command_queue.h"  // from @ml_drift
#include "ml_drift/cl/cl_context.h"  // from @ml_drift
#include "ml_drift/cl/environment.h"  // from @ml_drift
#include "ml_drift/cl/opencl_wrapper.h"  // from @ml_drift
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_tensor_buffer.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_any.h"
#include "litert/test/matchers.h"

namespace litert {
namespace {

class BufferHandlerOpenClTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // MSAN does not support GPU tests.
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
    GTEST_SKIP() << "GPU tests are not supported in MSAN";
#endif

    if (!ml_drift::cl::LoadOpenCL().ok()) {
      GTEST_SKIP() << "OpenCL not loaded for ml_drift";
    }

    ASSERT_OK(ml_drift::cl::CreateEnvironment(&env_));

    LITERT_ASSERT_OK_AND_ASSIGN(
        LiteRtAny context_id,
        litert::ToLiteRtAny(litert::LiteRtVariant(
            reinterpret_cast<int64_t>(env_.context().context()))));

    LITERT_ASSERT_OK_AND_ASSIGN(
        LiteRtAny queue_id,
        litert::ToLiteRtAny(litert::LiteRtVariant(
            reinterpret_cast<int64_t>(env_.queue()->queue()))));

    const std::array<LiteRtEnvOption, 2> environment_options = {
        LiteRtEnvOption{
            /*.tag=*/kLiteRtEnvOptionTagOpenClContext,
            /*.value=*/context_id,
        },
        LiteRtEnvOption{
            /*.tag=*/kLiteRtEnvOptionTagOpenClCommandQueue,
            /*.value=*/queue_id,
        },
    };
    ASSERT_EQ(
        LiteRtCreateEnvironment(environment_options.size(),
                                environment_options.data(), &environment_),
        kLiteRtStatusOk);
  }

  void TearDown() override {
    if (environment_) {
      LiteRtDestroyEnvironment(environment_);
    }
  }

  ml_drift::cl::Environment env_;
  LiteRtEnvironment environment_ = nullptr;
};

TEST_F(BufferHandlerOpenClTest, TestCreateOpenClMemory) {
  LiteRtRankedTensorType tensorType;
  tensorType.element_type = kLiteRtElementTypeFloat32;
  tensorType.layout.rank = 2;
  tensorType.layout.dimensions[0] = 4;
  tensorType.layout.dimensions[1] = 4;
  tensorType.layout.has_strides = false;

  size_t bytes = 4 * 4 * sizeof(float);
  HwMemoryInfoPtr memoryInfo = nullptr;

  auto status = LiteRtCreateOpenClMemory(
      env_.context().context(), env_.queue()->queue(), &tensorType,
      kLiteRtTensorBufferTypeOpenClBuffer, bytes, bytes, &memoryInfo);
  ASSERT_EQ(status, kLiteRtStatusOk);
  ASSERT_NE(memoryInfo, nullptr);

  status = LiteRtDestroyOpenClMemory(memoryInfo);
  ASSERT_EQ(status, kLiteRtStatusOk);
}

TEST_F(BufferHandlerOpenClTest, TestLockUnlockOpenClMemory) {
  LiteRtRankedTensorType tensorType;
  tensorType.element_type = kLiteRtElementTypeFloat32;
  tensorType.layout.rank = 1;
  tensorType.layout.dimensions[0] = 4;
  tensorType.layout.has_strides = false;

  size_t bytes = 4 * sizeof(float);
  HwMemoryInfoPtr memoryInfo = nullptr;

  auto status = LiteRtCreateOpenClMemory(
      env_.context().context(), env_.queue()->queue(), &tensorType,
      kLiteRtTensorBufferTypeOpenClBuffer, bytes, bytes, &memoryInfo);
  ASSERT_EQ(status, kLiteRtStatusOk);

  void* hostMemory = nullptr;
  status = LiteRtLockOpenClMemory(memoryInfo, kLiteRtTensorBufferLockModeWrite,
                                  &hostMemory);
  ASSERT_EQ(status, kLiteRtStatusOk);
  ASSERT_NE(hostMemory, nullptr);

  float* floatMemory = static_cast<float*>(hostMemory);
  for (int i = 0; i < 4; ++i) {
    floatMemory[i] = static_cast<float>(i);
  }

  status = LiteRtUnlockOpenClMemory(memoryInfo);
  ASSERT_EQ(status, kLiteRtStatusOk);

  // Read back
  hostMemory = nullptr;
  status = LiteRtLockOpenClMemory(memoryInfo, kLiteRtTensorBufferLockModeRead,
                                  &hostMemory);
  ASSERT_EQ(status, kLiteRtStatusOk);
  ASSERT_NE(hostMemory, nullptr);
  floatMemory = static_cast<float*>(hostMemory);
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(floatMemory[i], static_cast<float>(i), 0.0001f);
  }

  status = LiteRtUnlockOpenClMemory(memoryInfo);
  ASSERT_EQ(status, kLiteRtStatusOk);

  status = LiteRtDestroyOpenClMemory(memoryInfo);
  ASSERT_EQ(status, kLiteRtStatusOk);
}

TEST_F(BufferHandlerOpenClTest, TestClearOpenClMemory) {
  LiteRtRankedTensorType tensorType;
  tensorType.element_type = kLiteRtElementTypeFloat32;
  tensorType.layout.rank = 1;
  tensorType.layout.dimensions[0] = 4;
  tensorType.layout.has_strides = false;

  size_t bytes = 4 * sizeof(float);
  HwMemoryInfoPtr memoryInfo = nullptr;

  auto status = LiteRtCreateOpenClMemory(
      env_.context().context(), env_.queue()->queue(), &tensorType,
      kLiteRtTensorBufferTypeOpenClBuffer, bytes, bytes, &memoryInfo);
  ASSERT_EQ(status, kLiteRtStatusOk);

  // Write some data
  void* hostMemory = nullptr;
  status = LiteRtLockOpenClMemory(memoryInfo, kLiteRtTensorBufferLockModeWrite,
                                  &hostMemory);
  ASSERT_EQ(status, kLiteRtStatusOk);
  float* floatMemory = static_cast<float*>(hostMemory);
  for (int i = 0; i < 4; ++i) {
    floatMemory[i] = static_cast<float>(i + 1);
  }
  status = LiteRtUnlockOpenClMemory(memoryInfo);
  ASSERT_EQ(status, kLiteRtStatusOk);

  // Clear memory
  status = LiteRtClearOpenClMemory(memoryInfo);
  ASSERT_EQ(status, kLiteRtStatusOk);

  // Read back and verify it is zero
  status = LiteRtLockOpenClMemory(memoryInfo, kLiteRtTensorBufferLockModeRead,
                                  &hostMemory);
  ASSERT_EQ(status, kLiteRtStatusOk);
  floatMemory = static_cast<float*>(hostMemory);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(floatMemory[i], 0.0f);
  }
  status = LiteRtUnlockOpenClMemory(memoryInfo);
  ASSERT_EQ(status, kLiteRtStatusOk);

  status = LiteRtDestroyOpenClMemory(memoryInfo);
  ASSERT_EQ(status, kLiteRtStatusOk);
}

TEST_F(BufferHandlerOpenClTest, TestImportOpenClMemory) {
  LiteRtRankedTensorType tensorType;
  tensorType.element_type = kLiteRtElementTypeFloat32;
  tensorType.layout.rank = 1;
  tensorType.layout.dimensions[0] = 4;
  tensorType.layout.has_strides = false;
  size_t bytes = 4 * sizeof(float);

  // Create an explicit OpenCL buffer using ml_drift library backing open_cl
  ml_drift::cl::Buffer cl_buffer;
  ASSERT_OK(
      ml_drift::cl::CreateReadWriteBuffer(bytes, &env_.context(), &cl_buffer));

  HwMemoryInfoPtr memoryInfo = nullptr;

  auto status = LiteRtImportOpenClMemory(
      env_.context().context(), env_.queue()->queue(), &tensorType,
      kLiteRtTensorBufferTypeOpenClBufferPacked,
      reinterpret_cast<void*>(cl_buffer.GetMemoryPtr()), bytes, bytes,
      &memoryInfo);
  ASSERT_EQ(status, kLiteRtStatusOk);
  ASSERT_NE(memoryInfo, nullptr);

  status = LiteRtDestroyOpenClMemory(memoryInfo);
  ASSERT_EQ(status, kLiteRtStatusOk);
}

TEST_F(BufferHandlerOpenClTest, TestLockUnlockOpenClBufferPacked) {
  LiteRtRankedTensorType tensorType;
  tensorType.element_type = kLiteRtElementTypeFloat32;
  tensorType.layout.rank = 1;
  tensorType.layout.dimensions[0] = 4;
  tensorType.layout.has_strides = false;

  size_t bytes = 4 * sizeof(float);
  HwMemoryInfoPtr memoryInfo = nullptr;

  auto status = LiteRtCreateOpenClMemory(
      env_.context().context(), env_.queue()->queue(), &tensorType,
      kLiteRtTensorBufferTypeOpenClBufferPacked, bytes, bytes, &memoryInfo);
  ASSERT_EQ(status, kLiteRtStatusOk);

  void* hostMemory = nullptr;
  status = LiteRtLockOpenClMemory(memoryInfo, kLiteRtTensorBufferLockModeWrite,
                                  &hostMemory);
  ASSERT_EQ(status, kLiteRtStatusOk);
  ASSERT_NE(hostMemory, nullptr);

  float* floatMemory = static_cast<float*>(hostMemory);
  for (int i = 0; i < 4; ++i) {
    floatMemory[i] = static_cast<float>(i);
  }

  status = LiteRtUnlockOpenClMemory(memoryInfo);
  ASSERT_EQ(status, kLiteRtStatusOk);

  // Read back
  hostMemory = nullptr;
  status = LiteRtLockOpenClMemory(memoryInfo, kLiteRtTensorBufferLockModeRead,
                                  &hostMemory);
  ASSERT_EQ(status, kLiteRtStatusOk);
  ASSERT_NE(hostMemory, nullptr);
  floatMemory = static_cast<float*>(hostMemory);
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(floatMemory[i], static_cast<float>(i), 0.0001f);
  }

  status = LiteRtUnlockOpenClMemory(memoryInfo);
  ASSERT_EQ(status, kLiteRtStatusOk);

  status = LiteRtDestroyOpenClMemory(memoryInfo);
  ASSERT_EQ(status, kLiteRtStatusOk);
}

}  // namespace
}  // namespace litert
