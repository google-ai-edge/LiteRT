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

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_platform_support.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_any.h"
#include "litert/cc/litert_layout.h"
#include "litert/test/matchers.h"

#if LITERT_HAS_OPENCL_SUPPORT
#include "ml_drift/cl/cl_command_queue.h"  // from @ml_drift
#include "ml_drift/cl/cl_context.h"  // from @ml_drift
#include "ml_drift/cl/environment.h"  // from @ml_drift
#include "ml_drift/cl/opencl_wrapper.h"  // from @ml_drift
#include "third_party/odml/litert/ml_drift/delegate/buffer_handler_opencl.h"
#endif  // LITERT_HAS_OPENCL_SUPPORT

#include "litert/c/internal/litert_tensor_buffer_registry.h"

namespace {
constexpr const float kTensorData[] = {10, 20, 30, 40};

constexpr const int32_t kTensorDimensions[] = {sizeof(kTensorData) /
                                               sizeof(kTensorData[0])};

constexpr const LiteRtRankedTensorType kTensorType = {
    /*.element_type=*/kLiteRtElementTypeFloat32,
    ::litert::BuildLayout(kTensorDimensions)};

bool CanLoadOpenCl() {
#if LITERT_HAS_OPENCL_SUPPORT
  return ml_drift::cl::LoadOpenCL().ok();
#else
  return false;
#endif
}

TEST(TensorBuffer, OpenCL) {
// MSAN does not support GPU tests.
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported In msan or tsan";
#endif

  if (!LiteRtHasOpenClSupport()) {
    GTEST_SKIP() << "OpenCL buffers are not supported on this platform; "
                    "skipping the test";
  }
  if (!CanLoadOpenCl()) {
    GTEST_SKIP() << "OpenCL could not be loaded; skipping the test";
  }

  ml_drift::cl::Environment cl_env;
  LITERT_ASSERT_OK(ml_drift::cl::CreateEnvironment(&cl_env));

  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtAny context_id,
      litert::ToLiteRtAny(litert::LiteRtVariant(
          reinterpret_cast<int64_t>(cl_env.context().context()))));

  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtAny queue_id,
      litert::ToLiteRtAny(litert::LiteRtVariant(
          reinterpret_cast<int64_t>(cl_env.queue()->queue()))));

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

  LiteRtEnvironment env;
  LITERT_ASSERT_OK(LiteRtCreateEnvironment(environment_options.size(),
                                           environment_options.data(), &env));

  LiteRtRegisterTensorBufferHandlers(
      env, kLiteRtTensorBufferTypeOpenClBufferPacked, LiteRtCreateOpenClMemory,
      LiteRtDestroyOpenClMemory, LiteRtLockOpenClMemory,
      LiteRtUnlockOpenClMemory, LiteRtClearOpenClMemory,
      LiteRtImportOpenClMemory, kLiteRtEnvOptionTagOpenClContext,
      kLiteRtEnvOptionTagOpenClCommandQueue);

  // Use packed buffer to test Clear() easily. Otherwise, downloaded data may
  // have some garbage values due to strides.
  constexpr auto kTensorBufferType = kLiteRtTensorBufferTypeOpenClBufferPacked;
  LiteRtTensorBuffer tensor_buffer;
  ASSERT_EQ(
      LiteRtCreateManagedTensorBuffer(env, kTensorBufferType, &kTensorType,
                                      sizeof(kTensorData), &tensor_buffer),
      kLiteRtStatusOk);

  LiteRtTensorBufferType buffer_type;
  ASSERT_EQ(LiteRtGetTensorBufferType(tensor_buffer, &buffer_type),
            kLiteRtStatusOk);
  ASSERT_EQ(buffer_type, kTensorBufferType);

  LiteRtRankedTensorType tensor_type;
  ASSERT_EQ(LiteRtGetTensorBufferTensorType(tensor_buffer, &tensor_type),
            kLiteRtStatusOk);
  ASSERT_EQ(tensor_type.element_type, kLiteRtElementTypeFloat32);
  ASSERT_EQ(tensor_type.layout.rank, 1);
  ASSERT_EQ(tensor_type.layout.dimensions[0], kTensorType.layout.dimensions[0]);
  ASSERT_EQ(tensor_type.layout.has_strides, false);

  size_t size;
  ASSERT_EQ(LiteRtGetTensorBufferSize(tensor_buffer, &size), kLiteRtStatusOk);
  ASSERT_EQ(size, sizeof(kTensorData));

  size_t offset;
  ASSERT_EQ(LiteRtGetTensorBufferOffset(tensor_buffer, &offset),
            kLiteRtStatusOk);
  ASSERT_EQ(offset, 0);

  void* host_mem_addr;
  ASSERT_EQ(LiteRtLockTensorBuffer(tensor_buffer, &host_mem_addr,
                                   kLiteRtTensorBufferLockModeWrite),
            kLiteRtStatusOk);
  std::memcpy(host_mem_addr, kTensorData, sizeof(kTensorData));
  ASSERT_EQ(LiteRtUnlockTensorBuffer(tensor_buffer), kLiteRtStatusOk);

  ASSERT_EQ(LiteRtLockTensorBuffer(tensor_buffer, &host_mem_addr,
                                   kLiteRtTensorBufferLockModeRead),
            kLiteRtStatusOk);
  ASSERT_EQ(std::memcmp(host_mem_addr, kTensorData, sizeof(kTensorData)), 0);
  ASSERT_EQ(LiteRtUnlockTensorBuffer(tensor_buffer), kLiteRtStatusOk);

  ASSERT_EQ(LiteRtClearTensorBuffer(tensor_buffer), kLiteRtStatusOk);
  ASSERT_EQ(LiteRtLockTensorBuffer(tensor_buffer, &host_mem_addr,
                                   kLiteRtTensorBufferLockModeRead),
            kLiteRtStatusOk);
  std::vector<uint8_t> zero_data(sizeof(kTensorData), 0);
  ASSERT_EQ(std::memcmp(host_mem_addr, zero_data.data(), zero_data.size()), 0);
  ASSERT_EQ(LiteRtUnlockTensorBuffer(tensor_buffer), kLiteRtStatusOk);

  LiteRtDestroyTensorBuffer(tensor_buffer);
  LiteRtDestroyEnvironment(env);
}

}  // namespace
