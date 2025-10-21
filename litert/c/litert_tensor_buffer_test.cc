// Copyright 2024 Google LLC.
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

#include "litert/c/litert_tensor_buffer.h"

#include <any>
#include <array>
#include <cstdint>
#include <cstring>

#include <gtest/gtest.h>  // NOLINT: Need when ANDROID_API_LEVEL >= 26
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_platform_support.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_any.h"
#include "litert/cc/litert_layout.h"
#include "litert/runtime/event.h"
#include "litert/test/matchers.h"

#if LITERT_HAS_OPENCL_SUPPORT
#include "tflite/delegates/gpu/cl/opencl_wrapper.h"
#endif  // LITERT_HAS_OPENCL_SUPPORT

namespace {
constexpr const float kTensorData[] = {10, 20, 30, 40};

constexpr const int32_t kTensorDimensions[] = {sizeof(kTensorData) /
                                               sizeof(kTensorData[0])};

constexpr const LiteRtRankedTensorType kTensorType = {
    /*.element_type=*/kLiteRtElementTypeFloat32,
    ::litert::BuildLayout(kTensorDimensions)};

}  // namespace

TEST(TensorBuffer, HostMemory) {
  constexpr auto kTensorBufferType = kLiteRtTensorBufferTypeHostMemory;

  LiteRtEnvironment env;
  LITERT_ASSERT_OK(
      LiteRtCreateEnvironment(/*num_options=*/0, /*options=*/nullptr, &env));

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

  LiteRtDestroyTensorBuffer(tensor_buffer);
  LiteRtDestroyEnvironment(env);
}

TEST(TensorBuffer, Ahwb) {
  if (!LiteRtHasAhwbSupport()) {
    GTEST_SKIP() << "AHardwareBuffers are not supported on this platform; "
                    "skipping the test";
  }

  LiteRtEnvironment env;
  LITERT_ASSERT_OK(
      LiteRtCreateEnvironment(/*num_options=*/0, /*options=*/nullptr, &env));
  constexpr auto kTensorBufferType = kLiteRtTensorBufferTypeAhwb;

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

  LiteRtDestroyTensorBuffer(tensor_buffer);
  LiteRtDestroyEnvironment(env);
}

TEST(TensorBuffer, Ion) {
  if (!LiteRtHasIonSupport()) {
    GTEST_SKIP()
        << "ION buffers are not supported on this platform; skipping the test";
  }

  LiteRtEnvironment env;
  LITERT_ASSERT_OK(
      LiteRtCreateEnvironment(/*num_options=*/0, /*options=*/nullptr, &env));
  constexpr auto kTensorBufferType = kLiteRtTensorBufferTypeIon;

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

  LiteRtDestroyTensorBuffer(tensor_buffer);
  LiteRtDestroyEnvironment(env);
}

TEST(TensorBuffer, DmaBuf) {
  if (!LiteRtHasDmaBufSupport()) {
    GTEST_SKIP()
        << "DMA-BUF buffers are not supported on this platform; skipping "
           "the test";
  }

  LiteRtEnvironment env;
  LITERT_ASSERT_OK(
      LiteRtCreateEnvironment(/*num_options=*/0, /*options=*/nullptr, &env));
  constexpr auto kTensorBufferType = kLiteRtTensorBufferTypeDmaBuf;

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

  LiteRtDestroyTensorBuffer(tensor_buffer);
  LiteRtDestroyEnvironment(env);
}

TEST(TensorBuffer, FastRpc) {
  if (!LiteRtHasFastRpcSupport()) {
    GTEST_SKIP()
        << "FastRPC buffers are not supported on this platform; skipping "
           "the test";
  }

  LiteRtEnvironment env;
  LITERT_ASSERT_OK(
      LiteRtCreateEnvironment(/*num_options=*/0, /*options=*/nullptr, &env));
  constexpr auto kTensorBufferType = kLiteRtTensorBufferTypeFastRpc;

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

  LiteRtDestroyTensorBuffer(tensor_buffer);
  LiteRtDestroyEnvironment(env);
}

TEST(TensorBuffer, Event) {
  LiteRtEnvironment env;
  LITERT_ASSERT_OK(
      LiteRtCreateEnvironment(/*num_options=*/0, /*options=*/nullptr, &env));
  constexpr auto kTensorBufferType = kLiteRtTensorBufferTypeHostMemory;

  LiteRtTensorBuffer tensor_buffer;
  ASSERT_EQ(
      LiteRtCreateManagedTensorBuffer(env, kTensorBufferType, &kTensorType,
                                      sizeof(kTensorData), &tensor_buffer),
      kLiteRtStatusOk);

  bool has_event = true;
  ASSERT_EQ(LiteRtHasTensorBufferEvent(tensor_buffer, &has_event),
            kLiteRtStatusOk);
  EXPECT_FALSE(has_event);

  LiteRtEvent event = new LiteRtEventT;
  ASSERT_EQ(LiteRtSetTensorBufferEvent(tensor_buffer, event), kLiteRtStatusOk);

  has_event = false;
  ASSERT_EQ(LiteRtHasTensorBufferEvent(tensor_buffer, &has_event),
            kLiteRtStatusOk);
  EXPECT_TRUE(has_event);

  LiteRtEvent actual_event;
  ASSERT_EQ(LiteRtGetTensorBufferEvent(tensor_buffer, &actual_event),
            kLiteRtStatusOk);
  ASSERT_EQ(actual_event, event);

  ASSERT_EQ(LiteRtClearTensorBufferEvent(tensor_buffer), kLiteRtStatusOk);
  ASSERT_EQ(actual_event, event);

  has_event = true;
  ASSERT_EQ(LiteRtHasTensorBufferEvent(tensor_buffer, &has_event),
            kLiteRtStatusOk);
  EXPECT_FALSE(has_event);

  LiteRtDestroyTensorBuffer(tensor_buffer);
  LiteRtDestroyEnvironment(env);
}

bool CanLoadOpenCl() {
#if LITERT_HAS_OPENCL_SUPPORT
  return tflite::gpu::cl::LoadOpenCL().ok();
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
  if (!CanLoadOpenCl()){
    GTEST_SKIP() << "OpenCL could not be loaded; skipping the test";
  }

  // Create an option with opencl device id zero. This trick initializes the
  // OpenCL environment at the LiteRtEnvironment creation time.
  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtAny null_deivce_id,
      litert::ToLiteRtAny(litert::LiteRtVariant(INT64_C(0))));
  const std::array<LiteRtEnvOption, 1> environment_options = {
      LiteRtEnvOption{
          /*.tag=*/kLiteRtEnvOptionTagOpenClDeviceId,
          /*.value=*/null_deivce_id,
      },
  };
  LiteRtEnvironment env;
  LITERT_ASSERT_OK(LiteRtCreateEnvironment(environment_options.size(),
                                           environment_options.data(), &env));

  constexpr auto kTensorBufferType = kLiteRtTensorBufferTypeOpenClBuffer;
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

  LiteRtDestroyTensorBuffer(tensor_buffer);
  LiteRtDestroyEnvironment(env);
}

#if LITERT_HAS_OPENGL_SUPPORT
TEST(TensorBuffer, GlBuffer) {
// MSAN does not support GPU tests.
#if defined(MEMORY_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported In msan";
#endif

  if (!LiteRtHasOpenGlSupport()) {
    GTEST_SKIP() << "OpenGL buffers are not supported on this platform; "
                    "skipping the test";
  }

  // Create an option with opengl display id zero. This trick initializes the
  // OpenGL environment at the LiteRtEnvironment creation time.
  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtAny null_display_id,
      litert::ToLiteRtAny(litert::LiteRtVariant(INT64_C(0))));
  const std::array<LiteRtEnvOption, 1> environment_options = {
      LiteRtEnvOption{
          /*.tag=*/kLiteRtEnvOptionTagEglDisplay,
          /*.value=*/null_display_id,
      },
  };
  LiteRtEnvironment env;
  LITERT_ASSERT_OK(LiteRtCreateEnvironment(environment_options.size(),
                                           environment_options.data(), &env));

  constexpr auto kTensorBufferType = kLiteRtTensorBufferTypeGlBuffer;

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
  ASSERT_EQ(LiteRtLockTensorBuffer(tensor_buffer, &host_mem_addr),
            kLiteRtStatusOk);
  std::memcpy(host_mem_addr, kTensorData, sizeof(kTensorData));
  ASSERT_EQ(LiteRtUnlockTensorBuffer(tensor_buffer), kLiteRtStatusOk);

  ASSERT_EQ(LiteRtLockTensorBuffer(tensor_buffer, &host_mem_addr),
            kLiteRtStatusOk);
  ASSERT_EQ(std::memcmp(host_mem_addr, kTensorData, sizeof(kTensorData)), 0);
  ASSERT_EQ(LiteRtUnlockTensorBuffer(tensor_buffer), kLiteRtStatusOk);

  LiteRtDestroyTensorBuffer(tensor_buffer);
  LiteRtDestroyEnvironment(env);
}
#endif  // LITERT_HAS_OPENGL_SUPPORT
