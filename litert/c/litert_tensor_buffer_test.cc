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

#include <array>  // NOLINT
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>  // NOLINT: Need when ANDROID_API_LEVEL >= 26
#include "litert/c/internal/litert_tensor_buffer_registry.h"  // NOLINT: Required for custom tensor buffer tests.
#include "litert/c/litert_any.h"  // NOLINT
#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_tensor_buffer.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"  // NOLINT
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_platform_support.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_any.h"  // NOLINT
#include "litert/cc/litert_layout.h"
#include "litert/runtime/event.h"
#include "litert/test/matchers.h"

#if LITERT_CUSTOM_TENSOR_BUFFER_TEST

#if LITERT_HAS_WEBGPU_SUPPORT
#include "ml_drift/webgpu/environment.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/buffer_handler_webgpu.h"
#endif  // LITERT_HAS_WEBGPU_SUPPORT

#if LITERT_HAS_VULKAN_SUPPORT
#include "ml_drift/syrtis/environment.h"  // from @ml_drift
#include "ml_drift/syrtis/vulkan_wrapper.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/buffer_handler_vulkan.h"
#include "ml_drift_delegate/delegate/shared_vulkan_env.h"
#endif  // LITERT_HAS_VULKAN_SUPPORT
#endif  // LITERT_CUSTOM_TENSOR_BUFFER_TEST

namespace {
constexpr const float kTensorData[] = {10, 20, 30, 40};

constexpr const int32_t kTensorDimensions[] = {sizeof(kTensorData) /
                                               sizeof(kTensorData[0])};

constexpr const LiteRtRankedTensorType kTensorType = {
    /*.element_type=*/kLiteRtElementTypeFloat32,
    ::litert::BuildLayout(kTensorDimensions)};

// A host-memory backed custom buffer type, used to test
// LiteRtCreateTensorBufferFromCustomHandle without vendor hardware. Buffers
// created through `create_func` own their memory; buffers created through
// `import_func` only wrap the caller's pointer.
struct FakeHostMemoryInfo : HwMemoryInfo {
  void* data;
  bool owns_data;
};

LiteRtStatus CreateFakeHostMemory(LiteRtGpuDeviceId device_id,
                                  LiteRtGpuQueueId queue_id,
                                  const LiteRtRankedTensorType* tensor_type,
                                  LiteRtTensorBufferType buffer_type,
                                  size_t bytes, size_t packed_bytes,
                                  HwMemoryInfoPtr* hw_memory_info) {
  auto* info = new FakeHostMemoryInfo();
  info->data = std::calloc(bytes, 1);
  info->owns_data = true;
  info->memory_handle = info->data;
  *hw_memory_info = info;
  return kLiteRtStatusOk;
}

LiteRtStatus ImportFakeHostMemory(LiteRtGpuDeviceId device_id,
                                  LiteRtGpuQueueId queue_id,
                                  const LiteRtRankedTensorType* tensor_type,
                                  LiteRtTensorBufferType buffer_type,
                                  HwMemoryHandle hw_buffer_handle,
                                  size_t bytes, size_t packed_bytes,
                                  HwMemoryInfoPtr* hw_memory_info) {
  if (hw_buffer_handle == nullptr || packed_bytes > bytes) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto* info = new FakeHostMemoryInfo();
  info->data = hw_buffer_handle;
  info->owns_data = false;
  info->memory_handle = hw_buffer_handle;
  *hw_memory_info = info;
  return kLiteRtStatusOk;
}

LiteRtStatus DestroyFakeHostMemory(HwMemoryInfoPtr hw_memory_info) {
  auto* info = static_cast<FakeHostMemoryInfo*>(hw_memory_info);
  if (info->owns_data) {
    std::free(info->data);
  }
  delete info;
  return kLiteRtStatusOk;
}

LiteRtStatus LockFakeHostMemory(HwMemoryInfoPtr hw_memory_info,
                                LiteRtTensorBufferLockMode mode,
                                void** host_memory_ptr) {
  *host_memory_ptr = static_cast<FakeHostMemoryInfo*>(hw_memory_info)->data;
  return kLiteRtStatusOk;
}

LiteRtStatus UnlockFakeHostMemory(HwMemoryInfoPtr hw_memory_info) {
  return kLiteRtStatusOk;
}

constexpr auto kFakeBufferType = kLiteRtTensorBufferTypeUserCustomBuffer;

LiteRtStatus RegisterFakeHostMemory(LiteRtEnvironment env,
                                    ImportCustomTensorBuffer import_func) {
  return LiteRtRegisterTensorBufferHandlers(
      env, kFakeBufferType, CreateFakeHostMemory, DestroyFakeHostMemory,
      LockFakeHostMemory, UnlockFakeHostMemory, /*clear_func=*/nullptr,
      import_func, kLiteRtEnvOptionTagNull, kLiteRtEnvOptionTagNull);
}

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

TEST(TensorBuffer, CreateFromCustomHandleWrapsMemoryWithoutCopy) {
  LiteRtEnvironment env;
  LITERT_ASSERT_OK(
      LiteRtCreateEnvironment(/*num_options=*/0, /*options=*/nullptr, &env));
  LITERT_ASSERT_OK(RegisterFakeHostMemory(env, ImportFakeHostMemory));

  float data[] = {10, 20, 30, 40};
  LiteRtTensorBuffer tensor_buffer;
  LITERT_ASSERT_OK(LiteRtCreateTensorBufferFromCustomHandle(
      env, &kTensorType, kFakeBufferType, data, sizeof(data), &tensor_buffer));

  LiteRtTensorBufferType buffer_type;
  LITERT_ASSERT_OK(LiteRtGetTensorBufferType(tensor_buffer, &buffer_type));
  EXPECT_EQ(buffer_type, kFakeBufferType);
  size_t size;
  LITERT_ASSERT_OK(LiteRtGetTensorBufferSize(tensor_buffer, &size));
  EXPECT_EQ(size, sizeof(data));

  // Locking returns the wrapped memory itself, and writes go to it.
  void* host_mem_addr;
  LITERT_ASSERT_OK(LiteRtLockTensorBuffer(tensor_buffer, &host_mem_addr,
                                          kLiteRtTensorBufferLockModeWrite));
  EXPECT_EQ(host_mem_addr, data);
  static_cast<float*>(host_mem_addr)[0] = 99;
  LITERT_ASSERT_OK(LiteRtUnlockTensorBuffer(tensor_buffer));
  EXPECT_EQ(data[0], 99);

  // Destroying the buffer must not free the caller's memory.
  LiteRtDestroyTensorBuffer(tensor_buffer);
  EXPECT_EQ(data[3], 40);
  LiteRtDestroyEnvironment(env);
}

TEST(TensorBuffer, CreateFromCustomHandleAliasesLargerBuffer) {
  LiteRtEnvironment env;
  LITERT_ASSERT_OK(
      LiteRtCreateEnvironment(/*num_options=*/0, /*options=*/nullptr, &env));
  LITERT_ASSERT_OK(RegisterFakeHostMemory(env, ImportFakeHostMemory));

  // A managed custom buffer twice the size of kTensorType, like the KV cache
  // allocated for the largest context size.
  constexpr int32_t kLargeDimensions[] = {2 * kTensorDimensions[0]};
  constexpr LiteRtRankedTensorType kLargeTensorType = {
      kLiteRtElementTypeFloat32, ::litert::BuildLayout(kLargeDimensions)};
  constexpr size_t kLargeSize = 2 * sizeof(kTensorData);
  LiteRtTensorBuffer large_buffer;
  LITERT_ASSERT_OK(LiteRtCreateManagedTensorBuffer(
      env, kFakeBufferType, &kLargeTensorType, kLargeSize, &large_buffer));

  void* large_data;
  LITERT_ASSERT_OK(LiteRtLockTensorBuffer(large_buffer, &large_data,
                                          kLiteRtTensorBufferLockModeRead));
  LITERT_ASSERT_OK(LiteRtUnlockTensorBuffer(large_buffer));

  // Wrap the start of the large buffer as a smaller buffer of the same type.
  LiteRtTensorBuffer alias_buffer;
  LITERT_ASSERT_OK(LiteRtCreateTensorBufferFromCustomHandle(
      env, &kTensorType, kFakeBufferType, large_data, sizeof(kTensorData),
      &alias_buffer));

  void* alias_data;
  LITERT_ASSERT_OK(LiteRtLockTensorBuffer(alias_buffer, &alias_data,
                                          kLiteRtTensorBufferLockModeWrite));
  EXPECT_EQ(alias_data, large_data);
  std::memcpy(alias_data, kTensorData, sizeof(kTensorData));
  LITERT_ASSERT_OK(LiteRtUnlockTensorBuffer(alias_buffer));

  // The alias is destroyed first; the large buffer still owns its memory and
  // sees the data written through the alias.
  LiteRtDestroyTensorBuffer(alias_buffer);
  LITERT_ASSERT_OK(LiteRtLockTensorBuffer(large_buffer, &large_data,
                                          kLiteRtTensorBufferLockModeRead));
  EXPECT_EQ(std::memcmp(large_data, kTensorData, sizeof(kTensorData)), 0);
  LITERT_ASSERT_OK(LiteRtUnlockTensorBuffer(large_buffer));

  LiteRtDestroyTensorBuffer(large_buffer);
  LiteRtDestroyEnvironment(env);
}

TEST(TensorBuffer, CreateFromCustomHandleRequiresImportHandler) {
  LiteRtEnvironment env;
  LITERT_ASSERT_OK(
      LiteRtCreateEnvironment(/*num_options=*/0, /*options=*/nullptr, &env));
  LITERT_ASSERT_OK(RegisterFakeHostMemory(env, /*import_func=*/nullptr));

  float data[] = {10, 20, 30, 40};
  LiteRtTensorBuffer tensor_buffer;
  EXPECT_EQ(LiteRtCreateTensorBufferFromCustomHandle(
                env, &kTensorType, kFakeBufferType, data, sizeof(data),
                &tensor_buffer),
            kLiteRtStatusErrorInvalidArgument);
  LiteRtDestroyEnvironment(env);
}

TEST(TensorBuffer, CreateFromCustomHandleRejectsNullArguments) {
  LiteRtEnvironment env;
  LITERT_ASSERT_OK(
      LiteRtCreateEnvironment(/*num_options=*/0, /*options=*/nullptr, &env));
  LITERT_ASSERT_OK(RegisterFakeHostMemory(env, ImportFakeHostMemory));

  float data[] = {10, 20, 30, 40};
  LiteRtTensorBuffer tensor_buffer;
  EXPECT_EQ(LiteRtCreateTensorBufferFromCustomHandle(
                env, /*tensor_type=*/nullptr, kFakeBufferType, data,
                sizeof(data), &tensor_buffer),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(LiteRtCreateTensorBufferFromCustomHandle(
                env, &kTensorType, kFakeBufferType, data, sizeof(data),
                /*buffer=*/nullptr),
            kLiteRtStatusErrorInvalidArgument);
  LiteRtDestroyEnvironment(env);
}

TEST(TensorBuffer, EmptyScalarBuffer) {
  constexpr const int32_t kEmptyDimensions[] = {0};
  constexpr const LiteRtRankedTensorType kEmptyTensorType = {
      kLiteRtElementTypeFloat32,
      ::litert::BuildLayout(kEmptyDimensions)};

  LiteRtEnvironment env;
  LITERT_ASSERT_OK(
      LiteRtCreateEnvironment(/*num_options=*/0, /*options=*/nullptr, &env));

  LiteRtTensorBuffer tensor_buffer;
  ASSERT_EQ(
      LiteRtCreateManagedTensorBuffer(env, kLiteRtTensorBufferTypeHostMemory,
                                      &kEmptyTensorType,
                                      /*buffer_size=*/0, &tensor_buffer),
      kLiteRtStatusOk);

  LiteRtTensorBufferType buffer_type;
  ASSERT_EQ(LiteRtGetTensorBufferType(tensor_buffer, &buffer_type),
            kLiteRtStatusOk);
  ASSERT_EQ(buffer_type, kLiteRtTensorBufferTypeHostMemory);

  LiteRtRankedTensorType tensor_type;
  ASSERT_EQ(LiteRtGetTensorBufferTensorType(tensor_buffer, &tensor_type),
            kLiteRtStatusOk);
  ASSERT_EQ(tensor_type.element_type, kLiteRtElementTypeFloat32);
  ASSERT_EQ(tensor_type.layout.rank, 1);
  ASSERT_EQ(tensor_type.layout.dimensions[0], 0);

  LiteRtDestroyTensorBuffer(tensor_buffer);
  LiteRtDestroyEnvironment(env);
}

TEST(TensorBuffer, DestroyNullIsNoOp) { LiteRtDestroyTensorBuffer(nullptr); }

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

#if LITERT_CUSTOM_TENSOR_BUFFER_TEST
#if LITERT_HAS_WEBGPU_SUPPORT
TEST(TensorBuffer, WebGpu) {
// MSAN does not support GPU tests.
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported In msan or tsan";
#endif

  ml_drift::webgpu::Environment wgpu_env;
  LITERT_ASSERT_OK(wgpu_env.Initialize());

  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtAny device_id,
      litert::ToLiteRtAny(reinterpret_cast<int64_t>(wgpu_env.device().Get())));
  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtAny command_queue,
      litert::ToLiteRtAny(reinterpret_cast<int64_t>(wgpu_env.queue().Get())));
  const std::array<LiteRtEnvOption, 2> environment_options = {
      LiteRtEnvOption{.tag = kLiteRtEnvOptionTagWebGpuDevice,
                      .value = device_id},
      LiteRtEnvOption{.tag = kLiteRtEnvOptionTagWebGpuQueue,
                      .value = command_queue},
  };
  LiteRtEnvironment env;
  LITERT_ASSERT_OK(LiteRtCreateEnvironment(environment_options.size(),
                                           environment_options.data(), &env));

  constexpr auto kTensorBufferType = kLiteRtTensorBufferTypeWebGpuBuffer;
  LiteRtRegisterTensorBufferHandlers(
      env, kTensorBufferType, LiteRtCreateWebGpuMemory,
      LiteRtDestroyWebGpuMemory, LiteRtLockWebGpuMemory,
      LiteRtUnlockWebGpuMemory, LiteRtClearWebGpuMemory,
      LiteRtImportWebGpuMemory, kLiteRtEnvOptionTagWebGpuDevice,
      kLiteRtEnvOptionTagWebGpuQueue);

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
#endif  // LITERT_HAS_WEBGPU_SUPPORT

#if LITERT_HAS_VULKAN_SUPPORT
TEST(TensorBuffer, Vulkan) {
// MSAN does not support GPU tests.
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported In msan or tsan";
#endif

  ASSERT_EQ(ml_drift::syrtis::InitVulkan(), 1);
  litert::ml_drift::SharedVulkanEnv vk_env;
  LITERT_ASSERT_OK(ml_drift::syrtis::CreateEnvironment(&vk_env.vulkan_env()));
  const VkCommandPoolCreateInfo command_pool_create_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = vk_env.vulkan_env().GetQueue().QueueFamilyIndex()};
  ASSERT_EQ(ml_drift::syrtis::vkCreateCommandPool(
                vk_env.vulkan_env().GetDevice(), &command_pool_create_info,
                nullptr, &vk_env.command_pool()),
            VK_SUCCESS);

  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtAny vulkan_env,
      litert::ToLiteRtAny(reinterpret_cast<int64_t>(&vk_env)));
  const std::array<LiteRtEnvOption, 1> environment_options = {
      LiteRtEnvOption{.tag = kLiteRtEnvOptionTagVulkanEnvironment,
                      .value = vulkan_env},
  };
  LiteRtEnvironment env;
  LITERT_ASSERT_OK(LiteRtCreateEnvironment(environment_options.size(),
                                           environment_options.data(), &env));

  constexpr auto kTensorBufferType = kLiteRtTensorBufferTypeVulkanBuffer;
  LiteRtRegisterTensorBufferHandlers(
      env, kTensorBufferType, LiteRtCreateVulkanMemory,
      LiteRtDestroyVulkanMemory, LiteRtLockVulkanMemory,
      LiteRtUnlockVulkanMemory, LiteRtClearVulkanMemory,
      /*import_handler=*/nullptr, kLiteRtEnvOptionTagVulkanEnvironment,
      (LiteRtEnvOptionTag)0);

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
  LITERT_ASSERT_OK(vk_env.SubmitCommandBuffer(/*wait_for_completion=*/true));

  ASSERT_EQ(LiteRtLockTensorBuffer(tensor_buffer, &host_mem_addr,
                                   kLiteRtTensorBufferLockModeRead),
            kLiteRtStatusOk);
  ASSERT_EQ(std::memcmp(host_mem_addr, kTensorData, sizeof(kTensorData)), 0);
  ASSERT_EQ(LiteRtUnlockTensorBuffer(tensor_buffer), kLiteRtStatusOk);
  LITERT_ASSERT_OK(vk_env.SubmitCommandBuffer(/*wait_for_completion=*/true));

  ASSERT_EQ(LiteRtClearTensorBuffer(tensor_buffer), kLiteRtStatusOk);
  LITERT_ASSERT_OK(vk_env.SubmitCommandBuffer(/*wait_for_completion=*/true));

  ASSERT_EQ(LiteRtLockTensorBuffer(tensor_buffer, &host_mem_addr,
                                   kLiteRtTensorBufferLockModeRead),
            kLiteRtStatusOk);
  std::vector<uint8_t> zero_data(sizeof(kTensorData), 0);
  ASSERT_EQ(std::memcmp(host_mem_addr, zero_data.data(), zero_data.size()), 0);
  ASSERT_EQ(LiteRtUnlockTensorBuffer(tensor_buffer), kLiteRtStatusOk);
  LITERT_ASSERT_OK(vk_env.SubmitCommandBuffer(/*wait_for_completion=*/true));

  LiteRtDestroyTensorBuffer(tensor_buffer);
  LiteRtDestroyEnvironment(env);
}
#endif  // LITERT_HAS_VULKAN_SUPPORT
#endif  // LITERT_CUSTOM_TENSOR_BUFFER_TEST
