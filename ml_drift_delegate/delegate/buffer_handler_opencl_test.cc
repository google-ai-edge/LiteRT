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

#include "ml_drift_delegate/delegate/buffer_handler_opencl.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "ml_drift/cl/buffer.h"  // from @ml_drift
#include "ml_drift/cl/cl_command_queue.h"  // from @ml_drift
#include "ml_drift/cl/cl_context.h"  // from @ml_drift
#include "ml_drift/cl/cl_device.h"  // from @ml_drift
#include "ml_drift/cl/environment.h"  // from @ml_drift
#include "ml_drift/cl/opencl_wrapper.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_tensor_buffer.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_any.h"
#include "litert/test/matchers.h"
#include <CL/cl.h>

namespace litert {
namespace {

using ::absl_testing::IsOkAndHolds;

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

struct OpenClMemoryDeleter {
  void operator()(HwMemoryInfoPtr ptr) const {
    EXPECT_EQ(LiteRtDestroyOpenClMemory(ptr), kLiteRtStatusOk);
  }
};
using ScopedOpenClMemory = std::unique_ptr<HwMemoryInfo, OpenClMemoryDeleter>;

absl::StatusOr<cl_uint> GetQueueRefCount(cl_command_queue queue) {
  cl_uint count = 0;
  const cl_int err = ::ml_drift::cl::clGetCommandQueueInfo(
      queue, CL_QUEUE_REFERENCE_COUNT, sizeof(count), &count, nullptr);
  if (err != CL_SUCCESS) {
    return absl::InternalError(
        absl::StrCat("clGetCommandQueueInfo failed with: ", err));
  }
  return count;
}

TEST_F(BufferHandlerOpenClTest, TestOpenClBufferPackedHostAllocFlag) {
  LiteRtRankedTensorType tensor_type{};
  tensor_type.element_type = kLiteRtElementTypeFloat32;
  tensor_type.layout.rank = 1;
  tensor_type.layout.dimensions[0] = 16;
  tensor_type.layout.has_strides = false;
  const size_t bytes = 16 * sizeof(float);

  HwMemoryInfoPtr default_packed_raw = nullptr;
  ASSERT_EQ(LiteRtCreateOpenClMemory(env_.context().context(),
                                     env_.queue()->queue(), &tensor_type,
                                     kLiteRtTensorBufferTypeOpenClBufferPacked,
                                     bytes, bytes, &default_packed_raw),
            kLiteRtStatusOk);
  ASSERT_NE(default_packed_raw, nullptr);
  ScopedOpenClMemory default_packed_info(default_packed_raw);

  cl_mem default_mem = static_cast<cl_mem>(default_packed_info->raw_handle);
  cl_mem_flags default_flags = 0;
  ASSERT_EQ(::ml_drift::cl::clGetMemObjectInfo(default_mem, CL_MEM_FLAGS,
                                               sizeof(default_flags),
                                               &default_flags, nullptr),
            CL_SUCCESS);
  EXPECT_EQ((default_flags & CL_MEM_ALLOC_HOST_PTR) != 0,
            env_.device().GetInfo().IsPowerVR());

  ::ml_drift::delegate::testing::ScopedOpenClPackedHostAllocForTesting
      force_host_alloc;

  HwMemoryInfoPtr packed_raw = nullptr;
  ASSERT_EQ(LiteRtCreateOpenClMemory(env_.context().context(),
                                     env_.queue()->queue(), &tensor_type,
                                     kLiteRtTensorBufferTypeOpenClBufferPacked,
                                     bytes, bytes, &packed_raw),
            kLiteRtStatusOk);
  ASSERT_NE(packed_raw, nullptr);
  ScopedOpenClMemory packed_info(packed_raw);

  cl_mem packed_mem = static_cast<cl_mem>(packed_info->raw_handle);
  cl_mem_flags packed_flags = 0;
  ASSERT_EQ(::ml_drift::cl::clGetMemObjectInfo(packed_mem, CL_MEM_FLAGS,
                                               sizeof(packed_flags),
                                               &packed_flags, nullptr),
            CL_SUCCESS);
  EXPECT_NE(packed_flags & CL_MEM_ALLOC_HOST_PTR, 0u);
  EXPECT_NE(packed_flags & CL_MEM_READ_WRITE, 0u);

  // Non-packed OpenClBuffer should NOT set CL_MEM_ALLOC_HOST_PTR.
  HwMemoryInfoPtr buffer_raw = nullptr;
  ASSERT_EQ(LiteRtCreateOpenClMemory(
                env_.context().context(), env_.queue()->queue(), &tensor_type,
                kLiteRtTensorBufferTypeOpenClBuffer, bytes, bytes, &buffer_raw),
            kLiteRtStatusOk);
  ASSERT_NE(buffer_raw, nullptr);
  ScopedOpenClMemory buffer_info(buffer_raw);

  cl_mem buffer_mem = static_cast<cl_mem>(buffer_info->raw_handle);
  cl_mem_flags buffer_flags = 0;
  ASSERT_EQ(::ml_drift::cl::clGetMemObjectInfo(buffer_mem, CL_MEM_FLAGS,
                                               sizeof(buffer_flags),
                                               &buffer_flags, nullptr),
            CL_SUCCESS);
  EXPECT_EQ(buffer_flags & CL_MEM_ALLOC_HOST_PTR, 0u);
}

TEST_F(BufferHandlerOpenClTest, TestOpenClBufferPackedReadWriteClearCoherence) {
  ::ml_drift::delegate::testing::ScopedOpenClPackedHostAllocForTesting
      force_host_alloc;

  constexpr size_t kNumElements = 16;
  constexpr size_t kBytes = kNumElements * sizeof(float);
  LiteRtRankedTensorType tensor_type{};
  tensor_type.element_type = kLiteRtElementTypeFloat32;
  tensor_type.layout.rank = 1;
  tensor_type.layout.dimensions[0] = kNumElements;
  tensor_type.layout.has_strides = false;

  HwMemoryInfoPtr raw_info = nullptr;
  ASSERT_EQ(LiteRtCreateOpenClMemory(env_.context().context(),
                                     env_.queue()->queue(), &tensor_type,
                                     kLiteRtTensorBufferTypeOpenClBufferPacked,
                                     kBytes, kBytes, &raw_info),
            kLiteRtStatusOk);
  ASSERT_NE(raw_info, nullptr);
  ScopedOpenClMemory memory_info(raw_info);

  cl_mem cl_buf = static_cast<cl_mem>(memory_info->raw_handle);

  // 1. Lock(Write) -> Unlock -> GPU clEnqueueReadBuffer sees CPU writes.
  void* host_ptr = nullptr;
  ASSERT_EQ(LiteRtLockOpenClMemory(memory_info.get(),
                                   kLiteRtTensorBufferLockModeWrite, &host_ptr),
            kLiteRtStatusOk);
  ASSERT_NE(host_ptr, nullptr);
  float* cpu_floats = static_cast<float*>(host_ptr);
  for (size_t i = 0; i < kNumElements; ++i) {
    cpu_floats[i] = static_cast<float>(i) * 1.5f + 2.0f;
  }
  // Clearing while locked must be rejected.
  EXPECT_EQ(LiteRtClearOpenClMemory(memory_info.get()),
            kLiteRtStatusErrorRuntimeFailure);
  ASSERT_EQ(LiteRtUnlockOpenClMemory(memory_info.get()), kLiteRtStatusOk);

  std::array<float, kNumElements> gpu_readback{};
  ASSERT_EQ(::ml_drift::cl::clEnqueueReadBuffer(
                env_.queue()->queue(), cl_buf, CL_TRUE, /*offset=*/0, kBytes,
                gpu_readback.data(), 0, nullptr, nullptr),
            CL_SUCCESS);
  for (size_t i = 0; i < kNumElements; ++i) {
    EXPECT_FLOAT_EQ(gpu_readback[i], static_cast<float>(i) * 1.5f + 2.0f);
  }

  // 2. GPU clEnqueueWriteBuffer -> Lock(Read) sees GPU writes.
  std::array<float, kNumElements> gpu_upload{};
  for (size_t i = 0; i < kNumElements; ++i) {
    gpu_upload[i] = 100.0f - static_cast<float>(i);
  }
  ASSERT_EQ(::ml_drift::cl::clEnqueueWriteBuffer(
                env_.queue()->queue(), cl_buf, CL_TRUE, /*offset=*/0, kBytes,
                gpu_upload.data(), 0, nullptr, nullptr),
            CL_SUCCESS);

  host_ptr = nullptr;
  ASSERT_EQ(LiteRtLockOpenClMemory(memory_info.get(),
                                   kLiteRtTensorBufferLockModeRead, &host_ptr),
            kLiteRtStatusOk);
  ASSERT_NE(host_ptr, nullptr);
  cpu_floats = static_cast<float*>(host_ptr);
  for (size_t i = 0; i < kNumElements; ++i) {
    EXPECT_FLOAT_EQ(cpu_floats[i], 100.0f - static_cast<float>(i));
  }
  ASSERT_EQ(LiteRtUnlockOpenClMemory(memory_info.get()), kLiteRtStatusOk);

  // 3. LiteRtClearOpenClMemory zeroes buffer contents cleanly.
  ASSERT_EQ(LiteRtClearOpenClMemory(memory_info.get()), kLiteRtStatusOk);

  host_ptr = nullptr;
  ASSERT_EQ(LiteRtLockOpenClMemory(memory_info.get(),
                                   kLiteRtTensorBufferLockModeRead, &host_ptr),
            kLiteRtStatusOk);
  ASSERT_NE(host_ptr, nullptr);
  cpu_floats = static_cast<float*>(host_ptr);
  for (size_t i = 0; i < kNumElements; ++i) {
    EXPECT_FLOAT_EQ(cpu_floats[i], 0.0f);
  }
  ASSERT_EQ(LiteRtUnlockOpenClMemory(memory_info.get()), kLiteRtStatusOk);
}

TEST_F(BufferHandlerOpenClTest, TestOpenClBufferPackedZeroBytes) {
  ::ml_drift::delegate::testing::ScopedOpenClPackedHostAllocForTesting
      force_host_alloc;

  LiteRtRankedTensorType zero_tensor_type{};
  zero_tensor_type.element_type = kLiteRtElementTypeFloat32;
  zero_tensor_type.layout.rank = 1;
  zero_tensor_type.layout.dimensions[0] = 0;
  zero_tensor_type.layout.has_strides = false;

  HwMemoryInfoPtr zero_raw = nullptr;
  ASSERT_EQ(LiteRtCreateOpenClMemory(
                env_.context().context(), env_.queue()->queue(),
                &zero_tensor_type, kLiteRtTensorBufferTypeOpenClBufferPacked,
                /*bytes=*/0, /*packed_bytes=*/0, &zero_raw),
            kLiteRtStatusOk);
  ASSERT_NE(zero_raw, nullptr);
  ScopedOpenClMemory zero_info(zero_raw);

  void* zero_ptr = nullptr;
  ASSERT_EQ(LiteRtLockOpenClMemory(zero_info.get(),
                                   kLiteRtTensorBufferLockModeWrite, &zero_ptr),
            kLiteRtStatusOk);
  EXPECT_NE(zero_ptr, nullptr);
  ASSERT_EQ(LiteRtUnlockOpenClMemory(zero_info.get()), kLiteRtStatusOk);
  ASSERT_EQ(LiteRtClearOpenClMemory(zero_info.get()), kLiteRtStatusOk);
}

TEST_F(BufferHandlerOpenClTest, TestOpenClCommandQueueRetention) {
  ::ml_drift::delegate::testing::ScopedOpenClPackedHostAllocForTesting
      force_host_alloc;

  cl_command_queue queue_handle = env_.queue()->queue();
  LiteRtRankedTensorType tensor_type{};
  tensor_type.element_type = kLiteRtElementTypeFloat32;
  tensor_type.layout.rank = 1;
  tensor_type.layout.dimensions[0] = 4;
  tensor_type.layout.has_strides = false;
  constexpr size_t kBytes = 4 * sizeof(float);

  ASSERT_OK_AND_ASSIGN(const cl_uint baseline_refcount,
                       GetQueueRefCount(queue_handle));
  {
    HwMemoryInfoPtr created_raw = nullptr;
    ASSERT_EQ(LiteRtCreateOpenClMemory(
                  env_.context().context(), queue_handle, &tensor_type,
                  kLiteRtTensorBufferTypeOpenClBufferPacked, kBytes, kBytes,
                  &created_raw),
              kLiteRtStatusOk);
    ASSERT_NE(created_raw, nullptr);
    ScopedOpenClMemory created_info(created_raw);
    EXPECT_THAT(GetQueueRefCount(queue_handle),
                IsOkAndHolds(baseline_refcount + 1));
  }
  EXPECT_THAT(GetQueueRefCount(queue_handle), IsOkAndHolds(baseline_refcount));

  ::ml_drift::cl::Buffer external_buf;
  ASSERT_OK(::ml_drift::cl::CreateReadWriteBuffer(kBytes, &env_.context(),
                                                  &external_buf));
  {
    HwMemoryInfoPtr imported_raw = nullptr;
    ASSERT_EQ(LiteRtImportOpenClMemory(
                  env_.context().context(), queue_handle, &tensor_type,
                  kLiteRtTensorBufferTypeOpenClBufferPacked,
                  reinterpret_cast<void*>(external_buf.GetMemoryPtr()), kBytes,
                  kBytes, &imported_raw),
              kLiteRtStatusOk);
    ASSERT_NE(imported_raw, nullptr);
    ScopedOpenClMemory imported_info(imported_raw);
    EXPECT_THAT(GetQueueRefCount(queue_handle),
                IsOkAndHolds(baseline_refcount + 1));
  }
  EXPECT_THAT(GetQueueRefCount(queue_handle), IsOkAndHolds(baseline_refcount));
}

TEST_F(BufferHandlerOpenClTest,
       TestOpenClOrphanCommandQueueDestroyWhileMapped) {
  ::ml_drift::delegate::testing::ScopedOpenClPackedHostAllocForTesting
      force_host_alloc;

  LiteRtRankedTensorType tensor_type{};
  tensor_type.element_type = kLiteRtElementTypeFloat32;
  tensor_type.layout.rank = 1;
  tensor_type.layout.dimensions[0] = 4;
  tensor_type.layout.has_strides = false;
  constexpr size_t kBytes = 4 * sizeof(float);

  ::ml_drift::cl::CLCommandQueue temp_queue;
  ASSERT_OK(::ml_drift::cl::CreateCLCommandQueue(env_.device(), env_.context(),
                                                 &temp_queue));
  HwMemoryInfoPtr orphan_queue_raw = nullptr;
  ASSERT_EQ(LiteRtCreateOpenClMemory(env_.context().context(),
                                     temp_queue.queue(), &tensor_type,
                                     kLiteRtTensorBufferTypeOpenClBufferPacked,
                                     kBytes, kBytes, &orphan_queue_raw),
            kLiteRtStatusOk);
  ASSERT_NE(orphan_queue_raw, nullptr);
  ScopedOpenClMemory orphan_queue_info(orphan_queue_raw);

  void* mapped_ptr = nullptr;
  ASSERT_EQ(
      LiteRtLockOpenClMemory(orphan_queue_info.get(),
                             kLiteRtTensorBufferLockModeReadWrite, &mapped_ptr),
      kLiteRtStatusOk);
  // Release caller's sole reference to `temp_queue` before buffer destruction.
  temp_queue = ::ml_drift::cl::CLCommandQueue();
  orphan_queue_info.reset();
}

TEST_F(BufferHandlerOpenClTest, TestInvalidArgumentsAndNullQueueErrors) {
  LiteRtRankedTensorType tensor_type{};
  tensor_type.element_type = kLiteRtElementTypeFloat32;
  tensor_type.layout.rank = 1;
  tensor_type.layout.dimensions[0] = 16;
  tensor_type.layout.has_strides = false;
  constexpr size_t kBytes = 16 * sizeof(float);

  HwMemoryInfoPtr out_info = nullptr;
  EXPECT_EQ(
      LiteRtCreateOpenClMemory(env_.context().context(), env_.queue()->queue(),
                               /*tensor_type=*/nullptr,
                               kLiteRtTensorBufferTypeOpenClBufferPacked,
                               kBytes, kBytes, &out_info),
      kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(LiteRtCreateOpenClMemory(
                env_.context().context(), env_.queue()->queue(), &tensor_type,
                kLiteRtTensorBufferTypeOpenClBufferPacked, kBytes, kBytes,
                /*opencl_memory_info=*/nullptr),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(
      LiteRtCreateOpenClMemory(
          /*device_id=*/nullptr, env_.queue()->queue(), &tensor_type,
          kLiteRtTensorBufferTypeOpenClBufferPacked, kBytes, kBytes, &out_info),
      kLiteRtStatusErrorInvalidArgument);
  // Oversized packed_bytes exceeding the 4-channel-padded BHWC(16,1,1,1)
  // tensor descriptor capacity (256 bytes).
  EXPECT_EQ(LiteRtCreateOpenClMemory(env_.context().context(),
                                     env_.queue()->queue(), &tensor_type,
                                     kLiteRtTensorBufferTypeOpenClBufferPacked,
                                     kBytes, kBytes * 8, &out_info),
            kLiteRtStatusErrorInvalidArgument);
  if (out_info != nullptr) {
    ScopedOpenClMemory unexpected_info(out_info);
    out_info = nullptr;
  }

  ::ml_drift::cl::Buffer external_buf;
  ASSERT_OK(::ml_drift::cl::CreateReadWriteBuffer(kBytes, &env_.context(),
                                                  &external_buf));
  void* raw_buf_handle = reinterpret_cast<void*>(external_buf.GetMemoryPtr());
  EXPECT_EQ(
      LiteRtImportOpenClMemory(env_.context().context(), env_.queue()->queue(),
                               /*tensor_type=*/nullptr,
                               kLiteRtTensorBufferTypeOpenClBufferPacked,
                               raw_buf_handle, kBytes, kBytes, &out_info),
      kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(LiteRtImportOpenClMemory(
                env_.context().context(), env_.queue()->queue(), &tensor_type,
                kLiteRtTensorBufferTypeOpenClBufferPacked,
                /*hw_buffer_handle=*/nullptr, kBytes, kBytes, &out_info),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(LiteRtImportOpenClMemory(
                env_.context().context(), env_.queue()->queue(), &tensor_type,
                kLiteRtTensorBufferTypeOpenClBufferPacked, raw_buf_handle,
                kBytes, kBytes, /*hw_memory_info=*/nullptr),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(LiteRtImportOpenClMemory(
                /*device_id=*/nullptr, env_.queue()->queue(), &tensor_type,
                kLiteRtTensorBufferTypeOpenClBufferPacked, raw_buf_handle,
                kBytes, kBytes, &out_info),
            kLiteRtStatusErrorInvalidArgument);

  // Packed buffer created with a null command queue fails cleanly on Lock/Clear
  // and destroys safely.
  HwMemoryInfoPtr null_queue_raw = nullptr;
  ASSERT_EQ(LiteRtCreateOpenClMemory(env_.context().context(),
                                     /*queue_id=*/nullptr, &tensor_type,
                                     kLiteRtTensorBufferTypeOpenClBufferPacked,
                                     kBytes, kBytes, &null_queue_raw),
            kLiteRtStatusOk);
  ASSERT_NE(null_queue_raw, nullptr);
  ScopedOpenClMemory null_queue_info(null_queue_raw);

  void* mapped_ptr = nullptr;
  EXPECT_EQ(
      LiteRtLockOpenClMemory(null_queue_info.get(),
                             kLiteRtTensorBufferLockModeWrite, &mapped_ptr),
      kLiteRtStatusErrorRuntimeFailure);
  EXPECT_EQ(LiteRtClearOpenClMemory(null_queue_info.get()),
            kLiteRtStatusErrorRuntimeFailure);
}

}  // namespace
}  // namespace litert
