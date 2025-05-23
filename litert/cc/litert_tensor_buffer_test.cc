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

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>  // NOLINT: Used for OpenCL logic.
#include <utility>
#include <vector>  // NOLINT: Used for OpenCL logic.

#include <gmock/gmock.h>
#include <gtest/gtest.h>  // NOLINT: Need when ANDROID_API_LEVEL >= 26
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_event.h"
#include "litert/cc/litert_handle.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_platform_support.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/runtime/tensor_buffer.h"
#include "litert/test/matchers.h"

#if LITERT_HAS_AHWB_SUPPORT
#include <android/hardware_buffer.h>
#endif  // LITERT_HAS_AHWB_SUPPORT

#if LITERT_HAS_OPENCL_SUPPORT
#include <CL/cl.h>  // NOLINT: Used for OpenCL logic.
#include "tflite/delegates/gpu/cl/buffer.h"  // NOLINT: Used for OpenCL logic.
#include "tflite/delegates/gpu/cl/cl_command_queue.h"  // NOLINT: Used for OpenCL logic.
#include "tflite/delegates/gpu/cl/cl_context.h"  // NOLINT: Used for OpenCL logic.
#include "tflite/delegates/gpu/cl/cl_device.h"  // NOLINT: Used for OpenCL logic.
#include "tflite/delegates/gpu/cl/environment.h"  // NOLINT: Used for OpenCL logic.
#include "tflite/delegates/gpu/cl/opencl_wrapper.h"  // NOLINT: Used for OpenCL logic.
#endif  // LITERT_HAS_OPENCL_SUPPORT

#if LITERT_HAS_OPENGL_SUPPORT
#include "tflite/delegates/gpu/cl/gl_interop.h"
#include "tflite/delegates/gpu/gl/egl_environment.h"
#endif  // LITERT_HAS_OPENGL_SUPPORT

namespace litert {
namespace {

using ::testing::Eq;
using ::testing::Ne;
using ::testing::litert::IsError;
using OptionTag = litert::Environment::OptionTag;

constexpr const float kTensorData[] = {10, 20, 30, 40};

constexpr const int32_t kTensorDimensions[] = {sizeof(kTensorData) /
                                               sizeof(kTensorData[0])};

constexpr int kFakeSyncFenceFd = 1;

constexpr const LiteRtRankedTensorType kTestTensorType = {
    /*.element_type=*/kLiteRtElementTypeFloat32,
    BuildLayout(kTensorDimensions)};

int GetReferenceCount(const TensorBuffer& tensor_buffer) {
  LiteRtTensorBufferT* internal_tensor_buffer =
      static_cast<LiteRtTensorBufferT*>(tensor_buffer.Get());
  return internal_tensor_buffer->RefCount();
}

#if LITERT_HAS_OPENGL_SUPPORT
struct GpuEnvironmentOptions {
  // If any of these objects are set, created environment will use them instead
  // of creating/choosing own instances.
  cl_device_id device_id = nullptr;
  cl_platform_id platform_id = nullptr;
  cl_context context = nullptr;
  cl_command_queue command_queue = nullptr;

  // Whenever input and/or output is GL object, EGL display and context must be
  // set to create GL aware OpenCL context. Do not set these variables whenever
  // GL interoperability is not needed.
  // It is the error to set egl_display, egl_context AND context at the same
  // time. If egl_display and egl_context are set, they will be used to create
  // GL-aware CL context.
  EGLDisplay egl_display = EGL_NO_DISPLAY;
  EGLContext egl_context = EGL_NO_CONTEXT;

  bool IsGlAware() const {
    return egl_context != EGL_NO_CONTEXT && egl_display != EGL_NO_DISPLAY;
  }
};

class UserGpuEnvironment {
 public:
  explicit UserGpuEnvironment(
      std::unique_ptr<tflite::gpu::gl::EglEnvironment> gl_env,
      std::unique_ptr<tflite::gpu::cl::CLDevice> device,
      std::unique_ptr<tflite::gpu::cl::CLContext> context,
      std::unique_ptr<tflite::gpu::cl::CLCommandQueue> command_queue,
      litert::Environment env)
      : gl_env_(std::move(gl_env)),
        device_(std::move(device)),
        context_(std::move(context)),
        command_queue_(std::move(command_queue)),
        env_(std::move(env)) {}

  static std::unique_ptr<UserGpuEnvironment> Create() {
    std::vector<litert::Environment::Option> environment_options;
    auto options = std::make_unique<GpuEnvironmentOptions>();

    // Create GL environment.
    std::unique_ptr<tflite::gpu::gl::EglEnvironment> gl_env;
    if (tflite::gpu::gl::EglEnvironment::NewEglEnvironment(&gl_env).ok()) {
      environment_options.push_back(litert::Environment::Option{
          OptionTag::EglDisplay, reinterpret_cast<int64_t>(gl_env->display())});
      environment_options.push_back(litert::Environment::Option{
          OptionTag::EglContext,
          reinterpret_cast<int64_t>(gl_env->context().context()),
      });
    }

    // Create CL environment.
    auto device = std::make_unique<tflite::gpu::cl::CLDevice>();
    auto context = std::make_unique<tflite::gpu::cl::CLContext>();
    auto command_queue = std::make_unique<tflite::gpu::cl::CLCommandQueue>();
    if (tflite::gpu::cl::LoadOpenCL().ok()) {
      EXPECT_OK(tflite::gpu::cl::CreateDefaultGPUDevice(device.get()));
      if (tflite::gpu::cl::IsGlSharingSupported(*device)) {
        EXPECT_OK(tflite::gpu::cl::CreateCLGLContext(
            *device,
            reinterpret_cast<cl_context_properties>(
                gl_env->context().context()),
            reinterpret_cast<cl_context_properties>(gl_env->display()),
            context.get()));
      } else {
        EXPECT_OK(tflite::gpu::cl::CreateCLContext(*device, context.get()));
      }

      EXPECT_OK(tflite::gpu::cl::CreateCLCommandQueue(*device, *context,
                                                      command_queue.get()));

      environment_options.push_back(litert::Environment::Option{
          OptionTag::ClDeviceId,
          reinterpret_cast<int64_t>(device->id()),
      });
      environment_options.push_back(litert::Environment::Option{
          OptionTag::ClPlatformId,
          reinterpret_cast<int64_t>(device->platform()),
      });
      environment_options.push_back(litert::Environment::Option{
          OptionTag::ClContext,
          reinterpret_cast<int64_t>(context->context()),
      });
      environment_options.push_back(litert::Environment::Option{
          OptionTag::ClCommandQueue,
          reinterpret_cast<int64_t>(command_queue->queue()),
      });
    }

    // Create LiteRt environment from GL and CL options.
    auto env = litert::Environment::Create(environment_options);
    return std::make_unique<UserGpuEnvironment>(
        std::move(gl_env), std::move(device), std::move(context),
        std::move(command_queue), std::move(*env));
  }

  LiteRtEnvironment GetEnvironment() { return env_.Get(); }
  tflite::gpu::cl::CLCommandQueue* GetCommandQueue() {
    return command_queue_.get();
  }

 private:
  std::unique_ptr<tflite::gpu::gl::EglEnvironment> gl_env_;
  std::unique_ptr<tflite::gpu::cl::CLDevice> device_;
  std::unique_ptr<tflite::gpu::cl::CLContext> context_;
  std::unique_ptr<tflite::gpu::cl::CLCommandQueue> command_queue_;
  litert::Environment env_;
};
#endif  // LITERT_HAS_OPENGL_SUPPORT

TEST(TensorBuffer, HostMemory) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  const RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = kLiteRtTensorBufferTypeHostMemory;

  auto tensor_buffer = TensorBuffer::CreateManaged(
      env.Get(), kTensorBufferType, kTensorType, sizeof(kTensorData));
  ASSERT_TRUE(tensor_buffer);

  auto tensor_buffer_type = tensor_buffer->BufferType();
  ASSERT_TRUE(tensor_buffer_type);
  ASSERT_EQ(*tensor_buffer_type, kTensorBufferType);

  auto tensor_type = tensor_buffer->TensorType();
  ASSERT_TRUE(tensor_type);

  ASSERT_EQ(tensor_type->ElementType(), ElementType::Float32);
  ASSERT_EQ(tensor_type->Layout().Rank(), 1);
  ASSERT_EQ(tensor_type->Layout().Dimensions()[0],
            kTensorType.Layout().Dimensions()[0]);
  ASSERT_FALSE(tensor_type->Layout().HasStrides());

  auto size = tensor_buffer->Size();
  ASSERT_TRUE(size);
  ASSERT_EQ(*size, sizeof(kTensorData));

  auto offset = tensor_buffer->Offset();
  ASSERT_TRUE(offset);
  ASSERT_EQ(*offset, 0);

  {
    auto lock_and_addr = TensorBufferScopedLock::Create(
        *tensor_buffer, TensorBuffer::LockMode::kWrite);
    ASSERT_TRUE(lock_and_addr);
    std::memcpy(lock_and_addr->second, kTensorData, sizeof(kTensorData));
  }

  {
    auto lock_and_addr = TensorBufferScopedLock::Create(
        *tensor_buffer, TensorBuffer::LockMode::kRead);
    ASSERT_TRUE(lock_and_addr);
    ASSERT_EQ(
        std::memcmp(lock_and_addr->second, kTensorData, sizeof(kTensorData)),
        0);
  }
}

TEST(TensorBuffer, DoubleLockOrUnlock) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  const RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = kLiteRtTensorBufferTypeHostMemory;

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto tensor_buffer,
      TensorBuffer::CreateManaged(env.Get(), kTensorBufferType, kTensorType,
                                  sizeof(kTensorData)));
  LITERT_EXPECT_OK(tensor_buffer.Lock());
  EXPECT_THAT(tensor_buffer.Lock(), IsError());
  LITERT_EXPECT_OK(tensor_buffer.Unlock());
  EXPECT_THAT(tensor_buffer.Unlock(), IsError());
}

TEST(TensorBuffer, TensorBufferScopedLock) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  const RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = kLiteRtTensorBufferTypeHostMemory;

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto tensor_buffer,
      TensorBuffer::CreateManaged(env.Get(), kTensorBufferType, kTensorType,
                                  sizeof(kTensorData)));

  {
    auto lock_and_addr = TensorBufferScopedLock::Create(tensor_buffer);
    LITERT_EXPECT_OK(lock_and_addr);
  }
}

TEST(TensorBuffer, Ahwb) {
  if (!HasAhwbSupport()) {
    GTEST_SKIP() << "AHardwareBuffers are not supported on this platform; "
                    "skipping the test";
  }
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));

  const RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = kLiteRtTensorBufferTypeAhwb;

  auto tensor_buffer = TensorBuffer::CreateManaged(
      env.Get(), kTensorBufferType, kTensorType, sizeof(kTensorData));
  ASSERT_TRUE(tensor_buffer);

  auto tensor_buffer_type = tensor_buffer->BufferType();
  ASSERT_TRUE(tensor_buffer_type);
  ASSERT_EQ(*tensor_buffer_type, kTensorBufferType);

  auto tensor_type = tensor_buffer->TensorType();
  ASSERT_TRUE(tensor_type);

  ASSERT_EQ(tensor_type->ElementType(), ElementType::Float32);
  ASSERT_EQ(tensor_type->Layout().Rank(), 1);
  ASSERT_EQ(tensor_type->Layout().Dimensions()[0],
            kTensorType.Layout().Dimensions()[0]);
  ASSERT_FALSE(tensor_type->Layout().HasStrides());

  auto size = tensor_buffer->Size();
  ASSERT_TRUE(size);
  ASSERT_EQ(*size, sizeof(kTensorData));

  auto offset = tensor_buffer->Offset();
  ASSERT_TRUE(offset);
  ASSERT_EQ(*offset, 0);

  {
    auto lock_and_addr = TensorBufferScopedLock::Create(
        *tensor_buffer, TensorBuffer::LockMode::kWrite);
    ASSERT_TRUE(lock_and_addr);
    std::memcpy(lock_and_addr->second, kTensorData, sizeof(kTensorData));
  }

  {
    auto lock_and_addr = TensorBufferScopedLock::Create(
        *tensor_buffer, TensorBuffer::LockMode::kRead);
    ASSERT_TRUE(lock_and_addr);
    ASSERT_EQ(
        std::memcmp(lock_and_addr->second, kTensorData, sizeof(kTensorData)),
        0);
  }
}

TEST(TensorBuffer, Ion) {
  if (!HasIonSupport()) {
    GTEST_SKIP()
        << "ION buffers are not supported on this platform; skipping the test";
  }
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));

  const RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = kLiteRtTensorBufferTypeIon;

  auto tensor_buffer = TensorBuffer::CreateManaged(
      env.Get(), kTensorBufferType, kTensorType, sizeof(kTensorData));
  ASSERT_TRUE(tensor_buffer);

  auto tensor_buffer_type = tensor_buffer->BufferType();
  ASSERT_TRUE(tensor_buffer_type);
  ASSERT_EQ(*tensor_buffer_type, kTensorBufferType);

  auto tensor_type = tensor_buffer->TensorType();
  ASSERT_TRUE(tensor_type);

  ASSERT_EQ(tensor_type->ElementType(), ElementType::Float32);
  ASSERT_EQ(tensor_type->Layout().Rank(), 1);
  ASSERT_EQ(tensor_type->Layout().Dimensions()[0],
            kTensorType.Layout().Dimensions()[0]);
  ASSERT_FALSE(tensor_type->Layout().HasStrides());

  auto size = tensor_buffer->Size();
  ASSERT_TRUE(size);
  ASSERT_EQ(*size, sizeof(kTensorData));

  auto offset = tensor_buffer->Offset();
  ASSERT_TRUE(offset);
  ASSERT_EQ(*offset, 0);

  {
    auto lock_and_addr = TensorBufferScopedLock::Create(
        *tensor_buffer, TensorBuffer::LockMode::kWrite);
    ASSERT_TRUE(lock_and_addr);
    std::memcpy(lock_and_addr->second, kTensorData, sizeof(kTensorData));
  }

  {
    auto lock_and_addr = TensorBufferScopedLock::Create(
        *tensor_buffer, TensorBuffer::LockMode::kRead);
    ASSERT_TRUE(lock_and_addr);
    ASSERT_EQ(
        std::memcmp(lock_and_addr->second, kTensorData, sizeof(kTensorData)),
        0);
  }
}

TEST(TensorBuffer, DmaBuf) {
  if (!HasDmaBufSupport()) {
    GTEST_SKIP()
        << "DMA-BUF buffers are not supported on this platform; skipping "
           "the test";
  }
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));

  const RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = kLiteRtTensorBufferTypeDmaBuf;

  auto tensor_buffer = TensorBuffer::CreateManaged(
      env.Get(), kTensorBufferType, kTensorType, sizeof(kTensorData));
  ASSERT_TRUE(tensor_buffer);

  auto tensor_buffer_type = tensor_buffer->BufferType();
  ASSERT_TRUE(tensor_buffer_type);
  ASSERT_EQ(*tensor_buffer_type, kTensorBufferType);

  auto tensor_type = tensor_buffer->TensorType();
  ASSERT_TRUE(tensor_type);

  ASSERT_EQ(tensor_type->ElementType(), ElementType::Float32);
  ASSERT_EQ(tensor_type->Layout().Rank(), 1);
  ASSERT_EQ(tensor_type->Layout().Dimensions()[0],
            kTensorType.Layout().Dimensions()[0]);
  ASSERT_FALSE(tensor_type->Layout().HasStrides());

  auto size = tensor_buffer->Size();
  ASSERT_TRUE(size);
  ASSERT_EQ(*size, sizeof(kTensorData));

  auto offset = tensor_buffer->Offset();
  ASSERT_TRUE(offset);
  ASSERT_EQ(*offset, 0);

  {
    auto lock_and_addr = TensorBufferScopedLock::Create(
        *tensor_buffer, TensorBuffer::LockMode::kWrite);
    ASSERT_TRUE(lock_and_addr);
    std::memcpy(lock_and_addr->second, kTensorData, sizeof(kTensorData));
  }

  {
    auto lock_and_addr = TensorBufferScopedLock::Create(
        *tensor_buffer, TensorBuffer::LockMode::kRead);
    ASSERT_TRUE(lock_and_addr);
    ASSERT_EQ(
        std::memcmp(lock_and_addr->second, kTensorData, sizeof(kTensorData)),
        0);
  }
}

TEST(TensorBuffer, FastRpc) {
  if (!HasFastRpcSupport()) {
    GTEST_SKIP()
        << "FastRPC buffers are not supported on this platform; skipping "
           "the test";
  }
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));

  const RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = kLiteRtTensorBufferTypeFastRpc;

  auto tensor_buffer = TensorBuffer::CreateManaged(
      env.Get(), kTensorBufferType, kTensorType, sizeof(kTensorData));
  ASSERT_TRUE(tensor_buffer);

  auto tensor_buffer_type = tensor_buffer->BufferType();
  ASSERT_TRUE(tensor_buffer_type);
  ASSERT_EQ(*tensor_buffer_type, kTensorBufferType);

  auto tensor_type = tensor_buffer->TensorType();
  ASSERT_TRUE(tensor_type);

  ASSERT_EQ(tensor_type->ElementType(), ElementType::Float32);
  ASSERT_EQ(tensor_type->Layout().Rank(), 1);
  ASSERT_EQ(tensor_type->Layout().Dimensions()[0],
            kTensorType.Layout().Dimensions()[0]);
  ASSERT_FALSE(tensor_type->Layout().HasStrides());

  auto size = tensor_buffer->Size();
  ASSERT_TRUE(size);
  ASSERT_EQ(*size, sizeof(kTensorData));

  auto offset = tensor_buffer->Offset();
  ASSERT_TRUE(offset);
  ASSERT_EQ(*offset, 0);

  {
    auto lock_and_addr = TensorBufferScopedLock::Create(
        *tensor_buffer, TensorBuffer::LockMode::kWrite);
    ASSERT_TRUE(lock_and_addr);
    std::memcpy(lock_and_addr->second, kTensorData, sizeof(kTensorData));
  }

  {
    auto lock_and_addr = TensorBufferScopedLock::Create(
        *tensor_buffer, TensorBuffer::LockMode::kRead);
    ASSERT_TRUE(lock_and_addr);
    ASSERT_EQ(
        std::memcmp(lock_and_addr->second, kTensorData, sizeof(kTensorData)),
        0);
  }
}

TEST(TensorBuffer, NotOwned) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  LiteRtTensorBuffer litert_tensor_buffer;
  ASSERT_EQ(LiteRtCreateManagedTensorBuffer(
                env.Get(), kLiteRtTensorBufferTypeHostMemory, &kTestTensorType,
                sizeof(kTensorData), &litert_tensor_buffer),
            kLiteRtStatusOk);

  TensorBuffer tensor_buffer(litert_tensor_buffer, litert::OwnHandle::kNo);
  ASSERT_EQ(tensor_buffer.Get(), litert_tensor_buffer);

  LiteRtDestroyTensorBuffer(litert_tensor_buffer);
}

TEST(TensorBuffer, CreateFromExternalHostMemory) {
  // Allocate a tensor buffer with host memory.
  const int kTensorBufferSize =
      std::max<int>(sizeof(kTensorData), LITERT_HOST_MEMORY_BUFFER_ALIGNMENT);
  const RankedTensorType kTensorType(kTestTensorType);
  void* host_memory_ptr;
  ASSERT_EQ(
      ::posix_memalign(&host_memory_ptr, LITERT_HOST_MEMORY_BUFFER_ALIGNMENT,
                       kTensorBufferSize),
      0);

  std::memcpy(host_memory_ptr, kTensorData, sizeof(kTensorData));

  // Create a tensor buffer that wraps the host memory.
  auto tensor_buffer_from_external_memory = TensorBuffer::CreateFromHostMemory(
      kTensorType, host_memory_ptr, kTensorBufferSize);

  auto lock_and_addr_external_memory = TensorBufferScopedLock::Create(
      *tensor_buffer_from_external_memory, TensorBuffer::LockMode::kWrite);
  ASSERT_TRUE(lock_and_addr_external_memory);
  ASSERT_EQ(std::memcmp(lock_and_addr_external_memory->second, kTensorData,
                        sizeof(kTensorData)),
            0);

  free(host_memory_ptr);
}

#if LITERT_HAS_AHWB_SUPPORT
TEST(TensorBuffer, CreateFromAhwb) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  AHardwareBuffer* ahw_buffer = nullptr;
  if (__builtin_available(android 26, *)) {
    int error = 0;
    AHardwareBuffer_Desc desc = {
        .width = LITERT_HOST_MEMORY_BUFFER_ALIGNMENT,
        .height = 1,
        .layers = 1,
        .format = AHARDWAREBUFFER_FORMAT_BLOB,
        .usage = AHARDWAREBUFFER_USAGE_CPU_WRITE_RARELY |
                 AHARDWAREBUFFER_USAGE_CPU_READ_RARELY};
    error = AHardwareBuffer_allocate(&desc, &ahw_buffer);
    ASSERT_EQ(error, 0);

    void* host_memory_ptr = nullptr;
    error =
        AHardwareBuffer_lock(ahw_buffer, AHARDWAREBUFFER_USAGE_CPU_WRITE_RARELY,
                             -1, nullptr, &host_memory_ptr);
    ASSERT_EQ(error, 0);

    std::memcpy(host_memory_ptr, kTensorData, sizeof(kTensorData));

    int fence_file_descriptor = -1;
    error = AHardwareBuffer_unlock(ahw_buffer, &fence_file_descriptor);
    ASSERT_EQ(error, 0);
  } else {
    GTEST_SKIP() << "AHardwareBuffers are not supported on this platform; "
                    "skipping the test";
  }

  {
    // Create a tensor buffer that wraps the AHardwareBuffer.
    const RankedTensorType kTensorType(kTestTensorType);
    auto tensor_buffer_from_ahwb =
        TensorBuffer::CreateFromAhwb(kTensorType, ahw_buffer,
                                     /*ahwb_offset=*/0);

    auto lock_and_addr_external_memory = TensorBufferScopedLock::Create(
        *tensor_buffer_from_ahwb, TensorBuffer::LockMode::kWrite);
    ASSERT_TRUE(lock_and_addr_external_memory);
    ASSERT_EQ(std::memcmp(lock_and_addr_external_memory->second, kTensorData,
                          sizeof(kTensorData)),
              0);
  }

  if (__builtin_available(android 26, *)) {
    AHardwareBuffer_release(ahw_buffer);
  }
}
#endif  // LITERT_HAS_AHWB_SUPPORT

TEST(TensorBuffer, Duplicate) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  LiteRtTensorBuffer litert_tensor_buffer;
  ASSERT_EQ(LiteRtCreateManagedTensorBuffer(
                env.Get(), kLiteRtTensorBufferTypeHostMemory, &kTestTensorType,
                sizeof(kTensorData), &litert_tensor_buffer),
            kLiteRtStatusOk);

  TensorBuffer tensor_buffer(litert_tensor_buffer, litert::OwnHandle::kYes);
  ASSERT_EQ(GetReferenceCount(tensor_buffer), 1);
  {
    auto duplicated_tensor_buffer = tensor_buffer.Duplicate();
    ASSERT_TRUE(duplicated_tensor_buffer);
    ASSERT_EQ(GetReferenceCount(*duplicated_tensor_buffer), 2);
    // The duplicated tensor buffer should point to the same underlying
    // LiteRtTensorBuffer object.
    ASSERT_EQ(duplicated_tensor_buffer->Get(), tensor_buffer.Get());

    // Update tensor buffer using the duplicated tensor buffer.
    auto lock_and_addr = TensorBufferScopedLock::Create(
        *duplicated_tensor_buffer, TensorBuffer::LockMode::kWrite);
    ASSERT_TRUE(lock_and_addr);
    std::memcpy(lock_and_addr->second, kTensorData, sizeof(kTensorData));

    // When the scope ends, the duplicated tensor buffer should be destroyed.
    // This should not affect the original tensor buffer.
  }

  ASSERT_EQ(GetReferenceCount(tensor_buffer), 1);
  // Check that the original tensor buffer is not affected.
  {
    auto lock_and_addr = TensorBufferScopedLock::Create(
        tensor_buffer, TensorBuffer::LockMode::kRead);
    ASSERT_TRUE(lock_and_addr);
    ASSERT_EQ(
        std::memcmp(lock_and_addr->second, kTensorData, sizeof(kTensorData)),
        0);
  }
}

TEST(TensorBuffer, ReadWriteBasic) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  LiteRtTensorBuffer litert_tensor_buffer;
  ASSERT_EQ(LiteRtCreateManagedTensorBuffer(
                env.Get(), kLiteRtTensorBufferTypeHostMemory, &kTestTensorType,
                sizeof(kTensorData), &litert_tensor_buffer),
            kLiteRtStatusOk);

  TensorBuffer tensor_buffer(litert_tensor_buffer, litert::OwnHandle::kYes);
  auto write_success = tensor_buffer.Write<float>(absl::MakeSpan(
      kTensorData, sizeof(kTensorData) / sizeof(kTensorData[0])));
  ASSERT_TRUE(write_success);
  float read_data[sizeof(kTensorData) / sizeof(kTensorData[0])];
  auto read_success = tensor_buffer.Read<float>(absl::MakeSpan(read_data));
  ASSERT_TRUE(read_success);
  ASSERT_EQ(std::memcmp(read_data, kTensorData, sizeof(kTensorData)), 0);
}

TEST(TensorBuffer, ReadWriteBufferSizeMismatch) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer tensor_buffer,
      TensorBuffer::CreateManaged(env.Get(), kLiteRtTensorBufferTypeHostMemory,
                                  RankedTensorType(kTestTensorType),
                                  sizeof(kTensorData)));
  {
    // Write with smaller size of data.
    auto write_success =
        tensor_buffer.Write<float>(absl::MakeSpan(kTensorData, 1));
    ASSERT_TRUE(write_success);
  }
  {
    constexpr const float big_data[] = {10, 20, 30, 40, 50};
    // Write with larger size of data.
    auto write_success =
        tensor_buffer.Write<float>(absl::MakeSpan(big_data, 5));
    ASSERT_FALSE(write_success);
  }
  auto write_success = tensor_buffer.Write<float>(absl::MakeSpan(
      kTensorData, sizeof(kTensorData) / sizeof(kTensorData[0])));
  ASSERT_TRUE(write_success);
  {
    // Read with smaller size of buffer.
    float read_data[1];
    auto read_success = tensor_buffer.Read<float>(absl::MakeSpan(read_data, 1));
    ASSERT_TRUE(read_success);
    ASSERT_EQ(read_data[0], kTensorData[0]);
  }
  {
    // Read with larger size of buffer.
    float read_data[5];
    auto read_success = tensor_buffer.Read<float>(absl::MakeSpan(read_data, 5));
    ASSERT_FALSE(read_success);
  }
}

#if LITERT_HAS_OPENGL_SUPPORT
TEST(TensorBuffer, CreateFromGlTexture) {
  // User provides EGL environment.
  auto user_gpu_env = UserGpuEnvironment::Create();

  // Create GL texture.
  tflite::gpu::gl::GlTexture gl_texture(GL_TEXTURE_2D, 1, GL_RGBA8, 1, 1,
                                        /*has_ownership=*/true);
  ASSERT_TRUE(gl_texture.is_valid());

  // Create tensor buffer from existing GL texture (e.g. this could be from
  // Android Camera API).
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer tensor_buffer,
      TensorBuffer::CreateFromGlTexture(
          user_gpu_env->GetEnvironment(), RankedTensorType(kTestTensorType),
          gl_texture.target(), gl_texture.id(), gl_texture.format(),
          gl_texture.bytes_size(), gl_texture.layer()));
}

tflite::gpu::gl::GlBuffer CreateTestGlBuffer(size_t size_bytes) {
  tflite::gpu::gl::GlBuffer gl_buffer;
  CHECK_OK(tflite::gpu::gl::CreateReadWriteShaderStorageBuffer<std::byte>(
      size_bytes, &gl_buffer));
  return gl_buffer;
}

TEST(TensorBuffer, CreateFromGlBuffer) {
  // User provides EGL environment.
  auto user_gpu_env = UserGpuEnvironment::Create();

  // Create GL buffer.
  tflite::gpu::gl::GlBuffer gl_buffer = CreateTestGlBuffer(sizeof(kTensorData));
  EXPECT_TRUE(gl_buffer.is_valid());
  EXPECT_EQ(gl_buffer.target(), GL_SHADER_STORAGE_BUFFER);

  // Create tensor buffer from existing GL buffer.
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer tensor_buffer,
      TensorBuffer::CreateFromGlBuffer(
          user_gpu_env->GetEnvironment(), RankedTensorType(kTestTensorType),
          gl_buffer.target(), gl_buffer.id(), gl_buffer.bytes_size(),
          gl_buffer.offset()));

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer::GlBuffer gl_buffer_from_tensor_buffer,
      tensor_buffer.GetGlBuffer());
  EXPECT_THAT(gl_buffer_from_tensor_buffer.target, Eq(gl_buffer.target()));
  EXPECT_THAT(gl_buffer_from_tensor_buffer.id, Eq(gl_buffer.id()));
  EXPECT_THAT(gl_buffer_from_tensor_buffer.size_bytes,
              Eq(gl_buffer.bytes_size()));
  EXPECT_THAT(gl_buffer_from_tensor_buffer.offset, Eq(gl_buffer.offset()));
}

TEST(TensorBuffer, CreateManagedGlBuffer) {
  // User provides EGL environment.
  auto user_gpu_env = UserGpuEnvironment::Create();

  // TensorBuffer::CreateManaged() is usually used with CompiledModel which
  // initializes the GPU environment. If there is no CompiledModel, user needs
  // to provide the GPU environment via LiteRtEnvironment.
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer tensor_buffer,
      TensorBuffer::CreateManaged(
          user_gpu_env->GetEnvironment(), kLiteRtTensorBufferTypeGlBuffer,
          RankedTensorType(kTestTensorType), sizeof(kTensorData)));
  LITERT_ASSERT_OK_AND_ASSIGN(TensorBuffer::GlBuffer gl_buffer,
                              tensor_buffer.GetGlBuffer());
  EXPECT_THAT(gl_buffer.target, Eq(GL_SHADER_STORAGE_BUFFER));
  EXPECT_THAT(gl_buffer.id, Ne(0));
  EXPECT_THAT(gl_buffer.size_bytes, Eq(sizeof(kTensorData)));
  EXPECT_THAT(gl_buffer.offset, Eq(0));
}

TEST(TensorBuffer, ClBufferFromGlBuffer) {
  // User provides EGL environment.
  auto user_gpu_env = UserGpuEnvironment::Create();

  // TensorBuffer::CreateManaged() is usually used with CompiledModel which
  // initializes the GPU environment. If there is no CompiledModel, user needs
  // to provide the GPU environment via LiteRtEnvironment.

  // TODO(b/383176413) Add check for GLSharing.
  if (!HasOpenClSupport() || !HasOpenGlSupport()) {
    GTEST_SKIP() << "OpenCL and/or GL are not supported on this platform; "
                    "skipping the test";
  }
  // Create GL Tensor buffer.
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer gl_tensor_buffer,
      TensorBuffer::CreateManaged(
          user_gpu_env->GetEnvironment(), kLiteRtTensorBufferTypeGlBuffer,
          RankedTensorType(kTestTensorType), sizeof(kTensorData)));

  LITERT_ASSERT_OK_AND_ASSIGN(cl_mem cl_buffer,
                              gl_tensor_buffer.GetOpenClMemory());
  EXPECT_THAT(cl_buffer, Ne(nullptr));
}

#if LITERT_HAS_AHWB_SUPPORT
TEST(TensorBuffer, GetGlBufferFromAhwb) {
  // User provides EGL environment.
  auto user_gpu_env = UserGpuEnvironment::Create();

  // Create AHWB Tensor buffer.
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer ahwb_tensor_buffer,
      TensorBuffer::CreateManaged(
          user_gpu_env->GetEnvironment(), kLiteRtTensorBufferTypeAhwb,
          RankedTensorType(kTestTensorType), sizeof(kTensorData)));

  // Write to AHWB Tensor buffer.
  LITERT_ASSERT_OK(ahwb_tensor_buffer.Write<float>(absl::MakeConstSpan(
      kTensorData, sizeof(kTensorData) / sizeof(kTensorData[0]))));

  LITERT_ASSERT_OK_AND_ASSIGN(TensorBuffer::GlBuffer gl_buffer,
                              ahwb_tensor_buffer.GetGlBuffer());
  EXPECT_THAT(gl_buffer.target, Eq(GL_SHADER_STORAGE_BUFFER));
  EXPECT_THAT(gl_buffer.id, Ne(0));
  EXPECT_THAT(gl_buffer.size_bytes, Eq(sizeof(kTensorData)));
  EXPECT_THAT(gl_buffer.offset, Eq(0));

  // Read from GL buffer.
  // TODO(gcarranza): Add GlBuffer ReadLock functionality to LiteRT
  // TensorBuffer. GlBuffer::Unlock currently writes to GL buffer.
  tflite::gpu::gl::GlBuffer gl_buffer_from_ahwb(
      gl_buffer.target, gl_buffer.id, gl_buffer.size_bytes, gl_buffer.offset,
      /*has_ownership=*/false);
  float read_data[sizeof(kTensorData) / sizeof(kTensorData[0])];
  auto status = gl_buffer_from_ahwb.Read<float>(absl::MakeSpan(read_data));
  ASSERT_TRUE(status.ok());
  ASSERT_EQ(std::memcmp(read_data, kTensorData, sizeof(kTensorData)), 0);
}
#endif  // LITERT_HAS_AHWB_SUPPORT

TEST(TensorBuffer, GetClBufferFromAhwb) {
  if (!HasOpenClSupport() || !HasAhwbSupport()) {
    GTEST_SKIP() << "OpenCL and/or AHWB are not supported on this platform; "
                    "skipping the "
                    "test";
  }
  // User provides EGL environment.
  auto user_gpu_env = UserGpuEnvironment::Create();
  // Create AHWB Tensor buffer.
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer ahwb_tensor_buffer,
      TensorBuffer::CreateManaged(
          user_gpu_env->GetEnvironment(), kLiteRtTensorBufferTypeAhwb,
          RankedTensorType(kTestTensorType), sizeof(kTensorData)));

  // Write to AHWB Tensor buffer.
  LITERT_ASSERT_OK(ahwb_tensor_buffer.Write<float>(absl::MakeConstSpan(
      kTensorData, sizeof(kTensorData) / sizeof(kTensorData[0]))));

  LITERT_ASSERT_OK_AND_ASSIGN(cl_mem cl_buffer,
                              ahwb_tensor_buffer.GetOpenClMemory());
  EXPECT_THAT(cl_buffer, Ne(nullptr));

  // Read from CL buffer.
  // TODO(gcarranza): Add ClBuffer ReadLock functionality to LiteRT
  // TensorBuffer. ClBuffer::Unlock currently writes to CL buffer.

  tflite::gpu::cl::Buffer cl_buffer_from_ahwb(cl_buffer, sizeof(kTensorData));

  tflite::gpu::cl::CLCommandQueue* queue = user_gpu_env->GetCommandQueue();
  std::vector<float> read_data;
  auto status = cl_buffer_from_ahwb.ReadData(queue, &read_data);
  ASSERT_TRUE(status.ok());
  ASSERT_EQ(std::memcmp(read_data.data(), kTensorData, sizeof(kTensorData)), 0);
}
#endif  // LITERT_HAS_OPENGL_SUPPORT

TEST(TensorBuffer, GetAhwb) {
  if (!HasAhwbSupport()) {
    GTEST_SKIP() << "AHardwareBuffers are not supported on this platform; "
                    "skipping the test";
  }
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer tensor_buffer,
      TensorBuffer::CreateManaged(env.Get(), kLiteRtTensorBufferTypeAhwb,
                                  RankedTensorType(kTestTensorType),
                                  sizeof(kTensorData)));
  LITERT_ASSERT_OK_AND_ASSIGN(AHardwareBuffer * ahwb, tensor_buffer.GetAhwb());
  EXPECT_THAT(ahwb, Ne(nullptr));
}

TEST(TensorBuffer, Event) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer tensor_buffer,
      TensorBuffer::CreateManaged(env.Get(), kLiteRtTensorBufferTypeHostMemory,
                                  RankedTensorType(kTestTensorType),
                                  sizeof(kTensorData)));
  // Create event.
  LITERT_ASSERT_OK_AND_ASSIGN(
      Event event,
      Event::CreateFromSyncFenceFd(env.Get(), kFakeSyncFenceFd, true));
  // Move event into tensor buffer.
  LITERT_EXPECT_OK(tensor_buffer.SetEvent(std::move(event)));
  EXPECT_TRUE(tensor_buffer.HasEvent());
  LITERT_ASSERT_OK_AND_ASSIGN(Event tensor_buffer_event,
                              tensor_buffer.GetEvent());
  LITERT_ASSERT_OK_AND_ASSIGN(int fence_fd,
                              tensor_buffer_event.GetSyncFenceFd());
  EXPECT_THAT(fence_fd, Eq(kFakeSyncFenceFd));
  // Clear event.
  LITERT_ASSERT_OK(tensor_buffer.ClearEvent());
  EXPECT_FALSE(tensor_buffer.HasEvent());
}

}  // namespace
}  // namespace litert
