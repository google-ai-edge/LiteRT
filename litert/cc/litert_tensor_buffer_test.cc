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
#include <cstring>
#include <memory>  // NOLINT: Used for OpenCL logic.
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>  // NOLINT: Need when ANDROID_API_LEVEL >= 26
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/internal/litert_platform_support.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_event.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_types.h"
#include "litert/runtime/tensor_buffer.h"
#include "litert/test/matchers.h"

#if LITERT_HAS_AHWB_SUPPORT
#include <android/hardware_buffer.h>
#endif  // LITERT_HAS_AHWB_SUPPORT

#if LITERT_HAS_OPENCL_SUPPORT
#include "tflite/delegates/gpu/cl/cl_command_queue.h"
#include "tflite/delegates/gpu/cl/cl_context.h"
#include "tflite/delegates/gpu/cl/cl_device.h"
#include "tflite/delegates/gpu/cl/opencl_wrapper.h"
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
class GlEnvironment {
 public:
  explicit GlEnvironment(
      std::unique_ptr<tflite::gpu::gl::EglEnvironment> egl_env)
      : egl_env_(std::move(egl_env)) {}

  static std::unique_ptr<GlEnvironment> Create() {
    std::unique_ptr<tflite::gpu::gl::EglEnvironment> egl_env;
    if (tflite::gpu::gl::EglEnvironment::NewEglEnvironment(&egl_env).ok()) {
      return std::make_unique<GlEnvironment>(std::move(egl_env));
    }
    return nullptr;
  }

  std::vector<litert::Environment::Option> GetEnvironmentOptions() {
    std::vector<litert::Environment::Option> environment_options;
    environment_options.push_back(litert::Environment::Option{
        OptionTag::EglDisplay,
        reinterpret_cast<int64_t>(egl_env_->display()),
    });
    environment_options.push_back(litert::Environment::Option{
        OptionTag::EglContext,
        reinterpret_cast<int64_t>(egl_env_->context().context()),
    });
    return environment_options;
  }
  tflite::gpu::gl::EglEnvironment& GetEglEnvironment() { return *egl_env_; }

 private:
  std::unique_ptr<tflite::gpu::gl::EglEnvironment> egl_env_;
};
#else
class GlEnvironment {
 public:
  static std::unique_ptr<GlEnvironment> Create() { return nullptr; }
  std::vector<litert::Environment::Option> GetEnvironmentOptions() {
    return {};
  }
};
#endif  // LITERT_HAS_OPENGL_SUPPORT

#if LITERT_HAS_OPENCL_SUPPORT
class ClEnvironment {
 public:
  explicit ClEnvironment(
      std::unique_ptr<tflite::gpu::cl::CLDevice> device,
      std::unique_ptr<tflite::gpu::cl::CLContext> context,
      std::unique_ptr<tflite::gpu::cl::CLCommandQueue> command_queue)
      : device_(std::move(device)),
        context_(std::move(context)),
        command_queue_(std::move(command_queue)) {}

  static std::unique_ptr<ClEnvironment> Create(GlEnvironment* gl_env) {
    auto device = std::make_unique<tflite::gpu::cl::CLDevice>();
    auto context = std::make_unique<tflite::gpu::cl::CLContext>();
    auto command_queue = std::make_unique<tflite::gpu::cl::CLCommandQueue>();
    if (tflite::gpu::cl::LoadOpenCL().ok()) {
      ABSL_CHECK_OK(tflite::gpu::cl::CreateDefaultGPUDevice(device.get()));
      ABSL_CHECK(CreateContext(gl_env, *device, context.get()));
      ABSL_CHECK_OK(tflite::gpu::cl::CreateCLCommandQueue(*device, *context,
                                                          command_queue.get()));
      return std::make_unique<ClEnvironment>(
          std::move(device), std::move(context), std::move(command_queue));
    }
    return nullptr;
  }

  std::vector<litert::Environment::Option> GetEnvironmentOptions() {
    std::vector<litert::Environment::Option> environment_options;
    environment_options.push_back(litert::Environment::Option{
        OptionTag::ClDeviceId,
        reinterpret_cast<int64_t>(device_->id()),
    });
    environment_options.push_back(litert::Environment::Option{
        OptionTag::ClPlatformId,
        reinterpret_cast<int64_t>(device_->platform()),
    });
    environment_options.push_back(litert::Environment::Option{
        OptionTag::ClContext,
        reinterpret_cast<int64_t>(context_->context()),
    });
    environment_options.push_back(litert::Environment::Option{
        OptionTag::ClCommandQueue,
        reinterpret_cast<int64_t>(command_queue_->queue()),
    });
    return environment_options;
  }

 private:
  static Expected<void> CreateContext(GlEnvironment* gl_env,
                                      tflite::gpu::cl::CLDevice& device,
                                      tflite::gpu::cl::CLContext* context) {
    if (gl_env == nullptr) {
      if (!tflite::gpu::cl::CreateCLContext(device, context).ok()) {
        return litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                                  "Failed to create CL context");
      }
    } else {
#if LITERT_HAS_OPENGL_SUPPORT
      if (!tflite::gpu::cl::IsGlSharingSupported(device)) {
        if (!tflite::gpu::cl::CreateCLContext(device, context).ok()) {
          return litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                                    "Failed to create CL context");
        }
      } else {
        if (!tflite::gpu::cl::CreateCLGLContext(
                 device,
                 reinterpret_cast<cl_context_properties>(
                     gl_env->GetEglEnvironment().context().context()),
                 reinterpret_cast<cl_context_properties>(
                     gl_env->GetEglEnvironment().display()),
                 context)
                 .ok()) {
          return litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                                    "Failed to create CL-GL context");
        }
      }
#else
      return litert::Unexpected(
          kLiteRtStatusErrorInvalidArgument,
          "LiteRT OpenGL support is disabled but gl_env is not null.");
#endif  // LITERT_HAS_OPENGL_SUPPORT
    }
    return {};
  }

  std::unique_ptr<tflite::gpu::cl::CLDevice> device_;
  std::unique_ptr<tflite::gpu::cl::CLContext> context_;
  std::unique_ptr<tflite::gpu::cl::CLCommandQueue> command_queue_;
};
#else
class ClEnvironment {
 public:
  static std::unique_ptr<ClEnvironment> Create(GlEnvironment* gl_env) {
    return nullptr;
  }
  std::vector<litert::Environment::Option> GetEnvironmentOptions() {
    return {};
  }
};
#endif  // LITERT_HAS_OPENCL_SUPPORT

class UserGpuEnvironment {
 public:
  explicit UserGpuEnvironment(std::unique_ptr<GlEnvironment> gl_env,
                              std::unique_ptr<ClEnvironment> cl_env,
                              litert::Environment env)
      : gl_env_(std::move(gl_env)),
        cl_env_(std::move(cl_env)),
        env_(std::move(env)) {}

  static std::unique_ptr<UserGpuEnvironment> Create(bool create_gl_env = true) {
    std::vector<litert::Environment::Option> environment_options;

    std::unique_ptr<GlEnvironment> gl_env =
        create_gl_env ? GlEnvironment::Create() : nullptr;
    std::unique_ptr<ClEnvironment> cl_env = ClEnvironment::Create(gl_env.get());

    if (gl_env != nullptr) {
      auto gl_options = gl_env->GetEnvironmentOptions();
      environment_options.insert(environment_options.end(), gl_options.begin(),
                                 gl_options.end());
    }
    if (cl_env != nullptr) {
      auto cl_options = cl_env->GetEnvironmentOptions();
      environment_options.insert(environment_options.end(), cl_options.begin(),
                                 cl_options.end());
    }

    // Create LiteRt environment from GL and CL options.
    LITERT_ASSIGN_OR_ABORT(auto env,
                           litert::Environment::Create(environment_options));
    return std::make_unique<UserGpuEnvironment>(
        std::move(gl_env), std::move(cl_env), std::move(env));
  }

  litert::Environment& GetEnvironment() { return env_; }

 private:
  std::unique_ptr<GlEnvironment> gl_env_;
  std::unique_ptr<ClEnvironment> cl_env_;
  litert::Environment env_;
};

TEST(TensorBuffer, HostMemory) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  const RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = TensorBufferType::kHostMemory;

  auto tensor_buffer = TensorBuffer::CreateManaged(
      env, kTensorBufferType, kTensorType, sizeof(kTensorData));
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

bool CanLoadOpenCl() {
#if LITERT_HAS_OPENCL_SUPPORT
  return tflite::gpu::cl::LoadOpenCL().ok();
#else
  return false;
#endif
}

TEST(TensorBuffer, ClBuffer) {
  if (!HasOpenClSupport()) {
    GTEST_SKIP() << "OpenCL buffers are not supported on this platform; "
                    "skipping the test";
  }
  if (!CanLoadOpenCl()) {
    GTEST_SKIP() << "OpenCL library could not be loaded; skipping the test";
  }
  auto user_gpu_env = UserGpuEnvironment::Create(/*create_gl_env=*/false);

  const RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = TensorBufferType::kOpenClBuffer;

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto tensor_buffer, TensorBuffer::CreateManaged(
                              user_gpu_env->GetEnvironment(), kTensorBufferType,
                              kTensorType, sizeof(kTensorData)));

  auto tensor_buffer_type = tensor_buffer.BufferType();
  ASSERT_TRUE(tensor_buffer_type);
  ASSERT_EQ(*tensor_buffer_type, kTensorBufferType);

  auto tensor_type = tensor_buffer.TensorType();
  ASSERT_TRUE(tensor_type);

  ASSERT_EQ(tensor_type->ElementType(), ElementType::Float32);
  ASSERT_EQ(tensor_type->Layout().Rank(), 1);
  ASSERT_EQ(tensor_type->Layout().Dimensions()[0],
            kTensorType.Layout().Dimensions()[0]);
  ASSERT_FALSE(tensor_type->Layout().HasStrides());

  auto size = tensor_buffer.Size();
  ASSERT_TRUE(size);
  ASSERT_EQ(*size, sizeof(kTensorData));

  auto offset = tensor_buffer.Offset();
  ASSERT_TRUE(offset);
  ASSERT_EQ(*offset, 0);

  {
    auto lock_and_addr = TensorBufferScopedLock::Create(
        tensor_buffer, TensorBuffer::LockMode::kWrite);
    ASSERT_TRUE(lock_and_addr);
    std::memcpy(lock_and_addr->second, kTensorData, sizeof(kTensorData));
  }

  {
    auto lock_and_addr = TensorBufferScopedLock::Create(
        tensor_buffer, TensorBuffer::LockMode::kRead);
    ASSERT_TRUE(lock_and_addr);
    ASSERT_EQ(
        std::memcmp(lock_and_addr->second, kTensorData, sizeof(kTensorData)),
        0);
  }
}

TEST(TensorBuffer, DoubleLockOrUnlock) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  const RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = TensorBufferType::kHostMemory;

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto tensor_buffer,
      TensorBuffer::CreateManaged(env, kTensorBufferType, kTensorType,
                                  sizeof(kTensorData)));
  LITERT_EXPECT_OK(tensor_buffer.Lock());
  EXPECT_THAT(tensor_buffer.Lock(), IsError());
  LITERT_EXPECT_OK(tensor_buffer.Unlock());
  EXPECT_THAT(tensor_buffer.Unlock(), IsError());
}

TEST(TensorBuffer, TensorBufferScopedLock_Read) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  const RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = TensorBufferType::kHostMemory;

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto tensor_buffer,
      TensorBuffer::CreateManaged(env, kTensorBufferType, kTensorType,
                                  sizeof(kTensorData)));

  {
    auto lock_and_addr = TensorBufferScopedLock::Create(
        tensor_buffer, TensorBuffer::LockMode::kRead);
    LITERT_EXPECT_OK(lock_and_addr);
  }
}

TEST(TensorBuffer, TensorBufferScopedLock_Write) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  const RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = TensorBufferType::kHostMemory;

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto tensor_buffer,
      TensorBuffer::CreateManaged(env, kTensorBufferType, kTensorType,
                                  sizeof(kTensorData)));

  {
    auto lock_and_addr = TensorBufferScopedLock::Create(
        tensor_buffer, TensorBuffer::LockMode::kWrite);
    LITERT_EXPECT_OK(lock_and_addr);
  }
}

TEST(TensorBuffer, TensorBufferScopedLock_ReadWrite) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  const RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = TensorBufferType::kHostMemory;

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto tensor_buffer,
      TensorBuffer::CreateManaged(env, kTensorBufferType, kTensorType,
                                  sizeof(kTensorData)));

  {
    auto lock_and_addr = TensorBufferScopedLock::Create(
        tensor_buffer, TensorBuffer::LockMode::kReadWrite);
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
  constexpr auto kTensorBufferType = TensorBufferType::kAhwb;

  auto tensor_buffer = TensorBuffer::CreateManaged(
      env, kTensorBufferType, kTensorType, sizeof(kTensorData));
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
  // TODO: Ion is failing on Android.
  if (!HasIonSupport() || true) {
    GTEST_SKIP() << "ION buffers are not supported on this platform; "
                    "skipping the test";
  }
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));

  const RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = TensorBufferType::kIon;

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto tensor_buffer,
      TensorBuffer::CreateManaged(env, kTensorBufferType, kTensorType,
                                  sizeof(kTensorData)));

  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_buffer_type,
                              tensor_buffer.BufferType());
  ASSERT_EQ(tensor_buffer_type, kTensorBufferType);

  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_type, tensor_buffer.TensorType());

  ASSERT_EQ(tensor_type.ElementType(), ElementType::Float32);
  ASSERT_EQ(tensor_type.Layout().Rank(), 1);
  ASSERT_EQ(tensor_type.Layout().Dimensions()[0],
            kTensorType.Layout().Dimensions()[0]);
  ASSERT_FALSE(tensor_type.Layout().HasStrides());

  LITERT_ASSERT_OK_AND_ASSIGN(auto size, tensor_buffer.Size());
  ASSERT_EQ(size, sizeof(kTensorData));

  LITERT_ASSERT_OK_AND_ASSIGN(size_t offset, tensor_buffer.Offset());
  ASSERT_EQ(offset, 0);

  {
    auto lock_and_addr = TensorBufferScopedLock::Create(
        tensor_buffer, TensorBuffer::LockMode::kWrite);
    ASSERT_TRUE(lock_and_addr);
    std::memcpy(lock_and_addr->second, kTensorData, sizeof(kTensorData));
  }

  {
    auto lock_and_addr = TensorBufferScopedLock::Create(
        tensor_buffer, TensorBuffer::LockMode::kRead);
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
  constexpr auto kTensorBufferType = TensorBufferType::kDmaBuf;

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto tensor_buffer,
      TensorBuffer::CreateManaged(env, kTensorBufferType, kTensorType,
                                  sizeof(kTensorData)));

  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_buffer_type,
                              tensor_buffer.BufferType());
  ASSERT_EQ(tensor_buffer_type, kTensorBufferType);

  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_type, tensor_buffer.TensorType());

  ASSERT_EQ(tensor_type.ElementType(), ElementType::Float32);
  ASSERT_EQ(tensor_type.Layout().Rank(), 1);
  ASSERT_EQ(tensor_type.Layout().Dimensions()[0],
            kTensorType.Layout().Dimensions()[0]);
  ASSERT_FALSE(tensor_type.Layout().HasStrides());

  LITERT_ASSERT_OK_AND_ASSIGN(size_t size, tensor_buffer.Size());
  ASSERT_EQ(size, sizeof(kTensorData));

  LITERT_ASSERT_OK_AND_ASSIGN(auto offset, tensor_buffer.Offset());
  ASSERT_EQ(offset, 0);

  {
    auto lock_and_addr = TensorBufferScopedLock::Create(
        tensor_buffer, TensorBuffer::LockMode::kWrite);
    ASSERT_TRUE(lock_and_addr);
    std::memcpy(lock_and_addr->second, kTensorData, sizeof(kTensorData));
  }

  {
    auto lock_and_addr = TensorBufferScopedLock::Create(
        tensor_buffer, TensorBuffer::LockMode::kRead);
    ASSERT_TRUE(lock_and_addr);
    ASSERT_EQ(
        std::memcmp(lock_and_addr->second, kTensorData, sizeof(kTensorData)),
        0);
  }
}

TEST(TensorBuffer, FastRpc) {
  // TODO: FastRPC is failing on Android.
  if (!HasFastRpcSupport() || true) {
    GTEST_SKIP()
        << "FastRPC buffers are not supported on this platform; skipping "
           "the test";
  }
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));

  const RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = TensorBufferType::kFastRpc;

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto tensor_buffer,
      TensorBuffer::CreateManaged(env, kTensorBufferType, kTensorType,
                                  sizeof(kTensorData)));

  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_buffer_type,
                              tensor_buffer.BufferType());
  ASSERT_EQ(tensor_buffer_type, kTensorBufferType);

  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_type, tensor_buffer.TensorType());

  ASSERT_EQ(tensor_type.ElementType(), ElementType::Float32);
  ASSERT_EQ(tensor_type.Layout().Rank(), 1);
  ASSERT_EQ(tensor_type.Layout().Dimensions()[0],
            kTensorType.Layout().Dimensions()[0]);
  ASSERT_FALSE(tensor_type.Layout().HasStrides());

  LITERT_ASSERT_OK_AND_ASSIGN(size_t size, tensor_buffer.Size());
  ASSERT_EQ(size, sizeof(kTensorData));

  LITERT_ASSERT_OK_AND_ASSIGN(size_t offset, tensor_buffer.Offset());
  ASSERT_EQ(offset, 0);

  {
    auto lock_and_addr = TensorBufferScopedLock::Create(
        tensor_buffer, TensorBuffer::LockMode::kWrite);
    ASSERT_TRUE(lock_and_addr);
    std::memcpy(lock_and_addr->second, kTensorData, sizeof(kTensorData));
  }

  {
    auto lock_and_addr = TensorBufferScopedLock::Create(
        tensor_buffer, TensorBuffer::LockMode::kRead);
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
                env.GetHolder().handle, kLiteRtTensorBufferTypeHostMemory,
                &kTestTensorType, sizeof(kTensorData), &litert_tensor_buffer),
            kLiteRtStatusOk);

  TensorBuffer tensor_buffer = TensorBuffer::WrapCObject(
      env.GetHolder(), litert_tensor_buffer, litert::OwnHandle::kNo);
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

  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  // Create a tensor buffer that wraps the host memory.
  auto tensor_buffer_from_external_memory = TensorBuffer::CreateFromHostMemory(
      env, kTensorType, host_memory_ptr, kTensorBufferSize);

  auto lock_and_addr_external_memory = TensorBufferScopedLock::Create(
      *tensor_buffer_from_external_memory, TensorBuffer::LockMode::kWrite);
  ASSERT_TRUE(lock_and_addr_external_memory);
  ASSERT_EQ(std::memcmp(lock_and_addr_external_memory->second, kTensorData,
                        sizeof(kTensorData)),
            0);

  litert_aligned_free(host_memory_ptr);
}

AHardwareBuffer* CreateTestAhwb() {
#if LITERT_HAS_AHWB_SUPPORT
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
    if (error != 0) {
      LOG(ERROR) << "Failed to allocate AHardwareBuffer: " << error;
      return nullptr;
    }

    void* host_memory_ptr = nullptr;
    error =
        AHardwareBuffer_lock(ahw_buffer, AHARDWAREBUFFER_USAGE_CPU_WRITE_RARELY,
                             -1, nullptr, &host_memory_ptr);
    if (error != 0) {
      LOG(ERROR) << "Failed to lock AHardwareBuffer: " << error;
      return nullptr;
    }

    std::memcpy(host_memory_ptr, kTensorData, sizeof(kTensorData));

    int fence_file_descriptor = -1;
    error = AHardwareBuffer_unlock(ahw_buffer, &fence_file_descriptor);
    if (error != 0) {
      LOG(ERROR) << "Failed to unlock AHardwareBuffer: " << error;
      return nullptr;
    }
    return ahw_buffer;
  }
  return nullptr;
#else
  return nullptr;
#endif  // LITERT_HAS_AHWB_SUPPORT
}

void ReleaseTestAhwb(AHardwareBuffer* ahw_buffer) {
#if LITERT_HAS_AHWB_SUPPORT
  if (__builtin_available(android 26, *)) {
    AHardwareBuffer_release(ahw_buffer);
  }
#endif
}

TEST(TensorBuffer, CreateFromAhwb) {
  if (!HasAhwbSupport()) {
    GTEST_SKIP() << "AHardwareBuffers are not supported on this platform; "
                    "skipping the test";
  }
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));

  AHardwareBuffer* ahw_buffer = CreateTestAhwb();
  ASSERT_NE(ahw_buffer, nullptr);

  // Create a tensor buffer that wraps the AHardwareBuffer.
  const RankedTensorType kTensorType(kTestTensorType);
  auto tensor_buffer_from_ahwb =
      TensorBuffer::CreateFromAhwb(env, kTensorType, ahw_buffer,
                                   /*ahwb_offset=*/0);

  auto lock_and_addr_external_memory = TensorBufferScopedLock::Create(
      *tensor_buffer_from_ahwb, TensorBuffer::LockMode::kWrite);
  ASSERT_TRUE(lock_and_addr_external_memory);
  ASSERT_EQ(std::memcmp(lock_and_addr_external_memory->second, kTensorData,
                        sizeof(kTensorData)),
            0);

  ReleaseTestAhwb(ahw_buffer);
}

TEST(TensorBuffer, Duplicate) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  LiteRtTensorBuffer litert_tensor_buffer;
  ASSERT_EQ(LiteRtCreateManagedTensorBuffer(
                env.GetHolder().handle, kLiteRtTensorBufferTypeHostMemory,
                &kTestTensorType, sizeof(kTensorData), &litert_tensor_buffer),
            kLiteRtStatusOk);

  TensorBuffer tensor_buffer = TensorBuffer::WrapCObject(
      env.GetHolder(), litert_tensor_buffer, litert::OwnHandle::kYes);
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
                env.GetHolder().handle, kLiteRtTensorBufferTypeHostMemory,
                &kTestTensorType, sizeof(kTensorData), &litert_tensor_buffer),
            kLiteRtStatusOk);

  TensorBuffer tensor_buffer = TensorBuffer::WrapCObject(
      env.GetHolder(), litert_tensor_buffer, litert::OwnHandle::kYes);
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
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory,
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

TEST(TensorBuffer, ClBufferFromGlBuffer) {
  if (!HasOpenClSupport() || !HasOpenGlSupport()) {
    GTEST_SKIP() << "OpenCL and/or GL are not supported on this platform; "
                    "skipping the test";
  }
  // User provides CL-GL environment.
  auto user_gpu_env = UserGpuEnvironment::Create();
  ASSERT_TRUE(user_gpu_env != nullptr);
  ASSERT_TRUE(user_gpu_env->GetEnvironment().GetHolder().handle != nullptr);
  bool is_cl_gl_sharing_supported = false;
  ASSERT_EQ(LiteRtEnvironmentSupportsClGlInterop(
                user_gpu_env->GetEnvironment().GetHolder().handle,
                &is_cl_gl_sharing_supported),
            kLiteRtStatusOk);

  if (!is_cl_gl_sharing_supported) {
    GTEST_SKIP() << "CL/GL sharing is not supported on this platform; "
                    "skipping the test";
  }

  // Create GL Tensor buffer.
  // TensorBuffer::CreateManaged() is usually used with CompiledModel which
  // initializes the GPU environment. If there is no CompiledModel, user needs
  // to provide the GPU environment via LiteRtEnvironment.
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer gl_tensor_buffer,
      TensorBuffer::CreateManaged(
          user_gpu_env->GetEnvironment(), TensorBufferType::kGlBuffer,
          RankedTensorType(kTestTensorType), sizeof(kTensorData)));

  LITERT_ASSERT_OK_AND_ASSIGN(auto cl_buffer,
                              gl_tensor_buffer.GetOpenClMemory());
  EXPECT_THAT(cl_buffer, Ne(nullptr));
}

TEST(TensorBuffer, CreateFromGlTexture) {
  if (!HasOpenGlSupport()) {
    GTEST_SKIP() << "GL is not supported on this platform; skipping the test";
  }
  // User provides EGL environment.
  auto user_gpu_env = UserGpuEnvironment::Create();

  // Create GL texture.
  // TODO: Remove ifdef after managed gl texture is supported.
#if LITERT_HAS_OPENGL_SUPPORT
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
#endif  // LITERT_HAS_OPENGL_SUPPORT
}

TEST(TensorBuffer, CreateManagedGlBuffer) {
  if (!HasOpenGlSupport()) {
    GTEST_SKIP() << "GL is not supported on this platform; skipping the test";
  }
  // User provides EGL environment.
  auto user_gpu_env = UserGpuEnvironment::Create();

  // TensorBuffer::CreateManaged() is usually used with CompiledModel which
  // initializes the GPU environment. If there is no CompiledModel, user needs
  // to provide the GPU environment via LiteRtEnvironment.
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer tensor_buffer,
      TensorBuffer::CreateManaged(
          user_gpu_env->GetEnvironment(), TensorBufferType::kGlBuffer,
          RankedTensorType(kTestTensorType), sizeof(kTensorData)));
  LITERT_ASSERT_OK_AND_ASSIGN(TensorBuffer::GlBuffer gl_buffer,
                              tensor_buffer.GetGlBuffer());
#if LITERT_HAS_OPENGL_SUPPORT
  EXPECT_THAT(gl_buffer.target, Eq(GL_SHADER_STORAGE_BUFFER));
#endif  // LITERT_HAS_OPENGL_SUPPORT
  EXPECT_THAT(gl_buffer.id, Ne(0));
  EXPECT_THAT(gl_buffer.size_bytes, Eq(sizeof(kTensorData)));
  EXPECT_THAT(gl_buffer.offset, Eq(0));
}

TEST(TensorBuffer, CreateFromGlBuffer) {
  if (!HasOpenGlSupport()) {
    GTEST_SKIP()
        << "OpenGl is not supported on this platform; skipping the test";
  }
  // User provides EGL environment.
  auto user_gpu_env = UserGpuEnvironment::Create();

  // TensorBuffer::CreateManaged() is usually used with CompiledModel which
  // initializes the GPU environment. If there is no CompiledModel, user needs
  // to provide the GPU environment via LiteRtEnvironment.
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer tensor_buffer,
      TensorBuffer::CreateManaged(
          user_gpu_env->GetEnvironment(), TensorBufferType::kGlBuffer,
          RankedTensorType(kTestTensorType), sizeof(kTensorData)));
  LITERT_ASSERT_OK_AND_ASSIGN(TensorBuffer::GlBuffer gl_buffer,
                              tensor_buffer.GetGlBuffer());

  // Create tensor buffer from existing GL buffer.
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer tensor_buffer_from_gl_buffer,
      TensorBuffer::CreateFromGlBuffer(user_gpu_env->GetEnvironment(),
                                       RankedTensorType(kTestTensorType),
                                       gl_buffer.target, gl_buffer.id,
                                       gl_buffer.size_bytes, gl_buffer.offset));

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer::GlBuffer gl_buffer_from_tensor_buffer,
      tensor_buffer.GetGlBuffer());
  EXPECT_THAT(gl_buffer_from_tensor_buffer.target, Eq(gl_buffer.target));
  EXPECT_THAT(gl_buffer_from_tensor_buffer.id, Eq(gl_buffer.id));
  EXPECT_THAT(gl_buffer_from_tensor_buffer.size_bytes,
              Eq(gl_buffer.size_bytes));
  EXPECT_THAT(gl_buffer_from_tensor_buffer.offset, Eq(gl_buffer.offset));
}

TEST(TensorBuffer, GetGlBufferFromAhwb) {
  if (!HasOpenGlSupport() || !HasAhwbSupport()) {
    GTEST_SKIP() << "OpenGl and/or AHWB are not supported on this platform; "
                    "skipping the test";
  }
  // User provides EGL environment.
  auto user_gpu_env = UserGpuEnvironment::Create();
  ASSERT_TRUE(user_gpu_env != nullptr);
  ASSERT_TRUE(user_gpu_env->GetEnvironment().GetHolder().handle != nullptr);
  bool is_ahwb_gl_interop_supported = false;
  ASSERT_EQ(LiteRtEnvironmentSupportsAhwbGlInterop(
                user_gpu_env->GetEnvironment().GetHolder().handle,
                &is_ahwb_gl_interop_supported),
            kLiteRtStatusOk);
  if (!is_ahwb_gl_interop_supported) {
    GTEST_SKIP() << "AHWB/GL interop is not supported on this platform; "
                    "skipping the test";
  }

  // Create AHWB Tensor buffer.
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer ahwb_tensor_buffer,
      TensorBuffer::CreateManaged(
          user_gpu_env->GetEnvironment(), TensorBufferType::kAhwb,
          RankedTensorType(kTestTensorType), sizeof(kTensorData)));

  // Write to AHWB Tensor buffer.
  LITERT_ASSERT_OK(ahwb_tensor_buffer.Write<float>(absl::MakeConstSpan(
      kTensorData, sizeof(kTensorData) / sizeof(kTensorData[0]))));

  LITERT_ASSERT_OK_AND_ASSIGN(TensorBuffer::GlBuffer gl_buffer,
                              ahwb_tensor_buffer.GetGlBuffer());
#if LITERT_HAS_OPENGL_SUPPORT
  EXPECT_THAT(gl_buffer.target, Eq(GL_SHADER_STORAGE_BUFFER));
#endif  // LITERT_HAS_OPENGL_SUPPORT
  EXPECT_THAT(gl_buffer.id, Ne(0));
  EXPECT_THAT(gl_buffer.size_bytes, Eq(sizeof(kTensorData)));
  EXPECT_THAT(gl_buffer.offset, Eq(0));

  // Read from GL buffer.
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer gl_buffer_from_ahwb,
      TensorBuffer::CreateFromGlBuffer(user_gpu_env->GetEnvironment(),
                                       RankedTensorType(kTestTensorType),
                                       gl_buffer.target, gl_buffer.id,
                                       gl_buffer.size_bytes, gl_buffer.offset));
  float read_data[sizeof(kTensorData) / sizeof(kTensorData[0])];
  LITERT_ASSERT_OK(gl_buffer_from_ahwb.Read<float>(absl::MakeSpan(read_data)));
  ASSERT_EQ(std::memcmp(read_data, kTensorData, sizeof(kTensorData)), 0);
}

TEST(TensorBuffer, CreateManagedClBuffer) {
  if (!HasOpenClSupport()) {
    GTEST_SKIP()
        << "OpenCL is not supported on this platform; skipping the test";
  }
  if (!CanLoadOpenCl()) {
    GTEST_SKIP() << "OpenCL library could not be loaded; skipping the test";
  }

  auto user_gpu_env = UserGpuEnvironment::Create(/*create_gl_env=*/false);

  // TensorBuffer::CreateManaged() is usually used with CompiledModel which
  // initializes the GPU environment. If there is no CompiledModel, user needs
  // to provide the GPU environment via LiteRtEnvironment.
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer tensor_buffer,
      TensorBuffer::CreateManaged(
          user_gpu_env->GetEnvironment(), TensorBufferType::kOpenClBuffer,
          RankedTensorType(kTestTensorType), sizeof(kTensorData)));
  LITERT_ASSERT_OK_AND_ASSIGN(auto cl_buffer, tensor_buffer.GetOpenClMemory());
  EXPECT_THAT(cl_buffer, Ne(nullptr));
}

TEST(TensorBuffer, CreateFromClBuffer) {
  if (!HasOpenClSupport()) {
    GTEST_SKIP() << "OpenCL is not supported on this platform; "
                    "skipping the test";
  }
  if (!CanLoadOpenCl()) {
    GTEST_SKIP() << "OpenCL library could not be loaded; skipping the test";
  }
  auto user_gpu_env = UserGpuEnvironment::Create(/*create_gl_env=*/false);

  const RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = TensorBufferType::kOpenClBuffer;

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto tensor_buffer, TensorBuffer::CreateManaged(
                              user_gpu_env->GetEnvironment(), kTensorBufferType,
                              kTensorType, sizeof(kTensorData)));

  LITERT_ASSERT_OK_AND_ASSIGN(auto cl_buffer, tensor_buffer.GetOpenClMemory());

  // Create a new tensor buffer with the same memory.
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto tensor_buffer_2,
      TensorBuffer::CreateFromClBuffer(user_gpu_env->GetEnvironment(),
                                       kTensorType, kTensorBufferType,
                                       cl_buffer, sizeof(kTensorData)));
  LITERT_ASSERT_OK_AND_ASSIGN(auto cl_buffer_2,
                              tensor_buffer_2.GetOpenClMemory());
  EXPECT_THAT(cl_buffer_2, Ne(nullptr));
  EXPECT_THAT(cl_buffer_2, Eq(cl_buffer));
}

TEST(TensorBuffer, GetClBufferFromAhwb) {
  if (!HasOpenClSupport() || !HasAhwbSupport()) {
    GTEST_SKIP() << "OpenCL and/or AHWB are not supported on this platform; "
                    "skipping the "
                    "test";
  }
  auto user_gpu_env = UserGpuEnvironment::Create(/*create_gl_env=*/false);
  ASSERT_TRUE(user_gpu_env != nullptr);
  ASSERT_TRUE(user_gpu_env->GetEnvironment().GetHolder().handle != nullptr);
  bool is_ahwb_cl_interop_supported = false;
  ASSERT_EQ(LiteRtEnvironmentSupportsAhwbClInterop(
                user_gpu_env->GetEnvironment().GetHolder().handle,
                &is_ahwb_cl_interop_supported),
            kLiteRtStatusOk);
  if (!is_ahwb_cl_interop_supported) {
    GTEST_SKIP() << "AHWB/CL interop is not supported on this platform; "
                    "skipping the test";
  }
  // Create AHWB Tensor buffer.
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer ahwb_tensor_buffer,
      TensorBuffer::CreateManaged(
          user_gpu_env->GetEnvironment(), TensorBufferType::kAhwb,
          RankedTensorType(kTestTensorType), sizeof(kTensorData)));

  // Write to AHWB Tensor buffer.
  LITERT_ASSERT_OK(ahwb_tensor_buffer.Write<float>(absl::MakeConstSpan(
      kTensorData, sizeof(kTensorData) / sizeof(kTensorData[0]))));

  LITERT_ASSERT_OK_AND_ASSIGN(cl_mem cl_buffer,
                              ahwb_tensor_buffer.GetOpenClMemory());
  EXPECT_THAT(cl_buffer, Ne(nullptr));

  // Read from CL buffer.
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer cl_buffer_from_ahwb,
      TensorBuffer::CreateFromClBuffer(user_gpu_env->GetEnvironment(),
                                       RankedTensorType(kTestTensorType),
                                       TensorBufferType::kOpenClBufferPacked,
                                       cl_buffer, sizeof(kTensorData)));

  {
    auto lock_and_addr = TensorBufferScopedLock::Create(
        cl_buffer_from_ahwb, TensorBuffer::LockMode::kRead);
    ASSERT_TRUE(lock_and_addr);
    ASSERT_EQ(
        std::memcmp(lock_and_addr->second, kTensorData, sizeof(kTensorData)),
        0);
  }
}

TEST(TensorBuffer, ClBufferWriteOnReadLockIsNoOp) {
  if (!HasOpenClSupport()) {
    GTEST_SKIP() << "OpenCL buffers are not supported on this platform; "
                    "skipping the test";
  }
  if (!CanLoadOpenCl()) {
    GTEST_SKIP() << "OpenCL library could not be loaded; skipping the test";
  }
  auto user_gpu_env = UserGpuEnvironment::Create(/*create_gl_env=*/false);

  const RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = TensorBufferType::kOpenClBuffer;

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto tensor_buffer, TensorBuffer::CreateManaged(
                              user_gpu_env->GetEnvironment(), kTensorBufferType,
                              kTensorType, sizeof(kTensorData)));

  // Write to the buffer with a read lock (should be no-op).
  float tensor_data[] = {0, 0, 0, 0};
  {
    auto lock_and_addr = TensorBufferScopedLock::Create(
        tensor_buffer, TensorBuffer::LockMode::kRead);
    ASSERT_TRUE(lock_and_addr);
    // Keep a copy of the data before writing.
    std::memcpy(tensor_data, lock_and_addr->second, sizeof(tensor_data));
    // Write to the buffer. This should not update the underlying buffer due to
    // LockMode::kRead.
    std::memcpy(lock_and_addr->second, kTensorData, sizeof(kTensorData));
  }

  // We expect the buffer to be unchanged due to write on read lock.
  {
    auto lock_and_addr = TensorBufferScopedLock::Create(
        tensor_buffer, TensorBuffer::LockMode::kRead);
    ASSERT_TRUE(lock_and_addr);
    ASSERT_EQ(
        std::memcmp(lock_and_addr->second, tensor_data, sizeof(tensor_data)),
        0);
  }
}

TEST(TensorBuffer, ClBufferReadOnWriteLockIsInvalid) {
  if (!HasOpenClSupport()) {
    GTEST_SKIP() << "OpenCL buffers are not supported on this platform; "
                    "skipping the test";
  }
  if (!CanLoadOpenCl()) {
    GTEST_SKIP() << "OpenCL library could not be loaded; skipping the test";
  }

  auto user_gpu_env = UserGpuEnvironment::Create(/*create_gl_env=*/false);

  const RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = TensorBufferType::kOpenClBuffer;

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto tensor_buffer, TensorBuffer::CreateManaged(
                              user_gpu_env->GetEnvironment(), kTensorBufferType,
                              kTensorType, sizeof(kTensorData)));

  {
    auto lock_and_addr = TensorBufferScopedLock::Create(
        tensor_buffer, TensorBuffer::LockMode::kWrite);
    ASSERT_TRUE(lock_and_addr);
    std::memcpy(lock_and_addr->second, kTensorData, sizeof(kTensorData));
  }

  LITERT_ASSERT_OK_AND_ASSIGN(auto memory, tensor_buffer.GetOpenClMemory());

  // Create a new tensor buffer with the same memory.
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto tensor_buffer_2,
      TensorBuffer::CreateFromClBuffer(user_gpu_env->GetEnvironment(),
                                       kTensorType, kTensorBufferType, memory,
                                       sizeof(kTensorData)));

  // Read on write lock is invalid, meaning that GPU memory that was previously
  // written to is not downloaded to host.
  {
    auto lock_and_addr = TensorBufferScopedLock::Create(
        tensor_buffer_2, TensorBuffer::LockMode::kWrite);
    ASSERT_TRUE(lock_and_addr);
    ASSERT_NE(
        std::memcmp(lock_and_addr->second, kTensorData, sizeof(kTensorData)),
        0);
  }
}

TEST(TensorBuffer, GetAhwb) {
  if (!HasAhwbSupport()) {
    GTEST_SKIP() << "AHardwareBuffers are not supported on this platform; "
                    "skipping the test";
  }
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer tensor_buffer,
      TensorBuffer::CreateManaged(env, TensorBufferType::kAhwb,
                                  RankedTensorType(kTestTensorType),
                                  sizeof(kTensorData)));
  LITERT_ASSERT_OK_AND_ASSIGN(AHardwareBuffer * ahwb, tensor_buffer.GetAhwb());
  EXPECT_THAT(ahwb, Ne(nullptr));
}

TEST(TensorBuffer, Event) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer tensor_buffer,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory,
                                  RankedTensorType(kTestTensorType),
                                  sizeof(kTensorData)));
  // Create event.
  LITERT_ASSERT_OK_AND_ASSIGN(
      Event event, Event::CreateFromSyncFenceFd(env, kFakeSyncFenceFd, true));
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
