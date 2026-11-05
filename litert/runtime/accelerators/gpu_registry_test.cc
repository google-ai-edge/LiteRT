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

#include "litert/runtime/accelerators/gpu_registry.h"

#include <filesystem>  // NOLINT
#include <memory>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "litert/c/internal/litert_accelerator_def.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/core/environment.h"
#include "litert/runtime/accelerators/gpu_static_registry.h"
#include "litert/test/common.h"

namespace litert::internal {
namespace {

#if defined(LITERT_WINDOWS_OS)
#define SO_EXT ".dll"
#elif defined(__APPLE__)
#define SO_EXT ".dylib"
#else
#define SO_EXT ".so"
#endif

LiteRtStatus StaticGetName(LiteRtAcceleratorConst accelerator,
                           const char** name) {
  *name = "GPU Static";
  return kLiteRtStatusOk;
}

LiteRtStatus StaticGetVersion(LiteRtAcceleratorConst accelerator,
                              LiteRtApiVersion* version) {
  version->major = 1;
  version->minor = 0;
  version->patch = 0;
  return kLiteRtStatusOk;
}

LiteRtStatus StaticGetHardwareSupport(
    LiteRtAcceleratorConst accelerator,
    LiteRtHwAcceleratorSet* supported_hardware) {
  *supported_hardware = kLiteRtHwAcceleratorGpu;
  return kLiteRtStatusOk;
}

LiteRtStatus StaticIsJit(LiteRtAcceleratorConst accelerator,
                         bool* does_jit_compilation) {
  *does_jit_compilation = false;
  return kLiteRtStatusOk;
}

LiteRtStatus StaticCreateDelegate(LiteRtRuntimeContext* runtime_context,
                                  LiteRtEnvironment env,
                                  LiteRtAcceleratorConst accelerator,
                                  LiteRtOptions options,
                                  LiteRtDelegateWrapper* delegate_wrapper) {
  *delegate_wrapper = nullptr;
  return kLiteRtStatusOk;
}

const LiteRtAcceleratorDef kStaticDef = {
    .abi_header =
        {
            .struct_size = sizeof(LiteRtAcceleratorDefV1),
            .major_version = 1,
            .minor_version = 0,
            .reserved = 0,
        },
    .get_name = StaticGetName,
    .get_version = StaticGetVersion,
    .get_hardware_support = StaticGetHardwareSupport,
    .is_tflite_delegate_responsible_for_jit_compilation = StaticIsJit,
    .create_delegate = StaticCreateDelegate,
    .start_metrics_collection = nullptr,
    .stop_metrics_collection = nullptr,
    .buffer_handlers = {},
};

class GpuRegistryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    litert::Expected<litert::testing::UniqueTestDirectory> test_dir =
        litert::testing::UniqueTestDirectory::Create();
    if (test_dir.HasValue()) {
      test_dir_ = std::make_unique<litert::testing::UniqueTestDirectory>(
          std::move(*test_dir));
    }

    webgpu_runfile_ = litert::testing::GetLiteRtPath(
        "runtime/accelerators/fake_webgpu_accelerator.so");
    opencl_runfile_ = litert::testing::GetLiteRtPath(
        "runtime/accelerators/fake_opencl_accelerator.so");
    unsupported_runfile_ = litert::testing::GetLiteRtPath(
        "runtime/accelerators/fake_unsupported_accelerator.so");
    LiteRtStaticLinkedAcceleratorGpuDef = nullptr;
  }

  void TearDown() override {
    LiteRtStaticLinkedAcceleratorGpuDef = nullptr;
    test_dir_.reset();
  }

  std::filesystem::path TestDir() const {
    if (!test_dir_) {
      return {};
    }
    return std::filesystem::path(std::string(test_dir_->Str()));
  }

  std::unique_ptr<litert::testing::UniqueTestDirectory> test_dir_;
  std::filesystem::path webgpu_runfile_;
  std::filesystem::path opencl_runfile_;
  std::filesystem::path unsupported_runfile_;
};

TEST_F(GpuRegistryTest, RegistersWebGpuFirstWhenBothExist) {
  ASSERT_NE(test_dir_, nullptr);
  ASSERT_TRUE(std::filesystem::exists(webgpu_runfile_));
  ASSERT_TRUE(std::filesystem::exists(opencl_runfile_));

  std::filesystem::copy_file(webgpu_runfile_,
                             TestDir() / ("libLiteRtWebGpuAccelerator" SO_EXT),
                             std::filesystem::copy_options::overwrite_existing);
  std::filesystem::copy_file(opencl_runfile_,
                             TestDir() / ("libLiteRtOpenClAccelerator" SO_EXT),
                             std::filesystem::copy_options::overwrite_existing);
  std::filesystem::copy_file(opencl_runfile_,
                             TestDir() / ("libLiteRtClGlAccelerator" SO_EXT),
                             std::filesystem::copy_options::overwrite_existing);

  std::string dir_str = TestDir().string();
  LiteRtEnvOption opt;
  opt.tag = kLiteRtEnvOptionTagRuntimeLibraryDir;
  opt.value.type = kLiteRtAnyTypeString;
  opt.value.str_value = dir_str.c_str();

  litert::Expected<LiteRtEnvironmentT::Ptr> env =
      LiteRtEnvironmentT::CreateWithOptions({opt});
  ASSERT_TRUE(env.HasValue());

  LiteRtStatus status = RegisterGpuAccelerator(env->get());
  EXPECT_EQ(status, kLiteRtStatusOk);

  ASSERT_GE((*env)->GetAcceleratorRegistry().size(), 1);
  litert::Expected<LiteRtAcceleratorT*> accelerator =
      (*env)->GetAcceleratorRegistry().Get(0);
  ASSERT_TRUE(accelerator.HasValue());
  const char* name = nullptr;
  EXPECT_EQ((*accelerator)->GetName(*accelerator, &name), kLiteRtStatusOk);
  EXPECT_STREQ(name, "GPU WebGPU");
}

TEST_F(GpuRegistryTest, FallsBackToOpenClWhenWebGpuAbsent) {
  ASSERT_NE(test_dir_, nullptr);
  ASSERT_TRUE(std::filesystem::exists(opencl_runfile_));

  std::filesystem::copy_file(opencl_runfile_,
                             TestDir() / ("libLiteRtOpenClAccelerator" SO_EXT),
                             std::filesystem::copy_options::overwrite_existing);
  std::filesystem::copy_file(opencl_runfile_,
                             TestDir() / ("libLiteRtClGlAccelerator" SO_EXT),
                             std::filesystem::copy_options::overwrite_existing);

  std::string dir_str = TestDir().string();
  LiteRtEnvOption opt;
  opt.tag = kLiteRtEnvOptionTagRuntimeLibraryDir;
  opt.value.type = kLiteRtAnyTypeString;
  opt.value.str_value = dir_str.c_str();

  litert::Expected<LiteRtEnvironmentT::Ptr> env =
      LiteRtEnvironmentT::CreateWithOptions({opt});
  ASSERT_TRUE(env.HasValue());

  LiteRtStatus status = RegisterGpuAccelerator(env->get());
  EXPECT_EQ(status, kLiteRtStatusOk);

  ASSERT_GE((*env)->GetAcceleratorRegistry().size(), 1);
  litert::Expected<LiteRtAcceleratorT*> accelerator =
      (*env)->GetAcceleratorRegistry().Get(0);
  ASSERT_TRUE(accelerator.HasValue());
  const char* name = nullptr;
  EXPECT_EQ((*accelerator)->GetName(*accelerator, &name), kLiteRtStatusOk);
  EXPECT_STREQ(name, "GPU OpenCL");
}

TEST_F(GpuRegistryTest, FallsBackToOpenClWhenWebGpuHardwareUnsupported) {
  ASSERT_NE(test_dir_, nullptr);
  ASSERT_TRUE(std::filesystem::exists(unsupported_runfile_));
  ASSERT_TRUE(std::filesystem::exists(opencl_runfile_));

  std::filesystem::copy_file(unsupported_runfile_,
                             TestDir() / ("libLiteRtWebGpuAccelerator" SO_EXT),
                             std::filesystem::copy_options::overwrite_existing);
  std::filesystem::copy_file(opencl_runfile_,
                             TestDir() / ("libLiteRtOpenClAccelerator" SO_EXT),
                             std::filesystem::copy_options::overwrite_existing);
  std::filesystem::copy_file(opencl_runfile_,
                             TestDir() / ("libLiteRtClGlAccelerator" SO_EXT),
                             std::filesystem::copy_options::overwrite_existing);

  std::string dir_str = TestDir().string();
  LiteRtEnvOption opt;
  opt.tag = kLiteRtEnvOptionTagRuntimeLibraryDir;
  opt.value.type = kLiteRtAnyTypeString;
  opt.value.str_value = dir_str.c_str();

  litert::Expected<LiteRtEnvironmentT::Ptr> env =
      LiteRtEnvironmentT::CreateWithOptions({opt});
  ASSERT_TRUE(env.HasValue());

  LiteRtStatus status = RegisterGpuAccelerator(env->get());
  EXPECT_EQ(status, kLiteRtStatusOk);

  ASSERT_GE((*env)->GetAcceleratorRegistry().size(), 1);
  litert::Expected<LiteRtAcceleratorT*> accelerator =
      (*env)->GetAcceleratorRegistry().Get(0);
  ASSERT_TRUE(accelerator.HasValue());
  const char* name = nullptr;
  EXPECT_EQ((*accelerator)->GetName(*accelerator, &name), kLiteRtStatusOk);
  EXPECT_STREQ(name, "GPU OpenCL");
}

TEST_F(GpuRegistryTest, FailsWhenNoAcceleratorPresent) {
  ASSERT_NE(test_dir_, nullptr);
  std::string dir_str = TestDir().string();
  LiteRtEnvOption opt;
  opt.tag = kLiteRtEnvOptionTagRuntimeLibraryDir;
  opt.value.type = kLiteRtAnyTypeString;
  opt.value.str_value = dir_str.c_str();

  litert::Expected<LiteRtEnvironmentT::Ptr> env =
      LiteRtEnvironmentT::CreateWithOptions({opt});
  ASSERT_TRUE(env.HasValue());

  LiteRtStatus status = RegisterGpuAccelerator(env->get());
  EXPECT_EQ(status, kLiteRtStatusErrorUnsupported);
}

TEST_F(GpuRegistryTest, RegistersFromHandleOption) {
  LiteRtEnvOption opt;
  opt.tag = kLiteRtEnvOptionTagSystemGpuAcceleratorHandle;
  opt.value.type = kLiteRtAnyTypeVoidPtr;
  opt.value.ptr_value = const_cast<LiteRtAcceleratorDef*>(&kStaticDef);

  litert::Expected<LiteRtEnvironmentT::Ptr> env =
      LiteRtEnvironmentT::CreateWithOptions({opt});
  ASSERT_TRUE(env.HasValue());

  LiteRtStatus status = RegisterGpuAccelerator(env->get());
  EXPECT_EQ(status, kLiteRtStatusOk);

  ASSERT_GE((*env)->GetAcceleratorRegistry().size(), 1);
  litert::Expected<LiteRtAcceleratorT*> accelerator =
      (*env)->GetAcceleratorRegistry().Get(0);
  ASSERT_TRUE(accelerator.HasValue());
  const char* name = nullptr;
  EXPECT_EQ((*accelerator)->GetName(*accelerator, &name), kLiteRtStatusOk);
  EXPECT_STREQ(name, "GPU Static");
}

TEST_F(GpuRegistryTest, RegistersFromStaticDef) {
  LiteRtStaticLinkedAcceleratorGpuDef = &kStaticDef;

  litert::Expected<LiteRtEnvironmentT::Ptr> env =
      LiteRtEnvironmentT::CreateWithOptions({});
  ASSERT_TRUE(env.HasValue());

  LiteRtStatus status = RegisterGpuAccelerator(env->get());
  EXPECT_EQ(status, kLiteRtStatusOk);

  ASSERT_GE((*env)->GetAcceleratorRegistry().size(), 1);
  litert::Expected<LiteRtAcceleratorT*> accelerator =
      (*env)->GetAcceleratorRegistry().Get(0);
  ASSERT_TRUE(accelerator.HasValue());
  const char* name = nullptr;
  EXPECT_EQ((*accelerator)->GetName(*accelerator, &name), kLiteRtStatusOk);
  EXPECT_STREQ(name, "GPU Static");
}

TEST_F(GpuRegistryTest, IgnoresInvalidHandleOptionType) {
  LiteRtEnvOption opt;
  opt.tag = kLiteRtEnvOptionTagSystemGpuAcceleratorHandle;
  opt.value.type = kLiteRtAnyTypeInt;
  opt.value.int_value = 12345;

  litert::Expected<LiteRtEnvironmentT::Ptr> env =
      LiteRtEnvironmentT::CreateWithOptions({opt});
  ASSERT_TRUE(env.HasValue());

  // Should fail because no valid accelerator is found.
  LiteRtStatus status = RegisterGpuAccelerator(env->get());
  EXPECT_EQ(status, kLiteRtStatusErrorUnsupported);
}

#undef SO_EXT

}  // namespace
}  // namespace litert::internal
