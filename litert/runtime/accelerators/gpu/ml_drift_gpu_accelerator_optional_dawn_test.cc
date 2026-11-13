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

#include <gtest/gtest.h>

#if defined(__linux__) && !defined(__ANDROID__)
#include <dlfcn.h>
#include <limits.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <functional>
#include <string>

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/c/internal/litert_accelerator_def.h"
#include "litert/c/internal/litert_accelerator_registration.h"
#include "litert/c/internal/litert_delegate_wrapper.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/litert_options.h"
#include "litert/c/options/litert_gpu_options.h"

namespace {

constexpr char kGpuAcceleratorSoPath[] =
    "third_party/odml/litert/litert/runtime/accelerators/gpu/"
    "libLiteRtGpuAccelerator.so";
constexpr char kFakeOpenClSoPath[] =
    "third_party/odml/litert/litert/runtime/accelerators/gpu/"
    "libfake_OpenCL.so";
constexpr char kFakeDawnSoPath[] =
    "ml_drift_delegate/delegate/libfake_webgpu_dawn.so";
constexpr char kStubEglSoPath[] =
    "third_party/odml/litert/litert/runtime/accelerators/gpu/libEGL.so.1";
constexpr char kStubGlesv2SoPath[] =
    "third_party/odml/litert/litert/runtime/accelerators/gpu/libGLESv2.so.2";

LiteRtStatus CreateGpuCompilationOptions(LiteRtGpuBackend backend,
                                         LiteRtOptions* out_options) {
  LrtGpuOptions* gpu_options = nullptr;
  LiteRtStatus status = LrtCreateGpuOptions(&gpu_options);
  if (status != kLiteRtStatusOk) {
    return status;
  }
  status = LrtSetGpuOptionsGpuBackend(gpu_options, backend);
  if (status != kLiteRtStatusOk) {
    LrtDestroyGpuOptions(gpu_options);
    return status;
  }
  const char* identifier = nullptr;
  void* payload = nullptr;
  void (*payload_deleter)(void*) = nullptr;
  status = LrtGetOpaqueGpuOptionsData(gpu_options, &identifier, &payload,
                                      &payload_deleter);
  LrtDestroyGpuOptions(gpu_options);
  if (status != kLiteRtStatusOk) {
    return status;
  }
  LiteRtOpaqueOptions opaque_options = nullptr;
  status = LiteRtCreateOpaqueOptions(identifier, payload, payload_deleter,
                                     &opaque_options);
  if (status != kLiteRtStatusOk) {
    if (payload_deleter && payload) {
      payload_deleter(payload);
    }
    return status;
  }
  status = LiteRtCreateOptions(out_options);
  if (status != kLiteRtStatusOk) {
    LiteRtDestroyOpaqueOptions(opaque_options);
    return status;
  }
  return LiteRtAddOpaqueOptions(*out_options, opaque_options);
}

// Runs `fn` in an isolated child process so process-wide static caches inside
// `LoadOpenCL()` and `EnsureWebGpuDawnLoaded()` start completely clean.
void RunInIsolatedProcess(const std::function<void()>& fn) {
  pid_t pid = fork();
  ASSERT_GE(pid, 0);
  if (pid == 0) {
    fn();
    _exit(testing::Test::HasFailure() ? 1 : 0);
  }
  int status = 0;
  ASSERT_EQ(waitpid(pid, &status, 0), pid);
  ASSERT_TRUE(WIFEXITED(status)) << "Child terminated abnormally: " << status;
  EXPECT_EQ(WEXITSTATUS(status), 0);
}

void PreloadGraphicsStubs() {
  void* egl_handle = dlopen(kStubEglSoPath, RTLD_NOW | RTLD_GLOBAL);
  ASSERT_NE(egl_handle, nullptr) << dlerror();
  void* glesv2_handle = dlopen(kStubGlesv2SoPath, RTLD_NOW | RTLD_GLOBAL);
  ASSERT_NE(glesv2_handle, nullptr) << dlerror();
}

void* PreloadFakeOpenCl(unsigned int platform_count) {
  void* cl_handle = dlopen(kFakeOpenClSoPath, RTLD_NOW | RTLD_GLOBAL);
  if (cl_handle == nullptr) {
    ADD_FAILURE() << dlerror();
    return nullptr;
  }
  using SetCountFn = void (*)(unsigned int);
  auto set_count = reinterpret_cast<SetCountFn>(
      dlsym(cl_handle, "FakeOpenClSetPlatformCount"));
  if (set_count == nullptr) {
    ADD_FAILURE() << dlerror();
    return nullptr;
  }
  set_count(platform_count);
  return cl_handle;
}

TEST(MlDriftGpuAcceleratorOptionalDawnTest,
     LoadsAcceleratorAndHandlesOptionalWebGpuDawnRuntime) {
  ASSERT_EQ(access(kGpuAcceleratorSoPath, R_OK), 0) << kGpuAcceleratorSoPath;

  RunInIsolatedProcess([] {
    const std::string base_temp = testing::TempDir();
    const std::string empty_dir =
        absl::StrCat(base_temp, "/explicit_empty_dir");
    ASSERT_EQ(mkdir(empty_dir.c_str(), 0755), 0);

    PreloadGraphicsStubs();

    // 1. Opening libLiteRtGpuAccelerator.so when libwebgpu_dawn.so is absent
    // from runfiles and LD_LIBRARY_PATH must succeed with RTLD_NOW |
    // RTLD_LOCAL.
    void* accel_handle = dlopen(kGpuAcceleratorSoPath, RTLD_NOW | RTLD_LOCAL);
    ASSERT_NE(accel_handle, nullptr) << dlerror();

    const auto* accel_def = reinterpret_cast<const LiteRtAcceleratorDef*>(
        dlsym(accel_handle, "LiteRtAcceleratorImpl"));
    ASSERT_NE(accel_def, nullptr) << dlerror();

    const char* accel_name = nullptr;
    ASSERT_EQ(accel_def->get_name(nullptr, &accel_name), kLiteRtStatusOk);
    ASSERT_NE(accel_name, nullptr);
    EXPECT_STREQ(accel_name, "LiteRT GPU");

    LiteRtAccelerator accelerator = nullptr;
    ASSERT_EQ(LiteRtCreateAccelerator(&accelerator), kLiteRtStatusOk);
    absl::Cleanup destroy_accelerator = [accelerator] {
      LiteRtDestroyAccelerator(accelerator);
    };

    LiteRtOptions webgpu_options = nullptr;
    ASSERT_EQ(
        CreateGpuCompilationOptions(kLiteRtGpuBackendWebGpu, &webgpu_options),
        kLiteRtStatusOk);
    absl::Cleanup destroy_options = [webgpu_options] {
      LiteRtDestroyOptions(webgpu_options);
    };

    // 2. When libwebgpu_dawn.so is absent from both default search paths and
    // kLiteRtEnvOptionTagRuntimeLibraryDir, requesting WebGPU returns
    // kLiteRtStatusErrorUnsupported cleanly without crashing or leaking
    // options.
    const LiteRtEnvOption missing_dawn_opt = {
        .tag = kLiteRtEnvOptionTagRuntimeLibraryDir,
        .value =
            LiteRtAny{
                .type = kLiteRtAnyTypeString,
                .str_value = empty_dir.c_str(),
            },
    };
    LiteRtEnvironment env_without_dawn = nullptr;
    ASSERT_EQ(LiteRtCreateEnvironment(1, &missing_dawn_opt, &env_without_dawn),
              kLiteRtStatusOk);
    absl::Cleanup destroy_env_without_dawn = [env_without_dawn] {
      LiteRtDestroyEnvironment(env_without_dawn);
    };

    LiteRtDelegateWrapper delegate_wrapper = nullptr;
    const LiteRtStatus without_dawn_status = accel_def->create_delegate(
        LrtGetRuntimeContext(), env_without_dawn, accelerator, webgpu_options,
        &delegate_wrapper);
#if defined(LITERT_WEBGPU_DAWN_DYNAMIC_LOADER)
    EXPECT_EQ(without_dawn_status, kLiteRtStatusErrorUnsupported);

    // 3. When libwebgpu_dawn.so is present in
    // kLiteRtEnvOptionTagRuntimeLibraryDir, create_delegate dynamically loads
    // it and dispatches through our proc-table trampolines into
    // wgpuCreateInstance.
    char resolved_fake_dawn[PATH_MAX];
    ASSERT_NE(realpath(kFakeDawnSoPath, resolved_fake_dawn), nullptr)
        << kFakeDawnSoPath;
    const std::string dawn_dir = absl::StrCat(base_temp, "/with_dawn");
    ASSERT_EQ(mkdir(dawn_dir.c_str(), 0755), 0);
    const std::string dawn_so = absl::StrCat(dawn_dir, "/libwebgpu_dawn.so");
    ASSERT_EQ(symlink(resolved_fake_dawn, dawn_so.c_str()), 0);

    EXPECT_EQ(dlopen(dawn_so.c_str(), RTLD_LAZY | RTLD_NOLOAD), nullptr);

    const LiteRtEnvOption with_dawn_opt = {
        .tag = kLiteRtEnvOptionTagRuntimeLibraryDir,
        .value =
            LiteRtAny{
                .type = kLiteRtAnyTypeString,
                .str_value = dawn_dir.c_str(),
            },
    };
    LiteRtEnvironment env_with_dawn = nullptr;
    ASSERT_EQ(LiteRtCreateEnvironment(1, &with_dawn_opt, &env_with_dawn),
              kLiteRtStatusOk);
    absl::Cleanup destroy_env_with_dawn = [env_with_dawn] {
      LiteRtDestroyEnvironment(env_with_dawn);
    };

    const LiteRtStatus with_dawn_status = accel_def->create_delegate(
        LrtGetRuntimeContext(), env_with_dawn, accelerator, webgpu_options,
        &delegate_wrapper);
    EXPECT_NE(with_dawn_status, kLiteRtStatusErrorUnsupported);
    if (with_dawn_status == kLiteRtStatusOk && delegate_wrapper != nullptr) {
      LiteRtDestroyDelegateWrapper(delegate_wrapper);
    }

    void* loaded_dawn = dlopen(dawn_so.c_str(), RTLD_LAZY | RTLD_NOLOAD);
    ASSERT_NE(loaded_dawn, nullptr);
    absl::Cleanup close_loaded_dawn = [loaded_dawn] { dlclose(loaded_dawn); };

    using GetCallsFn = int (*)();
    auto get_calls = reinterpret_cast<GetCallsFn>(
        dlsym(loaded_dawn, "FakeDawnGetCreateInstanceCalls"));
    ASSERT_NE(get_calls, nullptr) << dlerror();
    EXPECT_GE(get_calls(), 1);
#else
    EXPECT_NE(without_dawn_status, kLiteRtStatusErrorUnsupported);
    if (without_dawn_status == kLiteRtStatusOk && delegate_wrapper != nullptr) {
      LiteRtDestroyDelegateWrapper(delegate_wrapper);
    }
#endif
  });
}

TEST(MlDriftGpuAcceleratorOptionalDawnTest,
     AutomaticBackendPrefersOpenClOverWebGpuWhenOpenClAvailable) {
  RunInIsolatedProcess([] {
    PreloadGraphicsStubs();
    void* cl_handle = PreloadFakeOpenCl(/*platform_count=*/1);
    ASSERT_NE(cl_handle, nullptr);

    char resolved_fake_dawn[PATH_MAX];
    ASSERT_NE(realpath(kFakeDawnSoPath, resolved_fake_dawn), nullptr)
        << kFakeDawnSoPath;
    const std::string dawn_dir =
        absl::StrCat(testing::TempDir(), "/auto_cl_with_dawn");
    ASSERT_EQ(mkdir(dawn_dir.c_str(), 0755), 0);
    const std::string dawn_so = absl::StrCat(dawn_dir, "/libwebgpu_dawn.so");
    ASSERT_EQ(symlink(resolved_fake_dawn, dawn_so.c_str()), 0);

    void* accel_handle = dlopen(kGpuAcceleratorSoPath, RTLD_NOW | RTLD_LOCAL);
    ASSERT_NE(accel_handle, nullptr) << dlerror();
    const auto* accel_def = reinterpret_cast<const LiteRtAcceleratorDef*>(
        dlsym(accel_handle, "LiteRtAcceleratorImpl"));
    ASSERT_NE(accel_def, nullptr) << dlerror();

    LiteRtAccelerator accelerator = nullptr;
    ASSERT_EQ(LiteRtCreateAccelerator(&accelerator), kLiteRtStatusOk);
    LiteRtOptions auto_options = nullptr;
    ASSERT_EQ(
        CreateGpuCompilationOptions(kLiteRtGpuBackendAutomatic, &auto_options),
        kLiteRtStatusOk);

    const LiteRtEnvOption with_dawn_opt = {
        .tag = kLiteRtEnvOptionTagRuntimeLibraryDir,
        .value =
            LiteRtAny{
                .type = kLiteRtAnyTypeString,
                .str_value = dawn_dir.c_str(),
            },
    };
    LiteRtEnvironment env = nullptr;
    ASSERT_EQ(LiteRtCreateEnvironment(1, &with_dawn_opt, &env),
              kLiteRtStatusOk);

    LiteRtDelegateWrapper delegate_wrapper = nullptr;
    const LiteRtStatus status =
        accel_def->create_delegate(LrtGetRuntimeContext(), env, accelerator,
                                   auto_options, &delegate_wrapper);
    EXPECT_NE(status, kLiteRtStatusErrorUnsupported);

    using GetCallsFn = int (*)();
    auto get_cl_platform_calls = reinterpret_cast<GetCallsFn>(
        dlsym(cl_handle, "FakeOpenClGetPlatformIdsCalls"));
    auto get_cl_context_calls = reinterpret_cast<GetCallsFn>(
        dlsym(cl_handle, "FakeOpenClGetCreateContextCalls"));
    ASSERT_NE(get_cl_platform_calls, nullptr);
    ASSERT_NE(get_cl_context_calls, nullptr);
    EXPECT_GE(get_cl_platform_calls(), 1);
    EXPECT_GE(get_cl_context_calls(), 1);

    // Dawn must not even be dlopened when OpenCL is available.
    EXPECT_EQ(dlopen(dawn_so.c_str(), RTLD_LAZY | RTLD_NOLOAD), nullptr);
  });
}

#if defined(LITERT_WEBGPU_DAWN_DYNAMIC_LOADER)
TEST(MlDriftGpuAcceleratorOptionalDawnTest,
     AutomaticBackendFallsBackToWebGpuOrUnsupportedWhenOpenClUnavailable) {
  // Subcase A: OpenCL unavailable and libwebgpu_dawn.so absent -> Unsupported.
  RunInIsolatedProcess([] {
    PreloadGraphicsStubs();
    ASSERT_NE(PreloadFakeOpenCl(/*platform_count=*/0), nullptr);

    const std::string empty_dir =
        absl::StrCat(testing::TempDir(), "/auto_no_cl_no_dawn");
    ASSERT_EQ(mkdir(empty_dir.c_str(), 0755), 0);

    void* accel_handle = dlopen(kGpuAcceleratorSoPath, RTLD_NOW | RTLD_LOCAL);
    ASSERT_NE(accel_handle, nullptr) << dlerror();
    const auto* accel_def = reinterpret_cast<const LiteRtAcceleratorDef*>(
        dlsym(accel_handle, "LiteRtAcceleratorImpl"));
    ASSERT_NE(accel_def, nullptr) << dlerror();

    LiteRtAccelerator accelerator = nullptr;
    ASSERT_EQ(LiteRtCreateAccelerator(&accelerator), kLiteRtStatusOk);
    LiteRtOptions auto_options = nullptr;
    ASSERT_EQ(
        CreateGpuCompilationOptions(kLiteRtGpuBackendAutomatic, &auto_options),
        kLiteRtStatusOk);

    const LiteRtEnvOption empty_opt = {
        .tag = kLiteRtEnvOptionTagRuntimeLibraryDir,
        .value =
            LiteRtAny{
                .type = kLiteRtAnyTypeString,
                .str_value = empty_dir.c_str(),
            },
    };
    LiteRtEnvironment env = nullptr;
    ASSERT_EQ(LiteRtCreateEnvironment(1, &empty_opt, &env), kLiteRtStatusOk);

    LiteRtDelegateWrapper delegate_wrapper = nullptr;
    EXPECT_EQ(
        accel_def->create_delegate(LrtGetRuntimeContext(), env, accelerator,
                                   auto_options, &delegate_wrapper),
        kLiteRtStatusErrorUnsupported);
  });

  // Subcase B: OpenCL unavailable and libwebgpu_dawn.so present -> WebGPU.
  RunInIsolatedProcess([] {
    PreloadGraphicsStubs();
    ASSERT_NE(PreloadFakeOpenCl(/*platform_count=*/0), nullptr);

    char resolved_fake_dawn[PATH_MAX];
    ASSERT_NE(realpath(kFakeDawnSoPath, resolved_fake_dawn), nullptr)
        << kFakeDawnSoPath;
    const std::string dawn_dir =
        absl::StrCat(testing::TempDir(), "/auto_no_cl_with_dawn");
    ASSERT_EQ(mkdir(dawn_dir.c_str(), 0755), 0);
    const std::string dawn_so = absl::StrCat(dawn_dir, "/libwebgpu_dawn.so");
    ASSERT_EQ(symlink(resolved_fake_dawn, dawn_so.c_str()), 0);

    void* accel_handle = dlopen(kGpuAcceleratorSoPath, RTLD_NOW | RTLD_LOCAL);
    ASSERT_NE(accel_handle, nullptr) << dlerror();
    const auto* accel_def = reinterpret_cast<const LiteRtAcceleratorDef*>(
        dlsym(accel_handle, "LiteRtAcceleratorImpl"));
    ASSERT_NE(accel_def, nullptr) << dlerror();

    LiteRtAccelerator accelerator = nullptr;
    ASSERT_EQ(LiteRtCreateAccelerator(&accelerator), kLiteRtStatusOk);
    LiteRtOptions auto_options = nullptr;
    ASSERT_EQ(
        CreateGpuCompilationOptions(kLiteRtGpuBackendAutomatic, &auto_options),
        kLiteRtStatusOk);

    const LiteRtEnvOption with_dawn_opt = {
        .tag = kLiteRtEnvOptionTagRuntimeLibraryDir,
        .value =
            LiteRtAny{
                .type = kLiteRtAnyTypeString,
                .str_value = dawn_dir.c_str(),
            },
    };
    LiteRtEnvironment env = nullptr;
    ASSERT_EQ(LiteRtCreateEnvironment(1, &with_dawn_opt, &env),
              kLiteRtStatusOk);

    LiteRtDelegateWrapper delegate_wrapper = nullptr;
    const LiteRtStatus status =
        accel_def->create_delegate(LrtGetRuntimeContext(), env, accelerator,
                                   auto_options, &delegate_wrapper);
    EXPECT_NE(status, kLiteRtStatusErrorUnsupported);

    void* loaded_dawn = dlopen(dawn_so.c_str(), RTLD_LAZY | RTLD_NOLOAD);
    ASSERT_NE(loaded_dawn, nullptr);
    using GetCallsFn = int (*)();
    auto get_dawn_calls = reinterpret_cast<GetCallsFn>(
        dlsym(loaded_dawn, "FakeDawnGetCreateInstanceCalls"));
    ASSERT_NE(get_dawn_calls, nullptr);
    EXPECT_GE(get_dawn_calls(), 1);
  });
}
#endif  // defined(LITERT_WEBGPU_DAWN_DYNAMIC_LOADER)

}  // namespace
#endif  // defined(__linux__) && !defined(__ANDROID__)
