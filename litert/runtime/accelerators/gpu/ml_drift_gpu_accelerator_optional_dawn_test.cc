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
#if defined(LITERT_WEBGPU_DAWN_DYNAMIC_LOADER)
#include <limits.h>
#include <stdlib.h>
#endif
#include <sys/stat.h>
#include <unistd.h>

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
#if defined(LITERT_WEBGPU_DAWN_DYNAMIC_LOADER)
constexpr char kFakeDawnSoPath[] =
    "ml_drift_delegate/delegate/libfake_webgpu_dawn.so";
#endif
constexpr char kStubEglSoPath[] =
    "third_party/odml/litert/litert/runtime/accelerators/gpu/libEGL.so.1";
constexpr char kStubGlesv2SoPath[] =
    "third_party/odml/litert/litert/runtime/accelerators/gpu/libGLESv2.so.2";

LiteRtStatus CreateWebGpuCompilationOptions(LiteRtOptions* out_options) {
  LrtGpuOptions* gpu_options = nullptr;
  LiteRtStatus status = LrtCreateGpuOptions(&gpu_options);
  if (status != kLiteRtStatusOk) {
    return status;
  }
  status = LrtSetGpuOptionsGpuBackend(gpu_options, kLiteRtGpuBackendWebGpu);
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

TEST(MlDriftGpuAcceleratorOptionalDawnTest,
     LoadsAcceleratorAndHandlesOptionalWebGpuDawnRuntime) {
  ASSERT_EQ(access(kGpuAcceleratorSoPath, R_OK), 0) << kGpuAcceleratorSoPath;

  const std::string base_temp = testing::TempDir();
  const std::string empty_dir = absl::StrCat(base_temp, "/empty_runtime_dir");
  ASSERT_EQ(mkdir(empty_dir.c_str(), 0755), 0);

  // Preload EGL/GLESv2 stubs on headless Linux test workers that do not have
  // system graphics libraries installed. Keep shared libraries mapped until
  // process exit so HeapLeakChecker retains .bss roots for static singletons
  // inside the plugin.
  void* egl_handle = dlopen(kStubEglSoPath, RTLD_NOW | RTLD_GLOBAL);
  ASSERT_NE(egl_handle, nullptr) << dlerror();
  void* glesv2_handle = dlopen(kStubGlesv2SoPath, RTLD_NOW | RTLD_GLOBAL);
  ASSERT_NE(glesv2_handle, nullptr) << dlerror();

  // 1. Opening libLiteRtGpuAccelerator.so when libwebgpu_dawn.so is absent from
  // runfiles and LD_LIBRARY_PATH must succeed with RTLD_NOW | RTLD_LOCAL.
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
  ASSERT_EQ(CreateWebGpuCompilationOptions(&webgpu_options), kLiteRtStatusOk);
  absl::Cleanup destroy_options = [webgpu_options] {
    LiteRtDestroyOptions(webgpu_options);
  };

  // 2. When libwebgpu_dawn.so is absent from both default search paths and
  // kLiteRtEnvOptionTagRuntimeLibraryDir, requesting WebGPU returns
  // kLiteRtStatusErrorUnsupported cleanly without crashing or leaking options.
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
  // kLiteRtEnvOptionTagRuntimeLibraryDir, create_delegate dynamically loads it
  // and dispatches through our proc-table trampolines into wgpuCreateInstance.
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

  // Fake Dawn returns nullptr from wgpuCreateInstance, so delegate creation
  // moves past the Dawn availability check (not kLiteRtStatusErrorUnsupported)
  // and fails environment creation after invoking wgpuCreateInstance.
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
  // In static runtime link mode, Dawn is statically linked into
  // libLiteRtGpuAccelerator.so, so WebGPU delegate creation proceeds past the
  // Dawn loader check even when the runtime library directory is empty.
  EXPECT_NE(without_dawn_status, kLiteRtStatusErrorUnsupported);
  if (without_dawn_status == kLiteRtStatusOk && delegate_wrapper != nullptr) {
    LiteRtDestroyDelegateWrapper(delegate_wrapper);
  }
#endif
}

}  // namespace
#endif  // defined(__linux__) && !defined(__ANDROID__)
