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

#include "ml_drift_delegate/delegate/gpu_environment_util.h"

#include <cstdint>

#include "absl/status/status.h"  // from @com_google_absl
#include "third_party/gloop/util/status/status_macros.h"
#include "ml_drift/cl/cl_device.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "litert/c/litert_common.h"
#include <CL/cl.h>
#if LITERT_HAS_VULKAN_SUPPORT
#include "ml_drift/syrtis/environment.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/shared_vulkan_env.h"
#endif  // LITERT_HAS_VULKAN_SUPPORT
#if LITERT_HAS_WEBGPU_SUPPORT
#if !defined(__EMSCRIPTEN__)
#include "dawn/dawn_proc.h"  // from @dawn
#include "dawn/dawn_proc_table.h"  // from @dawn
#include "webgpu/webgpu.h"  // from @dawn
#endif  // !defined(__EMSCRIPTEN__)
#include "ml_drift/webgpu/environment.h"  // from @ml_drift
#include "ml_drift/webgpu/webgpu_api_util.h"  // from @ml_drift
#include "ml_drift/webgpu/webgpu_headers.h"  // from @ml_drift
#endif  // LITERT_HAS_WEBGPU_SUPPORT
#include "litert/c/litert_environment_options.h"
#include "litert/cc/litert_any.h"
#include "litert/core/environment.h"
#include "litert/runtime/gpu_environment.h"

namespace litert {
namespace ml_drift {

namespace {

absl::Status UpdateGpuEnvironmentOpenCl(LiteRtEnvironment environment) {
  const auto& env_options = environment->GetOptions();

  auto opencl_device_id_res =
      env_options.GetOption(kLiteRtEnvOptionTagOpenClDeviceId);
  if (!opencl_device_id_res.HasValue()) return absl::OkStatus();
  auto opencl_device_id_any = opencl_device_id_res.Value();
  auto opencl_device_id = reinterpret_cast<cl_device_id>(
      std::get<int64_t>(::litert::ToStdAny(opencl_device_id_any)));

  auto opencl_platform_id_res =
      env_options.GetOption(kLiteRtEnvOptionTagOpenClPlatformId);
  if (!opencl_platform_id_res.HasValue()) return absl::OkStatus();
  auto opencl_platform_id_any = opencl_platform_id_res.Value();
  auto opencl_platform_id = reinterpret_cast<cl_platform_id>(
      std::get<int64_t>(::litert::ToStdAny(opencl_platform_id_any)));

  ::ml_drift::cl::CLDevice cl_device(opencl_device_id, opencl_platform_id);
  const auto& gpu_info = cl_device.GetInfo();

  auto gpu_env_res = environment->GetGpuEnvironment();
  if (gpu_env_res.HasValue()) {
    gpu_env_res.Value()->SetFP16Supported(gpu_info.SupportsFP16());
  }

  return absl::OkStatus();
}

#if LITERT_HAS_WEBGPU_SUPPORT
absl::Status UpdateGpuEnvironmentWebGpu(LiteRtEnvironment environment) {
  const auto& env_options = environment->GetOptions();

  auto wgpu_device_res = env_options.GetOption(kLiteRtEnvOptionTagWebGpuDevice);
  if (!wgpu_device_res.HasValue()) return absl::OkStatus();

  WGPUDevice wgpu_device = reinterpret_cast<WGPUDevice>(
      std::get<int64_t>(::litert::ToStdAny(wgpu_device_res.Value())));

#if !defined(__EMSCRIPTEN__)
  auto wgpu_procs_res = env_options.GetOption(kLiteRtEnvOptionTagWebGpuProcs);
  if (wgpu_procs_res.HasValue()) {
    auto procs_int =
        std::get<int64_t>(::litert::ToStdAny(wgpu_procs_res.Value()));
    if (procs_int != 0) {
      dawnProcSetProcs(reinterpret_cast<const DawnProcTable*>(procs_int));
    }
  }
#endif  // !defined(__EMSCRIPTEN__)

  auto wgpu_flush_res =
      env_options.GetOption(kLiteRtEnvOptionTagWebGpuFlushCallback);
  if (wgpu_flush_res.HasValue()) {
    auto flush_cb_int =
        std::get<int64_t>(::litert::ToStdAny(wgpu_flush_res.Value()));
    if (flush_cb_int != 0) {
      ::ml_drift::webgpu::SetFlushCallback(
          reinterpret_cast<::ml_drift::webgpu::WebGpuFlushCallback>(
              flush_cb_int));
    }
  }

  if (wgpu_device == nullptr) return absl::OkStatus();

  auto wgpu_instance_res =
      env_options.GetOption(kLiteRtEnvOptionTagWebGpuInstance);
  WGPUInstance wgpu_instance = nullptr;
  if (wgpu_instance_res.HasValue()) {
    wgpu_instance = reinterpret_cast<WGPUInstance>(
        std::get<int64_t>(::litert::ToStdAny(wgpu_instance_res.Value())));
  }
#if !defined(__EMSCRIPTEN__)
  if (wgpu_instance == nullptr && wgpu_device != nullptr) {
    wgpu::Device device_tmp(wgpu_device);
    wgpu::Adapter adapter_tmp = device_tmp.GetAdapter();
    if (adapter_tmp) {
      wgpu::Instance instance_tmp = adapter_tmp.GetInstance();
      if (instance_tmp) {
        wgpu_instance = instance_tmp.Get();
      }
    }
  }
#endif  // !defined(__EMSCRIPTEN__)
  wgpu::Instance instance;
  if (wgpu_instance != nullptr) {
    instance = wgpu::Instance(wgpu_instance);
  }

  ::ml_drift::webgpu::Environment wgpu_env;
  wgpu::Device device(wgpu_device);
  wgpu::AdapterInfo adapter_info;
#if !defined(__EMSCRIPTEN__)
  device.GetAdapter().GetInfo(&adapter_info);
#endif  // !defined(__EMSCRIPTEN__)

  auto status = wgpu_env.Initialize(device, adapter_info, instance);
  if (!status.ok()) return status;

  auto gpu_env_res = environment->GetGpuEnvironment();
  if (gpu_env_res.HasValue()) {
    gpu_env_res.Value()->SetFP16Supported(wgpu_env.GetInfo().SupportsFP16());
  }

  return absl::OkStatus();
}
#endif  // LITERT_HAS_WEBGPU_SUPPORT

#if LITERT_HAS_VULKAN_SUPPORT
absl::Status UpdateGpuEnvironmentVulkan(LiteRtEnvironment environment) {
  const auto& env_options = environment->GetOptions();

  auto vulkan_env_res =
      env_options.GetOption(kLiteRtEnvOptionTagVulkanEnvironment);
  if (!vulkan_env_res.HasValue()) return absl::OkStatus();

  SharedVulkanEnv* shared_vulkan_env = reinterpret_cast<SharedVulkanEnv*>(
      std::get<int64_t>(::litert::ToStdAny(vulkan_env_res.Value())));

  if (shared_vulkan_env == nullptr) return absl::OkStatus();

  auto gpu_env_res = environment->GetGpuEnvironment();
  if (gpu_env_res.HasValue()) {
    gpu_env_res.Value()->SetFP16Supported(
        shared_vulkan_env->vulkan_env().GetInfo().SupportsFP16());
  }

  return absl::OkStatus();
}
#endif  // LITERT_HAS_VULKAN_SUPPORT

}  // namespace

#if LITERT_HAS_METAL_SUPPORT
// Declared in gpu_environment_util_metal.mm
extern absl::Status UpdateGpuEnvironmentMetal(LiteRtEnvironment environment);
#endif  // LITERT_HAS_METAL_SUPPORT

absl::Status UpdateGpuEnvironmentWithMlDriftCapabilities(
    LiteRtEnvironment environment) {
  if (environment == nullptr) {
    return absl::InvalidArgumentError("Environment pointer is null.");
  }

  RETURN_IF_ERROR(UpdateGpuEnvironmentOpenCl(environment));
#if LITERT_HAS_WEBGPU_SUPPORT
  RETURN_IF_ERROR(UpdateGpuEnvironmentWebGpu(environment));
#endif  // LITERT_HAS_WEBGPU_SUPPORT
#if LITERT_HAS_METAL_SUPPORT
  RETURN_IF_ERROR(UpdateGpuEnvironmentMetal(environment));
#endif  // LITERT_HAS_METAL_SUPPORT
#if LITERT_HAS_VULKAN_SUPPORT
  RETURN_IF_ERROR(UpdateGpuEnvironmentVulkan(environment));
#endif  // LITERT_HAS_VULKAN_SUPPORT

  return absl::OkStatus();
}

}  // namespace ml_drift
}  // namespace litert
