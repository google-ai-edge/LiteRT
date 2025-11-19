// Copyright 2025 Google LLC.
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

#include "litert/runtime/gpu_environment.h"

#include <cstdint>
#include <vector>

#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment_options.h"
#include "litert/cc/litert_any.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/environment.h"

#if LITERT_HAS_METAL_SUPPORT
#include "litert/runtime/metal_info.h"
#endif  // LITERT_HAS_METAL_SUPPORT

#if LITERT_HAS_OPENCL_SUPPORT
#include <CL/cl.h>
#include "tflite/delegates/gpu/cl/cl_command_queue.h"
#include "tflite/delegates/gpu/cl/cl_context.h"
#include "tflite/delegates/gpu/cl/cl_device.h"
#include "tflite/delegates/gpu/cl/opencl_wrapper.h"
#endif  // LITERT_HAS_OPENCL_SUPPORT

#if LITERT_HAS_OPENGL_SUPPORT
#include <EGL/egl.h>

#include "tflite/delegates/gpu/cl/gl_interop.h"
#endif  // LITERT_HAS_OPENGL_SUPPORT

namespace litert {
namespace internal {

GpuEnvironmentOptions CreateGpuEnvironmentOptions(
    LiteRtEnvironmentT* environment) {
  GpuEnvironmentOptions options;
  // If environment is not provided, return the default (empty) options.
  if (!environment) {
    return options;
  }

  auto callback_option =
      environment->GetOption(kLiteRtEnvOptionTagCallbackOnGpuEnvDestroy);
  if (callback_option.has_value() &&
      callback_option->type == kLiteRtAnyTypeVoidPtr) {
    options.callback_on_destroy = reinterpret_cast<void (*)(void*)>(
        const_cast<void*>(callback_option->ptr_value));
    auto user_data_option = environment->GetOption(
        kLiteRtEnvOptionTagCallbackUserDataOnGpuEnvDestroy);
    if (user_data_option.has_value() &&
        user_data_option->type == kLiteRtAnyTypeVoidPtr) {
      options.callback_user_data_on_destroy = reinterpret_cast<void*>(
          const_cast<void*>(user_data_option->ptr_value));
    }
  }

#if LITERT_HAS_OPENCL_SUPPORT
  auto device_option =
      environment->GetOption(kLiteRtEnvOptionTagOpenClDeviceId);
  if (device_option.has_value() && device_option->type == kLiteRtAnyTypeInt) {
    options.device_id =
        reinterpret_cast<cl_device_id>(device_option->int_value);
  }
  auto platform_option =
      environment->GetOption(kLiteRtEnvOptionTagOpenClPlatformId);
  if (platform_option.has_value() &&
      platform_option->type == kLiteRtAnyTypeInt) {
    options.platform_id =
        reinterpret_cast<cl_platform_id>(platform_option->int_value);
  }
  auto context_option =
      environment->GetOption(kLiteRtEnvOptionTagOpenClContext);
  if (context_option.has_value() && context_option->type == kLiteRtAnyTypeInt) {
    options.context = reinterpret_cast<cl_context>(context_option->int_value);
  }
  auto command_queue_option =
      environment->GetOption(kLiteRtEnvOptionTagOpenClCommandQueue);
  if (command_queue_option.has_value() &&
      command_queue_option->type == kLiteRtAnyTypeInt) {
    options.command_queue =
        reinterpret_cast<cl_command_queue>(command_queue_option->int_value);
  }
#endif  // LITERT_HAS_OPENCL_SUPPORT

#if LITERT_HAS_OPENGL_SUPPORT
  auto egl_display_option =
      environment->GetOption(kLiteRtEnvOptionTagEglDisplay);
  if (egl_display_option.has_value() &&
      egl_display_option->type == kLiteRtAnyTypeInt) {
    options.egl_display =
        reinterpret_cast<EGLDisplay>(egl_display_option->int_value);
  }
  auto egl_context_option =
      environment->GetOption(kLiteRtEnvOptionTagEglContext);
  if (egl_context_option.has_value() &&
      egl_context_option->type == kLiteRtAnyTypeInt) {
    options.egl_context =
        reinterpret_cast<EGLContext>(egl_context_option->int_value);
  }
#endif  // LITERT_HAS_OPENGL_SUPPORT

#if LITERT_HAS_WEBGPU_SUPPORT
  auto wgpu_device_option =
      environment->GetOption(kLiteRtEnvOptionTagWebGpuDevice);
  if (wgpu_device_option.has_value() &&
      wgpu_device_option->type == kLiteRtAnyTypeInt) {
    options.webgpu_device =
        reinterpret_cast<WGPUDevice>(wgpu_device_option->int_value);
  }
  auto wgpu_queue_option =
      environment->GetOption(kLiteRtEnvOptionTagWebGpuQueue);
  if (wgpu_queue_option.has_value() &&
      wgpu_queue_option->type == kLiteRtAnyTypeInt) {
    options.webgpu_queue =
        reinterpret_cast<WGPUQueue>(wgpu_queue_option->int_value);
  }
#endif  // LITERT_HAS_WEBGPU_SUPPORT

#if LITERT_HAS_METAL_SUPPORT
  auto metal_device_option =
      environment->GetOption(kLiteRtEnvOptionTagMetalDevice);
  auto metal_command_queue_option =
      environment->GetOption(kLiteRtEnvOptionTagMetalCommandQueue);
  if (metal_device_option.has_value() &&
      metal_device_option->type == kLiteRtAnyTypeVoidPtr &&
      metal_command_queue_option.has_value() &&
      metal_command_queue_option->type == kLiteRtAnyTypeVoidPtr) {
    LiteRtCreateWithCommandQueue(
        const_cast<void*>(metal_command_queue_option->ptr_value),
        const_cast<void*>(metal_device_option->ptr_value), &options.metal_info);
  }
#endif  // LITERT_HAS_METAL_SUPPORT

#if LITERT_HAS_VULKAN_SUPPORT
  auto vulkan_env_option =
      environment->GetOption(kLiteRtEnvOptionTagVulkanEnvironment);
  if (vulkan_env_option.has_value() &&
      vulkan_env_option->type == kLiteRtAnyTypeInt) {
    options.vulkan_env = reinterpret_cast<void*>(vulkan_env_option->int_value);
  }
#endif  // LITERT_HAS_VULKAN_SUPPORT

  return options;
}

#if LITERT_HAS_OPENCL_SUPPORT
bool SupportsAhwbClInteropHelper(tflite::gpu::cl::CLDevice device) {
#if LITERT_HAS_AHWB_SUPPORT
  return device.GetInfo().SupportsExtension("cl_arm_import_memory") &&
         ::tflite::gpu::cl::clImportMemoryARM != nullptr;
#else   // LITERT_HAS_AHWB_SUPPORT
  return false;
#endif  // LITERT_HAS_AHWB_SUPPORT
}
#endif  // LITERT_HAS_OPENCL_SUPPORT

namespace {
#if LITERT_HAS_OPENGL_SUPPORT && LITERT_HAS_AHWB_SUPPORT
PFNGLBUFFERSTORAGEEXTERNALEXTPROC glBufferStorageExternalEXT;
PFNEGLGETNATIVECLIENTBUFFERANDROIDPROC eglGetNativeClientBufferANDROID;

bool SupportsAhwbGlInteropHelper() {
  static const bool extensions_allowed = [] {
    eglGetNativeClientBufferANDROID =
        reinterpret_cast<PFNEGLGETNATIVECLIENTBUFFERANDROIDPROC>(
            eglGetProcAddress("eglGetNativeClientBufferANDROID"));
    glBufferStorageExternalEXT =
        reinterpret_cast<PFNGLBUFFERSTORAGEEXTERNALEXTPROC>(
            eglGetProcAddress("glBufferStorageExternalEXT"));
    return eglGetNativeClientBufferANDROID && glBufferStorageExternalEXT;
  }();
  return extensions_allowed;
}
#endif  // LITERT_HAS_OPENGL_SUPPORT && LITERT_HAS_AHWB_SUPPORT

}  // namespace

GpuEnvironment::~GpuEnvironment() {
  if (options_.callback_on_destroy) {
    options_.callback_on_destroy(options_.callback_user_data_on_destroy);
  }
}

Expected<void> GpuEnvironment::Initialize(LiteRtEnvironmentT* environment) {
#if LITERT_HAS_OPENCL_SUPPORT
  // Set up OpenCL.
  LITERT_RETURN_IF_ERROR(tflite::gpu::cl::LoadOpenCL().ok())
      << "Failed to load OpenCL for LiteRT.";
  properties_.is_opencl_available = true;
#endif  // LITERT_HAS_OPENCL_SUPPORT

  // Set up options.
  options_ = CreateGpuEnvironmentOptions(environment);
  // Options that will be added to the LiteRT Environment after GPU environment
  // is initialized.
  std::vector<LiteRtEnvOption> created_gpu_resources;

#if LITERT_HAS_OPENCL_SUPPORT
  // Set up device.
  if (options_.device_id && options_.platform_id) {
    device_ =
        tflite::gpu::cl::CLDevice(options_.device_id, options_.platform_id);
    LITERT_LOG(
        LITERT_INFO,
        "Created OpenCL device from provided device id and platform id.");
  } else {
    LITERT_RETURN_IF_ERROR(
        tflite::gpu::cl::CreateDefaultGPUDevice(&device_).ok())
        << "Failed to create default OpenCL device";
    // New option: cl_device_id
    LITERT_ASSIGN_OR_RETURN(
        auto device_id,
        ToLiteRtAny(LiteRtVariant(reinterpret_cast<int64_t>(device_.id()))));
    created_gpu_resources.push_back(LiteRtEnvOption{
        .tag = kLiteRtEnvOptionTagOpenClDeviceId, .value = device_id});
    options_.device_id = device_.id();
    // New option: cl_platform_id
    LITERT_ASSIGN_OR_RETURN(
        auto platform_id, ToLiteRtAny(LiteRtVariant(
                              reinterpret_cast<int64_t>(device_.platform()))));
    created_gpu_resources.push_back(LiteRtEnvOption{
        .tag = kLiteRtEnvOptionTagOpenClPlatformId, .value = platform_id});
    options_.platform_id = device_.platform();

    LITERT_LOG(LITERT_INFO, "Created default OpenCL device.");
  }
#endif  // LITERT_HAS_OPENCL_SUPPORT

  // Set up remaining properties.
#if LITERT_HAS_OPENCL_SUPPORT
#if LITERT_HAS_OPENGL_SUPPORT
  // Set up GL interop properties when OpenCL and OpenGL are both supported.
  properties_.is_gl_sharing_supported =
      tflite::gpu::cl::IsGlSharingSupported(device_);
  properties_.is_gl_to_cl_fast_sync_supported =
      tflite::gpu::cl::IsClEventFromEglSyncSupported(device_);
  properties_.is_cl_to_gl_fast_sync_supported =
      tflite::gpu::cl::IsEglSyncFromClEventSupported();
#endif  // LITERT_HAS_OPENGL_SUPPORT
  properties_.is_ahwb_cl_interop_supported =
      SupportsAhwbClInteropHelper(device_);
#endif  // LITERT_HAS_OPENCL_SUPPORT

#if LITERT_HAS_OPENGL_SUPPORT && LITERT_HAS_AHWB_SUPPORT
  properties_.is_ahwb_gl_interop_supported = SupportsAhwbGlInteropHelper();
#endif  // LITERT_HAS_OPENGL_SUPPORT && LITERT_HAS_AHWB_SUPPORT

#if LITERT_HAS_OPENCL_SUPPORT
  // Set up context.
  if (options_.context) {
    if (options_.IsGlAware()) {
      // TODO(b/383176413): Add check to confirm this context is GL-aware.
      // We currently assume that user configured context properly.
      context_ = tflite::gpu::cl::CLContext(options_.context,
                                            /*has_ownership=*/false);
#if LITERT_HAS_OPENGL_SUPPORT
      LITERT_RETURN_IF_ERROR(eglGetCurrentContext() == options_.egl_context)
          << "EGL context is not the same as provided context";
      LITERT_RETURN_IF_ERROR(eglGetCurrentDisplay() == options_.egl_display)
          << "EGL display is not the same as provided display";
      std::unique_ptr<tflite::gpu::gl::EglEnvironment> egl_env;
      // This function call implicitly reuses provided EGL context and display
      // present on this thread.
      LITERT_RETURN_IF_ERROR(
          tflite::gpu::gl::EglEnvironment::NewEglEnvironment(&egl_env).ok())
          << "Failed to create EGL environment";
      egl_env_ = std::move(egl_env);
      LITERT_LOG(LITERT_INFO, "Reusing provided EGL environment.");
#endif  // LITERT_HAS_OPENGL_SUPPORT
    } else {
      context_ = tflite::gpu::cl::CLContext(options_.context,
                                            /*has_ownership=*/false);
      LITERT_LOG(LITERT_INFO, "Created OpenCL context from provided context.");
    }
  } else {
    // If no OpenCL context is provided and no EGL options are set, attempt to
    // create a default EGL Environment.
    if (!options_.IsGlAware()) {
#if LITERT_HAS_OPENGL_SUPPORT
      std::unique_ptr<tflite::gpu::gl::EglEnvironment> egl_env;
      LITERT_RETURN_IF_ERROR(
          tflite::gpu::gl::EglEnvironment::NewEglEnvironment(&egl_env).ok())
          << "Failed to create EGL environment";
      egl_env_ = std::move(egl_env);
      // New option: egl_display
      LITERT_ASSIGN_OR_RETURN(
          auto egl_display, ToLiteRtAny(LiteRtVariant(reinterpret_cast<int64_t>(
                                egl_env_->display()))));
      created_gpu_resources.push_back(LiteRtEnvOption{
          .tag = kLiteRtEnvOptionTagEglDisplay, .value = egl_display});
      options_.egl_display = egl_env_->display();
      // New option: egl_context
      LITERT_ASSIGN_OR_RETURN(
          auto egl_context, ToLiteRtAny(LiteRtVariant(reinterpret_cast<int64_t>(
                                egl_env_->context().context()))));
      created_gpu_resources.push_back(LiteRtEnvOption{
          .tag = kLiteRtEnvOptionTagEglContext, .value = egl_context});
      options_.egl_context = egl_env_->context().context();

      LITERT_LOG(LITERT_INFO, "Created default EGL environment.");
#else
      LITERT_LOG(LITERT_INFO, "No default EGL environment created.");
#endif  // LITERT_HAS_OPENGL_SUPPORT
    }
    // If no OpenCL context is provided and EGL options are set, attempt to
    // create a default OpenCL context.
    if (options_.IsGlAware() && properties_.is_gl_sharing_supported) {
      LITERT_RETURN_IF_ERROR(
          tflite::gpu::cl::CreateCLGLContext(
              device_,
              reinterpret_cast<cl_context_properties>(options_.egl_context),
              reinterpret_cast<cl_context_properties>(options_.egl_display),
              &context_)
              .ok())
          << "Failed to create OpenGL-OpenCL shared context";
      LITERT_LOG(LITERT_INFO, "Created default OpenGL-OpenCL shared context.");
    } else {
      LITERT_RETURN_IF_ERROR(
          tflite::gpu::cl::CreateCLContext(device_, &context_).ok())
          << "Failed to create OpenCL context";
      LITERT_LOG(LITERT_INFO, "Created default OpenCL context.");
    }
    // New option: cl_context
    LITERT_ASSIGN_OR_RETURN(
        auto context_id, ToLiteRtAny(LiteRtVariant(
                             reinterpret_cast<int64_t>(context_.context()))));
    created_gpu_resources.push_back(LiteRtEnvOption{
        .tag = kLiteRtEnvOptionTagOpenClContext, .value = context_id});
    options_.context = context_.context();
  }
  // Set up command queue.
  if (options_.command_queue) {
    command_queue_ = tflite::gpu::cl::CLCommandQueue(options_.command_queue,
                                                     /*has_ownership=*/false);
  } else {
    LITERT_RETURN_IF_ERROR(tflite::gpu::cl::CreateCLCommandQueue(
                               device_, context_, &command_queue_)
                               .ok())
        << "Failed to create OpenCL command queue";
    // New option: cl_command_queue
    LITERT_ASSIGN_OR_RETURN(auto command_queue_id,
                            ToLiteRtAny(LiteRtVariant(reinterpret_cast<int64_t>(
                                command_queue_.queue()))));
    created_gpu_resources.push_back(
        LiteRtEnvOption{.tag = kLiteRtEnvOptionTagOpenClCommandQueue,
                        .value = command_queue_id});
    options_.command_queue = command_queue_.queue();

    LITERT_LOG(LITERT_INFO, "Created default OpenCL command queue.");
  }
#else
  LITERT_LOG(LITERT_INFO, "Failed to create OpenCL context.");
#endif  // LITERT_HAS_OPENCL_SUPPORT

#if LITERT_HAS_METAL_SUPPORT
  // Set up Metal.
  if (options_.metal_info) {
    metal_info_ = std::move(options_.metal_info);
    LITERT_LOG(LITERT_INFO, "Created Metal device from provided device id");
  } else {
    MetalInfoPtr metal_info_ptr = nullptr;
    LITERT_RETURN_IF_ERROR(LiteRtCreateMetalInfo(&metal_info_ptr));
    if (metal_info_ptr == nullptr) {
      LITERT_LOG(LITERT_ERROR, "Failed to create default Metal device.");
      return {};
    }
    metal_info_ = std::move(metal_info_ptr);
    LITERT_LOG(LITERT_INFO, "Created default Metal device.");
  }
#endif  // LITERT_HAS_METAL_SUPPORT

  // Add all new options to the LiteRT environment.
  if (!created_gpu_resources.empty()) {
    environment->AddOptions(created_gpu_resources);
  }
#if LITERT_HAS_OPENCL_SUPPORT
  LITERT_LOG(
      LITERT_DEBUG,
      "LiteRT GPU environment initialized: cl_device_id=%p, cl_platform_id=%p, "
      "cl_context=%p, cl_command_queue=%p, egl_context=%p, "
      "egl_display=%p",
      options_.device_id, options_.platform_id, options_.context,
      options_.command_queue, options_.egl_context, options_.egl_display);
#endif  // LITERT_HAS_OPENCL_SUPPORT
  return {};
}

Expected<void> GpuEnvironment::AddEnvironmentOptions(
    absl::Span<const LiteRtEnvOption> options) {
  for (const auto& opt : options) {
    if (opt.tag == kLiteRtEnvOptionTagCallbackOnGpuEnvDestroy) {
      options_.callback_on_destroy = reinterpret_cast<void (*)(void*)>(
          const_cast<void*>(opt.value.ptr_value));
      continue;
    }
    if (opt.tag == kLiteRtEnvOptionTagCallbackUserDataOnGpuEnvDestroy) {
      options_.callback_user_data_on_destroy =
          reinterpret_cast<void*>(const_cast<void*>(opt.value.ptr_value));
      continue;
    }
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      absl::StrFormat("Cannot add the following option to "
                                      "existing GPU environment. Tag: %d",
                                      opt.tag));
  }
  return {};
}

}  // namespace internal
}  // namespace litert
