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

#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_logging.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/environment.h"

#if LITERT_HAS_OPENCL_SUPPORT
#include <CL/cl.h>
#include "tflite/delegates/gpu/cl/cl_command_queue.h"
#include "tflite/delegates/gpu/cl/cl_context.h"
#include "tflite/delegates/gpu/cl/cl_device.h"
#include "tflite/delegates/gpu/cl/opencl_wrapper.h"
#endif  // LITERT_HAS_OPENCL_SUPPORT

#if LITERT_HAS_OPENGL_SUPPORT
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
  return options;
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
    LITERT_LOG(LITERT_INFO, "Created default OpenCL device.");
  }
#endif  // LITERT_HAS_OPENCL_SUPPORT

  // Set up remaining properties.
#if LITERT_HAS_OPENGL_SUPPORT
  properties_.is_gl_sharing_supported =
      tflite::gpu::cl::IsGlSharingSupported(device_);
  properties_.is_gl_to_cl_fast_sync_supported =
      tflite::gpu::cl::IsClEventFromEglSyncSupported(device_);
  properties_.is_cl_to_gl_fast_sync_supported =
      tflite::gpu::cl::IsEglSyncFromClEventSupported();
#endif  // LITERT_HAS_OPENGL_SUPPORT

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
#endif  // LITERT_HAS_OPENGL_SUPPORT
    } else {
      context_ = tflite::gpu::cl::CLContext(options_.context,
                                            /*has_ownership=*/false);
      LITERT_LOG(LITERT_INFO, "Created OpenCL context from provided context.");
    }
  } else {
    // If no CL context is provided and no EGL options are set, attempt to
    // create a default EGL Environment.
    if (!options_.IsGlAware()) {
#if LITERT_HAS_OPENGL_SUPPORT
      std::unique_ptr<tflite::gpu::gl::EglEnvironment> egl_env;
      LITERT_RETURN_IF_ERROR(
          tflite::gpu::gl::EglEnvironment::NewEglEnvironment(&egl_env).ok())
          << "Failed to create EGL environment";
      egl_env_ = std::move(egl_env);
      options_.egl_display = egl_env_->display();
      options_.egl_context = egl_env_->context().context();
      LITERT_LOG(LITERT_INFO, "Created default EGL environment.");
#else
      LITERT_LOG(LITERT_INFO, "No default EGL environment created.");
#endif  // LITERT_HAS_OPENGL_SUPPORT
    }
    if (options_.IsGlAware() && properties_.is_gl_sharing_supported) {
      auto status = tflite::gpu::cl::CreateCLGLContext(
          device_,
          reinterpret_cast<cl_context_properties>(options_.egl_context),
          reinterpret_cast<cl_context_properties>(options_.egl_display),
          &context_);
      LITERT_RETURN_IF_ERROR(
          tflite::gpu::cl::CreateCLGLContext(
              device_,
              reinterpret_cast<cl_context_properties>(options_.egl_context),
              reinterpret_cast<cl_context_properties>(options_.egl_display),
              &context_)
              .ok())
          << "Failed to create OpenGL-OpenCL shared context";
      LITERT_LOG(LITERT_INFO, "Created OpenGL-OpenCL shared context.");
    } else {
      LITERT_RETURN_IF_ERROR(
          tflite::gpu::cl::CreateCLContext(device_, &context_).ok())
          << "Failed to create OpenCL context";
      LITERT_LOG(LITERT_INFO, "Created OpenCL context.");
    }
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
    LITERT_LOG(LITERT_INFO, "Created OpenCL command queue.");
  }
#else
  LITERT_LOG(LITERT_INFO, "Failed to create OpenCL context.");
#endif  // LITERT_HAS_OPENCL_SUPPORT

  return {};
}

}  // namespace internal
}  // namespace litert
