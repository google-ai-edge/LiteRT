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
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_logging.h"
#include "litert/core/environment.h"
#include <CL/cl.h>
#include "tflite/delegates/gpu/cl/cl_command_queue.h"
#include "tflite/delegates/gpu/cl/cl_context.h"
#include "tflite/delegates/gpu/cl/cl_device.h"
#include "tflite/delegates/gpu/cl/opencl_wrapper.h"

namespace litert {
namespace internal {

GpuEnvironmentSingleton::GpuEnvironmentSingleton(
    LiteRtEnvironmentT* environment) {
// Set up OpenGL. Reuses context and display if created on this thread.
#if LITERT_HAS_OPENGL_SUPPORT
  std::unique_ptr<tflite::gpu::gl::EglEnvironment> egl_env;
  auto status = tflite::gpu::gl::EglEnvironment::NewEglEnvironment(&egl_env);
  if (!status.ok()) {
    LITERT_LOG(LITERT_ERROR, "Failed to create EGL environment");
  } else {
    egl_env_ = std::move(egl_env);
  }
#endif  // LITERT_HAS_OPENGL_SUPPORT

  // Set up OpenCL.
  cl_device_id device_id = nullptr;
  cl_platform_id platform_id = nullptr;
  cl_context context = nullptr;
  cl_command_queue command_queue = nullptr;
  if (!tflite::gpu::cl::LoadOpenCL().ok()) {
    LITERT_LOG(LITERT_ERROR, "Failed to load OpenCL for LiteRT.");
  }
  if (environment) {
    auto device_option =
        environment->GetOption(kLiteRtEnvOptionTagOpenClDeviceId);
    if (device_option.has_value() && device_option->type == kLiteRtAnyTypeInt) {
      device_id = reinterpret_cast<cl_device_id>(device_option->int_value);
    }
    auto platform_option =
        environment->GetOption(kLiteRtEnvOptionTagOpenClPlatformId);
    if (platform_option.has_value() &&
        platform_option->type == kLiteRtAnyTypeInt) {
      platform_id =
          reinterpret_cast<cl_platform_id>(platform_option->int_value);
    }
    auto context_option =
        environment->GetOption(kLiteRtEnvOptionTagOpenClContext);
    if (context_option.has_value() &&
        context_option->type == kLiteRtAnyTypeInt) {
      context = reinterpret_cast<cl_context>(context_option->int_value);
    }
    auto command_queue_option =
        environment->GetOption(kLiteRtEnvOptionTagOpenClCommandQueue);
    if (command_queue_option.has_value() &&
        command_queue_option->type == kLiteRtAnyTypeInt) {
      command_queue =
          reinterpret_cast<cl_command_queue>(command_queue_option->int_value);
    }
  }
  if (device_id && platform_id) {
    device_ = tflite::gpu::cl::CLDevice(device_id, platform_id);
  } else {
    auto status = tflite::gpu::cl::CreateDefaultGPUDevice(&device_);
    if (!status.ok()) {
      LITERT_LOG(LITERT_ERROR, "Failed to create OpenCL device");
    }
  }
  if (context) {
    context_ = tflite::gpu::cl::CLContext(context, /*has_ownership=*/false);
  } else {
    auto status = tflite::gpu::cl::CreateCLContext(device_, &context_);
    if (!status.ok()) {
      LITERT_LOG(LITERT_ERROR, "Failed to create OpenCL contxt");
    }
  }
  if (command_queue) {
    command_queue_ =
        tflite::gpu::cl::CLCommandQueue(command_queue, /*has_ownership=*/false);
  } else {
    auto status = tflite::gpu::cl::CreateCLCommandQueue(device_, context_,
                                                        &command_queue_);
    if (!status.ok()) {
      LITERT_LOG(LITERT_ERROR, "Failed to create OpenCL command queue");
    }
  }
}

GpuEnvironmentSingleton* GpuEnvironmentSingleton::instance_ = nullptr;

}  // namespace internal
}  // namespace litert
