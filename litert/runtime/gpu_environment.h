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

#ifndef ODML_LITERT_LITERT_RUNTIME_GPU_ENVIRONMENT_H_
#define ODML_LITERT_LITERT_RUNTIME_GPU_ENVIRONMENT_H_

#include <memory>
#include <utility>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_gl_types.h"
#include "litert/cc/litert_expected.h"

#if LITERT_HAS_OPENCL_SUPPORT
#include <CL/cl.h>
#include "tflite/delegates/gpu/cl/cl_command_queue.h"
#include "tflite/delegates/gpu/cl/cl_context.h"
#include "tflite/delegates/gpu/cl/cl_device.h"
#endif  // LITERT_HAS_OPENCL_SUPPORT

#if LITERT_HAS_OPENGL_SUPPORT
#include "tflite/delegates/gpu/gl/egl_environment.h"
#endif  // LITERT_HAS_OPENGL_SUPPORT

#if LITERT_HAS_WEBGPU_SUPPORT
// TODO b/422216124: Use the WebGPU headers directly.
typedef struct WGPUDeviceImpl* WGPUDevice;
typedef struct WGPUQueueImpl* WGPUQueue;
#endif  // LITERT_HAS_WEBGPU_SUPPORT

namespace litert::internal {

#if LITERT_HAS_METAL_SUPPORT
#include "litert/runtime/metal_info.h"
#endif  // LITERT_HAS_METAL_SUPPORT

struct GpuEnvironmentProperties {
  bool is_opencl_available = false;

  // Indicates whether GL objects (buffers and textures) can be shared with CL
  // context.
  bool is_gl_sharing_supported = false;

  // Indicates whether fast GL->CL synchronization is supported.
  bool is_gl_to_cl_fast_sync_supported = false;

  // Indicates whether fast CL->GL synchronization is supported.
  bool is_cl_to_gl_fast_sync_supported = false;

  // Indicates whether AHWB->CL interop is supported.
  bool is_ahwb_cl_interop_supported = false;

  // Indicates whether AHWB->GL interop is supported.
  bool is_ahwb_gl_interop_supported = false;

  // Indicates whether Metal is available.
  bool is_metal_available = false;
};

struct GpuEnvironmentOptions {
  void (*callback_on_destroy)(void*) = nullptr;
  void* callback_user_data_on_destroy = nullptr;

  // If any of these objects are set, created environment will use them instead
  // of creating/choosing own instances.
#if LITERT_HAS_OPENCL_SUPPORT
  cl_device_id device_id = nullptr;
  cl_platform_id platform_id = nullptr;
  cl_context context = nullptr;
  cl_command_queue command_queue = nullptr;
#endif  // LITERT_HAS_OPENCL_SUPPORT

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

#if LITERT_HAS_WEBGPU_SUPPORT
  WGPUDevice webgpu_device = nullptr;
  WGPUQueue webgpu_queue = nullptr;
#endif  // LITERT_HAS_WEBGPU_SUPPORT

#if LITERT_HAS_METAL_SUPPORT
  MetalInfoPtr metal_info;
#endif  // LITERT_HAS_METAL_SUPPORT

#if LITERT_HAS_VULKAN_SUPPORT
  void* vulkan_env = nullptr;
#endif  // LITERT_HAS_VULKAN_SUPPORT
};

// A class for storing the MLD global environment and kept in Environment.
// This class is used to store OpenCL, OpenGL environment objects.
class GpuEnvironment {
 public:
  GpuEnvironment(const GpuEnvironment&) = delete;
  GpuEnvironment& operator=(const GpuEnvironment&) = delete;
  GpuEnvironment() = default;
  ~GpuEnvironment();

#if LITERT_HAS_OPENCL_SUPPORT
  tflite::gpu::cl::CLDevice* GetDevice() { return &device_; }
  tflite::gpu::cl::CLContext* GetContext() { return &context_; }
  tflite::gpu::cl::CLCommandQueue* GetCommandQueue() { return &command_queue_; }
#endif  // LITERT_HAS_OPENCL_SUPPORT
  EGLDisplay GetEglDisplay() { return options_.egl_display; }
  EGLContext GetEglContext() { return options_.egl_context; }

#if LITERT_HAS_WEBGPU_SUPPORT
  WGPUDevice GetWebGpuDevice() { return options_.webgpu_device; }
  WGPUQueue GetWebGpuQueue() { return options_.webgpu_queue; }
#endif  // LITERT_HAS_WEBGPU_SUPPORT

#if LITERT_HAS_METAL_SUPPORT
  void* GetMetalDevice() { return metal_info_->metal_info; }
#endif  // LITERT_HAS_METAL_SUPPORT

#if LITERT_HAS_VULKAN_SUPPORT
  void* GetVulkanEnvironment() { return options_.vulkan_env; }
#endif  // LITERT_HAS_VULKAN_SUPPORT

  // Create a GpuEnvironment with the given environment.
  static Expected<std::unique_ptr<GpuEnvironment>> Create(
      LiteRtEnvironmentT* environment) {
    auto instance = std::make_unique<GpuEnvironment>();
    instance->Initialize(environment);
    LITERT_LOG(LITERT_INFO, "Created LiteRT GpuEnvironment.");
    return std::move(instance);
  }

  bool SupportsClGlInterop() { return properties_.is_gl_sharing_supported; }

  bool SupportsAhwbClInterop() {
    return properties_.is_ahwb_cl_interop_supported;
  }

  bool SupportsAhwbGlInterop() {
    return properties_.is_ahwb_gl_interop_supported;
  }

  // Adds options to the existing GPU environment.
  Expected<void> AddEnvironmentOptions(
      absl::Span<const LiteRtEnvOption> options);

 private:
  // Load the OpenCL device, context and command queue from the environment if
  // available. Otherwise, create the default device, context and command queue.
  Expected<void> Initialize(LiteRtEnvironment environment);

#if LITERT_HAS_OPENCL_SUPPORT
  tflite::gpu::cl::CLDevice device_;
  tflite::gpu::cl::CLContext context_;
  tflite::gpu::cl::CLCommandQueue command_queue_;
#endif  // LITERT_HAS_OPENCL_SUPPORT

#if LITERT_HAS_OPENGL_SUPPORT
  std::unique_ptr<tflite::gpu::gl::EglEnvironment> egl_env_;
#endif  // LITERT_HAS_OPENGL_SUPPORT

#if LITERT_HAS_METAL_SUPPORT
  MetalInfoPtr metal_info_;
#endif  // LITERT_HAS_METAL_SUPPORT

  GpuEnvironmentOptions options_;
  GpuEnvironmentProperties properties_;
};

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_RUNTIME_GPU_ENVIRONMENT_H_
