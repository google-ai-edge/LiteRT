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

#include "litert/c/litert_common.h"
#include "litert/c/litert_gl_types.h"
#include "litert/c/litert_logging.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

#if LITERT_HAS_OPENCL_SUPPORT
#include <CL/cl.h>
#include "tflite/delegates/gpu/cl/cl_command_queue.h"
#include "tflite/delegates/gpu/cl/cl_context.h"
#include "tflite/delegates/gpu/cl/cl_device.h"
#endif  // LITERT_HAS_OPENCL_SUPPORT

#if LITERT_HAS_OPENGL_SUPPORT
#include "tflite/delegates/gpu/gl/egl_environment.h"
#endif  // LITERT_HAS_OPENGL_SUPPORT

namespace litert::internal {

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
};

struct GpuEnvironmentOptions {
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
};

// A class for storing the MLD global environment and kept in Environment.
// This class is used to store OpenCL, OpenGL environment objects.
class GpuEnvironment {
 public:
  GpuEnvironment(const GpuEnvironment&) = delete;
  GpuEnvironment& operator=(const GpuEnvironment&) = delete;
  GpuEnvironment() = default;
  ~GpuEnvironment() = default;
#if LITERT_HAS_OPENCL_SUPPORT
  tflite::gpu::cl::CLDevice* getDevice() { return &device_; }
  tflite::gpu::cl::CLContext* getContext() { return &context_; }
  tflite::gpu::cl::CLCommandQueue* getCommandQueue() { return &command_queue_; }
#endif  // LITERT_HAS_OPENCL_SUPPORT
  EGLDisplay getEglDisplay() { return options_.egl_display; }
  EGLContext getEglContext() { return options_.egl_context; }

  // Create a GpuEnvironment with the given environment.
  static Expected<std::unique_ptr<GpuEnvironment>> Create(
      LiteRtEnvironmentT* environment) {
    auto instance = new GpuEnvironment();
    instance->Initialize(environment);
    LITERT_LOG(LITERT_INFO, "Created LiteRT GpuEnvironment.");
    return std::unique_ptr<GpuEnvironment>(instance);
  }

  bool SupportsClGlInterop() { return properties_.is_gl_sharing_supported; }

  bool SupportsAhwbClInterop() {
    return properties_.is_ahwb_cl_interop_supported;
  }

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
  GpuEnvironmentOptions options_;
  GpuEnvironmentProperties properties_;
};

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_RUNTIME_GPU_ENVIRONMENT_H_
