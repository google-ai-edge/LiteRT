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

#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_logging.h"
#include "litert/cc/litert_expected.h"
#include "litert/core/environment.h"
#include "tflite/delegates/gpu/cl/cl_command_queue.h"
#include "tflite/delegates/gpu/cl/cl_context.h"
#include "tflite/delegates/gpu/cl/cl_device.h"

#if LITERT_HAS_OPENGL_SUPPORT
#include "tflite/delegates/gpu/gl/egl_environment.h"
#endif  // LITERT_HAS_OPENGL_SUPPORT

namespace litert::internal {

// Inner singleton class that is for storing the MLD global environment.
// This class is used to store OpenCL, OpenGL environment objects.
class GpuEnvironmentSingleton {
 public:
  GpuEnvironmentSingleton(const GpuEnvironmentSingleton&) = delete;
  GpuEnvironmentSingleton& operator=(const GpuEnvironmentSingleton&) = delete;
  ~GpuEnvironmentSingleton() = default;
  tflite::gpu::cl::CLDevice* getDevice() { return &device_; }
  tflite::gpu::cl::CLContext* getContext() { return &context_; }
  tflite::gpu::cl::CLCommandQueue* getCommandQueue() { return &command_queue_; }
#if LITERT_HAS_OPENGL_SUPPORT
  tflite::gpu::gl::EglEnvironment* getEglEnvironment() {
    return egl_env_.get();
  }
#endif  // LITERT_HAS_OPENGL_SUPPORT

  static GpuEnvironmentSingleton& GetInstance() {
    if (instance_ == nullptr) {
      instance_ = new GpuEnvironmentSingleton(nullptr);
    }
    return *instance_;
  }

  // Create the singleton instance with the given environment.
  // It will fail if the singleton instance already exists.
  static Expected<GpuEnvironmentSingleton*> Create(
      LiteRtEnvironmentT* environment) {
    if (instance_ == nullptr) {
      instance_ = new GpuEnvironmentSingleton(environment);
      LITERT_LOG(LITERT_INFO, "Created LiteRT EnvironmentSingleton.");
    } else {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "EnvironmentSingleton already exists");
    }
    return instance_;
  }

 private:
  // Load the OpenCL device, context and command queue from the environment if
  // available. Otherwise, create the default device, context and command queue.
  explicit GpuEnvironmentSingleton(LiteRtEnvironmentT* environment);

  static GpuEnvironmentSingleton* instance_;
  tflite::gpu::cl::CLDevice device_;
  tflite::gpu::cl::CLContext context_;
  tflite::gpu::cl::CLCommandQueue command_queue_;
#if LITERT_HAS_OPENGL_SUPPORT
  std::unique_ptr<tflite::gpu::gl::EglEnvironment> egl_env_;
#endif  // LITERT_HAS_OPENGL_SUPPORT
};

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_RUNTIME_GPU_ENVIRONMENT_H_
