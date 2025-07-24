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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_GPU_ENVIRONMENT_METAL_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_GPU_ENVIRONMENT_METAL_H_

#import <Metal/Metal.h>

#include "litert/runtime/gpu_environment.h"
#include "tflite/delegates/gpu/metal/metal_device.h"

namespace litert::internal {

// using ::tflite::gpu::MetalInfo;

class MetalInfoImpl : public MetalInfo {
 public:
  ~MetalInfoImpl() override = default;
  explicit MetalInfoImpl(void* device) {
    device_ = (__bridge id<MTLDevice>)(device);
    metal_device_ = tflite::gpu::metal::MetalDevice(device_);
  }
  MetalInfoImpl() {
    metal_device_ = tflite::gpu::metal::MetalDevice();
    device_ = metal_device_.device();
  };
  void* GetDevice() override { return (__bridge void*)(device_); }

  tflite::gpu::metal::MetalDevice* GetMetalDevice() { return &metal_device_; }

  bool IsMetalAvailable() override { return device_ != nullptr; }

 private:
  id<MTLDevice> device_;
  tflite::gpu::metal::MetalDevice metal_device_;
};
}  // namespace litert::internal

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_GPU_ENVIRONMENT_METAL_H_
