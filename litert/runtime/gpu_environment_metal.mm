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

#import <Metal/Metal.h>

#include "litert/runtime/gpu_environment.h"
#include "tflite/delegates/gpu/metal/metal_device.h"

namespace litert::internal {
namespace {

class MetalInfoImpl : public MetalInfo {
 public:
  MetalInfoImpl() : metal_device_(tflite::gpu::metal::MetalDevice()) {}
  explicit MetalInfoImpl(id<MTLDevice> device)
      : metal_device_(tflite::gpu::metal::MetalDevice(device)) {}

  ~MetalInfoImpl() override = default;

  // Implementation of MetalInfo.
  void* GetDevice() override { return (__bridge void*)(metal_device_.device()); }
  bool IsMetalAvailable() override { return metal_device_.device() != nullptr; }

 private:
  const tflite::gpu::metal::MetalDevice metal_device_;
};

}  // namespace

MetalInfoPtr MetalInfo::Create() { return std::make_unique<MetalInfoImpl>(); }

MetalInfoPtr MetalInfo::CreateWithDevice(void* device) {
  return std::make_unique<MetalInfoImpl>((__bridge id<MTLDevice>)device);
}

}  // namespace litert::internal
