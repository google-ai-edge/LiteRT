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

#include "litert/runtime/metal_info.h"

#import <Metal/Metal.h>

#include <memory>

#include "base/casts.h"
#include "litert/c/litert_common.h"

namespace {

struct MetalInfoImpl : public MetalInfo {
  id<MTLDevice> metal_device;
};

}  // namespace

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

LiteRtStatus LiteRtCreateMetalInfo(MetalInfoPtr* metal_info) {
  auto metal_info_impl = std::make_unique<MetalInfoImpl>(
      MetalInfoImpl{.metal_device = MTLCreateSystemDefaultDevice()});
  metal_info_impl->metal_info = (__bridge void*)metal_info_impl->metal_device;
  *metal_info = metal_info_impl.release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCreateWithDevice(void* device, MetalInfoPtr* metal_info) {
  auto metal_info_impl = std::make_unique<MetalInfoImpl>(
      MetalInfoImpl{.metal_device = (__bridge id<MTLDevice>)device});
  metal_info_impl->metal_info = (__bridge void*)metal_info_impl->metal_device;
  *metal_info = metal_info_impl.release();
  return kLiteRtStatusOk;
}

void LiteRtDeleteMetalInfo(MetalInfoPtr metal_info) {
  if (metal_info) {
    delete metal_info;
  }
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
