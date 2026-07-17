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

#import "third_party/odml/litert/litert/objc/apis/LRTOptions.h"

#include <cstdint>

#include "litert/cc/litert_options.h"

namespace {

constexpr LRTHardwareAccelerators kValidAcceleratorsMask =
    LRTHardwareAcceleratorCPU | LRTHardwareAcceleratorGPU | LRTHardwareAcceleratorNPU;

}  // namespace

@implementation LRTOptions {
  litert::Options _cppOptions;
}

- (instancetype)initWithHardwareAccelerators:(LRTHardwareAccelerators)hardwareAccelerators {
  NSParameterAssert((hardwareAccelerators & ~kValidAcceleratorsMask) == 0);
  self = [super init];
  if (self) {
    _hardwareAccelerators = hardwareAccelerators;
    _cppOptions.SetHardwareAccelerators(
        litert::HwAcceleratorSet(static_cast<int>(hardwareAccelerators)));
  }
  return self;
}

- (instancetype)init {
  return [self initWithHardwareAccelerators:LRTHardwareAcceleratorNone];
}

- (litert::Options *)cppOptions {
  return &_cppOptions;
}

@end
