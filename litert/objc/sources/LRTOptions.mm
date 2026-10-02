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
#import "third_party/odml/litert/litert/objc/sources/LRTOptions+Internal.h"

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

- (BOOL)isEqual:(nullable id)object {
  if (self == object) {
    return YES;
  }
  if (![object isKindOfClass:[LRTOptions class]]) {
    return NO;
  }
  return [self isEqualToOptions:(LRTOptions *)object];
}

- (NSUInteger)hash {
  return _hardwareAccelerators ^ (_usesMetalArgumentBuffers ? (1UL << 8) : 0) ^
         (_enablesMetalResidencySet ? (1UL << 9) : 0);
}

#pragma mark - NSCopying

- (id)copyWithZone:(nullable NSZone *)zone {
  LRTOptions *copy =
      [[LRTOptions allocWithZone:zone] initWithHardwareAccelerators:_hardwareAccelerators];
  if (_usesMetalArgumentBuffers) {
    copy.usesMetalArgumentBuffers = YES;
  }
  if (_enablesMetalResidencySet) {
    copy.enablesMetalResidencySet = YES;
  }
  return copy;
}

#pragma mark - Properties

- (void)setUsesMetalArgumentBuffers:(BOOL)usesMetalArgumentBuffers {
  _usesMetalArgumentBuffers = usesMetalArgumentBuffers;
  [self applyMetalOptions];
}

- (void)setEnablesMetalResidencySet:(BOOL)enablesMetalResidencySet {
  _enablesMetalResidencySet = enablesMetalResidencySet;
  [self applyMetalOptions];
}

#pragma mark - Public

- (BOOL)isEqualToOptions:(LRTOptions *)otherOptions {
  if (!otherOptions) {
    return NO;
  }
  return _hardwareAccelerators == otherOptions.hardwareAccelerators &&
         _usesMetalArgumentBuffers == otherOptions.usesMetalArgumentBuffers &&
         _enablesMetalResidencySet == otherOptions.enablesMetalResidencySet;
}

#pragma mark - LRTOptions (Internal)

- (litert::Options *)cppOptions {
  return &_cppOptions;
}

#pragma mark - Private

/**
 * Mirrors the Metal properties onto the underlying C++ GPU options.
 *
 * The C++ API lazily creates the GPU options on first access, so this is only called from the
 * Metal property setters: options that never mention Metal keep their default, GPU-free C++
 * representation.
 */
- (void)applyMetalOptions {
#if defined(__APPLE__)
  auto gpuOptions = _cppOptions.GetGpuOptions();
  if (gpuOptions.HasValue()) {
    gpuOptions->SetUseMetalArgumentBuffers(_usesMetalArgumentBuffers);
    gpuOptions->EnableMetalResidencySet(_enablesMetalResidencySet);
  }
#endif  // defined(__APPLE__)
}

@end
