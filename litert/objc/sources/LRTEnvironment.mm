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

#import "third_party/odml/litert/litert/objc/apis/LRTEnvironment.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_expected.h"
#import "third_party/odml/litert/litert/objc/sources/LRTEnvironment+Internal.h"
#import "third_party/odml/litert/litert/objc/sources/LRTError+Internal.h"

namespace {

id GetBridgedObjectForOption(const litert::Environment &environment,
                             litert::EnvironmentOptions::Tag optionTag) {
  auto options = environment.GetOptions();
  if (!options.HasValue()) return nil;

  auto optionValue = options->GetOption(optionTag);
  if (!optionValue.HasValue()) return nil;

  if (std::holds_alternative<const void *>(*optionValue)) {
    return (__bridge id)std::get<const void *>(*optionValue);
  }
  if (std::holds_alternative<void *>(*optionValue)) {
    return (__bridge id)std::get<void *>(*optionValue);
  }
  return nil;
}

}  // namespace

@implementation LRTEnvironmentOptions

- (BOOL)isEqual:(id)object {
  if (self == object) {
    return YES;
  }
  if (![object isKindOfClass:[LRTEnvironmentOptions class]]) {
    return NO;
  }
  return [self isEqualToEnvironmentOptions:(LRTEnvironmentOptions *)object];
}

- (NSUInteger)hash {
  return [_metalDevice hash] ^ [_metalCommandQueue hash];
}

#pragma mark - NSCopying

- (id)copyWithZone:(NSZone *)zone {
  // Metal devices and command queues are shared GPU handles rather than value objects (they do not
  // conform to NSCopying), and buffers allocated by the caller can only be used with the device
  // that created them. The copy therefore intentionally references the same Metal objects.
  LRTEnvironmentOptions *copy = [[LRTEnvironmentOptions allocWithZone:zone] init];
  copy.metalDevice = _metalDevice;
  copy.metalCommandQueue = _metalCommandQueue;
  return copy;
}

#pragma mark - Public

- (BOOL)isEqualToEnvironmentOptions:(LRTEnvironmentOptions *)otherOptions {
  if (!otherOptions) {
    return NO;
  }
  BOOL devicesMatch =
      (_metalDevice == otherOptions.metalDevice) || [_metalDevice isEqual:otherOptions.metalDevice];
  BOOL queuesMatch = (_metalCommandQueue == otherOptions.metalCommandQueue) ||
                     [_metalCommandQueue isEqual:otherOptions.metalCommandQueue];
  return devicesMatch && queuesMatch;
}

@end

@implementation LRTEnvironment {
  std::unique_ptr<litert::Environment> _cppEnvironment;
}

+ (instancetype)environmentWithOptions:(LRTEnvironmentOptions *)options error:(NSError **)error {
  std::vector<litert::EnvironmentOptions::Option> cppOptions;

  if (options) {
    id<MTLDevice> metalDevice = options.metalDevice;
    if (metalDevice) {
      // LiteRT retains internal ownership of the raw pointer before Environment::Create returns.
      cppOptions.emplace_back(litert::EnvironmentOptions::Tag::kMetalDevice,
                              (__bridge const void *)metalDevice);
    }
    id<MTLCommandQueue> metalCommandQueue = options.metalCommandQueue;
    if (metalCommandQueue) {
      // LiteRT retains internal ownership of the raw pointer before Environment::Create returns.
      cppOptions.emplace_back(litert::EnvironmentOptions::Tag::kMetalCommandQueue,
                              (__bridge const void *)metalCommandQueue);
    }
  }

  auto environmentResult = litert::Environment::Create(litert::EnvironmentOptions(cppOptions));
  if (!environmentResult.HasValue()) {
    LRTSetErrorFromCppError(error, environmentResult.Error());
    return nil;
  }

  auto cppEnvironment = std::make_unique<litert::Environment>(std::move(environmentResult.Value()));
  return [[LRTEnvironment alloc] initInternalWithEnvironment:std::move(cppEnvironment)];
}

- (instancetype)initInternalWithEnvironment:(std::unique_ptr<litert::Environment>)cppEnvironment {
  self = [super init];
  if (self) {
    _cppEnvironment = std::move(cppEnvironment);
  }
  return self;
}

#pragma mark - Properties

- (id<MTLDevice>)metalDevice {
  if (!_cppEnvironment) return nil;
  id bridgedObject =
      GetBridgedObjectForOption(*_cppEnvironment, litert::EnvironmentOptions::Tag::kMetalDevice);
  return (id<MTLDevice>)bridgedObject;
}

- (id<MTLCommandQueue>)metalCommandQueue {
  if (!_cppEnvironment) return nil;
  id bridgedObject = GetBridgedObjectForOption(*_cppEnvironment,
                                               litert::EnvironmentOptions::Tag::kMetalCommandQueue);
  return (id<MTLCommandQueue>)bridgedObject;
}

#pragma mark - LRTEnvironment (Internal)

- (litert::Environment *)cppEnvironment {
  return _cppEnvironment.get();
}

@end
