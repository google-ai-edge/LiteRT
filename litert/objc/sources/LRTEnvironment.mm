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
#import "third_party/odml/litert/litert/objc/apis/LRTError.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_expected.h"

namespace {

id _Nullable GetBridgedObjectForOption(const litert::Environment &env,
                                       litert::EnvironmentOptions::Tag optionTag) {
  auto options = env.GetOptions();
  if (!options.HasValue()) return nil;

  auto optionVal = options->GetOption(optionTag);
  if (!optionVal.HasValue()) return nil;

  if (std::holds_alternative<const void *>(*optionVal)) {
    return (__bridge id)std::get<const void *>(*optionVal);
  }
  if (std::holds_alternative<void *>(*optionVal)) {
    return (__bridge id)std::get<void *>(*optionVal);
  }
  return nil;
}

}  // namespace

@implementation LRTEnvironmentOptions
@end

@interface LRTEnvironment ()

- (instancetype)initInternalWithEnvironment:(std::unique_ptr<litert::Environment>)cppEnvironment
    NS_DESIGNATED_INITIALIZER;

@end

@implementation LRTEnvironment {
  std::unique_ptr<litert::Environment> _cppEnvironment;
}
- (instancetype)initInternalWithEnvironment:(std::unique_ptr<litert::Environment>)cppEnvironment {
  self = [super init];
  if (self) {
    _cppEnvironment = std::move(cppEnvironment);
  }
  return self;
}

+ (nullable instancetype)environmentWithOptions:(nullable LRTEnvironmentOptions *)options
                                          error:(NSError **)error {
  std::vector<litert::EnvironmentOptions::Option> cppOptions;

  if (options) {
    if (options.metalDevice) {
      // LiteRT retains internal ownership of the raw pointer before Environment::Create returns.
      cppOptions.emplace_back(litert::EnvironmentOptions::Tag::kMetalDevice,
                              (__bridge const void *)options.metalDevice);
    }
    if (options.metalCommandQueue) {
      // LiteRT retains internal ownership of the raw pointer before Environment::Create returns.
      cppOptions.emplace_back(litert::EnvironmentOptions::Tag::kMetalCommandQueue,
                              (__bridge const void *)options.metalCommandQueue);
    }
  }

  auto envResult = litert::Environment::Create(litert::EnvironmentOptions(cppOptions));
  if (!envResult.HasValue()) {
    if (error) {
      NSDictionary *userInfo = @{
        NSLocalizedDescriptionKey :
            [NSString stringWithUTF8String:envResult.Error().Message().c_str()]
      };
      *error = [NSError errorWithDomain:LRTErrorDomain
                                   code:static_cast<NSInteger>(envResult.Error().Status())
                               userInfo:userInfo];
    }
    return nil;
  }

  auto cppEnv = std::make_unique<litert::Environment>(std::move(envResult.Value()));
  return [[LRTEnvironment alloc] initInternalWithEnvironment:std::move(cppEnv)];
}

- (nullable id<MTLDevice>)metalDevice {
  if (!_cppEnvironment) return nil;
  id obj =
      GetBridgedObjectForOption(*_cppEnvironment, litert::EnvironmentOptions::Tag::kMetalDevice);
  return (id<MTLDevice>)obj;
}

- (nullable id<MTLCommandQueue>)metalCommandQueue {
  if (!_cppEnvironment) return nil;
  id obj = GetBridgedObjectForOption(*_cppEnvironment,
                                     litert::EnvironmentOptions::Tag::kMetalCommandQueue);
  return (id<MTLCommandQueue>)obj;
}

@end
