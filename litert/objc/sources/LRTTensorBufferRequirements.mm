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

#import "third_party/odml/litert/litert/objc/apis/LRTTensorBufferRequirements.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_tensor_buffer_requirements.h"
#include "litert/cc/litert_tensor_buffer_types.h"
#import "third_party/odml/litert/litert/objc/sources/LRTError+Internal.h"
#import "third_party/odml/litert/litert/objc/sources/LRTTensorBufferRequirements+Internal.h"

@interface LRTTensorBufferRequirements ()

- (instancetype)initWithSupportedBufferTypes:(NSArray<NSNumber *> *)supportedBufferTypes
                                  bufferSize:(NSUInteger)bufferSize
                                   alignment:(NSUInteger)alignment
                                     strides:(NSArray<NSNumber *> *)strides
    NS_DESIGNATED_INITIALIZER;

@end

@implementation LRTTensorBufferRequirements

+ (instancetype)requirementsWithCppRequirements:
                    (const litert::TensorBufferRequirements &)cppRequirements
                                          error:(NSError **)error {
  litert::Expected<std::vector<litert::TensorBufferType>> supportedTypesResult =
      cppRequirements.SupportedTypes();
  if (!supportedTypesResult.HasValue()) {
    LRTSetErrorFromCppError(error, supportedTypesResult.Error());
    return nil;
  }

  litert::Expected<size_t> bufferSizeResult = cppRequirements.BufferSize();
  if (!bufferSizeResult.HasValue()) {
    LRTSetErrorFromCppError(error, bufferSizeResult.Error());
    return nil;
  }

  litert::Expected<size_t> alignmentResult = cppRequirements.Alignment();
  if (!alignmentResult.HasValue()) {
    LRTSetErrorFromCppError(error, alignmentResult.Error());
    return nil;
  }

  litert::Expected<litert::Span<const uint32_t>> stridesResult = cppRequirements.Strides();
  if (!stridesResult.HasValue()) {
    LRTSetErrorFromCppError(error, stridesResult.Error());
    return nil;
  }

  const std::vector<litert::TensorBufferType> &cppSupportedTypes = supportedTypesResult.Value();
  NSMutableArray<NSNumber *> *supportedBufferTypes =
      [NSMutableArray arrayWithCapacity:cppSupportedTypes.size()];
  for (litert::TensorBufferType bufferType : cppSupportedTypes) {
    [supportedBufferTypes addObject:@(static_cast<LRTTensorBufferType>(bufferType))];
  }

  litert::Span<const uint32_t> cppStrides = stridesResult.Value();
  NSMutableArray<NSNumber *> *strides = [NSMutableArray arrayWithCapacity:cppStrides.size()];
  for (uint32_t stride : cppStrides) {
    [strides addObject:@(stride)];
  }

  return
      [[self alloc] initWithSupportedBufferTypes:supportedBufferTypes
                                      bufferSize:static_cast<NSUInteger>(bufferSizeResult.Value())
                                       alignment:static_cast<NSUInteger>(alignmentResult.Value())
                                         strides:strides];
}

- (instancetype)initWithSupportedBufferTypes:(NSArray<NSNumber *> *)supportedBufferTypes
                                  bufferSize:(NSUInteger)bufferSize
                                   alignment:(NSUInteger)alignment
                                     strides:(NSArray<NSNumber *> *)strides {
  self = [super init];
  if (self) {
    _supportedBufferTypes = [supportedBufferTypes copy];
    _bufferSize = bufferSize;
    _alignment = alignment;
    _strides = [strides copy];
  }
  return self;
}

@end
