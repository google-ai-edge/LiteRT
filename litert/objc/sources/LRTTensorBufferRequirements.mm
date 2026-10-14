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
#import "third_party/odml/litert/litert/objc/apis/LRTTensorBuffer.h"
#import "third_party/odml/litert/litert/objc/sources/LRTError+Internal.h"
#import "third_party/odml/litert/litert/objc/sources/LRTTensorBufferRequirements+Internal.h"

@implementation LRTTensorBufferRequirements

+ (instancetype)requirementsWithCPPRequirements:
                    (const litert::TensorBufferRequirements &)cppRequirements
                                          error:(NSError **)error {
  litert::Expected<std::vector<litert::TensorBufferType>> supportedTypesResult =
      cppRequirements.SupportedTypes();
  if (!supportedTypesResult.HasValue()) {
    LRTSetErrorFromCPPError(error, supportedTypesResult.Error());
    return nil;
  }

  litert::Expected<size_t> bufferSizeResult = cppRequirements.BufferSize();
  if (!bufferSizeResult.HasValue()) {
    LRTSetErrorFromCPPError(error, bufferSizeResult.Error());
    return nil;
  }

  litert::Expected<size_t> alignmentResult = cppRequirements.Alignment();
  if (!alignmentResult.HasValue()) {
    LRTSetErrorFromCPPError(error, alignmentResult.Error());
    return nil;
  }

  litert::Expected<litert::Span<const uint32_t>> stridesResult = cppRequirements.Strides();
  if (!stridesResult.HasValue()) {
    LRTSetErrorFromCPPError(error, stridesResult.Error());
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

- (BOOL)isEqual:(id)object {
  if (self == object) {
    return YES;
  }
  if (![object isKindOfClass:[LRTTensorBufferRequirements class]]) {
    return NO;
  }
  return [self isEqualToRequirements:(LRTTensorBufferRequirements *)object];
}

- (NSUInteger)hash {
  const NSUInteger kHashPrime = 31;
  NSUInteger hash = _bufferSize;
  hash = (hash * kHashPrime) + _alignment;
  for (NSNumber *bufferType in _supportedBufferTypes) {
    hash = (hash * kHashPrime) + static_cast<NSUInteger>(bufferType.integerValue);
  }
  for (NSNumber *stride in _strides) {
    hash = (hash * kHashPrime) + stride.unsignedIntegerValue;
  }
  return hash;
}

- (NSString *)description {
  return [NSString
      stringWithFormat:@"<%@: %p, supportedBufferTypes=%@, bufferSize=%lu, alignment=%lu, "
                       @"strides=%@>",
                       NSStringFromClass([self class]), self, _supportedBufferTypes,
                       (unsigned long)_bufferSize, (unsigned long)_alignment, _strides];
}

#pragma mark - NSCopying

- (id)copyWithZone:(NSZone *)zone {
  // LRTTensorBufferRequirements is deeply immutable, so returning self is safe.
  return self;
}

#pragma mark - Private

- (BOOL)isEqualToRequirements:(LRTTensorBufferRequirements *)otherRequirements {
  if (!otherRequirements) {
    return NO;
  }
  return _bufferSize == otherRequirements.bufferSize && _alignment == otherRequirements.alignment &&
         [_supportedBufferTypes isEqualToArray:otherRequirements.supportedBufferTypes] &&
         [_strides isEqualToArray:otherRequirements.strides];
}

@end
