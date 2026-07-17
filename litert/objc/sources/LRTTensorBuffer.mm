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

#import "third_party/odml/litert/litert/objc/apis/LRTTensorBuffer.h"
#import "third_party/odml/litert/litert/objc/apis/LRTError.h"
#import "third_party/odml/litert/litert/objc/sources/LRTEnvironment+Internal.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>

#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_types.h"

@implementation LRTTensorBuffer {
  std::unique_ptr<litert::TensorBuffer> _cppTensorBuffer;
}

- (instancetype)initInternalWithCppTensorBuffer:
    (std::unique_ptr<litert::TensorBuffer>)cppTensorBuffer {
  self = [super init];
  if (self) {
    _cppTensorBuffer = std::move(cppTensorBuffer);
  }
  return self;
}

+ (nullable instancetype)tensorBufferWithCppTensorBuffer:(litert::TensorBuffer)cppTensorBuffer {
  auto cppPtr = std::make_unique<litert::TensorBuffer>(std::move(cppTensorBuffer));
  return [[LRTTensorBuffer alloc] initInternalWithCppTensorBuffer:std::move(cppPtr)];
}

+ (nullable instancetype)tensorBufferWithEnvironment:(LRTEnvironment *)environment
                                                size:(NSUInteger)size
                                         elementType:(LRTElementType)elementType
                                          dimensions:(NSArray<NSNumber *> *)dimensions
                                               error:(NSError **)error {
  if (!environment || ![environment cppEnvironment]) {
    if (error) {
      *error =
          [NSError errorWithDomain:LRTErrorDomain
                              code:LRTErrorCodeInvalidArgument
                          userInfo:@{NSLocalizedDescriptionKey : @"Valid LRTEnvironment required"}];
    }
    return nil;
  }

  litert::Dimensions dims;
  dims.reserve(dimensions.count);
  for (NSNumber *dim in dimensions) {
    dims.push_back(dim.intValue);
  }

  litert::RankedTensorType tensorType(static_cast<litert::ElementType>(elementType),
                                      litert::Layout(dims));

  auto bufferResult = litert::TensorBuffer::CreateManaged(
      *[environment cppEnvironment],
      static_cast<litert::TensorBufferType>(LRTTensorBufferTypeHostMemory), tensorType, size);

  if (!bufferResult.HasValue()) {
    if (error) {
      NSDictionary *userInfo =
          @{NSLocalizedDescriptionKey : @(bufferResult.Error().Message().c_str())};
      *error = [NSError errorWithDomain:LRTErrorDomain
                                   code:static_cast<NSInteger>(bufferResult.Error().Status())
                               userInfo:userInfo];
    }
    return nil;
  }

  auto cppPtr = std::make_unique<litert::TensorBuffer>(std::move(bufferResult.Value()));
  return [[LRTTensorBuffer alloc] initInternalWithCppTensorBuffer:std::move(cppPtr)];
}

+ (nullable instancetype)tensorBufferWithEnvironment:(LRTEnvironment *)environment
                                         metalBuffer:(id<MTLBuffer>)metalBuffer
                                         elementType:(LRTElementType)elementType
                                          dimensions:(NSArray<NSNumber *> *)dimensions
                                               error:(NSError **)error {
  if (!environment || ![environment cppEnvironment]) {
    if (error) {
      *error =
          [NSError errorWithDomain:LRTErrorDomain
                              code:LRTErrorCodeInvalidArgument
                          userInfo:@{NSLocalizedDescriptionKey : @"Valid LRTEnvironment required"}];
    }
    return nil;
  }

  litert::Dimensions dims;
  dims.reserve(dimensions.count);
  for (NSNumber *dim in dimensions) {
    dims.push_back(dim.intValue);
  }

  litert::RankedTensorType tensorType(static_cast<litert::ElementType>(elementType),
                                      litert::Layout(dims));

  auto bufferResult = litert::TensorBuffer::CreateFromMetalBuffer(
      *[environment cppEnvironment], tensorType,
      static_cast<litert::TensorBufferType>(LRTTensorBufferTypeMetalBuffer),
      (__bridge void *)metalBuffer, metalBuffer.length);

  if (!bufferResult.HasValue()) {
    if (error) {
      NSDictionary *userInfo =
          @{NSLocalizedDescriptionKey : @(bufferResult.Error().Message().c_str())};
      *error = [NSError errorWithDomain:LRTErrorDomain
                                   code:static_cast<NSInteger>(bufferResult.Error().Status())
                               userInfo:userInfo];
    }
    return nil;
  }

  auto cppPtr = std::make_unique<litert::TensorBuffer>(std::move(bufferResult.Value()));
  return [[LRTTensorBuffer alloc] initInternalWithCppTensorBuffer:std::move(cppPtr)];
}

- (LRTTensorBufferType)bufferType {
  if (!_cppTensorBuffer) return LRTTensorBufferTypeUnknown;
  auto typeResult = _cppTensorBuffer->BufferType();
  if (!typeResult.HasValue()) return LRTTensorBufferTypeUnknown;
  return static_cast<LRTTensorBufferType>(*typeResult);
}

- (LRTElementType)elementType {
  if (!_cppTensorBuffer) return LRTElementTypeUnknown;
  auto tensorTypeResult = _cppTensorBuffer->TensorType();
  if (!tensorTypeResult.HasValue()) return LRTElementTypeUnknown;
  return static_cast<LRTElementType>(tensorTypeResult->ElementType());
}

- (NSArray<NSNumber *> *)dimensions {
  if (!_cppTensorBuffer) return @[];
  auto tensorTypeResult = _cppTensorBuffer->TensorType();
  if (!tensorTypeResult.HasValue()) return @[];

  auto shape = tensorTypeResult->Layout().Dimensions();
  NSMutableArray<NSNumber *> *dims = [NSMutableArray arrayWithCapacity:shape.size()];
  for (auto dim : shape) {
    [dims addObject:@(dim)];
  }
  return [dims copy];
}

- (NSUInteger)size {
  if (!_cppTensorBuffer) return 0;
  auto sizeResult = _cppTensorBuffer->PackedSize();
  if (!sizeResult.HasValue()) return 0;
  return *sizeResult;
}

- (nullable id<MTLBuffer>)metalBuffer {
  if (!_cppTensorBuffer) return nil;
  if (self.bufferType != LRTTensorBufferTypeMetalBuffer) return nil;
  auto metalMemoryResult = _cppTensorBuffer->GetMetalBuffer();
  if (!metalMemoryResult.HasValue()) return nil;
  return (__bridge id<MTLBuffer>)*metalMemoryResult;
}

- (nullable NSData *)readDataWithError:(NSError **)error {
  if (!_cppTensorBuffer) return nil;

  auto lockResult = _cppTensorBuffer->Lock(litert::TensorBuffer::LockMode::kRead);
  if (!lockResult.HasValue()) {
    if (error) {
      *error =
          [NSError errorWithDomain:LRTErrorDomain
                              code:static_cast<NSInteger>(lockResult.Error().Status())
                          userInfo:@{NSLocalizedDescriptionKey : @"Failed to lock tensor buffer"}];
    }
    return nil;
  }

  void *hostAddr = *lockResult;
  NSData *data = [NSData dataWithBytes:hostAddr length:self.size];
  _cppTensorBuffer->Unlock();
  return data;
}

- (BOOL)writeData:(NSData *)data error:(NSError **)error {
  if (!_cppTensorBuffer) return NO;

  auto lockResult = _cppTensorBuffer->Lock(litert::TensorBuffer::LockMode::kWrite);
  if (!lockResult.HasValue()) {
    if (error) {
      *error =
          [NSError errorWithDomain:LRTErrorDomain
                              code:static_cast<NSInteger>(lockResult.Error().Status())
                          userInfo:@{NSLocalizedDescriptionKey : @"Failed to lock tensor buffer"}];
    }
    return NO;
  }

  void *hostAddr = *lockResult;
  size_t copyLength = std::min<size_t>(data.length, self.size);
  std::memcpy(hostAddr, data.bytes, copyLength);
  _cppTensorBuffer->Unlock();
  return YES;
}

- (nullable litert::TensorBuffer *)cppTensorBuffer {
  return _cppTensorBuffer.get();
}

@end
