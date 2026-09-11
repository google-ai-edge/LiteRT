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
#import "third_party/odml/litert/litert/objc/sources/LRTEnvironment+Internal.h"
#import "third_party/odml/litert/litert/objc/sources/LRTError+Internal.h"
#import "third_party/odml/litert/litert/objc/sources/LRTTensorBuffer+Internal.h"

namespace {

bool IsMetalBufferType(LRTTensorBufferType type) {
  switch (type) {
    case LRTTensorBufferTypeMetalBuffer:
    case LRTTensorBufferTypeMetalBufferFP16:
    case LRTTensorBufferTypeMetalBufferPacked:
      return true;
    default:
      return false;
  }
}

bool IsMetalTextureType(LRTTensorBufferType type) {
  switch (type) {
    case LRTTensorBufferTypeMetalTexture:
    case LRTTensorBufferTypeMetalTextureFP16:
      return true;
    default:
      return false;
  }
}

bool ValidateAndExtractDimensions(NSArray<NSNumber *> *dimensions,
                                  litert::Dimensions &outDimensions, NSError **error) {
  if (dimensions == nil) {
    if (error) {
      *error = CreateLRTError(LRTErrorCodeInvalidArgument, @"Dimensions array cannot be nil");
    }
    return false;
  }
  outDimensions.clear();
  outDimensions.reserve(dimensions.count);
  for (NSNumber *dimension in dimensions) {
    if (dimension == nil || dimension.intValue < 0) {
      if (error) {
        *error =
            CreateLRTError(LRTErrorCodeInvalidArgument, @"Dimension values must be non-negative");
      }
      return false;
    }
    outDimensions.push_back(dimension.intValue);
  }
  return true;
}

}  // namespace

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
  return [self tensorBufferWithEnvironment:environment
                                bufferType:LRTTensorBufferTypeHostMemory
                                      size:size
                               elementType:elementType
                                dimensions:dimensions
                                     error:error];
}

+ (nullable instancetype)managedMetalTensorBufferWithEnvironment:(LRTEnvironment *)environment
                                                            size:(NSUInteger)size
                                                     elementType:(LRTElementType)elementType
                                                      dimensions:(NSArray<NSNumber *> *)dimensions
                                                           error:(NSError **)error {
  return [self tensorBufferWithEnvironment:environment
                                bufferType:LRTTensorBufferTypeMetalBuffer
                                      size:size
                               elementType:elementType
                                dimensions:dimensions
                                     error:error];
}

+ (nullable instancetype)tensorBufferWithEnvironment:(LRTEnvironment *)environment
                                          bufferType:(LRTTensorBufferType)bufferType
                                                size:(NSUInteger)size
                                         elementType:(LRTElementType)elementType
                                          dimensions:(NSArray<NSNumber *> *)dimensions
                                               error:(NSError **)error {
  if (environment == nil) {
    if (error) {
      *error = CreateLRTError(LRTErrorCodeInvalidArgument, @"Valid LRTEnvironment required");
    }
    return nil;
  }

  litert::Environment *cppEnvironment = [environment cppEnvironment];
  if (!cppEnvironment) {
    if (error) {
      *error = CreateLRTError(LRTErrorCodeInvalidArgument, @"Valid LRTEnvironment required");
    }
    return nil;
  }

  if (bufferType == LRTTensorBufferTypeUnknown) {
    if (error) {
      *error = CreateLRTError(LRTErrorCodeInvalidArgument, @"Valid LRTTensorBufferType required");
    }
    return nil;
  }

  if (elementType == LRTElementTypeNone) {
    if (error) {
      *error = CreateLRTError(LRTErrorCodeInvalidArgument, @"Valid LRTElementType required");
    }
    return nil;
  }

  if (size == 0) {
    if (error) {
      *error = CreateLRTError(LRTErrorCodeInvalidArgument, @"Buffer size must be greater than 0");
    }
    return nil;
  }

  litert::Dimensions tensorDimensions;
  if (!ValidateAndExtractDimensions(dimensions, tensorDimensions, error)) {
    return nil;
  }

  litert::RankedTensorType tensorType(static_cast<litert::ElementType>(elementType),
                                      litert::Layout(tensorDimensions));

  auto bufferResult = litert::TensorBuffer::CreateManaged(
      *cppEnvironment, static_cast<litert::TensorBufferType>(bufferType), tensorType, size);

  if (!bufferResult.HasValue()) {
    if (error) {
      *error = CreateLRTError(static_cast<NSInteger>(bufferResult.Error().Status()),
                              @(bufferResult.Error().Message().c_str()));
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
  if (environment == nil) {
    if (error) {
      *error = CreateLRTError(LRTErrorCodeInvalidArgument, @"Valid LRTEnvironment required");
    }
    return nil;
  }

  litert::Environment *cppEnvironment = [environment cppEnvironment];
  if (!cppEnvironment) {
    if (error) {
      *error = CreateLRTError(LRTErrorCodeInvalidArgument, @"Valid LRTEnvironment required");
    }
    return nil;
  }

  if (metalBuffer == nil) {
    if (error) {
      *error = CreateLRTError(LRTErrorCodeInvalidArgument, @"Valid MTLBuffer required");
    }
    return nil;
  }

  if (elementType == LRTElementTypeNone) {
    if (error) {
      *error = CreateLRTError(LRTErrorCodeInvalidArgument, @"Valid LRTElementType required");
    }
    return nil;
  }

  litert::Dimensions tensorDimensions;
  if (!ValidateAndExtractDimensions(dimensions, tensorDimensions, error)) {
    return nil;
  }

  litert::RankedTensorType tensorType(static_cast<litert::ElementType>(elementType),
                                      litert::Layout(tensorDimensions));

  auto bufferResult = litert::TensorBuffer::CreateFromMetalBuffer(
      *cppEnvironment, tensorType,
      static_cast<litert::TensorBufferType>(LRTTensorBufferTypeMetalBuffer),
      (__bridge void *)metalBuffer, metalBuffer.length);

  if (!bufferResult.HasValue()) {
    if (error) {
      *error = CreateLRTError(static_cast<NSInteger>(bufferResult.Error().Status()),
                              @(bufferResult.Error().Message().c_str()));
    }
    return nil;
  }

  auto cppPtr = std::make_unique<litert::TensorBuffer>(std::move(bufferResult.Value()));
  return [[LRTTensorBuffer alloc] initInternalWithCppTensorBuffer:std::move(cppPtr)];
}

+ (nullable instancetype)tensorBufferWithEnvironment:(LRTEnvironment *)environment
                                        metalTexture:(id<MTLTexture>)metalTexture
                                         elementType:(LRTElementType)elementType
                                          dimensions:(NSArray<NSNumber *> *)dimensions
                                               error:(NSError **)error {
  if (environment == nil) {
    if (error) {
      *error = CreateLRTError(LRTErrorCodeInvalidArgument, @"Valid LRTEnvironment required");
    }
    return nil;
  }

  litert::Environment *cppEnvironment = [environment cppEnvironment];
  if (!cppEnvironment) {
    if (error) {
      *error = CreateLRTError(LRTErrorCodeInvalidArgument, @"Valid LRTEnvironment required");
    }
    return nil;
  }

  if (metalTexture == nil) {
    if (error) {
      *error = CreateLRTError(LRTErrorCodeInvalidArgument, @"Valid MTLTexture required");
    }
    return nil;
  }

  if (elementType == LRTElementTypeNone) {
    if (error) {
      *error = CreateLRTError(LRTErrorCodeInvalidArgument, @"Valid LRTElementType required");
    }
    return nil;
  }

  litert::Dimensions tensorDimensions;
  if (!ValidateAndExtractDimensions(dimensions, tensorDimensions, error)) {
    return nil;
  }

  litert::RankedTensorType tensorType(static_cast<litert::ElementType>(elementType),
                                      litert::Layout(tensorDimensions));

  auto bufferResult = litert::TensorBuffer::CreateFromMetalBuffer(
      *cppEnvironment, tensorType,
      static_cast<litert::TensorBufferType>(LRTTensorBufferTypeMetalTexture),
      (__bridge void *)metalTexture, /*size_bytes=*/0);

  if (!bufferResult.HasValue()) {
    if (error) {
      *error = CreateLRTError(static_cast<NSInteger>(bufferResult.Error().Status()),
                              @(bufferResult.Error().Message().c_str()));
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
  NSMutableArray<NSNumber *> *dimensions = [NSMutableArray arrayWithCapacity:shape.size()];
  for (auto dimension : shape) {
    [dimensions addObject:@(dimension)];
  }
  return [dimensions copy];
}

- (NSUInteger)size {
  if (!_cppTensorBuffer) return 0;
  auto sizeResult = _cppTensorBuffer->PackedSize();
  if (!sizeResult.HasValue()) return 0;
  return *sizeResult;
}

- (nullable id<MTLBuffer>)metalBuffer {
  if (!_cppTensorBuffer) return nil;
  if (!IsMetalBufferType(self.bufferType)) return nil;
  auto metalMemoryResult = _cppTensorBuffer->GetMetalBuffer();
  if (!metalMemoryResult.HasValue()) return nil;
  return (__bridge id<MTLBuffer>)*metalMemoryResult;
}

- (nullable id<MTLTexture>)metalTexture {
  if (!_cppTensorBuffer) return nil;
  if (!IsMetalTextureType(self.bufferType)) return nil;
  auto metalMemoryResult = _cppTensorBuffer->GetMetalBuffer();
  if (!metalMemoryResult.HasValue()) return nil;
  return (__bridge id<MTLTexture>)*metalMemoryResult;
}

- (nullable NSData *)readDataWithError:(NSError **)error {
  if (!_cppTensorBuffer) {
    if (error) {
      *error = CreateLRTError(LRTErrorCodeInvalidArgument, @"Invalid tensor buffer");
    }
    return nil;
  }

  auto lockResult = _cppTensorBuffer->Lock(litert::TensorBuffer::LockMode::kRead);
  if (!lockResult.HasValue()) {
    if (error) {
      *error = CreateLRTError(static_cast<NSInteger>(lockResult.Error().Status()),
                              @"Failed to lock tensor buffer");
    }
    return nil;
  }

  void *hostAddr = *lockResult;
  NSData *data = [NSData dataWithBytes:hostAddr length:self.size];
  _cppTensorBuffer->Unlock();
  return data;
}

- (BOOL)writeData:(NSData *)data error:(NSError **)error {
  if (!_cppTensorBuffer) {
    if (error) {
      *error = CreateLRTError(LRTErrorCodeInvalidArgument, @"Invalid tensor buffer");
    }
    return NO;
  }

  if (data == nil) {
    if (error) {
      *error = CreateLRTError(LRTErrorCodeInvalidArgument, @"Data cannot be nil");
    }
    return NO;
  }

  auto lockResult = _cppTensorBuffer->Lock(litert::TensorBuffer::LockMode::kWrite);
  if (!lockResult.HasValue()) {
    if (error) {
      *error = CreateLRTError(static_cast<NSInteger>(lockResult.Error().Status()),
                              @"Failed to lock tensor buffer");
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
