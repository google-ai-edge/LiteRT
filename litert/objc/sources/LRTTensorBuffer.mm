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
#include <optional>
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

/**
 * Returns the C++ environment backing @c environment, or nullptr if there is none.
 *
 * Messaging a nil @c environment returns nullptr as well, so this also covers a nil argument.
 */
litert::Environment *_Nullable ValidatedCppEnvironment(LRTEnvironment *_Nullable environment,
                                                       NSError *_Nullable *_Nullable error) {
  litert::Environment *cppEnvironment = [environment cppEnvironment];
  if (cppEnvironment == nullptr) {
    LRTSetError(error, LRTErrorCodeInvalidArgument, @"Valid LRTEnvironment required");
  }
  return cppEnvironment;
}

/** Returns the tensor type described by @c elementType and @c dimensions, or nullopt if invalid. */
std::optional<litert::RankedTensorType> RankedTensorTypeFromDimensions(
    LRTElementType elementType, NSArray<NSNumber *> *_Nullable dimensions,
    NSError *_Nullable *_Nullable error) {
  if (elementType == LRTElementTypeNone) {
    LRTSetError(error, LRTErrorCodeInvalidArgument, @"Valid LRTElementType required");
    return std::nullopt;
  }
  if (dimensions == nil) {
    LRTSetError(error, LRTErrorCodeInvalidArgument, @"Dimensions array cannot be nil");
    return std::nullopt;
  }

  litert::Dimensions tensorDimensions;
  tensorDimensions.reserve(dimensions.count);
  for (NSNumber *dimension in dimensions) {
    if (dimension == nil || dimension.intValue < 0) {
      LRTSetError(error, LRTErrorCodeInvalidArgument, @"Dimension values must be non-negative");
      return std::nullopt;
    }
    tensorDimensions.push_back(dimension.intValue);
  }
  return litert::RankedTensorType(static_cast<litert::ElementType>(elementType),
                                  litert::Layout(tensorDimensions));
}

}  // namespace

@interface LRTTensorBuffer ()

/**
 * Wraps Metal memory owned by the caller in a tensor buffer.
 *
 * @param environment LiteRT environment instance.
 * @param metalMemory The @c id<MTLBuffer> or @c id<MTLTexture> to wrap.
 * @param bufferType Buffer type describing @c metalMemory.
 * @param sizeBytes Size of @c metalMemory in bytes; pass 0 for textures, whose size LiteRT
 * derives from the tensor type.
 * @param elementType Data element type.
 * @param dimensions Tensor shape dimensions array.
 * @param error Out-parameter populated on failure.
 * @return A new LRTTensorBuffer instance, or nil on failure.
 */
+ (nullable instancetype)tensorBufferWithEnvironment:(LRTEnvironment *)environment
                                         metalMemory:(id)metalMemory
                                          bufferType:(LRTTensorBufferType)bufferType
                                           sizeBytes:(NSUInteger)sizeBytes
                                         elementType:(LRTElementType)elementType
                                          dimensions:(NSArray<NSNumber *> *)dimensions
                                               error:(NSError **)error;

@end

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
  litert::Environment *cppEnvironment = ValidatedCppEnvironment(environment, error);
  if (cppEnvironment == nullptr) {
    return nil;
  }

  if (bufferType == LRTTensorBufferTypeUnknown) {
    LRTSetError(error, LRTErrorCodeInvalidArgument, @"Valid LRTTensorBufferType required");
    return nil;
  }

  if (size == 0) {
    LRTSetError(error, LRTErrorCodeInvalidArgument, @"Buffer size must be greater than 0");
    return nil;
  }

  std::optional<litert::RankedTensorType> tensorType =
      RankedTensorTypeFromDimensions(elementType, dimensions, error);
  if (!tensorType.has_value()) {
    return nil;
  }

  auto bufferResult = litert::TensorBuffer::CreateManaged(
      *cppEnvironment, static_cast<litert::TensorBufferType>(bufferType), *tensorType, size);
  if (!bufferResult.HasValue()) {
    LRTSetErrorFromCppError(error, bufferResult.Error());
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
  if (metalBuffer == nil) {
    LRTSetError(error, LRTErrorCodeInvalidArgument, @"Valid MTLBuffer required");
    return nil;
  }
  return [self tensorBufferWithEnvironment:environment
                               metalMemory:metalBuffer
                                bufferType:LRTTensorBufferTypeMetalBuffer
                                 sizeBytes:metalBuffer.length
                               elementType:elementType
                                dimensions:dimensions
                                     error:error];
}

+ (nullable instancetype)tensorBufferWithEnvironment:(LRTEnvironment *)environment
                                        metalTexture:(id<MTLTexture>)metalTexture
                                         elementType:(LRTElementType)elementType
                                          dimensions:(NSArray<NSNumber *> *)dimensions
                                               error:(NSError **)error {
  if (metalTexture == nil) {
    LRTSetError(error, LRTErrorCodeInvalidArgument, @"Valid MTLTexture required");
    return nil;
  }
  return [self tensorBufferWithEnvironment:environment
                               metalMemory:metalTexture
                                bufferType:LRTTensorBufferTypeMetalTexture
                                 sizeBytes:0
                               elementType:elementType
                                dimensions:dimensions
                                     error:error];
}

+ (nullable instancetype)tensorBufferWithEnvironment:(LRTEnvironment *)environment
                                         metalMemory:(id)metalMemory
                                          bufferType:(LRTTensorBufferType)bufferType
                                           sizeBytes:(NSUInteger)sizeBytes
                                         elementType:(LRTElementType)elementType
                                          dimensions:(NSArray<NSNumber *> *)dimensions
                                               error:(NSError **)error {
  litert::Environment *cppEnvironment = ValidatedCppEnvironment(environment, error);
  if (cppEnvironment == nullptr) {
    return nil;
  }

  std::optional<litert::RankedTensorType> tensorType =
      RankedTensorTypeFromDimensions(elementType, dimensions, error);
  if (!tensorType.has_value()) {
    return nil;
  }

  auto bufferResult = litert::TensorBuffer::CreateFromMetalBuffer(
      *cppEnvironment, *tensorType, static_cast<litert::TensorBufferType>(bufferType),
      (__bridge void *)metalMemory, sizeBytes);
  if (!bufferResult.HasValue()) {
    LRTSetErrorFromCppError(error, bufferResult.Error());
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
    LRTSetError(error, LRTErrorCodeInvalidArgument, @"Invalid tensor buffer");
    return nil;
  }

  auto lockResult = _cppTensorBuffer->Lock(litert::TensorBuffer::LockMode::kRead);
  if (!lockResult.HasValue()) {
    LRTSetError(error, static_cast<NSInteger>(lockResult.Error().StatusValue()),
                @"Failed to lock tensor buffer");
    return nil;
  }

  void *hostAddr = *lockResult;
  NSData *data = [NSData dataWithBytes:hostAddr length:self.size];
  _cppTensorBuffer->Unlock();
  return data;
}

- (BOOL)writeData:(NSData *)data error:(NSError **)error {
  if (!_cppTensorBuffer) {
    LRTSetError(error, LRTErrorCodeInvalidArgument, @"Invalid tensor buffer");
    return NO;
  }

  if (data == nil) {
    LRTSetError(error, LRTErrorCodeInvalidArgument, @"Data cannot be nil");
    return NO;
  }

  auto lockResult = _cppTensorBuffer->Lock(litert::TensorBuffer::LockMode::kWrite);
  if (!lockResult.HasValue()) {
    LRTSetError(error, static_cast<NSInteger>(lockResult.Error().StatusValue()),
                @"Failed to lock tensor buffer");
    return NO;
  }

  void *hostAddr = *lockResult;
  // Writing a shorter or longer NSData than the buffer is allowed: extra bytes are dropped and
  // the remaining bytes of the buffer keep their previous contents.
  size_t copyLength = std::min<size_t>(data.length, self.size);
  std::memcpy(hostAddr, data.bytes, copyLength);
  _cppTensorBuffer->Unlock();
  return YES;
}

- (nullable litert::TensorBuffer *)cppTensorBuffer {
  return _cppTensorBuffer.get();
}

@end
