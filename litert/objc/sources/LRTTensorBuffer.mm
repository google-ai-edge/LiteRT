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
#include "litert/cc/litert_tensor_buffer_requirements.h"
#include "litert/cc/litert_tensor_buffer_types.h"
#import "third_party/odml/litert/litert/objc/sources/LRTEnvironment+Internal.h"
#import "third_party/odml/litert/litert/objc/sources/LRTError+Internal.h"
#import "third_party/odml/litert/litert/objc/sources/LRTTensorBuffer+Internal.h"
#import "third_party/odml/litert/litert/objc/sources/LRTTensorBufferRequirements+Internal.h"
#import "third_party/odml/litert/litert/objc/sources/LRTTensorType+Internal.h"

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
litert::Environment *ValidatedCPPEnvironment(LRTEnvironment *environment, NSError **error) {
  litert::Environment *cppEnvironment = [environment cppEnvironment];
  if (cppEnvironment == nullptr) {
    LRTSetError(error, LRTErrorCodeInvalidArgument, @"Valid LRTEnvironment required");
  }
  return cppEnvironment;
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
+ (instancetype)tensorBufferWithEnvironment:(LRTEnvironment *)environment
                                metalMemory:(id<MTLResource>)metalMemory
                                 bufferType:(LRTTensorBufferType)bufferType
                                  sizeBytes:(NSUInteger)sizeBytes
                                elementType:(LRTElementType)elementType
                                 dimensions:(NSArray<NSNumber *> *)dimensions
                                      error:(NSError **)error;

/** Initializes an @c LRTTensorBuffer taking ownership of a C++ @c litert::TensorBuffer. */
- (instancetype)initInternalWithCPPTensorBuffer:
    (std::unique_ptr<litert::TensorBuffer>)cppTensorBuffer NS_DESIGNATED_INITIALIZER;

@end

@implementation LRTTensorBuffer {
  std::unique_ptr<litert::TensorBuffer> _cppTensorBuffer;
}

+ (instancetype)tensorBufferWithEnvironment:(LRTEnvironment *)environment
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

+ (instancetype)managedMetalTensorBufferWithEnvironment:(LRTEnvironment *)environment
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

+ (instancetype)tensorBufferWithEnvironment:(LRTEnvironment *)environment
                                 bufferType:(LRTTensorBufferType)bufferType
                                       size:(NSUInteger)size
                                elementType:(LRTElementType)elementType
                                 dimensions:(NSArray<NSNumber *> *)dimensions
                                      error:(NSError **)error {
  LRTTensorType *tensorType = [[LRTTensorType alloc] initWithElementType:elementType
                                                              dimensions:dimensions];
  return [self tensorBufferWithEnvironment:environment
                                bufferType:bufferType
                                      size:size
                                tensorType:tensorType
                                     error:error];
}

+ (instancetype)tensorBufferWithEnvironment:(LRTEnvironment *)environment
                                 bufferType:(LRTTensorBufferType)bufferType
                                       size:(NSUInteger)size
                                 tensorType:(LRTTensorType *)tensorType
                                      error:(NSError **)error {
  litert::Environment *cppEnvironment = ValidatedCPPEnvironment(environment, error);
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

  if (tensorType == nil) {
    LRTSetError(error, LRTErrorCodeInvalidArgument, @"Valid LRTTensorType required");
    return nil;
  }

  std::unique_ptr<litert::RankedTensorType> cppTensorType =
      [tensorType cppTensorTypeWithError:error];
  if (cppTensorType == nullptr) {
    return nil;
  }

  litert::Expected<litert::TensorBuffer> bufferResult = litert::TensorBuffer::CreateManaged(
      *cppEnvironment, static_cast<litert::TensorBufferType>(bufferType), *cppTensorType, size);
  if (!bufferResult.HasValue()) {
    LRTSetErrorFromCPPError(error, bufferResult.Error());
    return nil;
  }

  auto cppTensorBuffer = std::make_unique<litert::TensorBuffer>(std::move(bufferResult.Value()));
  return [[LRTTensorBuffer alloc] initInternalWithCPPTensorBuffer:std::move(cppTensorBuffer)];
}

+ (instancetype)tensorBufferWithEnvironment:(LRTEnvironment *)environment
                                 tensorType:(LRTTensorType *)tensorType
                               requirements:(LRTTensorBufferRequirements *)requirements
                                      error:(NSError **)error {
  litert::Environment *cppEnvironment = ValidatedCPPEnvironment(environment, error);
  if (cppEnvironment == nullptr) {
    return nil;
  }

  if (tensorType == nil) {
    LRTSetError(error, LRTErrorCodeInvalidArgument, @"Valid LRTTensorType required");
    return nil;
  }

  if (requirements == nil) {
    LRTSetError(error, LRTErrorCodeInvalidArgument, @"Valid LRTTensorBufferRequirements required");
    return nil;
  }

  std::unique_ptr<litert::RankedTensorType> cppTensorType =
      [tensorType cppTensorTypeWithError:error];
  if (cppTensorType == nullptr) {
    return nil;
  }

  std::unique_ptr<litert::TensorBufferRequirements> cppRequirements =
      [requirements cppRequirementsWithError:error];
  if (cppRequirements == nullptr) {
    return nil;
  }

  litert::Expected<litert::TensorBuffer> bufferResult =
      litert::TensorBuffer::CreateManagedFromRequirements(*cppEnvironment, *cppTensorType,
                                                          *cppRequirements);
  if (!bufferResult.HasValue()) {
    LRTSetErrorFromCPPError(error, bufferResult.Error());
    return nil;
  }

  auto cppTensorBuffer = std::make_unique<litert::TensorBuffer>(std::move(bufferResult.Value()));
  return [[LRTTensorBuffer alloc] initInternalWithCPPTensorBuffer:std::move(cppTensorBuffer)];
}

+ (instancetype)tensorBufferWithEnvironment:(LRTEnvironment *)environment
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

+ (instancetype)tensorBufferWithEnvironment:(LRTEnvironment *)environment
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

+ (instancetype)tensorBufferWithEnvironment:(LRTEnvironment *)environment
                                metalMemory:(id<MTLResource>)metalMemory
                                 bufferType:(LRTTensorBufferType)bufferType
                                  sizeBytes:(NSUInteger)sizeBytes
                                elementType:(LRTElementType)elementType
                                 dimensions:(NSArray<NSNumber *> *)dimensions
                                      error:(NSError **)error {
  litert::Environment *cppEnvironment = ValidatedCPPEnvironment(environment, error);
  if (cppEnvironment == nullptr) {
    return nil;
  }

  LRTTensorType *tensorType = [[LRTTensorType alloc] initWithElementType:elementType
                                                              dimensions:dimensions];
  std::unique_ptr<litert::RankedTensorType> cppTensorType =
      [tensorType cppTensorTypeWithError:error];
  if (cppTensorType == nullptr) {
    return nil;
  }

  // The Metal buffer handler registered by ml_drift_metal_accelerator (LiteRtImportMetalMemory)
  // retains its own ARC reference to the bridged id<MTLResource> before CreateFromMetalBuffer
  // returns.
  litert::Expected<litert::TensorBuffer> bufferResult = litert::TensorBuffer::CreateFromMetalBuffer(
      *cppEnvironment, *cppTensorType, static_cast<litert::TensorBufferType>(bufferType),
      (__bridge void *)metalMemory, sizeBytes);
  if (!bufferResult.HasValue()) {
    LRTSetErrorFromCPPError(error, bufferResult.Error());
    return nil;
  }

  auto cppTensorBuffer = std::make_unique<litert::TensorBuffer>(std::move(bufferResult.Value()));
  return [[LRTTensorBuffer alloc] initInternalWithCPPTensorBuffer:std::move(cppTensorBuffer)];
}

- (instancetype)initInternalWithCPPTensorBuffer:
    (std::unique_ptr<litert::TensorBuffer>)cppTensorBuffer {
  self = [super init];
  if (self) {
    _cppTensorBuffer = std::move(cppTensorBuffer);
  }
  return self;
}

- (NSString *)description {
  return [NSString
      stringWithFormat:@"<%@: %p, bufferType=%ld, elementType=%ld, dimensions=%@, size=%lu>",
                       NSStringFromClass([self class]), self, (long)self.bufferType,
                       (long)self.elementType, self.dimensions, (unsigned long)self.size];
}

#pragma mark - Properties

- (LRTTensorBufferType)bufferType {
  if (!_cppTensorBuffer) return LRTTensorBufferTypeUnknown;
  litert::Expected<litert::TensorBufferType> typeResult = _cppTensorBuffer->BufferType();
  if (!typeResult.HasValue()) return LRTTensorBufferTypeUnknown;
  return static_cast<LRTTensorBufferType>(*typeResult);
}

- (LRTTensorType *)tensorType {
  if (!_cppTensorBuffer) return nil;
  litert::Expected<litert::RankedTensorType> tensorTypeResult = _cppTensorBuffer->TensorType();
  if (!tensorTypeResult.HasValue()) return nil;
  return [LRTTensorType tensorTypeWithCPPTensorType:tensorTypeResult.Value()];
}

- (LRTElementType)elementType {
  LRTTensorType *tensorType = self.tensorType;
  return tensorType ? tensorType.elementType : LRTElementTypeUnknown;
}

- (NSArray<NSNumber *> *)dimensions {
  LRTTensorType *tensorType = self.tensorType;
  return tensorType ? tensorType.dimensions : @[];
}

- (NSUInteger)size {
  if (!_cppTensorBuffer) return 0;
  auto sizeResult = _cppTensorBuffer->PackedSize();
  if (!sizeResult.HasValue()) return 0;
  return *sizeResult;
}

- (id<MTLBuffer>)metalBuffer {
  if (!_cppTensorBuffer) return nil;
  if (!IsMetalBufferType(self.bufferType)) return nil;
  auto metalMemoryResult = _cppTensorBuffer->GetMetalBuffer();
  if (!metalMemoryResult.HasValue()) return nil;
  return (__bridge id<MTLBuffer>)*metalMemoryResult;
}

- (id<MTLTexture>)metalTexture {
  if (!_cppTensorBuffer) return nil;
  if (!IsMetalTextureType(self.bufferType)) return nil;
  auto metalMemoryResult = _cppTensorBuffer->GetMetalBuffer();
  if (!metalMemoryResult.HasValue()) return nil;
  return (__bridge id<MTLTexture>)*metalMemoryResult;
}

#pragma mark - Public

- (NSData *)readDataWithError:(NSError **)error {
  if (!_cppTensorBuffer) {
    LRTSetError(error, LRTErrorCodeInvalidArgument, @"Invalid tensor buffer");
    return nil;
  }

  auto lockResult = _cppTensorBuffer->Lock(litert::TensorBuffer::LockMode::kRead);
  if (!lockResult.HasValue()) {
    LRTSetErrorFromCPPError(error, lockResult.Error());
    return nil;
  }

  void *hostAddress = *lockResult;
  NSData *data = [NSData dataWithBytes:hostAddress length:self.size];
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
    LRTSetErrorFromCPPError(error, lockResult.Error());
    return NO;
  }

  void *hostAddress = *lockResult;
  // Writing a shorter or longer NSData than the buffer is allowed: extra bytes are dropped and
  // the remaining bytes of the buffer keep their previous contents.
  size_t copyLength = std::min<size_t>(data.length, self.size);
  std::memcpy(hostAddress, data.bytes, copyLength);
  _cppTensorBuffer->Unlock();
  return YES;
}

#pragma mark - Internal

+ (instancetype)tensorBufferWithCPPTensorBuffer:(litert::TensorBuffer)cppTensorBuffer {
  auto ownedCppTensorBuffer = std::make_unique<litert::TensorBuffer>(std::move(cppTensorBuffer));
  return [[LRTTensorBuffer alloc] initInternalWithCPPTensorBuffer:std::move(ownedCppTensorBuffer)];
}

- (litert::TensorBuffer *)cppTensorBuffer {
  return _cppTensorBuffer.get();
}

@end
