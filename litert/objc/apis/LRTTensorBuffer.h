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

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#import "third_party/odml/litert/litert/objc/apis/LRTEnvironment.h"

NS_ASSUME_NONNULL_BEGIN

/** Element data types supported by LiteRT tensor buffers (bridging @c LiteRtElementType in
 * litert_model_types.h). */
typedef NS_ENUM(NSInteger, LRTElementType) {
  LRTElementTypeUnknown = 0,
  LRTElementTypeFloat32 = 1,
  LRTElementTypeInt32 = 2,
  LRTElementTypeUInt8 = 3,
  LRTElementTypeInt64 = 4,
  LRTElementTypeBool = 5,
  LRTElementTypeFloat16 = 6,
};

/** Underlying storage type of a LiteRT tensor buffer (bridging @c LiteRtTensorBufferType in
 * litert_tensor_buffer_types.h). */
typedef NS_ENUM(NSInteger, LRTTensorBufferType) {
  LRTTensorBufferTypeUnknown = 0,
  LRTTensorBufferTypeHostMemory = 1,
  LRTTensorBufferTypeMetalBuffer = 30,
};

/** Wraps a LiteRT tensor buffer holding model inputs or outputs. */
@interface LRTTensorBuffer : NSObject

- (instancetype)init NS_UNAVAILABLE;

/**
 * Creates a managed host memory tensor buffer.
 *
 * @param environment LiteRT environment instance.
 * @param size Buffer capacity in bytes.
 * @param elementType Data element type.
 * @param dimensions Tensor shape dimensions array.
 * @param error Out-parameter populated on failure.
 * @return A new LRTTensorBuffer instance, or nil on failure.
 */
+ (nullable instancetype)tensorBufferWithEnvironment:(LRTEnvironment *)environment
                                                size:(NSUInteger)size
                                         elementType:(LRTElementType)elementType
                                          dimensions:(NSArray<NSNumber *> *)dimensions
                                               error:(NSError **)error;

/**
 * Creates a tensor buffer wrapping an existing Metal buffer.
 *
 * @param environment LiteRT environment instance.
 * @param metalBuffer Metal buffer object.
 * @param elementType Data element type.
 * @param dimensions Tensor shape dimensions array.
 * @param error Out-parameter populated on failure.
 * @return A new LRTTensorBuffer instance, or nil on failure.
 */
+ (nullable instancetype)tensorBufferWithEnvironment:(LRTEnvironment *)environment
                                         metalBuffer:(id<MTLBuffer>)metalBuffer
                                         elementType:(LRTElementType)elementType
                                          dimensions:(NSArray<NSNumber *> *)dimensions
                                               error:(NSError **)error;

/** Buffer storage type (HostMemory, MetalBuffer, etc.). */
@property(nonatomic, readonly) LRTTensorBufferType bufferType;

/** Element type of tensor elements. */
@property(nonatomic, readonly) LRTElementType elementType;

/** Tensor shape dimension sizes. */
@property(nonatomic, readonly, copy) NSArray<NSNumber *> *dimensions;

/** Packed buffer size in bytes. */
@property(nonatomic, readonly) NSUInteger size;

/** Metal buffer reference if backing memory is Metal, or nil otherwise. */
@property(nonatomic, readonly, nullable) id<MTLBuffer> metalBuffer;

/**
 * Copies and returns the raw byte data from the tensor buffer.
 *
 * @param error Out-parameter populated on failure.
 * @return Data buffer copy, or nil on failure.
 */
- (nullable NSData *)readDataWithError:(NSError **)error;

/**
 * Overwrites the contents of the tensor buffer with the provided data.
 *
 * @param data Bytes to write into the tensor buffer.
 * @param error Out-parameter populated on failure.
 * @return YES on success, NO on failure.
 */
- (BOOL)writeData:(NSData *)data error:(NSError **)error;

@end

NS_ASSUME_NONNULL_END
