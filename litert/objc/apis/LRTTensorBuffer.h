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

// LINT.IfChange(element_types)
/** Element data types supported by LiteRT tensor buffers (matching @c LiteRtElementType in
 * litert_model_types.h). */
typedef NS_ENUM(NSInteger, LRTElementType) {
  LRTElementTypeNone = 0,
  LRTElementTypeUnknown = LRTElementTypeNone,
  LRTElementTypeFloat32 = 1,
  LRTElementTypeInt32 = 2,
  LRTElementTypeUInt8 = 3,
  LRTElementTypeInt64 = 4,
  LRTElementTypeString = 5,
  LRTElementTypeBool = 6,
  LRTElementTypeInt16 = 7,
  LRTElementTypeComplex64 = 8,
  LRTElementTypeInt8 = 9,
  LRTElementTypeFloat16 = 10,
  LRTElementTypeFloat64 = 11,
  LRTElementTypeComplex128 = 12,
  LRTElementTypeUInt64 = 13,
  LRTElementTypeResource = 14,
  LRTElementTypeVariant = 15,
  LRTElementTypeUInt32 = 16,
  LRTElementTypeUInt16 = 17,
  LRTElementTypeInt4 = 18,
  LRTElementTypeBFloat16 = 19,
  LRTElementTypeInt2 = 20,
  LRTElementTypeUInt4 = 21,
  LRTElementTypeFloat8E4M3FN = 22,
  LRTElementTypeFloat8E5M2 = 23,
};
// LINT.ThenChange(../../c/litert_model_types.h)

// LINT.IfChange(tensor_buffer_types)
/** Underlying storage type of a LiteRT tensor buffer (matching @c LiteRtTensorBufferType in
 * litert_tensor_buffer_types.h). */
typedef NS_ENUM(NSInteger, LRTTensorBufferType) {
  LRTTensorBufferTypeUnknown = 0,
  LRTTensorBufferTypeHostMemory = 1,
  LRTTensorBufferTypeAHWB = 2,
  LRTTensorBufferTypeION = 3,
  LRTTensorBufferTypeDMABuf = 4,
  LRTTensorBufferTypeFastRPC = 5,
  LRTTensorBufferTypeGLBuffer = 6,
  LRTTensorBufferTypeGLTexture = 7,
  LRTTensorBufferTypeOpenCLBuffer = 10,
  LRTTensorBufferTypeOpenCLBufferFP16 = 11,
  LRTTensorBufferTypeOpenCLTexture = 12,
  LRTTensorBufferTypeOpenCLTextureFP16 = 13,
  LRTTensorBufferTypeOpenCLBufferPacked = 14,
  LRTTensorBufferTypeOpenCLImageBuffer = 15,
  LRTTensorBufferTypeOpenCLImageBufferFP16 = 16,
  LRTTensorBufferTypeWebGPUBuffer = 20,
  LRTTensorBufferTypeWebGPUBufferFP16 = 21,
  LRTTensorBufferTypeWebGPUTexture = 22,
  LRTTensorBufferTypeWebGPUTextureFP16 = 23,
  LRTTensorBufferTypeWebGPUImageBuffer = 24,
  LRTTensorBufferTypeWebGPUImageBufferFP16 = 25,
  LRTTensorBufferTypeWebGPUBufferPacked = 26,
  LRTTensorBufferTypeMetalBuffer = 30,
  LRTTensorBufferTypeMetalBufferFP16 = 31,
  LRTTensorBufferTypeMetalTexture = 32,
  LRTTensorBufferTypeMetalTextureFP16 = 33,
  LRTTensorBufferTypeMetalBufferPacked = 34,
  LRTTensorBufferTypeVulkanBuffer = 40,
  LRTTensorBufferTypeVulkanBufferFP16 = 41,
  LRTTensorBufferTypeVulkanTexture = 42,
  LRTTensorBufferTypeVulkanTextureFP16 = 43,
  LRTTensorBufferTypeVulkanImageBuffer = 44,
  LRTTensorBufferTypeVulkanImageBufferFP16 = 45,
  LRTTensorBufferTypeVulkanBufferPacked = 46,
};
// LINT.ThenChange(../../c/litert_tensor_buffer_types.h:tensor_buffer_types)

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
