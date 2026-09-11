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

#import <Metal/Metal.h>
#import <XCTest/XCTest.h>

#import "third_party/odml/litert/litert/objc/apis/LRTEnvironment.h"
#import "third_party/odml/litert/litert/objc/apis/LRTError.h"
#import "third_party/odml/litert/litert/objc/apis/LRTTensorBuffer.h"

#include <cstring>
#include <vector>

@interface LRTTensorBufferTests : XCTestCase
@end

@implementation LRTTensorBufferTests

- (void)testManagedHostMemoryBufferCreation {
  NSError *error = nil;
  LRTEnvironment *env = [LRTEnvironment environmentWithOptions:nil error:&error];
  XCTAssertNotNil(env);
  XCTAssertNil(error);

  NSUInteger size = 16 * sizeof(float);
  LRTTensorBuffer *buffer = [LRTTensorBuffer tensorBufferWithEnvironment:env
                                                                    size:size
                                                             elementType:LRTElementTypeFloat32
                                                              dimensions:@[ @4, @4 ]
                                                                   error:&error];

  XCTAssertNotNil(buffer);
  XCTAssertNil(error);
  XCTAssertEqual(buffer.bufferType, LRTTensorBufferTypeHostMemory);
  XCTAssertEqual(buffer.elementType, LRTElementTypeFloat32);
  XCTAssertEqualObjects(buffer.dimensions, (@[ @4, @4 ]));
  XCTAssertGreaterThanOrEqual(buffer.size, size);
  XCTAssertNil(buffer.metalBuffer);
  XCTAssertNil(buffer.metalTexture);

  std::vector<float> inputValues(16, 42.0f);
  NSData *inputData = [NSData dataWithBytes:inputValues.data()
                                     length:inputValues.size() * sizeof(float)];
  BOOL writeSuccess = [buffer writeData:inputData error:&error];
  XCTAssertTrue(writeSuccess);
  XCTAssertNil(error);

  NSData *readData = [buffer readDataWithError:&error];
  XCTAssertNotNil(readData);
  XCTAssertEqual(std::memcmp(readData.bytes, inputData.bytes, inputData.length), 0);
}

- (void)testMetalBufferCreation {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  XCTSkipIf(device == nil, @"Metal is not supported on this device/simulator.");

  NSError *error = nil;
  LRTEnvironmentOptions *envOptions = [[LRTEnvironmentOptions alloc] init];
  envOptions.metalDevice = device;
  envOptions.metalCommandQueue = [device newCommandQueue];

  LRTEnvironment *env = [LRTEnvironment environmentWithOptions:envOptions error:&error];
  XCTAssertNotNil(env);
  XCTAssertNil(error);

  NSUInteger size = 16 * sizeof(float);
  id<MTLBuffer> metalBuffer = [device newBufferWithLength:size
                                                  options:MTLResourceStorageModeShared];
  XCTAssertNotNil(metalBuffer);

  LRTTensorBuffer *buffer = [LRTTensorBuffer tensorBufferWithEnvironment:env
                                                             metalBuffer:metalBuffer
                                                             elementType:LRTElementTypeFloat32
                                                              dimensions:@[ @4, @4 ]
                                                                   error:&error];

  XCTAssertNotNil(buffer);
  XCTAssertNil(error);
  XCTAssertEqual(buffer.bufferType, LRTTensorBufferTypeMetalBuffer);
  XCTAssertEqual(buffer.elementType, LRTElementTypeFloat32);
  XCTAssertEqualObjects(buffer.dimensions, (@[ @4, @4 ]));
  XCTAssertEqual(buffer.metalBuffer, metalBuffer);
  XCTAssertNil(buffer.metalTexture);
}

- (void)testMetalTextureCreation {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  XCTSkipIf(device == nil, @"Metal is not supported on this device/simulator.");

  NSError *error = nil;
  LRTEnvironmentOptions *envOptions = [[LRTEnvironmentOptions alloc] init];
  envOptions.metalDevice = device;
  envOptions.metalCommandQueue = [device newCommandQueue];

  LRTEnvironment *env = [LRTEnvironment environmentWithOptions:envOptions error:&error];
  XCTAssertNotNil(env);
  XCTAssertNil(error);

  MTLTextureDescriptor *textureDescriptor =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                         width:4
                                                        height:4
                                                     mipmapped:NO];
  textureDescriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  id<MTLTexture> metalTexture = [device newTextureWithDescriptor:textureDescriptor];
  XCTAssertNotNil(metalTexture);

  LRTTensorBuffer *buffer = [LRTTensorBuffer tensorBufferWithEnvironment:env
                                                            metalTexture:metalTexture
                                                             elementType:LRTElementTypeFloat32
                                                              dimensions:@[ @4, @4, @4 ]
                                                                   error:&error];

  XCTAssertNotNil(buffer);
  XCTAssertNil(error);
  XCTAssertEqual(buffer.bufferType, LRTTensorBufferTypeMetalTexture);
  XCTAssertEqual(buffer.elementType, LRTElementTypeFloat32);
  XCTAssertEqualObjects(buffer.dimensions, (@[ @4, @4, @4 ]));
  XCTAssertEqual(buffer.metalTexture, metalTexture);
  XCTAssertNil(buffer.metalBuffer);
}

- (void)testVariousElementTypes {
  NSError *error = nil;
  LRTEnvironment *env = [LRTEnvironment environmentWithOptions:nil error:&error];
  XCTAssertNotNil(env);
  XCTAssertNil(error);

  NSArray<NSNumber *> *elementTypes = @[
    @(LRTElementTypeFloat32),
    @(LRTElementTypeInt32),
    @(LRTElementTypeUInt8),
    @(LRTElementTypeInt64),
    @(LRTElementTypeBool),
    @(LRTElementTypeFloat16),
  ];

  for (NSNumber *typeNum in elementTypes) {
    LRTElementType type = (LRTElementType)typeNum.integerValue;
    LRTTensorBuffer *buf = [LRTTensorBuffer tensorBufferWithEnvironment:env
                                                                   size:32
                                                            elementType:type
                                                             dimensions:@[ @2, @4 ]
                                                                  error:&error];
    XCTAssertNotNil(buf, @"Failed for element type: %ld", (long)type);
    XCTAssertNil(error);
    XCTAssertEqual(buf.elementType, type);
    XCTAssertEqual(buf.bufferType, LRTTensorBufferTypeHostMemory);
  }
}

- (void)testManagedMetalBufferCreation {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  XCTSkipIf(device == nil, @"Metal is not supported on this device/simulator.");

  NSError *error = nil;
  LRTEnvironmentOptions *envOptions = [[LRTEnvironmentOptions alloc] init];
  envOptions.metalDevice = device;
  envOptions.metalCommandQueue = [device newCommandQueue];

  LRTEnvironment *env = [LRTEnvironment environmentWithOptions:envOptions error:&error];
  XCTAssertNotNil(env);
  XCTAssertNil(error);

  NSUInteger size = 16 * sizeof(float);
  LRTTensorBuffer *buffer =
      [LRTTensorBuffer managedMetalTensorBufferWithEnvironment:env
                                                          size:size
                                                   elementType:LRTElementTypeFloat32
                                                    dimensions:@[ @4, @4 ]
                                                         error:&error];

  XCTAssertNotNil(buffer);
  XCTAssertNil(error);
  XCTAssertEqual(buffer.bufferType, LRTTensorBufferTypeMetalBuffer);
  XCTAssertEqual(buffer.elementType, LRTElementTypeFloat32);
  XCTAssertEqualObjects(buffer.dimensions, (@[ @4, @4 ]));
  XCTAssertGreaterThanOrEqual(buffer.size, size);
  XCTAssertNotNil(buffer.metalBuffer);
  XCTAssertNil(buffer.metalTexture);

  std::vector<float> inputValues(16, 7.5f);
  NSData *inputData = [NSData dataWithBytes:inputValues.data()
                                     length:inputValues.size() * sizeof(float)];
  BOOL writeSuccess = [buffer writeData:inputData error:&error];
  XCTAssertTrue(writeSuccess);
  XCTAssertNil(error);

  NSData *readData = [buffer readDataWithError:&error];
  XCTAssertNotNil(readData);
  XCTAssertNil(error);
  XCTAssertEqual(std::memcmp(readData.bytes, inputData.bytes, inputData.length), 0);
}

- (void)testManagedTensorBufferWithExplicitType {
  NSError *error = nil;
  LRTEnvironment *env = [LRTEnvironment environmentWithOptions:nil error:&error];
  XCTAssertNotNil(env);
  XCTAssertNil(error);

  NSUInteger size = 8 * sizeof(int32_t);
  LRTTensorBuffer *buffer =
      [LRTTensorBuffer tensorBufferWithEnvironment:env
                                        bufferType:LRTTensorBufferTypeHostMemory
                                              size:size
                                       elementType:LRTElementTypeInt32
                                        dimensions:@[ @8 ]
                                             error:&error];

  XCTAssertNotNil(buffer);
  XCTAssertNil(error);
  XCTAssertEqual(buffer.bufferType, LRTTensorBufferTypeHostMemory);
  XCTAssertEqual(buffer.elementType, LRTElementTypeInt32);
  XCTAssertEqualObjects(buffer.dimensions, (@[ @8 ]));
}

#pragma mark - Parameter Validation Tests

// Suppress -Wnonnull warnings so we can explicitly test runtime defensive null-checks
// on parameters marked nonnull in public headers (e.g. from Swift or dynamic Objective-C
// callers).
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnonnull"

- (void)testTensorBufferWithNilEnvironmentFails {
  NSError *error = nil;
  LRTTensorBuffer *buffer = [LRTTensorBuffer tensorBufferWithEnvironment:nil
                                                                    size:16
                                                             elementType:LRTElementTypeFloat32
                                                              dimensions:@[ @4 ]
                                                                   error:&error];
  XCTAssertNil(buffer);
  XCTAssertNotNil(error);
  XCTAssertEqual(error.code, LRTErrorCodeInvalidArgument);
}

- (void)testTensorBufferWithZeroSizeFails {
  NSError *error = nil;
  LRTEnvironment *env = [LRTEnvironment environmentWithOptions:nil error:&error];
  XCTAssertNotNil(env);
  XCTAssertNil(error);

  LRTTensorBuffer *buffer = [LRTTensorBuffer tensorBufferWithEnvironment:env
                                                                    size:0
                                                             elementType:LRTElementTypeFloat32
                                                              dimensions:@[ @4 ]
                                                                   error:&error];
  XCTAssertNil(buffer);
  XCTAssertNotNil(error);
  XCTAssertEqual(error.code, LRTErrorCodeInvalidArgument);
}

- (void)testTensorBufferWithInvalidElementTypeFails {
  NSError *error = nil;
  LRTEnvironment *env = [LRTEnvironment environmentWithOptions:nil error:&error];
  XCTAssertNotNil(env);
  XCTAssertNil(error);

  LRTTensorBuffer *buffer = [LRTTensorBuffer tensorBufferWithEnvironment:env
                                                                    size:16
                                                             elementType:LRTElementTypeNone
                                                              dimensions:@[ @4 ]
                                                                   error:&error];
  XCTAssertNil(buffer);
  XCTAssertNotNil(error);
  XCTAssertEqual(error.code, LRTErrorCodeInvalidArgument);
}

- (void)testTensorBufferWithNilDimensionsFails {
  NSError *error = nil;
  LRTEnvironment *env = [LRTEnvironment environmentWithOptions:nil error:&error];
  XCTAssertNotNil(env);
  XCTAssertNil(error);

  LRTTensorBuffer *buffer = [LRTTensorBuffer tensorBufferWithEnvironment:env
                                                                    size:16
                                                             elementType:LRTElementTypeFloat32
                                                              dimensions:nil
                                                                   error:&error];
  XCTAssertNil(buffer);
  XCTAssertNotNil(error);
  XCTAssertEqual(error.code, LRTErrorCodeInvalidArgument);
}

- (void)testTensorBufferWithNegativeDimensionFails {
  NSError *error = nil;
  LRTEnvironment *env = [LRTEnvironment environmentWithOptions:nil error:&error];
  XCTAssertNotNil(env);
  XCTAssertNil(error);

  LRTTensorBuffer *buffer = [LRTTensorBuffer tensorBufferWithEnvironment:env
                                                                    size:16
                                                             elementType:LRTElementTypeFloat32
                                                              dimensions:@[ @(-1) ]
                                                                   error:&error];
  XCTAssertNil(buffer);
  XCTAssertNotNil(error);
  XCTAssertEqual(error.code, LRTErrorCodeInvalidArgument);
}

- (void)testTensorBufferWithNilMetalBufferFails {
  NSError *error = nil;
  LRTEnvironment *env = [LRTEnvironment environmentWithOptions:nil error:&error];
  XCTAssertNotNil(env);
  XCTAssertNil(error);

  LRTTensorBuffer *buffer = [LRTTensorBuffer tensorBufferWithEnvironment:env
                                                             metalBuffer:nil
                                                             elementType:LRTElementTypeFloat32
                                                              dimensions:@[ @4 ]
                                                                   error:&error];
  XCTAssertNil(buffer);
  XCTAssertNotNil(error);
  XCTAssertEqual(error.code, LRTErrorCodeInvalidArgument);
}

- (void)testTensorBufferWithNilMetalTextureFails {
  NSError *error = nil;
  LRTEnvironment *env = [LRTEnvironment environmentWithOptions:nil error:&error];
  XCTAssertNotNil(env);
  XCTAssertNil(error);

  LRTTensorBuffer *buffer = [LRTTensorBuffer tensorBufferWithEnvironment:env
                                                            metalTexture:nil
                                                             elementType:LRTElementTypeFloat32
                                                              dimensions:@[ @4 ]
                                                                   error:&error];
  XCTAssertNil(buffer);
  XCTAssertNotNil(error);
  XCTAssertEqual(error.code, LRTErrorCodeInvalidArgument);
}

- (void)testTensorBufferWriteNilDataFails {
  NSError *error = nil;
  LRTEnvironment *env = [LRTEnvironment environmentWithOptions:nil error:&error];
  XCTAssertNotNil(env);
  XCTAssertNil(error);

  LRTTensorBuffer *buffer = [LRTTensorBuffer tensorBufferWithEnvironment:env
                                                                    size:16
                                                             elementType:LRTElementTypeFloat32
                                                              dimensions:@[ @4 ]
                                                                   error:&error];
  XCTAssertNotNil(buffer);
  XCTAssertNil(error);

  BOOL writeSuccess = [buffer writeData:nil error:&error];
  XCTAssertFalse(writeSuccess);
  XCTAssertNotNil(error);
  XCTAssertEqual(error.code, LRTErrorCodeInvalidArgument);
}

#pragma clang diagnostic pop

@end
