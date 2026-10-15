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
#import "third_party/odml/litert/litert/objc/apis/LRTTensorBufferRequirements.h"

#include <cstring>
#include <vector>

@interface LRTTensorBufferTests : XCTestCase
@end

@implementation LRTTensorBufferTests

- (void)testManagedHostMemoryBufferCreation {
  NSError *error = nil;
  LRTEnvironment *environment = [LRTEnvironment environmentWithOptions:nil error:&error];
  XCTAssertNotNil(environment);
  XCTAssertNil(error);

  NSUInteger size = 16 * sizeof(float);
  LRTTensorBuffer *buffer = [LRTTensorBuffer tensorBufferWithEnvironment:environment
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
  LRTEnvironmentOptions *environmentOptions = [[LRTEnvironmentOptions alloc] init];
  environmentOptions.metalDevice = device;
  environmentOptions.metalCommandQueue = [device newCommandQueue];

  LRTEnvironment *environment = [LRTEnvironment environmentWithOptions:environmentOptions
                                                                 error:&error];
  XCTAssertNotNil(environment);
  XCTAssertNil(error);

  NSUInteger size = 16 * sizeof(float);
  id<MTLBuffer> metalBuffer = [device newBufferWithLength:size
                                                  options:MTLResourceStorageModeShared];
  XCTAssertNotNil(metalBuffer);

  LRTTensorBuffer *buffer = [LRTTensorBuffer tensorBufferWithEnvironment:environment
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
  LRTEnvironmentOptions *environmentOptions = [[LRTEnvironmentOptions alloc] init];
  environmentOptions.metalDevice = device;
  environmentOptions.metalCommandQueue = [device newCommandQueue];

  LRTEnvironment *environment = [LRTEnvironment environmentWithOptions:environmentOptions
                                                                 error:&error];
  XCTAssertNotNil(environment);
  XCTAssertNil(error);

  MTLTextureDescriptor *textureDescriptor =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                         width:4
                                                        height:4
                                                     mipmapped:NO];
  textureDescriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  id<MTLTexture> metalTexture = [device newTextureWithDescriptor:textureDescriptor];
  XCTAssertNotNil(metalTexture);

  LRTTensorBuffer *buffer = [LRTTensorBuffer tensorBufferWithEnvironment:environment
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
  LRTEnvironment *environment = [LRTEnvironment environmentWithOptions:nil error:&error];
  XCTAssertNotNil(environment);
  XCTAssertNil(error);

  NSArray<NSNumber *> *elementTypes = @[
    @(LRTElementTypeFloat32),
    @(LRTElementTypeInt32),
    @(LRTElementTypeUInt8),
    @(LRTElementTypeInt64),
    @(LRTElementTypeBool),
    @(LRTElementTypeFloat16),
  ];

  for (NSNumber *typeNumber in elementTypes) {
    LRTElementType type = (LRTElementType)typeNumber.integerValue;
    LRTTensorBuffer *buffer = [LRTTensorBuffer tensorBufferWithEnvironment:environment
                                                                      size:32
                                                               elementType:type
                                                                dimensions:@[ @2, @4 ]
                                                                     error:&error];
    XCTAssertNotNil(buffer, @"Failed for element type: %ld", (long)type);
    XCTAssertNil(error);
    XCTAssertEqual(buffer.elementType, type);
    XCTAssertEqual(buffer.bufferType, LRTTensorBufferTypeHostMemory);
  }
}

- (void)testManagedMetalBufferCreation {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  XCTSkipIf(device == nil, @"Metal is not supported on this device/simulator.");

  NSError *error = nil;
  LRTEnvironmentOptions *environmentOptions = [[LRTEnvironmentOptions alloc] init];
  environmentOptions.metalDevice = device;
  environmentOptions.metalCommandQueue = [device newCommandQueue];

  LRTEnvironment *environment = [LRTEnvironment environmentWithOptions:environmentOptions
                                                                 error:&error];
  XCTAssertNotNil(environment);
  XCTAssertNil(error);

  NSUInteger size = 16 * sizeof(float);
  LRTTensorBuffer *buffer =
      [LRTTensorBuffer managedMetalTensorBufferWithEnvironment:environment
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
  LRTEnvironment *environment = [LRTEnvironment environmentWithOptions:nil error:&error];
  XCTAssertNotNil(environment);
  XCTAssertNil(error);

  NSUInteger size = 8 * sizeof(int32_t);
  LRTTensorBuffer *buffer =
      [LRTTensorBuffer tensorBufferWithEnvironment:environment
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
  LRTTensorType *expectedTensorType = [[LRTTensorType alloc] initWithElementType:LRTElementTypeInt32
                                                                      dimensions:@[ @8 ]];
  XCTAssertEqualObjects(buffer.tensorType, expectedTensorType);
}

- (void)testManagedTensorBufferWithTensorTypeAndRequirements {
  NSError *error = nil;
  LRTEnvironment *environment = [LRTEnvironment environmentWithOptions:nil error:&error];
  XCTAssertNotNil(environment);
  XCTAssertNil(error);

  LRTTensorType *tensorType = [[LRTTensorType alloc] initWithElementType:LRTElementTypeFloat32
                                                              dimensions:@[ @2, @4 ]];
  LRTTensorBufferRequirements *requirements = [[LRTTensorBufferRequirements alloc]
      initWithSupportedBufferTypes:@[ @(LRTTensorBufferTypeHostMemory) ]
                        bufferSize:8 * sizeof(float)
                         alignment:64
                           strides:@[]];

  LRTTensorBuffer *buffer = [LRTTensorBuffer tensorBufferWithEnvironment:environment
                                                              tensorType:tensorType
                                                            requirements:requirements
                                                                   error:&error];
  XCTAssertNotNil(buffer);
  XCTAssertNil(error);
  XCTAssertEqual(buffer.bufferType, LRTTensorBufferTypeHostMemory);
  XCTAssertEqualObjects(buffer.tensorType, tensorType);
  XCTAssertGreaterThanOrEqual(buffer.size, 8 * sizeof(float));
  XCTAssertTrue([buffer.description containsString:@"LRTTensorBuffer"]);
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
  LRTEnvironment *environment = [LRTEnvironment environmentWithOptions:nil error:&error];
  XCTAssertNotNil(environment);
  XCTAssertNil(error);

  LRTTensorBuffer *buffer = [LRTTensorBuffer tensorBufferWithEnvironment:environment
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
  LRTEnvironment *environment = [LRTEnvironment environmentWithOptions:nil error:&error];
  XCTAssertNotNil(environment);
  XCTAssertNil(error);

  LRTTensorBuffer *buffer = [LRTTensorBuffer tensorBufferWithEnvironment:environment
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
  LRTEnvironment *environment = [LRTEnvironment environmentWithOptions:nil error:&error];
  XCTAssertNotNil(environment);
  XCTAssertNil(error);

  LRTTensorBuffer *buffer = [LRTTensorBuffer tensorBufferWithEnvironment:environment
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
  LRTEnvironment *environment = [LRTEnvironment environmentWithOptions:nil error:&error];
  XCTAssertNotNil(environment);
  XCTAssertNil(error);

  LRTTensorBuffer *buffer = [LRTTensorBuffer tensorBufferWithEnvironment:environment
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
  LRTEnvironment *environment = [LRTEnvironment environmentWithOptions:nil error:&error];
  XCTAssertNotNil(environment);
  XCTAssertNil(error);

  LRTTensorBuffer *buffer = [LRTTensorBuffer tensorBufferWithEnvironment:environment
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
  LRTEnvironment *environment = [LRTEnvironment environmentWithOptions:nil error:&error];
  XCTAssertNotNil(environment);
  XCTAssertNil(error);

  LRTTensorBuffer *buffer = [LRTTensorBuffer tensorBufferWithEnvironment:environment
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
  LRTEnvironment *environment = [LRTEnvironment environmentWithOptions:nil error:&error];
  XCTAssertNotNil(environment);
  XCTAssertNil(error);

  LRTTensorBuffer *buffer = [LRTTensorBuffer tensorBufferWithEnvironment:environment
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

#pragma mark - LRTTensorBufferRequirements Tests

- (void)testTensorBufferRequirementsInitializationAndDefensiveCopy {
  NSMutableArray<NSNumber *> *mutableTypes =
      [NSMutableArray arrayWithArray:@[ @(LRTTensorBufferTypeHostMemory) ]];
  NSMutableArray<NSNumber *> *mutableStrides = [NSMutableArray arrayWithArray:@[ @4, @1 ]];

  LRTTensorBufferRequirements *requirements =
      [[LRTTensorBufferRequirements alloc] initWithSupportedBufferTypes:mutableTypes
                                                             bufferSize:128
                                                              alignment:64
                                                                strides:mutableStrides];

  [mutableTypes addObject:@(LRTTensorBufferTypeMetalBuffer)];
  [mutableStrides addObject:@1];

  XCTAssertEqualObjects(requirements.supportedBufferTypes, (@[ @(LRTTensorBufferTypeHostMemory) ]));
  XCTAssertEqual(requirements.bufferSize, 128U);
  XCTAssertEqual(requirements.alignment, 64U);
  XCTAssertEqualObjects(requirements.strides, (@[ @4, @1 ]));
}

- (void)testTensorBufferRequirementsEqualityHashAndCopy {
  LRTTensorBufferRequirements *firstRequirements = [[LRTTensorBufferRequirements alloc]
      initWithSupportedBufferTypes:@[ @(LRTTensorBufferTypeHostMemory) ]
                        bufferSize:256
                         alignment:64
                           strides:@[ @16, @4, @1 ]];
  LRTTensorBufferRequirements *matchingRequirements = [[LRTTensorBufferRequirements alloc]
      initWithSupportedBufferTypes:@[ @(LRTTensorBufferTypeHostMemory) ]
                        bufferSize:256
                         alignment:64
                           strides:@[ @16, @4, @1 ]];
  LRTTensorBufferRequirements *differentSizeRequirements = [[LRTTensorBufferRequirements alloc]
      initWithSupportedBufferTypes:@[ @(LRTTensorBufferTypeHostMemory) ]
                        bufferSize:512
                         alignment:64
                           strides:@[ @16, @4, @1 ]];
  LRTTensorBufferRequirements *permutedStridesRequirements = [[LRTTensorBufferRequirements alloc]
      initWithSupportedBufferTypes:@[ @(LRTTensorBufferTypeHostMemory) ]
                        bufferSize:256
                         alignment:64
                           strides:@[ @4, @16, @1 ]];

  XCTAssertEqualObjects(firstRequirements, matchingRequirements);
  XCTAssertEqual(firstRequirements.hash, matchingRequirements.hash);
  XCTAssertNotEqualObjects(firstRequirements, differentSizeRequirements);
  XCTAssertNotEqualObjects(firstRequirements, permutedStridesRequirements);
  XCTAssertNotEqual(firstRequirements.hash, permutedStridesRequirements.hash);

  LRTTensorBufferRequirements *copiedRequirements = [firstRequirements copy];
  XCTAssertEqualObjects(firstRequirements, copiedRequirements);
  XCTAssertEqual(firstRequirements.hash, copiedRequirements.hash);
}

- (void)testTensorBufferRequirementsDescription {
  LRTTensorBufferRequirements *requirements = [[LRTTensorBufferRequirements alloc]
      initWithSupportedBufferTypes:@[ @(LRTTensorBufferTypeHostMemory) ]
                        bufferSize:64
                         alignment:32
                           strides:@[]];
  NSString *description = requirements.description;
  XCTAssertTrue([description containsString:@"LRTTensorBufferRequirements"]);
  XCTAssertTrue([description containsString:@"bufferSize=64"]);
  XCTAssertTrue([description containsString:@"alignment=32"]);
}

@end
