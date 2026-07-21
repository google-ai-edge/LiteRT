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
}

@end
