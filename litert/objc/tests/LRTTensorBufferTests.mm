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

#import "third_party/odml/litert/litert/objc/apis/LRTEnvironment.h"
#import "third_party/odml/litert/litert/objc/apis/LRTError.h"
#import "third_party/odml/litert/litert/objc/apis/LRTTensorBuffer.h"

#include <vector>

#include "testing/base/public/gunit.h"

TEST(LRTTensorBufferTest, ManagedHostMemoryBufferCreation) {
  NSError *error = nil;
  LRTEnvironment *env = [LRTEnvironment environmentWithOptions:nil error:&error];
  ASSERT_NE(env, nil);

  NSUInteger size = 16 * sizeof(float);
  LRTTensorBuffer *buffer = [LRTTensorBuffer tensorBufferWithEnvironment:env
                                                                    size:size
                                                             elementType:LRTElementTypeFloat32
                                                              dimensions:@[ @4, @4 ]
                                                                   error:&error];

  EXPECT_NE(buffer, nil);
  EXPECT_EQ(error, nil);
  EXPECT_EQ(buffer.bufferType, LRTTensorBufferTypeHostMemory);
  EXPECT_EQ(buffer.elementType, LRTElementTypeFloat32);
  EXPECT_EQ(buffer.dimensions, (@[ @4, @4 ]));
  EXPECT_GE(buffer.size, size);

  std::vector<float> inputValues(16, 42.0f);
  NSData *inputData = [NSData dataWithBytes:inputValues.data()
                                     length:inputValues.size() * sizeof(float)];
  BOOL writeSuccess = [buffer writeData:inputData error:&error];
  EXPECT_TRUE(writeSuccess);
  EXPECT_EQ(error, nil);

  NSData *readData = [buffer readDataWithError:&error];
  EXPECT_NE(readData, nil);
  EXPECT_EQ(std::memcmp(readData.bytes, inputData.bytes, inputData.length), 0);
}

TEST(LRTTensorBufferTest, MetalBufferCreation) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device) {
    GTEST_SKIP() << "Metal device is unavailable on this system";
  }

  NSError *error = nil;
  LRTEnvironment *env = [LRTEnvironment environmentWithOptions:nil error:&error];
  ASSERT_NE(env, nil);

  NSUInteger size = 16 * sizeof(float);
  id<MTLBuffer> metalBuffer = [device newBufferWithLength:size
                                                  options:MTLResourceStorageModeShared];
  ASSERT_NE(metalBuffer, nil);

  LRTTensorBuffer *buffer = [LRTTensorBuffer tensorBufferWithEnvironment:env
                                                             metalBuffer:metalBuffer
                                                             elementType:LRTElementTypeFloat32
                                                              dimensions:@[ @4, @4 ]
                                                                   error:&error];

  EXPECT_NE(buffer, nil);
  EXPECT_EQ(error, nil);
  EXPECT_EQ(buffer.bufferType, LRTTensorBufferTypeMetalBuffer);
  EXPECT_EQ(buffer.elementType, LRTElementTypeFloat32);
  EXPECT_EQ(buffer.dimensions, (@[ @4, @4 ]));
  EXPECT_EQ(buffer.metalBuffer, metalBuffer);
}
