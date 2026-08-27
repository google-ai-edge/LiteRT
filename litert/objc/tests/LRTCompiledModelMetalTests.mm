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

#include <cstring>
#include <string>
#include <vector>

#import "third_party/odml/litert/litert/objc/apis/LRTCompiledModel.h"
#import "third_party/odml/litert/litert/objc/apis/LRTEnvironment.h"
#import "third_party/odml/litert/litert/objc/apis/LRTError.h"
#import "third_party/odml/litert/litert/objc/apis/LRTOptions.h"
#import "third_party/odml/litert/litert/objc/apis/LRTTensorBuffer.h"
#include "litert/test/common.h"
#include "litert/test/testdata/simple_model_test_vectors.h"

@interface LRTCompiledModelMetalTests : XCTestCase
@end

static constexpr float kTestAccuracy = 1e-5f;

static NSString *GetTestModelPath() {
  NSBundle *bundle = [NSBundle bundleForClass:[LRTCompiledModelMetalTests class]];
  NSString *path = [bundle pathForResource:@"simple_model" ofType:@"tflite"];
  if (path) {
    return path;
  }
  std::string modelPath = litert::testing::GetTestFilePath(kModelFileName);
  return @(modelPath.c_str());
}

@implementation LRTCompiledModelMetalTests

- (void)testCustomEnvironmentOptionsWithMetalDeviceAndQueue {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  XCTSkipIf(device == nil, @"Metal is not supported on this device/simulator.");

  NSError *error = nil;
  LRTEnvironmentOptions *envOptions = [[LRTEnvironmentOptions alloc] init];
  envOptions.metalDevice = device;
  envOptions.metalCommandQueue = [device newCommandQueue];

  LRTEnvironment *env = [LRTEnvironment environmentWithOptions:envOptions error:&error];
  XCTAssertNotNil(env);
  XCTAssertNil(error);
  XCTAssertEqual(env.metalDevice, envOptions.metalDevice);
  XCTAssertEqual(env.metalCommandQueue, envOptions.metalCommandQueue);

  LRTOptions *options = [[LRTOptions alloc] initWithHardwareAccelerators:LRTHardwareAcceleratorGPU];
  NSString *modelPath = GetTestModelPath();

  LRTCompiledModel *compiledModel = [LRTCompiledModel compiledModelWithModelFilePath:modelPath
                                                                         environment:env
                                                                             options:options
                                                                               error:&error];
  XCTAssertNotNil(compiledModel);
  XCTAssertNil(error);

  NSArray<LRTTensorBuffer *> *inputs = [compiledModel createInputTensorBuffersWithError:&error];
  XCTAssertNotNil(inputs);
  XCTAssertNil(error);
  XCTAssertEqual(inputs.count, 2);
  XCTAssertEqual(inputs[0].bufferType, LRTTensorBufferTypeMetalBuffer);
  XCTAssertNotNil(inputs[0].metalBuffer);
  XCTAssertEqual(inputs[1].bufferType, LRTTensorBufferTypeMetalBuffer);
  XCTAssertNotNil(inputs[1].metalBuffer);

  NSArray<LRTTensorBuffer *> *outputs = [compiledModel createOutputTensorBuffersWithError:&error];
  XCTAssertNotNil(outputs);
  XCTAssertNil(error);
  XCTAssertEqual(outputs.count, 1);
  XCTAssertEqual(outputs[0].bufferType, LRTTensorBufferTypeMetalBuffer);
  XCTAssertNotNil(outputs[0].metalBuffer);

  NSData *input0Data = [NSData dataWithBytes:kTestInput0Tensor length:sizeof(kTestInput0Tensor)];
  NSData *input1Data = [NSData dataWithBytes:kTestInput1Tensor length:sizeof(kTestInput1Tensor)];

  XCTAssertTrue([inputs[0] writeData:input0Data error:&error]);
  XCTAssertTrue([inputs[1] writeData:input1Data error:&error]);

  BOOL runSuccess = [compiledModel runWithInputs:inputs outputs:outputs error:&error];
  XCTAssertTrue(runSuccess);
  XCTAssertNil(error);

  NSData *outputData = [outputs[0] readDataWithError:&error];
  XCTAssertNotNil(outputData);
  XCTAssertEqual(outputData.length, sizeof(kTestOutputTensor));

  const float *outputFloat = static_cast<const float *>(outputData.bytes);
  for (size_t i = 0; i < kTestOutputSize; ++i) {
    XCTAssertEqualWithAccuracy(outputFloat[i], kTestOutputTensor[i], kTestAccuracy);
  }
}

- (void)testMetalPipelineChaining {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  XCTSkipIf(device == nil, @"Metal is not supported on this device/simulator.");

  constexpr const float kTestOutputTensorForPipelineTest[] = {21.0f, 42.0f};

  NSError *error = nil;
  LRTEnvironment *env = [LRTEnvironment environmentWithOptions:nil error:&error];
  XCTAssertNotNil(env);
  XCTAssertNil(error);

  LRTOptions *options = [[LRTOptions alloc] initWithHardwareAccelerators:LRTHardwareAcceleratorGPU];
  NSString *modelPath = GetTestModelPath();

  // Create 1st model.
  LRTCompiledModel *compiledModel1 = [LRTCompiledModel compiledModelWithModelFilePath:modelPath
                                                                          environment:env
                                                                              options:options
                                                                                error:&error];
  XCTAssertNotNil(compiledModel1);
  XCTAssertNil(error);

  NSArray<LRTTensorBuffer *> *inputs1 = [compiledModel1 createInputTensorBuffersWithError:&error];
  XCTAssertNotNil(inputs1);
  XCTAssertNil(error);
  XCTAssertEqual(inputs1.count, 2);

  NSArray<LRTTensorBuffer *> *outputs1 = [compiledModel1 createOutputTensorBuffersWithError:&error];
  XCTAssertNotNil(outputs1);
  XCTAssertNil(error);
  XCTAssertEqual(outputs1.count, 1);

  // Create 2nd model.
  LRTCompiledModel *compiledModel2 = [LRTCompiledModel compiledModelWithModelFilePath:modelPath
                                                                          environment:env
                                                                              options:options
                                                                                error:&error];
  XCTAssertNotNil(compiledModel2);
  XCTAssertNil(error);

  // One of input buffers of 2nd model is same as output of 1st model.
  // Set rest of the input buffers of 2nd model same as 1st model's input buffer 1.
  NSArray<LRTTensorBuffer *> *inputs2 = @[ outputs1[0], inputs1[1] ];
  NSArray<LRTTensorBuffer *> *outputs2 = [compiledModel2 createOutputTensorBuffersWithError:&error];
  XCTAssertNotNil(outputs2);
  XCTAssertNil(error);
  XCTAssertEqual(outputs2.count, 1);

  // Fill model inputs for 1st model.
  NSData *input0Data = [NSData dataWithBytes:kTestInput0Tensor length:sizeof(kTestInput0Tensor)];
  NSData *input1Data = [NSData dataWithBytes:kTestInput1Tensor length:sizeof(kTestInput1Tensor)];

  XCTAssertTrue([inputs1[0] writeData:input0Data error:&error]);
  XCTAssertTrue([inputs1[1] writeData:input1Data error:&error]);

  // Execute 1st model.
  BOOL runSuccess1 = [compiledModel1 runWithInputs:inputs1 outputs:outputs1 error:&error];
  XCTAssertTrue(runSuccess1);
  XCTAssertNil(error);

  // Execute 2nd model using output of 1st model as input 0.
  BOOL runSuccess2 = [compiledModel2 runWithInputs:inputs2 outputs:outputs2 error:&error];
  XCTAssertTrue(runSuccess2);
  XCTAssertNil(error);

  // Check 2nd model output.
  NSData *outputData2 = [outputs2[0] readDataWithError:&error];
  XCTAssertNotNil(outputData2);
  XCTAssertEqual(outputData2.length, sizeof(kTestOutputTensorForPipelineTest));

  const float *outputFloat2 = static_cast<const float *>(outputData2.bytes);
  for (size_t i = 0; i < kTestOutputSize; ++i) {
    XCTAssertEqualWithAccuracy(outputFloat2[i], kTestOutputTensorForPipelineTest[i], kTestAccuracy);
  }
}

- (void)testCustomMetalBufferInputsAndOutputs {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  XCTSkipIf(device == nil, @"Metal is not supported on this device/simulator.");

  NSError *error = nil;
  LRTEnvironmentOptions *envOptions = [[LRTEnvironmentOptions alloc] init];
  envOptions.metalDevice = device;
  envOptions.metalCommandQueue = [device newCommandQueue];

  LRTEnvironment *env = [LRTEnvironment environmentWithOptions:envOptions error:&error];
  XCTAssertNotNil(env);
  XCTAssertNil(error);

  LRTOptions *options = [[LRTOptions alloc] initWithHardwareAccelerators:LRTHardwareAcceleratorGPU];
  NSString *modelPath = GetTestModelPath();

  LRTCompiledModel *compiledModel = [LRTCompiledModel compiledModelWithModelFilePath:modelPath
                                                                         environment:env
                                                                             options:options
                                                                               error:&error];
  XCTAssertNotNil(compiledModel);
  XCTAssertNil(error);

  // Allocate explicit MTLBuffers.
  id<MTLBuffer> mtlInput0 = [device newBufferWithLength:sizeof(kTestInput0Tensor)
                                                options:MTLResourceStorageModeShared];
  XCTAssertNotNil(mtlInput0);
  std::memcpy(mtlInput0.contents, kTestInput0Tensor, sizeof(kTestInput0Tensor));

  id<MTLBuffer> mtlInput1 = [device newBufferWithLength:sizeof(kTestInput1Tensor)
                                                options:MTLResourceStorageModeShared];
  XCTAssertNotNil(mtlInput1);
  std::memcpy(mtlInput1.contents, kTestInput1Tensor, sizeof(kTestInput1Tensor));

  id<MTLBuffer> mtlOutput0 = [device newBufferWithLength:sizeof(kTestOutputTensor)
                                                 options:MTLResourceStorageModeShared];
  XCTAssertNotNil(mtlOutput0);

  LRTTensorBuffer *input0 = [LRTTensorBuffer tensorBufferWithEnvironment:env
                                                             metalBuffer:mtlInput0
                                                             elementType:LRTElementTypeFloat32
                                                              dimensions:@[ @2 ]
                                                                   error:&error];
  XCTAssertNotNil(input0);
  XCTAssertNil(error);

  LRTTensorBuffer *input1 = [LRTTensorBuffer tensorBufferWithEnvironment:env
                                                             metalBuffer:mtlInput1
                                                             elementType:LRTElementTypeFloat32
                                                              dimensions:@[ @2 ]
                                                                   error:&error];
  XCTAssertNotNil(input1);
  XCTAssertNil(error);

  LRTTensorBuffer *output0 = [LRTTensorBuffer tensorBufferWithEnvironment:env
                                                              metalBuffer:mtlOutput0
                                                              elementType:LRTElementTypeFloat32
                                                               dimensions:@[ @2 ]
                                                                    error:&error];
  XCTAssertNotNil(output0);
  XCTAssertNil(error);

  BOOL runSuccess = [compiledModel runWithInputs:@[ input0, input1 ]
                                         outputs:@[ output0 ]
                                           error:&error];
  XCTAssertTrue(runSuccess);
  XCTAssertNil(error);

  const float *outputFloat = static_cast<const float *>(mtlOutput0.contents);
  for (size_t i = 0; i < kTestOutputSize; ++i) {
    XCTAssertEqualWithAccuracy(outputFloat[i], kTestOutputTensor[i], kTestAccuracy);
  }
}

@end
