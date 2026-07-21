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

#import <XCTest/XCTest.h>

#include <string>
#include <vector>

#import "third_party/odml/litert/litert/objc/apis/LRTCompiledModel.h"
#import "third_party/odml/litert/litert/objc/apis/LRTEnvironment.h"
#import "third_party/odml/litert/litert/objc/apis/LRTError.h"
#import "third_party/odml/litert/litert/objc/apis/LRTOptions.h"
#import "third_party/odml/litert/litert/objc/apis/LRTTensorBuffer.h"
#include "litert/test/common.h"
#include "litert/test/testdata/simple_model_test_vectors.h"

@interface LRTCompiledModelTests : XCTestCase
@end

static constexpr float kTestAccuracy = 1e-5f;

static NSString *GetTestModelPath() {
  NSBundle *bundle = [NSBundle bundleForClass:[LRTCompiledModelTests class]];
  NSString *path = [bundle pathForResource:@"simple_model" ofType:@"tflite"];
  if (path) {
    return path;
  }
  std::string modelPath = litert::testing::GetTestFilePath(kModelFileName);
  return @(modelPath.c_str());
}

@implementation LRTCompiledModelTests

- (void)testCreateAndRunFromFilePath {
  NSError *error = nil;
  LRTEnvironment *env = [LRTEnvironment environmentWithOptions:nil error:&error];
  XCTAssertNotNil(env);
  XCTAssertNil(error);

  LRTOptions *options = [[LRTOptions alloc] initWithHardwareAccelerators:LRTHardwareAcceleratorCPU];

  NSString *filePath = GetTestModelPath();

  LRTCompiledModel *model = [LRTCompiledModel compiledModelWithModelFilePath:filePath
                                                                 environment:env
                                                                     options:options
                                                                       error:&error];
  XCTAssertNotNil(model);
  XCTAssertNil(error);
  XCTAssertEqual(model.environment, env);
  XCTAssertEqual(model.options, options);

  NSArray<LRTTensorBuffer *> *inputs = [model createInputTensorBuffersWithError:&error];
  XCTAssertNotNil(inputs);
  XCTAssertNil(error);
  XCTAssertEqual(inputs.count, 2);

  NSArray<LRTTensorBuffer *> *outputs = [model createOutputTensorBuffersWithError:&error];
  XCTAssertNotNil(outputs);
  XCTAssertNil(error);
  XCTAssertEqual(outputs.count, 1);

  NSData *input0Data = [NSData dataWithBytes:kTestInput0Tensor length:sizeof(kTestInput0Tensor)];
  NSData *input1Data = [NSData dataWithBytes:kTestInput1Tensor length:sizeof(kTestInput1Tensor)];

  XCTAssertTrue([inputs[0] writeData:input0Data error:&error]);
  XCTAssertTrue([inputs[1] writeData:input1Data error:&error]);

  BOOL runSuccess = [model runWithInputs:inputs outputs:outputs error:&error];
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

@end
