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

#include <cstring>
#include <string>

#import "third_party/odml/litert/litert/objc/apis/LRTEnvironment.h"
#import "third_party/odml/litert/litert/objc/apis/LRTError.h"
#import "third_party/odml/litert/litert/objc/apis/LRTModel.h"
#include "litert/test/common.h"
#include "litert/test/testdata/simple_model_test_vectors.h"

@interface LRTModelTests : XCTestCase
@end

static NSString *GetTestModelPath() {
  NSBundle *bundle = [NSBundle bundleForClass:[LRTModelTests class]];
  NSString *path = [bundle pathForResource:@"simple_model" ofType:@"tflite"];
  if (path) {
    return path;
  }
  std::string modelPath = litert::testing::GetTestFilePath(kModelFileName);
  return @(modelPath.c_str());
}

@implementation LRTModelTests

- (void)testLoadModelFromFilePath {
  NSError *error = nil;
  LRTEnvironment *environment = [LRTEnvironment environmentWithOptions:nil error:&error];
  XCTAssertNotNil(environment);
  XCTAssertNil(error);

  NSString *modelPath = GetTestModelPath();
  LRTModel *model = [LRTModel modelWithModelFilePath:modelPath
                                         environment:environment
                                               error:&error];
  XCTAssertNotNil(model);
  XCTAssertNil(error);
  XCTAssertEqual(model.environment, environment);

  NSArray<NSString *> *signatures = model.signatureKeys;
  XCTAssertNotNil(signatures);
  XCTAssertEqual(signatures.count, 1);

  NSArray<NSString *> *inputNames = [model inputNamesForSignatureIndex:0 error:&error];
  XCTAssertNotNil(inputNames);
  XCTAssertNil(error);
  XCTAssertEqual(inputNames.count, 2);

  NSArray<NSString *> *outputNames = [model outputNamesForSignatureIndex:0 error:&error];
  XCTAssertNotNil(outputNames);
  XCTAssertNil(error);
  XCTAssertEqual(outputNames.count, 1);

  NSString *signatureKey = signatures.firstObject;
  NSArray<NSString *> *keyInputNames = [model inputNamesForSignatureKey:signatureKey error:&error];
  XCTAssertNotNil(keyInputNames);
  XCTAssertNil(error);
  XCTAssertEqualObjects(keyInputNames, inputNames);

  NSArray<NSString *> *keyOutputNames = [model outputNamesForSignatureKey:signatureKey
                                                                    error:&error];
  XCTAssertNotNil(keyOutputNames);
  XCTAssertNil(error);
  XCTAssertEqualObjects(keyOutputNames, outputNames);
}

- (void)testLoadModelFromData {
  NSError *error = nil;
  LRTEnvironment *environment = [LRTEnvironment environmentWithOptions:nil error:&error];
  XCTAssertNotNil(environment);
  XCTAssertNil(error);

  NSString *modelPath = GetTestModelPath();
  NSData *modelData = [NSData dataWithContentsOfFile:modelPath];
  XCTAssertNotNil(modelData);

  LRTModel *model = [LRTModel modelWithModelData:modelData environment:environment error:&error];
  XCTAssertNotNil(model);
  XCTAssertNil(error);
  XCTAssertEqual(model.environment, environment);

  NSArray<NSString *> *signatures = model.signatureKeys;
  XCTAssertNotNil(signatures);
  XCTAssertEqual(signatures.count, 1);

  NSArray<NSString *> *inputNames = [model inputNamesForSignatureIndex:0 error:&error];
  XCTAssertNotNil(inputNames);
  XCTAssertNil(error);
  XCTAssertEqual(inputNames.count, 2);

  NSArray<NSString *> *outputNames = [model outputNamesForSignatureIndex:0 error:&error];
  XCTAssertNotNil(outputNames);
  XCTAssertNil(error);
  XCTAssertEqual(outputNames.count, 1);
}

- (void)testLoadModelFromMutableDataThatCallerLaterClobbers {
  NSError *error = nil;
  LRTEnvironment *environment = [LRTEnvironment environmentWithOptions:nil error:&error];
  XCTAssertNotNil(environment);
  XCTAssertNil(error);

  NSMutableData *modelData = [NSMutableData dataWithContentsOfFile:GetTestModelPath()];
  XCTAssertNotNil(modelData);

  LRTModel *model = [LRTModel modelWithModelData:modelData environment:environment error:&error];
  XCTAssertNotNil(model);
  XCTAssertNil(error);

  // The model owns its own copy of the bytes, so neither clobbering nor releasing the caller's
  // buffer may affect it.
  std::memset(modelData.mutableBytes, 0, modelData.length);
  modelData = nil;

  XCTAssertEqual(model.signatureKeys.count, 1);

  NSArray<NSString *> *inputNames = [model inputNamesForSignatureIndex:0 error:&error];
  XCTAssertNotNil(inputNames);
  XCTAssertNil(error);
  XCTAssertEqual(inputNames.count, 2);
}

#pragma mark - Parameter Validation Tests

// Suppress -Wnonnull warnings so we can explicitly test runtime defensive null-checks
// on parameters marked nonnull in public headers (e.g. from Swift or dynamic Objective-C
// callers).
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnonnull"

- (void)testModelWithNilFilePathFails {
  NSError *error = nil;
  LRTEnvironment *environment = [LRTEnvironment environmentWithOptions:nil error:&error];
  XCTAssertNotNil(environment);
  XCTAssertNil(error);

  XCTAssertNil([LRTModel modelWithModelFilePath:nil environment:environment error:&error]);
  XCTAssertNotNil(error);
  XCTAssertEqual(error.code, LRTErrorCodeInvalidArgument);
}

- (void)testModelWithInvalidFilePathFails {
  NSError *error = nil;
  LRTEnvironment *environment = [LRTEnvironment environmentWithOptions:nil error:&error];
  XCTAssertNotNil(environment);
  XCTAssertNil(error);

  XCTAssertNil([LRTModel modelWithModelFilePath:@"/invalid/path/model.tflite"
                                    environment:environment
                                          error:&error]);
  XCTAssertNotNil(error);
}

- (void)testModelWithNilDataFails {
  NSError *error = nil;
  LRTEnvironment *environment = [LRTEnvironment environmentWithOptions:nil error:&error];
  XCTAssertNotNil(environment);
  XCTAssertNil(error);

  XCTAssertNil([LRTModel modelWithModelData:nil environment:environment error:&error]);
  XCTAssertNotNil(error);
  XCTAssertEqual(error.code, LRTErrorCodeInvalidArgument);
}

- (void)testModelWithEmptyDataFails {
  NSError *error = nil;
  LRTEnvironment *environment = [LRTEnvironment environmentWithOptions:nil error:&error];
  XCTAssertNotNil(environment);
  XCTAssertNil(error);

  XCTAssertNil([LRTModel modelWithModelData:[NSData data] environment:environment error:&error]);
  XCTAssertNotNil(error);
  XCTAssertEqual(error.code, LRTErrorCodeInvalidArgument);
}

- (void)testInputNamesWithNilSignatureKeyFails {
  NSError *error = nil;
  LRTEnvironment *environment = [LRTEnvironment environmentWithOptions:nil error:&error];
  XCTAssertNotNil(environment);
  XCTAssertNil(error);

  LRTModel *model = [LRTModel modelWithModelFilePath:GetTestModelPath()
                                         environment:environment
                                               error:&error];
  XCTAssertNotNil(model);
  XCTAssertNil(error);

  XCTAssertNil([model inputNamesForSignatureKey:nil error:&error]);
  XCTAssertNotNil(error);
  XCTAssertEqual(error.code, LRTErrorCodeInvalidArgument);
}

- (void)testOutputNamesWithNilSignatureKeyFails {
  NSError *error = nil;
  LRTEnvironment *environment = [LRTEnvironment environmentWithOptions:nil error:&error];
  XCTAssertNotNil(environment);
  XCTAssertNil(error);

  LRTModel *model = [LRTModel modelWithModelFilePath:GetTestModelPath()
                                         environment:environment
                                               error:&error];
  XCTAssertNotNil(model);
  XCTAssertNil(error);

  XCTAssertNil([model outputNamesForSignatureKey:nil error:&error]);
  XCTAssertNotNil(error);
  XCTAssertEqual(error.code, LRTErrorCodeInvalidArgument);
}

- (void)testInputNamesWithOutOfBoundsSignatureIndexFails {
  NSError *error = nil;
  LRTEnvironment *environment = [LRTEnvironment environmentWithOptions:nil error:&error];
  XCTAssertNotNil(environment);
  XCTAssertNil(error);

  LRTModel *model = [LRTModel modelWithModelFilePath:GetTestModelPath()
                                         environment:environment
                                               error:&error];
  XCTAssertNotNil(model);
  XCTAssertNil(error);

  XCTAssertNil([model inputNamesForSignatureIndex:999 error:&error]);
  XCTAssertNotNil(error);
}

- (void)testOutputNamesWithOutOfBoundsSignatureIndexFails {
  NSError *error = nil;
  LRTEnvironment *environment = [LRTEnvironment environmentWithOptions:nil error:&error];
  XCTAssertNotNil(environment);
  XCTAssertNil(error);

  LRTModel *model = [LRTModel modelWithModelFilePath:GetTestModelPath()
                                         environment:environment
                                               error:&error];
  XCTAssertNotNil(model);
  XCTAssertNil(error);

  XCTAssertNil([model outputNamesForSignatureIndex:999 error:&error]);
  XCTAssertNotNil(error);
}

- (void)testMetadataWithNilKeyFails {
  NSError *error = nil;
  LRTEnvironment *environment = [LRTEnvironment environmentWithOptions:nil error:&error];
  XCTAssertNotNil(environment);
  XCTAssertNil(error);

  LRTModel *model = [LRTModel modelWithModelFilePath:GetTestModelPath()
                                         environment:environment
                                               error:&error];
  XCTAssertNotNil(model);
  XCTAssertNil(error);

  XCTAssertNil([model metadataForKey:nil error:&error]);
  XCTAssertNotNil(error);
  XCTAssertEqual(error.code, LRTErrorCodeInvalidArgument);
}

- (void)testMetadataWithUnknownKeyFails {
  NSError *error = nil;
  LRTEnvironment *environment = [LRTEnvironment environmentWithOptions:nil error:&error];
  XCTAssertNotNil(environment);
  XCTAssertNil(error);

  LRTModel *model = [LRTModel modelWithModelFilePath:GetTestModelPath()
                                         environment:environment
                                               error:&error];
  XCTAssertNotNil(model);
  XCTAssertNil(error);

  XCTAssertNil([model metadataForKey:@"non_existent_metadata_key" error:&error]);
  XCTAssertNotNil(error);
}

#pragma clang diagnostic pop

@end
