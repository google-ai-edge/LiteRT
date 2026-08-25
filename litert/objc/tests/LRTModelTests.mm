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
  LRTEnvironment *env = [LRTEnvironment environmentWithOptions:nil error:&error];
  XCTAssertNotNil(env);
  XCTAssertNil(error);

  NSString *modelPath = GetTestModelPath();
  LRTModel *model = [LRTModel modelWithModelFilePath:modelPath environment:env error:&error];
  XCTAssertNotNil(model);
  XCTAssertNil(error);
  XCTAssertEqual(model.environment, env);

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

  NSString *sigKey = signatures.firstObject;
  NSArray<NSString *> *keyInputNames = [model inputNamesForSignatureKey:sigKey error:&error];
  XCTAssertNotNil(keyInputNames);
  XCTAssertNil(error);
  XCTAssertEqualObjects(keyInputNames, inputNames);

  NSArray<NSString *> *keyOutputNames = [model outputNamesForSignatureKey:sigKey error:&error];
  XCTAssertNotNil(keyOutputNames);
  XCTAssertNil(error);
  XCTAssertEqualObjects(keyOutputNames, outputNames);
}

- (void)testLoadModelFromData {
  NSError *error = nil;
  LRTEnvironment *env = [LRTEnvironment environmentWithOptions:nil error:&error];
  XCTAssertNotNil(env);
  XCTAssertNil(error);

  NSString *modelPath = GetTestModelPath();
  NSData *modelData = [NSData dataWithContentsOfFile:modelPath];
  XCTAssertNotNil(modelData);

  LRTModel *model = [LRTModel modelWithModelData:modelData environment:env error:&error];
  XCTAssertNotNil(model);
  XCTAssertNil(error);
  XCTAssertEqual(model.environment, env);

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

- (void)testModelErrorHandling {
  NSError *error = nil;
  LRTEnvironment *env = [LRTEnvironment environmentWithOptions:nil error:&error];
  XCTAssertNotNil(env);
  XCTAssertNil(error);

  // Nil file path
  NSString *nilPath = (id)nil;
  XCTAssertNil([LRTModel modelWithModelFilePath:nilPath environment:env error:&error]);
  XCTAssertNotNil(error);
  XCTAssertEqual(error.code, LRTErrorCodeInvalidArgument);
  error = nil;

  // Invalid file path
  XCTAssertNil([LRTModel modelWithModelFilePath:@"/invalid/path/model.tflite"
                                    environment:env
                                          error:&error]);
  XCTAssertNotNil(error);
  error = nil;

  // Nil model data
  NSData *nilData = (id)nil;
  XCTAssertNil([LRTModel modelWithModelData:nilData environment:env error:&error]);
  XCTAssertNotNil(error);
  XCTAssertEqual(error.code, LRTErrorCodeInvalidArgument);
  error = nil;

  // Empty model data
  XCTAssertNil([LRTModel modelWithModelData:[NSData data] environment:env error:&error]);
  XCTAssertNotNil(error);
  XCTAssertEqual(error.code, LRTErrorCodeInvalidArgument);
  error = nil;

  // Valid model inspection error paths
  LRTModel *model = [LRTModel modelWithModelFilePath:GetTestModelPath()
                                         environment:env
                                               error:&error];
  XCTAssertNotNil(model);
  XCTAssertNil(error);

  // Nil signature key
  NSString *nilKey = (id)nil;
  XCTAssertNil([model inputNamesForSignatureKey:nilKey error:&error]);
  XCTAssertNotNil(error);
  XCTAssertEqual(error.code, LRTErrorCodeInvalidArgument);
  error = nil;

  XCTAssertNil([model outputNamesForSignatureKey:nilKey error:&error]);
  XCTAssertNotNil(error);
  XCTAssertEqual(error.code, LRTErrorCodeInvalidArgument);
  error = nil;

  // Non-existent signature index
  XCTAssertNil([model inputNamesForSignatureIndex:999 error:&error]);
  XCTAssertNotNil(error);
  error = nil;

  XCTAssertNil([model outputNamesForSignatureIndex:999 error:&error]);
  XCTAssertNotNil(error);
  error = nil;

  // Nil metadata key
  NSString *nilMetaKey = (id)nil;
  XCTAssertNil([model metadataForKey:nilMetaKey error:&error]);
  XCTAssertNotNil(error);
  XCTAssertEqual(error.code, LRTErrorCodeInvalidArgument);
  error = nil;

  // Non-existent metadata key
  XCTAssertNil([model metadataForKey:@"non_existent_metadata_key" error:&error]);
  XCTAssertNotNil(error);
}

@end
