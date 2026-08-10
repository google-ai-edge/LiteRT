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

#import "third_party/odml/litert/litert/objc/apis/LRTEnvironment.h"
#import "third_party/odml/litert/litert/objc/apis/LRTError.h"

@interface LRTEnvironmentTests : XCTestCase
@end

@implementation LRTEnvironmentTests

- (void)testCreateWithDefaultOptionsSuccess {
  NSError *error = nil;
  LRTEnvironment *env = [LRTEnvironment environmentWithOptions:nil error:&error];
  XCTAssertNotNil(env);
  XCTAssertNil(error);
}

- (void)testCreateWithCustomOptionsSuccess {
  LRTEnvironmentOptions *options = [[LRTEnvironmentOptions alloc] init];
  NSError *error = nil;
  LRTEnvironment *env = [LRTEnvironment environmentWithOptions:options error:&error];
  XCTAssertNotNil(env);
  XCTAssertNil(error);
}

@end
