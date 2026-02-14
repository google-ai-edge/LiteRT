// Copyright 2017 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#import <Foundation/Foundation.h>
#import <XCTest/XCTest.h>

#import "examples/ios/Squarer/Sources/Squarer.h"

@interface SquarerTests : XCTestCase
@end

@implementation SquarerTests

- (void)testNumberIsSquared {
  Squarer *squarer = [[Squarer alloc] init];
  XCTAssertEqual(49, [squarer squareInteger:7], @"Number should be squared");
}

- (void)testEnvIsSet {
  XCTAssertNotNil([[NSProcessInfo processInfo] environment][@"TEST_ENV_VAR"],
                  @"TEST_ENV_VAR should be set");
}

- (void)testSrcdirEnv {
  XCTAssertNotNil([[NSProcessInfo processInfo] environment][@"TEST_SRCDIR"],
                  @"TEST_SRCDIR should be set");
}

@end
