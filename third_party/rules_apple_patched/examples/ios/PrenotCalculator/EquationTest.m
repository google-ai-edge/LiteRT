// Copyright 2015 The Bazel Authors. All rights reserved.
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

#import "examples/ios/PrenotCalculator/Equation.h"
#import "examples/ios/PrenotCalculator/Literal.h"

#import <XCTest/XCTest.h>

@interface EquationTests : XCTestCase
@end

@implementation EquationTests

- (void)testCalculateAdd {
  Equation *equation = [[Equation alloc] initWithOperation:kAdd];
  [equation addExpressionAsChild:[[Literal alloc] initWithDouble:6]];
  [equation addExpressionAsChild:[[Literal alloc] initWithDouble:7]];
  XCTAssertEqual([equation calculate], 13);
}

- (void)testCalculateSubtract {
  Equation *equation = [[Equation alloc] initWithOperation:kSubtract];
  [equation addExpressionAsChild:[[Literal alloc] initWithDouble:6]];
  [equation addExpressionAsChild:[[Literal alloc] initWithDouble:7]];
  XCTAssertEqual([equation calculate], -1);
}

- (void)testCalculateMultiply {
  Equation *equation = [[Equation alloc] initWithOperation:kMultiply];
  [equation addExpressionAsChild:[[Literal alloc] initWithDouble:6]];
  [equation addExpressionAsChild:[[Literal alloc] initWithDouble:7]];
  XCTAssertEqual([equation calculate], 42);
}

- (void)testCalculateDivide {
  Equation *equation = [[Equation alloc] initWithOperation:kDivide];
  [equation addExpressionAsChild:[[Literal alloc] initWithDouble:42]];
  [equation addExpressionAsChild:[[Literal alloc] initWithDouble:7]];
  XCTAssertEqual([equation calculate], 6);
}

@end
