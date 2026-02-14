#import <XCTest/XCTest.h>
#import "test/starlark_tests/resources/common.h"

@interface CommonTests: XCTestCase
@end

@implementation CommonTests

- (void)testAnything {
  // Call something in test host to ensure bundle loading works.
  [[[ObjectiveCCommonClass alloc] init] doSomethingCommon];
  XCTAssertNil(nil);
}

@end
