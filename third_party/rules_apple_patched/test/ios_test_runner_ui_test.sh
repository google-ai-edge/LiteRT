#!/bin/bash

# Copyright 2018 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Integration tests for iOS test runner.

function set_up() {
  mkdir -p ios
}

function tear_down() {
  rm -rf ios
}

function create_sim_runners() {
  cat > ios/BUILD <<EOF
load(
    "@build_bazel_rules_apple//apple:ios.bzl",
    "ios_application",
    "ios_ui_test"
 )
load("@build_bazel_rules_swift//swift:swift.bzl",
     "swift_library"
)
load(
    "@build_bazel_rules_apple//apple/testing/default_runner:ios_test_runner.bzl",
    "ios_test_runner"
)

ios_test_runner(
    name = "ios_x86_64_sim_runner",
    device_type = "iPhone 12",
)
EOF
}

function create_ios_app() {
  if [[ ! -f ios/BUILD ]]; then
    fail "create_sim_runners must be called first."
  fi

  cat > ios/main.m <<EOF
#import <UIKit/UIKit.h>

int main(int argc, char * argv[]) {
  @autoreleasepool {
    return UIApplicationMain(argc, argv, nil, nil);
  }
}
EOF

  cat > ios/Info.plist <<EOF
{
  CFBundleIdentifier = "\${PRODUCT_BUNDLE_IDENTIFIER}";
  CFBundleName = "\${PRODUCT_NAME}";
  CFBundlePackageType = "APPL";
  CFBundleShortVersionString = "1.0";
  CFBundleVersion = "1.0";
}
EOF

  cat >> ios/BUILD <<EOF
objc_library(
    name = "app_lib",
    srcs = [
        "main.m"
    ],
)

ios_application(
    name = "app",
    bundle_id = "my.bundle.id",
    families = ["iphone"],
    infoplists = ["Info.plist"],
    minimum_os_version = "${MIN_OS_IOS_NPLUS1}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    deps = [":app_lib"],
)
EOF
}

function create_ios_ui_tests() {
  if [[ ! -f ios/BUILD ]]; then
    fail "create_sim_runners must be called first."
  fi

  cat > ios/pass_ui_test.m <<EOF
#import <XCTest/XCTest.h>

@interface PassingUiTest : XCTestCase

@end

@implementation PassingUiTest

- (void)setUp {
    [super setUp];
    self.continueAfterFailure = NO;
    [[[XCUIApplication alloc] init] launch];
}

- (void)tearDown {
    [super tearDown];
}

- (void)testPass {
  XCTAssertEqual(1, 1, @"should pass");
}

- (void)testPass2 {
  XCTAssertEqual(1, 1, @"should pass");
}

@end
EOF

  cat > ios/pass_ui_test.swift <<EOF
import XCTest

class PassingUiTest: XCTestCase {

  override func setUp() {
    super.setUp()
    XCUIApplication().launch()
  }

  func testPass() throws {
    let result = 1 + 1;
    XCTAssertEqual(result, 2);
  }
}
EOF

  cat > ios/fail_ui_test.m <<EOF
#import <XCTest/XCTest.h>

@interface FailingUiTest : XCTestCase

@end

@implementation FailingUiTest

- (void)setUp {
    [super setUp];
    self.continueAfterFailure = NO;
    [[[XCUIApplication alloc] init] launch];
}

- (void)tearDown {
    [super tearDown];
}

- (void)testFail {
  XCTAssertEqual(0, 1, @"should fail");
}

@end
EOF

  cat > ios/PassUiTest-Info.plist <<EOF
<plist version="1.0">
<dict>
        <key>CFBundleExecutable</key>
        <string>PassUiTest</string>
</dict>
</plist>
EOF

  cat > ios/PassUiSwiftTest-Info.plist <<EOF
<plist version="1.0">
<dict>
        <key>CFBundleExecutable</key>
        <string>PassUiSwiftTest</string>
</dict>
</plist>
EOF

  cat > ios/FailUiTest-Info.plist <<EOF
<plist version="1.0">
<dict>
        <key>CFBundleExecutable</key>
        <string>FailingUiTest</string>
</dict>
</plist>
EOF

  cat >> ios/BUILD <<EOF
objc_library(
    name = "pass_ui_test_lib",
    srcs = ["pass_ui_test.m"],
)

ios_ui_test(
    name = "PassingUiTest",
    infoplists = ["PassUiTest-Info.plist"],
    deps = [":pass_ui_test_lib"],
    minimum_os_version = "${MIN_OS_IOS}",
    test_host = ":app",
    runner = ":ios_x86_64_sim_runner",
)

swift_library(
    name = "pass_ui_swift_test_lib",
    testonly = True,
    srcs = ["pass_ui_test.swift"],
)

ios_ui_test(
    name = "PassingUiSwiftTest",
    infoplists = ["PassUiSwiftTest-Info.plist"],
    deps = [":pass_ui_swift_test_lib"],
    minimum_os_version = "${MIN_OS_IOS_NPLUS1}",
    test_host = ":app",
    runner = ":ios_x86_64_sim_runner",
)

objc_library(
    name = "fail_ui_test_lib",
    srcs = ["fail_ui_test.m"],
)

ios_ui_test(
    name = 'FailingUiTest',
    infoplists = ["FailUiTest-Info.plist"],
    deps = [":fail_ui_test_lib"],
    minimum_os_version = "${MIN_OS_IOS}",
    test_host = ":app",
    runner = ":ios_x86_64_sim_runner",
)
EOF
}

function create_ios_ui_envtest() {
  if [[ ! -f ios/BUILD ]]; then
    fail "create_sim_runners must be called first."
  fi

  cat > ios/env_ui_test.m <<EOF
#import <XCTest/XCTest.h>
#include <assert.h>
#include <stdlib.h>

@interface EnvUiTest : XCTestCase

@end

@implementation EnvUiTest

- (void)setUp {
    [super setUp];
    self.continueAfterFailure = NO;
    [[[XCUIApplication alloc] init] launch];
}

- (void)tearDown {
    [super tearDown];
}

- (void)testEnv {
  NSString *var_value = [[[NSProcessInfo processInfo] environment] objectForKey:@"$1"];
  XCTAssertEqualObjects(var_value, @"$2", @"env $1 should be %@, instead is %@", @"$2", var_value);
}

@end
EOF

  cat >ios/EnvUiTest-Info.plist <<EOF
<plist version="1.0">
<dict>
        <key>CFBundleExecutable</key>
        <string>EnvUiTest</string>
</dict>
</plist>
EOF

  cat >> ios/BUILD <<EOF
objc_library(
    name = "env_ui_test_lib",
    srcs = ["env_ui_test.m"],
)

ios_ui_test(
    name = 'EnvUiTest',
    infoplists = ["EnvUiTest-Info.plist"],
    deps = [":env_ui_test_lib"],
    minimum_os_version = "${MIN_OS_IOS}",
    test_host = ":app",
    runner = ":ios_x86_64_sim_runner",
)
EOF
}

function do_ios_test() {
  do_test ios "--test_output=all" "--spawn_strategy=local" "--ios_minimum_os=11.0" "$@"
}

function test_ios_ui_test_pass() {
  create_sim_runners
  create_ios_app
  create_ios_ui_tests
  do_ios_test //ios:PassingUiTest || fail "should pass"

  expect_log "Test Suite 'PassingUiTest' passed"
  expect_log "Test Suite 'PassingUiTest.xctest' passed"
  expect_log "Executed 2 tests, with 0 failures"
}

function test_ios_ui_swift_test_pass() {
  create_sim_runners
  create_ios_app
  create_ios_ui_tests
  do_ios_test //ios:PassingUiSwiftTest || fail "should pass"

  expect_log "Test Suite 'PassingUiTest' passed"
  expect_log "Test Suite 'PassingUiSwiftTest.xctest' passed"
  expect_log "Executed 1 test, with 0 failures"
}

function test_ios_ui_test_fail() {
  create_sim_runners
  create_ios_app
  create_ios_ui_tests
  ! do_ios_test //ios:FailingUiTest || fail "should fail"

  expect_log "Test Suite 'FailingUiTest' failed"
  expect_log "Test Suite 'FailingUiTest.xctest' failed"
  expect_log "Executed 1 test, with 1 failure"
}

function test_ios_ui_test_with_filter() {
  create_sim_runners
  create_ios_app
  create_ios_ui_tests
  do_ios_test --test_filter=PassingUiTest/testPass2 //ios:PassingUiTest || fail "should pass"

  expect_log "Test Case '-\[PassingUiTest testPass2\]' passed"
  expect_log "Test Suite 'PassingUiTest' passed"
  expect_log "Test Suite 'PassingUiTest.xctest' passed"
  expect_log "Executed 1 test, with 0 failures"
}

function test_ios_ui_test_with_env() {
  create_sim_runners
  create_ios_app
  create_ios_ui_envtest ENV_KEY1 ENV_VALUE2
  do_ios_test --test_env=ENV_KEY1=ENV_VALUE2 //ios:EnvUiTest || fail "should pass"

  expect_log "Test Suite 'EnvUiTest' passed"
}

run_suite "ios_ui_test with iOS test runner bundling tests"
