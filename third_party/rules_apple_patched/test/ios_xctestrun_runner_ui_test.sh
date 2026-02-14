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

# Integration tests for iOS xctestrun runner.

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
    "@build_bazel_rules_apple//apple/testing/default_runner:ios_xctestrun_runner.bzl",
    "ios_xctestrun_runner"
)

ios_xctestrun_runner(
    name = "ios_x86_64_sim_runner",
    device_type = "iPhone Xs",
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

@interface PassingUITest : XCTestCase

@end

@implementation PassingUITest

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

class PassingUITest: XCTestCase {

  override func setUp() {
    super.setUp()
    XCUIApplication().launch()
  }

  func testPass() throws {
    let result = 1 + 1;
    XCTAssertEqual(result, 2);
  }

  func testPass2() throws {
    let result = 1 + 1;
    XCTAssertEqual(result, 2);
  }
}
EOF

  cat > ios/fail_ui_test.m <<EOF
#import <XCTest/XCTest.h>

@interface FailingUITest : XCTestCase

@end

@implementation FailingUITest

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

  cat > ios/PassUITest-Info.plist <<EOF
<plist version="1.0">
<dict>
        <key>CFBundleExecutable</key>
        <string>PassUITest</string>
</dict>
</plist>
EOF

  cat > ios/PassUISwiftTest-Info.plist <<EOF
<plist version="1.0">
<dict>
        <key>CFBundleExecutable</key>
        <string>PassUISwiftTest</string>
</dict>
</plist>
EOF

  cat > ios/FailUITest-Info.plist <<EOF
<plist version="1.0">
<dict>
        <key>CFBundleExecutable</key>
        <string>FailingUITest</string>
</dict>
</plist>
EOF

  cat >> ios/BUILD <<EOF
objc_library(
    name = "pass_ui_test_lib",
    srcs = ["pass_ui_test.m"],
)

ios_ui_test(
    name = "PassingUITest",
    infoplists = ["PassUITest-Info.plist"],
    deps = [":pass_ui_test_lib"],
    minimum_os_version = "${MIN_OS_IOS}",
    test_host = ":app",
    runner = ":ios_x86_64_sim_runner",
)

ios_ui_test(
    name = "PassingUITest_WithUnderscore",
    infoplists = ["PassUITest-Info.plist"],
    deps = [":pass_ui_test_lib"],
    minimum_os_version = "${MIN_OS_IOS}",
    test_host = ":app",
    runner = ":ios_x86_64_sim_runner",
)

ios_ui_test(
    name = "PassingUITest-WithDash",
    infoplists = ["PassUITest-Info.plist"],
    deps = [":pass_ui_test_lib"],
    minimum_os_version = "${MIN_OS_IOS}",
    test_host = ":app",
    runner = ":ios_x86_64_sim_runner",
)

swift_library(
    name = "pass_ui_swift_test_lib",
    module_name = "PassingUISwiftTest",
    testonly = True,
    srcs = ["pass_ui_test.swift"],
)

ios_ui_test(
    name = "PassingUISwiftTest",
    infoplists = ["PassUISwiftTest-Info.plist"],
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
    name = 'FailingUITest',
    infoplists = ["FailUITest-Info.plist"],
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

@interface EnvUITest : XCTestCase

@end

@implementation EnvUITest

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

  cat >ios/EnvUITest-Info.plist <<EOF
<plist version="1.0">
<dict>
        <key>CFBundleExecutable</key>
        <string>EnvUITest</string>
</dict>
</plist>
EOF

  cat >> ios/BUILD <<EOF
objc_library(
    name = "env_ui_test_lib",
    srcs = ["env_ui_test.m"],
)

ios_ui_test(
    name = 'EnvUITest',
    infoplists = ["EnvUITest-Info.plist"],
    deps = [":env_ui_test_lib"],
    minimum_os_version = "${MIN_OS_IOS}",
    test_host = ":app",
    runner = ":ios_x86_64_sim_runner",
)
EOF
}

function do_ios_test() {
  do_test ios "--test_output=all" "--spawn_strategy=local" "--ios_minimum_os=9.0" "$@"
}

function test_ios_ui_test_pass() {
  create_sim_runners
  create_ios_app
  create_ios_ui_tests
  do_ios_test //ios:PassingUITest || fail "should pass"

  expect_log "Test Suite 'PassingUITest' passed"
  expect_log "Test Suite 'PassingUITest.xctest' passed"
  expect_log "Executed 2 tests, with 0 failures"
}

function test_ios_ui_swift_test_pass() {
  create_sim_runners
  create_ios_app
  create_ios_ui_tests
  do_ios_test //ios:PassingUISwiftTest || fail "should pass"

  expect_log "Test Suite 'PassingUITest' passed"
  expect_log "Test Suite 'PassingUISwiftTest.xctest' passed"
  expect_log "Executed 2 tests, with 0 failures"
}

function test_ios_ui_test_fail() {
  create_sim_runners
  create_ios_app
  create_ios_ui_tests
  ! do_ios_test //ios:FailingUITest || fail "should fail"

  expect_log "Test Suite 'FailingUITest' failed"
  expect_log "Test Suite 'FailingUITest.xctest' failed"
  expect_log "Executed 1 test, with 1 failure"
}

function test_ios_ui_test_with_filter() {
  create_sim_runners
  create_ios_app
  create_ios_ui_tests
  do_ios_test --test_filter=PassingUITest/testPass2 //ios:PassingUITest || fail "should pass"

  expect_log "Test Case '-\[PassingUITest testPass2\]' passed"
  expect_log "Test Suite 'PassingUITest' passed"
  expect_log "Test Suite 'PassingUITest.xctest' passed"
  expect_log "Executed 1 test, with 0 failures"
}

function test_ios_ui_test_with_filter_no_tests_ran_fail() {
  create_sim_runners
  create_ios_app
  create_ios_ui_tests
  ! do_ios_test --test_filter=PassingUITest/testInvalid //ios:PassingUITest|| fail "should fail"

  expect_log "Test Suite 'PassingUITest.xctest' passed"
  expect_log "Test Suite 'Selected tests' passed"
  expect_log "Executed 0 tests, with 0 failures"
  expect_log "error: no tests were executed, is the test bundle empty?"
}

function test_ios_ui_test_with_filter_no_tests_ran_pass() {
  create_sim_runners
  create_ios_app
  create_ios_ui_tests
  do_ios_test \
    --test_env=ERROR_ON_NO_TESTS_RAN=0 \
    --test_filter=PassingUITest/testInvalid \
    //ios:PassingUITest \
    || fail "should pass"

  expect_log "Test Suite 'PassingUITest.xctest' passed"
  expect_log "Test Suite 'Selected tests' passed"
  expect_log "Executed 0 tests, with 0 failures"
}

function test_ios_ui_test_underscore_with_filter() {
  create_sim_runners
  create_ios_app
  create_ios_ui_tests
  do_ios_test --test_filter=PassingUITest/testPass2 //ios:PassingUITest_WithUnderscore || fail "should pass"

  expect_log "Test Case '-\[PassingUITest testPass2\]' passed"
  expect_log "Test Suite 'PassingUITest' passed"
  expect_log "Test Suite 'PassingUITest_WithUnderscore.xctest' passed"
  expect_log "Executed 1 test, with 0 failures"
}

function test_ios_ui_test_dash_with_filter() {
  create_sim_runners
  create_ios_app
  create_ios_ui_tests
  do_ios_test --test_filter=PassingUITest/testPass2 //ios:PassingUITest-WithDash || fail "should pass"

  expect_log "Test Case '-\[PassingUITest testPass2\]' passed"
  expect_log "Test Suite 'PassingUITest' passed"
  expect_log "Test Suite 'PassingUITest-WithDash.xctest' passed"
  expect_log "Executed 1 test, with 0 failures"
}

function test_ios_ui_swift_test_with_filter() {
  create_sim_runners
  create_ios_app
  create_ios_ui_tests
  do_ios_test --test_filter=PassingUITest/testPass2 //ios:PassingUISwiftTest || fail "should pass"

  expect_log "Test Case '-\[PassingUISwiftTest.PassingUITest testPass2\]' passed"
  expect_log "Test Suite 'PassingUITest' passed"
  expect_log "Test Suite 'PassingUISwiftTest.xctest' passed"
  expect_log "Executed 1 test, with 0 failures"
}

function test_ios_ui_test_with_env() {
  create_sim_runners
  create_ios_app
  create_ios_ui_envtest ENV_KEY1 ENV_VALUE2
  do_ios_test --test_env=ENV_KEY1=ENV_VALUE2 //ios:EnvUITest || fail "should pass"

  expect_log "Test Suite 'EnvUITest' passed"
}

function test_ios_ui_test_default_attachment_lifetime_arg() {
  create_sim_runners
  create_ios_app
  create_ios_ui_tests
  do_ios_test \
    --test_env=DEBUG_XCTESTRUNNER=1 \
    --test_filter=PassingUITest/testPass2 \
    //ios:PassingUITest || fail "should pass"

    expect_log "<key>SystemAttachmentLifetime</key>"
    expect_log "<string>keepNever</string>"
    expect_log "<key>UserAttachmentLifetime</key>"
    expect_log "<string>keepNever</string>"
}

function test_ios_ui_test_attachment_lifetime_arg() {
  create_sim_runners
  create_ios_app
  create_ios_ui_tests
  do_ios_test \
    --test_env=DEBUG_XCTESTRUNNER=1 \
    --test_filter=PassingUITest/testPass2 \
    --test_arg=--xctestrun_attachment_lifetime=deleteOnSuccess \
    //ios:PassingUITest || fail "should pass"

    expect_log "<key>SystemAttachmentLifetime</key>"
    expect_log "<string>deleteOnSuccess</string>"
    expect_log "<key>UserAttachmentLifetime</key>"
    expect_log "<string>deleteOnSuccess</string>"
}

run_suite "ios_ui_test with iOS xctestrun runner bundling tests"
