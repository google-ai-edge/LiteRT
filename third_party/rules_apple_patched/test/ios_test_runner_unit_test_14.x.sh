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
    "ios_unit_test"
)
load("@build_bazel_rules_swift//swift:swift.bzl",
     "swift_library"
)
load(
    "@build_bazel_rules_apple//apple/testing/default_runner:ios_test_runner.bzl",
    "ios_test_runner"
)

ios_test_runner(
    name = "ios_x86_64_sim_runner_14",
    device_type = "iPhone 8",
    os_version = "14.5",
)
EOF
}

function create_ios_unit_tests() {
  if [[ ! -f ios/BUILD ]]; then
    fail "create_sim_runners must be called first."
  fi

  cat > ios/pass_unit_test.m <<EOF
#import <XCTest/XCTest.h>
#import <XCTest/XCUIApplication.h>

@interface PassingUnitTest : XCTestCase

@end

@implementation PassingUnitTest {
  XCUIApplication *_app;
}

- (void)testPass {
  XCTAssertEqual(1, 1, @"should pass");
}

- (void)testPass2 {
  XCTAssertEqual(1, 1, @"should pass");
}

- (void)testPass3 {
  XCTAssertEqual(1, 1, @"should pass");
}

- (void)testPassEnvVariable {
  XCTAssertEqualObjects([NSProcessInfo processInfo].environment[@"SomeVariable1"], @"Its My First Variable", @"should pass");
  XCTAssertEqualObjects([NSProcessInfo processInfo].environment[@"SomeVariable2"], @"Its My Second Variable", @"should pass");
  XCTAssertEqualObjects([NSProcessInfo processInfo].environment[@"REFERENCE_DIR"], @"/Project/My Tests/ReferenceImages", @"should pass");
  XCTAssertEqualObjects([NSProcessInfo processInfo].environment[@"IMAGE_DIR"], @"/Project/My Tests/Images", @"should pass");
}

- (void)uiTestSymbols {
  // This function triggers https://github.com/google/xctestrunner/blob/7f8fc81b10c8d93f09f6fe38b2a3f37ba25336a6/test_runner/xctest_session.py#L382
  _app = [[XCUIApplication alloc] init];
}

@end
EOF

  cat > ios/PassUnitTest-Info.plist <<EOF
<plist version="1.0">
<dict>
        <key>CFBundleExecutable</key>
        <string>PassingUnitTest</string>
</dict>
</plist>
EOF

  cat >> ios/BUILD <<EOF
test_env = {
    "SomeVariable1": "Its My First Variable",
    "SomeVariable2": "Its My Second Variable",
    "REFERENCE_DIR": "/Project/My Tests/ReferenceImages",
    "IMAGE_DIR": "/Project/My Tests/Images"
}

objc_library(
    name = "pass_unit_test_lib",
    srcs = ["pass_unit_test.m"],
)

ios_unit_test(
    name = "PassingUnitTest",
    infoplists = ["PassUnitTest-Info.plist"],
    deps = [":pass_unit_test_lib"],
    minimum_os_version = "${MIN_OS_IOS}",
    env = test_env,
    runner = ":ios_x86_64_sim_runner_14",
)
EOF
}

function do_ios_test() {
  do_test ios "--test_output=all" "--spawn_strategy=local" "$@"
}

function test_ios_unit_test_pass() {
  create_sim_runners
  create_ios_unit_tests
  do_ios_test //ios:PassingUnitTest || fail "should pass"

  expect_log "Test Suite 'PassingUnitTest' passed"
  expect_log "Test Suite 'PassingUnitTest.xctest' passed"
  expect_log "Executed 4 tests, with 0 failures"
}

run_suite "ios_unit_test with iOS 14.x test runner bundling tests"
