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
load("@rules_shell//shell:sh_binary.bzl", "sh_binary")

ios_test_runner(
    name = "ios_x86_64_sim_runner",
    device_type = "iPhone Xs",
)

genrule(
  name = "pre_action_gen",
  executable = True,
  outs = ["pre_action.bash"],
  cmd = """
echo 'echo "PRE-ACTION: TEST_TARGET=\$\$TEST_TARGET"' > \$@
""",
)

sh_binary(
  name = "pre_action",
  srcs = [":pre_action_gen"],
)

genrule(
  name = "post_action_gen",
  executable = True,
  outs = ["post_action.bash"],
  cmd = """
echo 'echo "POST-ACTION: TEST_TARGET=\$\$TEST_TARGET"' > \$@
""",
)

sh_binary(
  name = "post_action",
  srcs = [":post_action_gen"],
)

ios_test_runner(
    name = "ios_x86_64_sim_runner_with_hooks",
    device_type = "iPhone Xs",
    pre_action = ":pre_action",
    post_action = ":post_action",
)

genrule(
  name = "post_action_soft_fail_gen",
  executable = True,
  outs = ["post_action_soft_fail.bash"],
  cmd = """
echo 'echo "POST-ACTION: Soft failing." && exit 0' > \$@
""",
)

sh_binary(
  name = "post_action_soft_fail",
  srcs = [":post_action_soft_fail_gen"],
)

ios_test_runner(
    name = "ios_x86_64_sim_runner_with_soft_fail",
    device_type = "iPhone Xs",
    post_action = ":post_action_soft_fail",
    post_action_determines_exit_code = True,
)

EOF
}

function create_test_host_app() {
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
    minimum_os_version = "${MIN_OS_IOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    deps = [":app_lib"],
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

  cat > ios/pass_unit_test.swift <<EOF
import XCTest

class PassingUnitTest : XCTestCase {
  func testPass() throws {
    let result = 1 + 1;
    XCTAssertEqual(result, 2, "should pass");
  }

  func testSrcdirSet() {
    XCTAssertNotNil(ProcessInfo.processInfo.environment["TEST_SRCDIR"])
  }
}
EOF

  cat > ios/fail_unit_test.m <<EOF
#import <XCTest/XCTest.h>

@interface FailingUnitTest : XCTestCase

@end

@implementation FailingUnitTest

- (void)testFail {
  XCTAssertEqual(0, 1, @"should fail");
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

  cat > ios/PassUnitSwiftTest-Info.plist <<EOF
<plist version="1.0">
<dict>
        <key>CFBundleExecutable</key>
        <string>PassUnitSwiftTest</string>
</dict>
</plist>
EOF

  cat > ios/FailUnitTest-Info.plist <<EOF
<plist version="1.0">
<dict>
        <key>CFBundleExecutable</key>
        <string>FailingUnitTest</string>
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
    runner = ":ios_x86_64_sim_runner",
)

ios_unit_test(
    name = "PassingWithHost",
    infoplists = ["PassUnitTest-Info.plist"],
    deps = [":pass_unit_test_lib"],
    minimum_os_version = "${MIN_OS_IOS}",
    test_host = ":app",
    env = test_env,
    runner = ":ios_x86_64_sim_runner",
)

ios_unit_test(
    name = "PassingUnitTestWithHooks",
    infoplists = ["PassUnitTest-Info.plist"],
    deps = [":pass_unit_test_lib"],
    minimum_os_version = "${MIN_OS_IOS}",
    test_host = ":app",
    env = test_env,
    runner = ":ios_x86_64_sim_runner_with_hooks",
)

swift_library(
    name = "pass_unit_swift_test_lib",
    testonly = True,
    srcs = ["pass_unit_test.swift"],
)

ios_unit_test(
    name = "PassingUnitSwiftTest",
    infoplists = ["PassUnitSwiftTest-Info.plist"],
    deps = [":pass_unit_swift_test_lib"],
    minimum_os_version = "${MIN_OS_IOS}",
    test_host = ":app",
    runner = ":ios_x86_64_sim_runner",
)

objc_library(
    name = "fail_unit_test_lib",
    srcs = ["fail_unit_test.m"],
)

ios_unit_test(
    name = 'FailingUnitTest',
    infoplists = ["FailUnitTest-Info.plist"],
    deps = [":fail_unit_test_lib"],
    minimum_os_version = "${MIN_OS_IOS}",
    runner = ":ios_x86_64_sim_runner",
)

ios_unit_test(
    name = 'SoftFailingUnitTest',
    infoplists = ["FailUnitTest-Info.plist"],
    deps = [":fail_unit_test_lib"],
    minimum_os_version = "${MIN_OS_IOS}",
    runner = ":ios_x86_64_sim_runner_with_soft_fail",
)

ios_unit_test(
    name = 'FailingWithHost',
    infoplists = ["FailUnitTest-Info.plist"],
    deps = [":fail_unit_test_lib"],
    minimum_os_version = "${MIN_OS_IOS}",
    test_host = ":app",
    runner = ":ios_x86_64_sim_runner",
)
EOF
}

function create_ios_unit_env_test() {
  if [[ ! -f ios/BUILD ]]; then
    fail "create_sim_runners must be called first."
  fi

  cat > ios/env_unit_test.m <<EOF
#import <XCTest/XCTest.h>
#include <assert.h>
#include <stdlib.h>

@interface EnvUnitTest : XCTestCase

@end

@implementation EnvUnitTest

- (void)testEnv {
  NSString *var_value = [[[NSProcessInfo processInfo] environment] objectForKey:@"$1"];
  XCTAssertEqualObjects(var_value, @"$2", @"env $1 should be %@, instead is %@", @"$2", var_value);
}

@end
EOF

  cat >ios/EnvUnitTest-Info.plist <<EOF
<plist version="1.0">
<dict>
        <key>CFBundleExecutable</key>
        <string>EnvUnitTest</string>
</dict>
</plist>
EOF

  cat >> ios/BUILD <<EOF
objc_library(
    name = "env_unit_test_lib",
    srcs = ["env_unit_test.m"],
)

ios_unit_test(
    name = 'EnvUnitTest',
    infoplists = ["EnvUnitTest-Info.plist"],
    deps = [":env_unit_test_lib"],
    minimum_os_version = "${MIN_OS_IOS}",
    runner = ":ios_x86_64_sim_runner",
)

ios_unit_test(
    name = 'EnvWithHost',
    infoplists = ["EnvUnitTest-Info.plist"],
    deps = [":env_unit_test_lib"],
    minimum_os_version = "${MIN_OS_IOS}",
    test_host = ":app",
    runner = ":ios_x86_64_sim_runner",
)
EOF
}

function create_ios_unit_env_inherit_test() {
  if [[ ! -f ios/BUILD ]]; then
    fail "create_sim_runners must be called first."
  fi

  cat > ios/env_unit_test.m <<EOF
#import <XCTest/XCTest.h>
#include <assert.h>
#include <stdlib.h>

@interface EnvInheritUnitTest : XCTestCase

@end

@implementation EnvInheritUnitTest

- (void)testEnv {
  NSString *var_value = [[[NSProcessInfo processInfo] environment] objectForKey:@"$1"];
  XCTAssertEqualObjects(var_value, @"$2", @"env $1 should be %@, instead is %@", @"$2", var_value);
}

@end
EOF

  cat >ios/EnvInheritUnitTest-Info.plist <<EOF
<plist version="1.0">
<dict>
        <key>CFBundleExecutable</key>
        <string>EnvInheritUnitTest</string>
</dict>
</plist>
EOF

  cat >> ios/BUILD <<EOF
objc_library(
    name = "env_unit_test_lib",
    srcs = ["env_unit_test.m"],
)

ios_unit_test(
    name = 'EnvInheritUnitTest',
    infoplists = ["EnvInheritUnitTest-Info.plist"],
    deps = [":env_unit_test_lib"],
    env_inherit = ["$1"],
    minimum_os_version = "${MIN_OS_IOS}",
    runner = ":ios_x86_64_sim_runner",
)

ios_unit_test(
    name = 'EnvInheritWithHost',
    infoplists = ["EnvInheritUnitTest-Info.plist"],
    deps = [":env_unit_test_lib"],
    env_inherit = ["$1"],
    minimum_os_version = "${MIN_OS_IOS}",
    test_host = ":app",
    runner = ":ios_x86_64_sim_runner",
)
EOF
}

function create_ios_unit_make_var_test() {
  if [[ ! -f ios/BUILD ]]; then
    fail "create_sim_runners must be called first."
  fi

  cat > ios/make_var_unit_test.m <<EOF
#import <XCTest/XCTest.h>
#include <assert.h>
#include <stdlib.h>

@interface MakeVarUnitTest : XCTestCase

@end

@implementation MakeVarUnitTest

- (void)testMakeVar {
  XCTAssertEqualObjects([NSProcessInfo processInfo].environment[@"MY_MAKE_VAR"], @"$1", @"should pass");
}

@end
EOF

  cat >ios/MakeVarUnitTest-Info.plist <<EOF
<plist version="1.0">
<dict>
        <key>CFBundleExecutable</key>
        <string>MakeVarUnitTest</string>
</dict>
</plist>
EOF

  cat >> ios/BUILD <<EOF
load("@bazel_skylib//rules:common_settings.bzl", "string_flag")

string_flag(
    name = "my_make_var",
    build_setting_default = "",
    make_variable = "MY_MAKE_VAR",
)

objc_library(
    name = "make_var_unit_test_lib",
    srcs = ["make_var_unit_test.m"],
)

ios_unit_test(
    name = 'MakeVarUnitTest',
    infoplists = ["MakeVarUnitTest-Info.plist"],
    deps = [":make_var_unit_test_lib"],
    minimum_os_version = "${MIN_OS_IOS}",
    runner = ":ios_x86_64_sim_runner",
    env = {
        "MY_MAKE_VAR": "\$(MY_MAKE_VAR)",
    },
    toolchains = [":my_make_var"],
)

ios_unit_test(
    name = 'MakeVarWithHost',
    infoplists = ["MakeVarUnitTest-Info.plist"],
    deps = [":make_var_unit_test_lib"],
    minimum_os_version = "${MIN_OS_IOS}",
    test_host = ":app",
    runner = ":ios_x86_64_sim_runner",
    env = {
        "MY_MAKE_VAR": "\$(MY_MAKE_VAR)",
    },
    toolchains = [":my_make_var"],
)
EOF
}

function create_ios_unit_argtest() {
  if [[ ! -f ios/BUILD ]]; then
    fail "create_sim_runners must be called first."
  fi

  cat > ios/arg_unit_test.m <<EOF
#import <XCTest/XCTest.h>
#include <assert.h>
#include <stdlib.h>

@interface ArgUnitTest : XCTestCase

@end

@implementation ArgUnitTest

- (void)testArg {
  XCTAssertTrue([[NSProcessInfo processInfo].arguments containsObject: @"--flag"], @"should pass");
  XCTAssertTrue([[NSProcessInfo processInfo].arguments containsObject: @"First"], @"should pass");
  XCTAssertTrue([[NSProcessInfo processInfo].arguments containsObject: @"Second"], @"should pass");
$([ $# != 0 ] && printf "  XCTAssertTrue([[NSProcessInfo processInfo].arguments containsObject: @\"%s\"], @\"should pass\");\n" "$@")
}

@end
EOF

  cat >ios/ArgUnitTest-Info.plist <<EOF
<plist version="1.0">
<dict>
        <key>CFBundleExecutable</key>
        <string>ArgUnitTest</string>
</dict>
</plist>
EOF

  cat >> ios/BUILD <<EOF
objc_library(
    name = "arg_unit_test_lib",
    srcs = ["arg_unit_test.m"],
)

ios_unit_test(
    name = 'ArgUnitTest',
    infoplists = ["ArgUnitTest-Info.plist"],
    deps = [":arg_unit_test_lib"],
    minimum_os_version = "${MIN_OS_IOS}",
    runner = ":ios_x86_64_sim_runner",
    args = [
      "--command_line_args=--flag",
      "--command_line_args=First,Second",
    ],
)
EOF
}

function create_ios_unit_tests_test_filter() {
  if [[ ! -f ios/BUILD ]]; then
    fail "create_sim_runners must be called first."
  fi

  cat > ios/test_filter_unit_test.m <<EOF
#import <XCTest/XCTest.h>
#import <XCTest/XCUIApplication.h>

@interface TestFilterUnitTest : XCTestCase

@end

@implementation TestFilterUnitTest {
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

@end
EOF

  cat > ios/TestFilterUnitTest-Info.plist <<EOF
<plist version="1.0">
<dict>
        <key>CFBundleExecutable</key>
        <string>TestFilterUnitTest</string>
</dict>
</plist>
EOF

  cat >> ios/BUILD <<EOF
objc_library(
    name = "test_filter_unit_test_lib",
    srcs = ["test_filter_unit_test.m"],
)

ios_unit_test(
    name = "TestFilterUnitTest",
    infoplists = ["TestFilterUnitTest-Info.plist"],
    deps = [":test_filter_unit_test_lib"],
    minimum_os_version = "${MIN_OS_IOS}",
    test_host = ":app",
    runner = ":ios_x86_64_sim_runner",
    test_filter = "$1",
)
EOF
}

function create_ios_unit_main_thread_checker_tests() {
  cat > ios/main_thread_checker_violation.swift <<EOF
import XCTest

class MainThreadCheckerViolationTest : XCTestCase {
  func testTriggerMainThreadChecker() {
        let expectation = self.expectation(description: "Background operation")

        DispatchQueue.global().async {
            // This will trigger the Main Thread Checker because we are
            // trying to update the UI from a background thread.
            let label = UILabel()
            label.text = "Test"

            expectation.fulfill()
        }

        waitForExpectations(timeout: 5) { (error) in
            if let error = error {
                XCTFail("waitForExpectations errored: \(error)")
            }
        }
    }
}
EOF

  cat > ios/MainThreadCheckerViolationTest-Info.plist <<EOF
<plist version="1.0">
<dict>
        <key>CFBundleExecutable</key>
        <string>MainThreadCheckerViolationTest</string>
</dict>
</plist>
EOF

  cat >> ios/BUILD <<EOF
swift_library(
    name = "main_thread_checker_violation_test_lib",
    testonly = True,
    srcs = ["main_thread_checker_violation.swift"],
)

ios_unit_test(
    name = "MainThreadCheckerViolationTest",
    infoplists = ["MainThreadCheckerViolationTest-Info.plist"],
    deps = [":main_thread_checker_violation_test_lib"],
    minimum_os_version = "${MIN_OS_IOS}",
    test_host = ":app",
    runner = ":ios_x86_64_sim_runner",
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

function test_ios_unit_test_with_host_pass() {
  create_sim_runners
  create_test_host_app
  create_ios_unit_tests
  do_ios_test //ios:PassingWithHost || fail "should pass"

  expect_log "Test Suite 'PassingUnitTest' passed"
  expect_log "Test Suite 'PassingWithHost.xctest' passed"
  expect_log "Executed 4 tests, with 0 failures"
}

function test_ios_unit_test_with_hooks_pass() {
  create_sim_runners
  create_ios_unit_tests
  do_ios_test //ios:PassingUnitTestWithHooks || fail "should pass"

  expect_log "PRE-ACTION: TEST_TARGET=//ios:PassingUnitTestWithHooks"
  expect_log "Test Suite 'PassingUnitTest' passed"
  expect_log "Test Suite 'PassingUnitTestWithHooks.xctest' passed"
  expect_log "Executed 4 tests, with 0 failures"
  expect_log "POST-ACTION: TEST_TARGET=//ios:PassingUnitTestWithHooks"
}

function test_ios_unit_swift_test_pass() {
  create_sim_runners
  create_test_host_app
  create_ios_unit_tests
  do_ios_test //ios:PassingUnitSwiftTest || fail "should pass"

  expect_log "Test Suite 'PassingUnitTest' passed"
  expect_log "Test Suite 'PassingUnitSwiftTest.xctest' passed"
  expect_log "Executed 2 tests, with 0 failures"
}

function test_ios_unit_test_fail() {
  create_sim_runners
  create_ios_unit_tests
  ! do_ios_test //ios:FailingUnitTest || fail "should fail"

  expect_log "Test Suite 'FailingUnitTest' failed"
  expect_log "Test Suite 'FailingUnitTest.xctest' failed"
  expect_log "Executed 1 test, with 1 failure"
}

function test_ios_unit_test_soft_fail() {
  create_sim_runners
  create_ios_unit_tests
  do_ios_test //ios:SoftFailingUnitTest || fail "should pass"

  expect_log "Test Suite 'FailingUnitTest' failed"
  expect_log "Test Suite 'SoftFailingUnitTest.xctest' failed"
  expect_log "Executed 1 test, with 1 failure"
  expect_log "POST-ACTION: Soft failing."
}

function test_ios_unit_test_with_host_fail() {
  create_sim_runners
  create_test_host_app
  create_ios_unit_tests
  ! do_ios_test //ios:FailingWithHost || fail "should fail"

  expect_log "Test Suite 'FailingUnitTest' failed"
  expect_log "Test Suite 'FailingWithHost.xctest' failed"
  expect_log "Executed 1 test, with 1 failure"
}

function test_ios_unit_test_with_filter() {
  create_sim_runners
  create_ios_unit_tests
  do_ios_test --test_filter=PassingUnitTest/testPass2 //ios:PassingUnitTest || fail "should pass"

  expect_log "Test Case '-\[PassingUnitTest testPass2\]' passed"
  expect_log "Test Suite 'PassingUnitTest' passed"
  expect_log "Test Suite 'PassingUnitTest.xctest' passed"
  expect_log "Executed 1 test, with 0 failures"
}

function test_ios_unit_test_with_multi_filter() {
  create_sim_runners
  create_ios_unit_tests
  do_ios_test --test_filter=PassingUnitTest/testPass2,PassingUnitTest/testPass3 //ios:PassingUnitTest || fail "should pass"

  expect_log "Test Case '-\[PassingUnitTest testPass2\]' passed"
  expect_log "Test Case '-\[PassingUnitTest testPass3\]' passed"
  expect_log "Test Suite 'PassingUnitTest' passed"
  expect_log "Test Suite 'PassingUnitTest.xctest' passed"
  expect_log "Executed 2 tests, with 0 failures"
}

function test_ios_unit_test_with_host_with_filter() {
  create_sim_runners
  create_test_host_app
  create_ios_unit_tests
  do_ios_test --test_filter=PassingUnitTest/testPass2 //ios:PassingWithHost || fail "should pass"

  expect_log "Test Case '-\[PassingUnitTest testPass2\]' passed"
  expect_log "Test Suite 'PassingUnitTest' passed"
  expect_log "Test Suite 'PassingWithHost.xctest' passed"
  expect_log "Executed 1 test, with 0 failures"
}

function test_ios_unit_test_with_host_and_skip_filter() {
  create_sim_runners
  create_test_host_app
  create_ios_unit_tests
  do_ios_test --test_filter=-PassingUnitTest/testPass //ios:PassingWithHost || fail "should pass"

  expect_not_log "Test Case '-\[PassingUnitTest testPass\]' passed"
  expect_log "Test Case '-\[PassingUnitTest testPass2\]' passed"
  expect_log "Test Case '-\[PassingUnitTest testPass3\]' passed"
  expect_log "Test Suite 'PassingUnitTest' passed"
  expect_log "Test Suite 'PassingWithHost.xctest' passed"
  expect_log "Executed 3 tests, with 0 failures"
}

function test_ios_unit_test_with_host_and_multi_skip_filter() {
  create_sim_runners
  create_test_host_app
  create_ios_unit_tests
  do_ios_test --test_filter=-PassingUnitTest/testPass,-PassingUnitTest/testPass2 //ios:PassingWithHost || fail "should pass"

  expect_not_log "Test Case '-\[PassingUnitTest testPass\]' passed"
  expect_not_log "Test Case '-\[PassingUnitTest testPass2\]' passed"
  expect_log "Test Case '-\[PassingUnitTest testPass3\]' passed"
  expect_log "Test Suite 'PassingUnitTest' passed"
  expect_log "Test Suite 'PassingWithHost.xctest' passed"
  expect_log "Executed 2 tests, with 0 failures"
}

function test_ios_unit_test_with_host_and_skip_and_only_filters() {
  create_sim_runners
  create_test_host_app
  create_ios_unit_tests
  do_ios_test --test_filter=PassingUnitTest,-PassingUnitTest/testPass2 //ios:PassingWithHost || fail "should pass"

  expect_log "Test Case '-\[PassingUnitTest testPass\]' passed"
  expect_not_log "Test Case '-\[PassingUnitTest testPass2\]' passed"
  expect_log "Test Case '-\[PassingUnitTest testPass3\]' passed"
  expect_log "Test Suite 'PassingUnitTest' passed"
  expect_log "Test Suite 'PassingWithHost.xctest' passed"
  expect_log "Executed 3 tests, with 0 failures"
}

function test_ios_unit_test_with_env() {
  create_sim_runners
  create_ios_unit_env_test ENV_KEY1 ENV_VALUE2
  do_ios_test --test_env=ENV_KEY1=ENV_VALUE2 //ios:EnvUnitTest || fail "should pass"

  expect_log "Test Suite 'EnvUnitTest' passed"
}

function test_ios_unit_test_with_host_with_env() {
  create_sim_runners
  create_test_host_app
  create_ios_unit_env_test ENV_KEY1 ENV_VALUE2
  do_ios_test --test_env=ENV_KEY1=ENV_VALUE2 //ios:EnvWithHost || fail "should pass"

  expect_log "Test Suite 'EnvUnitTest' passed"
}

function test_ios_unit_test_with_env_inherit() {
  create_sim_runners
  create_ios_unit_env_inherit_test ENV_INHERIT_KEY1 ENV_INHERIT_VALUE2
  ENV_INHERIT_KEY1=ENV_INHERIT_VALUE2 do_ios_test //ios:EnvInheritUnitTest || fail "should pass"

  expect_log "Test Suite 'EnvInheritUnitTest' passed"
}

function test_ios_unit_test_with_host_env_inherit() {
  create_sim_runners
  create_test_host_app
  create_ios_unit_env_inherit_test ENV_INHERIT_KEY1 ENV_INHERIT_VALUE2
  ENV_INHERIT_KEY1=ENV_INHERIT_VALUE2 do_ios_test //ios:EnvInheritWithHost || fail "should pass"

  expect_log "Test Suite 'EnvInheritUnitTest' passed"
}

function test_ios_unit_test_with_make_var_empty() {
  create_sim_runners
  create_ios_unit_make_var_test ""
  do_ios_test //ios:MakeVarUnitTest || fail "should pass"

  expect_log "Test Suite 'MakeVarUnitTest' passed"
}

function test_ios_unit_test_with_make_var_set() {
  create_sim_runners
  create_ios_unit_make_var_test MAKE_VAR_VALUE1
  do_ios_test --//ios:my_make_var=MAKE_VAR_VALUE1 //ios:MakeVarUnitTest || fail "should pass"

  expect_log "Test Suite 'MakeVarUnitTest' passed"
}

function test_ios_unit_test_with_host_with_make_var_empty() {
  create_sim_runners
  create_test_host_app
  create_ios_unit_make_var_test ""
  do_ios_test //ios:MakeVarWithHost || fail "should pass"

  expect_log "Test Suite 'MakeVarUnitTest' passed"
}

function test_ios_unit_test_with_host_with_make_var_set() {
  create_sim_runners
  create_test_host_app
  create_ios_unit_make_var_test MAKE_VAR_VALUE1
  do_ios_test --//ios:my_make_var=MAKE_VAR_VALUE1 //ios:MakeVarWithHost || fail "should pass"

  expect_log "Test Suite 'MakeVarUnitTest' passed"
}

function test_ios_unit_simulator_id() {
  create_sim_runners
  create_ios_unit_tests
  readonly simulator_id=$(xcrun simctl create custom-sim "iPhone X" 2> >(grep -v "No runtime specified" || true))
  do_ios_test //ios:PassingUnitTest \
    --test_arg="--destination=platform=ios_simulator,id=$simulator_id" \
    || fail "should pass"

  xcrun simctl delete "$simulator_id"

  # Custom logs from xctestrunner
  expect_not_log "Creating a new simulator"
  expect_not_log "Created new simulator"
  expect_log "Executed 4 tests, with 0 failures"
}

function test_ios_unit_test_dot_separated_command_line_args() {
  create_sim_runners
  create_ios_unit_argtest arg1 arg2 arg3
  do_ios_test //ios:ArgUnitTest \
    --test_arg="--command_line_args=arg1,arg2,arg3" || fail "should pass"

  expect_log "Test Suite 'ArgUnitTest' passed"
}

function test_ios_unit_test_multiple_command_line_args() {
  create_sim_runners
  create_ios_unit_argtest arg1 arg2
  do_ios_test //ios:ArgUnitTest \
    --test_arg="--command_line_args=arg1" \
    --test_arg="--command_line_args=arg2" || fail "should pass"

  expect_log "Test Suite 'ArgUnitTest' passed"
}

function test_ios_unit_other_arg() {
  create_sim_runners
  create_ios_unit_tests
  ! do_ios_test //ios:PassingUnitTest --test_arg=invalid_arg || fail "should fail"

  # Error comes from xctestrunner
  expect_log "error: unrecognized arguments: invalid_arg"
}

function test_ios_unit_test_with_multi_equal_env() {
  create_sim_runners
  create_ios_unit_envtest ENV_KEY1 ENV_VALUE2=ENV_VALUE3
  do_ios_test --test_env=ENV_KEY1=ENV_VALUE2=ENV_VALUE3 //ios:EnvUnitTest || fail "should pass"

  expect_log "Test Suite 'EnvUnitTest' passed"
}

function test_ios_unit_test_pass_asan() {
  create_sim_runners
  create_ios_unit_tests
  do_ios_test --features=asan //ios:PassingUnitTest || fail "should pass"

  expect_log "Test Suite 'PassingUnitTest' passed"
  expect_log "Test Suite 'PassingUnitTest.xctest' passed"
  expect_log "Executed 4 tests, with 0 failures"
}

function test_with_test_filter_build_attribute() {
  create_sim_runners
  create_test_host_app
  create_ios_unit_tests_test_filter TestFilterUnitTest/testPass2
  do_ios_test //ios:TestFilterUnitTest || fail "should pass"

  expect_log "Test Case '-\[TestFilterUnitTest testPass2\]' passed"
  expect_log "Test Suite 'TestFilterUnitTest' passed"
  expect_log "Test Suite 'TestFilterUnitTest.xctest' passed"
  expect_log "Executed 1 test, with 0 failures"
}

function test_ios_unit_test_with_multi_test_filter_build_attribute() {
  create_sim_runners
  create_test_host_app
  create_ios_unit_tests_test_filter TestFilterUnitTest/testPass2,TestFilterUnitTest/testPass3
  do_ios_test //ios:TestFilterUnitTest || fail "should pass"

  expect_log "Test Case '-\[TestFilterUnitTest testPass2\]' passed"
  expect_log "Test Case '-\[TestFilterUnitTest testPass3\]' passed"
  expect_log "Test Suite 'TestFilterUnitTest' passed"
  expect_log "Test Suite 'TestFilterUnitTest.xctest' passed"
  expect_log "Executed 2 tests, with 0 failures"
}

function test_ios_unit_test_with_skip_test_filter_build_attribute() {
  create_sim_runners
  create_test_host_app
  create_ios_unit_tests_test_filter -TestFilterUnitTest/testPass
  do_ios_test //ios:TestFilterUnitTest || fail "should pass"

  expect_not_log "Test Case '-\[TestFilterUnitTest testPass\]' passed"
  expect_log "Test Case '-\[TestFilterUnitTest testPass2\]' passed"
  expect_log "Test Case '-\[TestFilterUnitTest testPass3\]' passed"
  expect_log "Test Suite 'TestFilterUnitTest' passed"
  expect_log "Test Suite 'TestFilterUnitTest.xctest' passed"
  expect_log "Executed 2 tests, with 0 failures"
}

function test_ios_unit_test_multi_skip_test_filter_build_attribute() {
  create_sim_runners
  create_test_host_app
  create_ios_unit_tests_test_filter -TestFilterUnitTest/testPass,-TestFilterUnitTest/testPass2
  do_ios_test //ios:TestFilterUnitTest || fail "should pass"

  expect_not_log "Test Case '-\[TestFilterUnitTest testPass\]' passed"
  expect_not_log "Test Case '-\[TestFilterUnitTest testPass2\]' passed"
  expect_log "Test Case '-\[TestFilterUnitTest testPass3\]' passed"
  expect_log "Test Suite 'TestFilterUnitTest' passed"
  expect_log "Test Suite 'TestFilterUnitTest.xctest' passed"
  expect_log "Executed 1 test, with 0 failures"
}

function test_ios_unit_test_with_skip_and_only_filters_build_attribute() {
  create_sim_runners
  create_test_host_app
  create_ios_unit_tests_test_filter TestFilterUnitTest,-TestFilterUnitTest/testPass2
  do_ios_test //ios:TestFilterUnitTest || fail "should pass"

  expect_log "Test Case '-\[TestFilterUnitTest testPass\]' passed"
  expect_not_log "Test Case '-\[TestFilterUnitTest testPass2\]' passed"
  expect_log "Test Case '-\[TestFilterUnitTest testPass3\]' passed"
  expect_log "Test Suite 'TestFilterUnitTest' passed"
  expect_log "Test Suite 'TestFilterUnitTest.xctest' passed"
  expect_log "Executed 2 tests, with 0 failures"
}

function test_ios_unit_test_with_build_attribute_and_test_env_filters() {
  create_sim_runners
  create_test_host_app
  create_ios_unit_tests_test_filter TestFilterUnitTest/testPass2
  do_ios_test --test_filter=TestFilterUnitTest/testPass3 //ios:TestFilterUnitTest || fail "should pass"

  expect_not_log "Test Case '-\[PassingUnitTest testPass\]' passed"
  expect_log "Test Case '-\[TestFilterUnitTest testPass2\]' passed"
  expect_log "Test Case '-\[TestFilterUnitTest testPass3\]' passed"
  expect_log "Test Suite 'TestFilterUnitTest' passed"
  expect_log "Test Suite 'TestFilterUnitTest.xctest' passed"
  expect_log "Executed 2 tests, with 0 failures"
}

function test_ios_unit_test_pass_main_thread_checker_without_crash_on_report_pass() {
  create_sim_runners
  create_test_host_app
  create_ios_unit_main_thread_checker_tests
  do_ios_test --features=apple.include_main_thread_checker //ios:MainThreadCheckerViolationTest || fail "should pass"

  expect_log "Test Suite 'MainThreadCheckerViolationTest' passed"
  expect_log "Test Suite 'MainThreadCheckerViolationTest.xctest' passed"
  expect_log "Executed 1 test, with 0 failures"
}

function test_ios_unit_test_pass_main_thread_checker_with_crash_on_report_fail() {
  create_sim_runners
  create_test_host_app
  create_ios_unit_main_thread_checker_tests
  ! do_ios_test --features=apple.include_main_thread_checker --features=apple.fail_on_main_thread_checker //ios:MainThreadCheckerViolationTest || fail "should fail"

  expect_log "Test Suite 'MainThreadCheckerViolationTest' failed"
  expect_log "Test Suite 'MainThreadCheckerViolationTest.xctest' failed"
  expect_log "Executed 1 test, with 1 failure"
}

run_suite "ios_unit_test with iOS test runner bundling tests"
