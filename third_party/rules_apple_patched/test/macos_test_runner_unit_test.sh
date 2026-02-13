#!/bin/bash

# Copyright 2025 The Bazel Authors. All rights reserved.
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

# Integration tests for macOS test runner.

function set_up() {
  mkdir -p macos
}

function tear_down() {
  rm -rf macos
}

function create_runners() {
  cat > macos/BUILD <<EOF
load(
    "@build_bazel_rules_apple//apple:macos.bzl",
    "macos_unit_test"
)
load("@build_bazel_rules_swift//swift:swift.bzl",
     "swift_library"
)
load(
    "@build_bazel_rules_apple//apple/testing/default_runner:macos_test_runner.bzl",
    "macos_test_runner"
)
load("@rules_shell//shell:sh_binary.bzl", "sh_binary")

macos_test_runner(
    name = "macos_runner",
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

macos_test_runner(
    name = "macos_runner_with_hooks",
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

macos_test_runner(
    name = "macos_runner_with_soft_fail",
    post_action = ":post_action_soft_fail",
    post_action_determines_exit_code = True,
)
EOF
}

function create_macos_unit_tests() {
  if [[ ! -f macos/BUILD ]]; then
    fail "create_runners must be called first."
  fi

  cat > macos/small_unit_test_1.m <<EOF
#import <XCTest/XCTest.h>
#import <XCTest/XCUIApplication.h>

@interface SmallUnitTest1 : XCTestCase

@end

@implementation SmallUnitTest1
- (void)testPass {
  XCTAssertEqual(1, 1, @"should pass");
}
@end
EOF

  cat > macos/small_unit_test_2.m <<EOF
#import <XCTest/XCTest.h>
#import <XCTest/XCUIApplication.h>

@interface SmallUnitTest2 : XCTestCase

@end

@implementation SmallUnitTest2
- (void)testPass {
  XCTAssertEqual(1, 1, @"should pass");
}
@end
EOF

  cat > macos/pass_unit_test.m <<EOF
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

  cat > macos/pass_unit_test.swift <<EOF
import XCTest

class PassingUnitTest : XCTestCase {
  func testPass() throws {
    let result = 1 + 1;
    XCTAssertEqual(result, 2, "should pass");
  }

  func testSrcdirSet() {
    XCTAssertNotNil(ProcessInfo.processInfo.environment["TEST_SRCDIR"])
  }

  func testUndeclaredOutputsSet() {
    XCTAssertNotNil(ProcessInfo.processInfo.environment["TEST_UNDECLARED_OUTPUTS_DIR"])
  }
}
EOF

  cat > macos/fail_unit_test.m <<EOF
#import <XCTest/XCTest.h>

@interface FailingUnitTest : XCTestCase

@end

@implementation FailingUnitTest

- (void)testFail {
  XCTAssertEqual(0, 1, @"should fail");
}

@end
EOF

  cat > macos/SmallUnitTest-Info.plist <<EOF
<plist version="1.0">
<dict>
        <key>CFBundleExecutable</key>
        <string>SmallUnitTest</string>
</dict>
</plist>
EOF

  cat > macos/PassUnitTest-Info.plist <<EOF
<plist version="1.0">
<dict>
        <key>CFBundleExecutable</key>
        <string>PassingUnitTest</string>
</dict>
</plist>
EOF

  cat > macos/PassUnitSwiftTest-Info.plist <<EOF
<plist version="1.0">
<dict>
        <key>CFBundleExecutable</key>
        <string>PassUnitSwiftTest</string>
</dict>
</plist>
EOF

  cat > macos/FailUnitTest-Info.plist <<EOF
<plist version="1.0">
<dict>
        <key>CFBundleExecutable</key>
        <string>FailingUnitTest</string>
</dict>
</plist>
EOF

  cat >> macos/BUILD <<EOF
test_env = {
    "SomeVariable1": "Its My First Variable",
    "SomeVariable2": "Its My Second Variable",
    "REFERENCE_DIR": "/Project/My Tests/ReferenceImages",
    "IMAGE_DIR": "/Project/My Tests/Images"
}

objc_library(
    name = "small_unit_test_lib",
    srcs = ["small_unit_test_1.m", "small_unit_test_2.m"],
)

macos_unit_test(
    name = "SmallUnitTest",
    infoplists = ["SmallUnitTest-Info.plist"],
    deps = [":small_unit_test_lib"],
    minimum_os_version = "${MIN_OS_MACOS}",
    env = test_env,
    runner = ":macos_runner",
)

objc_library(
    name = "pass_unit_test_lib",
    srcs = ["pass_unit_test.m"],
)

macos_unit_test(
    name = "PassingUnitTest",
    infoplists = ["PassUnitTest-Info.plist"],
    deps = [":pass_unit_test_lib"],
    minimum_os_version = "${MIN_OS_MACOS}",
    env = test_env,
    runner = ":macos_runner",
)

macos_unit_test(
    name = "PassingUnitTestWithHooks",
    infoplists = ["PassUnitTest-Info.plist"],
    deps = [":pass_unit_test_lib"],
    minimum_os_version = "${MIN_OS_MACOS}",
    env = test_env,
    runner = ":macos_runner_with_hooks",
)

objc_library(
    name = "fail_unit_test_lib",
    srcs = ["fail_unit_test.m"],
)

macos_unit_test(
    name = 'FailingUnitTest',
    infoplists = ["FailUnitTest-Info.plist"],
    deps = [":fail_unit_test_lib"],
    minimum_os_version = "${MIN_OS_MACOS}",
    runner = ":macos_runner",
)

macos_unit_test(
    name = 'SoftFailingUnitTest',
    infoplists = ["FailUnitTest-Info.plist"],
    deps = [":fail_unit_test_lib"],
    minimum_os_version = "${MIN_OS_MACOS}",
    runner = ":macos_runner_with_soft_fail",
)
EOF
}

function create_macos_unit_envtest() {
  if [[ ! -f macos/BUILD ]]; then
    fail "create_runners must be called first."
  fi

  cat > macos/env_unit_test.m <<EOF
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

  cat >macos/EnvUnitTest-Info.plist <<EOF
<plist version="1.0">
<dict>
        <key>CFBundleExecutable</key>
        <string>EnvUnitTest</string>
</dict>
</plist>
EOF

  cat >> macos/BUILD <<EOF
objc_library(
    name = "env_unit_test_lib",
    srcs = ["env_unit_test.m"],
)

macos_unit_test(
    name = 'EnvUnitTest',
    infoplists = ["EnvUnitTest-Info.plist"],
    deps = [":env_unit_test_lib"],
    minimum_os_version = "${MIN_OS_MACOS}",
    runner = ":macos_runner",
)
EOF
}

function create_macos_unit_make_var_test() {
  if [[ ! -f macos/BUILD ]]; then
    fail "create_runners must be called first."
  fi

  cat > macos/make_var_unit_test.m <<EOF
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

  cat >macos/MakeVarUnitTest-Info.plist <<EOF
<plist version="1.0">
<dict>
        <key>CFBundleExecutable</key>
        <string>MakeVarUnitTest</string>
</dict>
</plist>
EOF

  cat >> macos/BUILD <<EOF
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

macos_unit_test(
    name = 'MakeVarUnitTest',
    infoplists = ["MakeVarUnitTest-Info.plist"],
    deps = [":make_var_unit_test_lib"],
    minimum_os_version = "${MIN_OS_MACOS}",
    runner = ":macos_runner",
    env = {
        "MY_MAKE_VAR": "\$(MY_MAKE_VAR)",
    },
    toolchains = [":my_make_var"],
)
EOF
}

function create_macos_unit_argtest() {
  if [[ ! -f macos/BUILD ]]; then
    fail "create_runners must be called first."
  fi

  cat > macos/arg_unit_test.m <<EOF
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

  cat >macos/ArgUnitTest-Info.plist <<EOF
<plist version="1.0">
<dict>
        <key>CFBundleExecutable</key>
        <string>ArgUnitTest</string>
</dict>
</plist>
EOF

  cat >> macos/BUILD <<EOF
objc_library(
    name = "arg_unit_test_lib",
    srcs = ["arg_unit_test.m"],
)

macos_unit_test(
    name = 'ArgUnitTest',
    infoplists = ["ArgUnitTest-Info.plist"],
    deps = [":arg_unit_test_lib"],
    minimum_os_version = "${MIN_OS_MACOS}",
    runner = ":macos_runner",
    args = [
      "--command_line_args=--flag",
      "--command_line_args=First,Second",
    ],
)
EOF
}

function do_macos_test() {
  do_test macos "--test_output=all" "--spawn_strategy=local" "$@"
}

function test_macos_unit_test_small_pass() {
  create_runners
  create_macos_unit_tests
  do_macos_test //macos:SmallUnitTest || fail "should pass"

  expect_log "Test Suite 'SmallUnitTest1' passed"
  expect_log "Test Suite 'SmallUnitTest2' passed"
  expect_log "Test Suite 'SmallUnitTest.xctest' passed"
  expect_log "Executed 2 tests, with 0 failures"
}

# Test bundle has tests with one test class with all tests filtered.
function test_macos_unit_test_small_empty_test_class_filter_pass() {
  create_runners
  create_macos_unit_tests
  do_macos_test --test_filter="-SmallUnitTest1/testPass" //macos:SmallUnitTest || fail "should pass"

  expect_log "Test Suite 'SmallUnitTest1' passed"
  expect_log "Test Suite 'SmallUnitTest2' passed"
  expect_log "Test Suite 'SmallUnitTest.xctest' passed"
  expect_log "Executed 1 test, with 0 failures"
}

# Test bundle has tests but filter excludes all of them.
function test_macos_unit_test_small_empty_fail() {
  create_runners
  create_macos_unit_tests

  ! do_macos_test --test_filter="BadFilter" //macos:SmallUnitTest || fail "should fail"

  expect_log "Test Suite 'SmallUnitTest.xctest' passed"
  expect_log "Executed 0 tests, with 0 failures"
}

function test_macos_unit_test_pass() {
  create_runners
  create_macos_unit_tests
  do_macos_test //macos:PassingUnitTest || fail "should pass"

  expect_log "Test Suite 'PassingUnitTest' passed"
  expect_log "Test Suite 'PassingUnitTest.xctest' passed"
  expect_log "Executed 4 tests, with 0 failures"
}

function test_macos_unit_test_with_hooks_pass() {
  create_runners
  create_macos_unit_tests
  do_macos_test //macos:PassingUnitTestWithHooks || fail "should pass"

  expect_log "PRE-ACTION: TEST_TARGET=//macos:PassingUnitTestWithHooks"
  expect_log "Test Suite 'PassingUnitTest' passed"
  expect_log "Test Suite 'PassingUnitTestWithHooks.xctest' passed"
  expect_log "Executed 4 tests, with 0 failures"
  expect_log "POST-ACTION: TEST_TARGET=//macos:PassingUnitTestWithHooks"
}

function test_macos_unit_test_fail() {
  create_runners
  create_macos_unit_tests
  ! do_macos_test //macos:FailingUnitTest || fail "should fail"

  expect_log "Test Suite 'FailingUnitTest' failed"
  expect_log "Test Suite 'FailingUnitTest.xctest' failed"
  expect_log "Executed 1 test, with 1 failure"
}

function test_macos_unit_test_soft_fail() {
  create_runners
  create_macos_unit_tests
  do_macos_test //macos:SoftFailingUnitTest || fail "should pass"

  expect_log "Test Suite 'FailingUnitTest' failed"
  expect_log "Test Suite 'SoftFailingUnitTest.xctest' failed"
  expect_log "Executed 1 test, with 1 failure"
  expect_log "POST-ACTION: Soft failing."
}

function test_macos_unit_test_with_filter() {
  create_runners
  create_macos_unit_tests
  do_macos_test --test_filter=PassingUnitTest/testPass2 //macos:PassingUnitTest || fail "should pass"

  expect_log "Test Case '-\[PassingUnitTest testPass2\]' passed"
  expect_log "Test Suite 'PassingUnitTest' passed"
  expect_log "Test Suite 'PassingUnitTest.xctest' passed"
  expect_log "Executed 1 test, with 0 failures"
}

function test_macos_unit_test_with_multi_filter() {
  create_runners
  create_macos_unit_tests
  do_macos_test --test_filter=PassingUnitTest/testPass2,PassingUnitTest/testPass3 //macos:PassingUnitTest || fail "should pass"

  expect_log "Test Case '-\[PassingUnitTest testPass2\]' passed"
  expect_log "Test Case '-\[PassingUnitTest testPass3\]' passed"
  expect_log "Test Suite 'PassingUnitTest' passed"
  expect_log "Test Suite 'PassingUnitTest.xctest' passed"
  expect_log "Executed 2 tests, with 0 failures"
}

function test_macos_unit_test_with_env() {
  create_runners
  create_macos_unit_envtest ENV_KEY1 ENV_VALUE2
  do_macos_test --test_env=ENV_KEY1=ENV_VALUE2 //macos:EnvUnitTest || fail "should pass"

  expect_log "Test Suite 'EnvUnitTest' passed"
}

function test_macos_unit_test_with_make_var_empty() {
  create_runners
  create_macos_unit_make_var_test ""
  do_macos_test //macos:MakeVarUnitTest || fail "should pass"

  expect_log "Test Suite 'MakeVarUnitTest' passed"
}

function test_macos_unit_test_with_make_var_set() {
  create_runners
  create_macos_unit_make_var_test MAKE_VAR_VALUE1
  do_macos_test --//macos:my_make_var=MAKE_VAR_VALUE1 //macos:MakeVarUnitTest || fail "should pass"

  expect_log "Test Suite 'MakeVarUnitTest' passed"
}

function test_macos_unit_test_dot_separated_command_line_args() {
  create_runners
  create_macos_unit_argtest arg1 arg2 arg3
  do_macos_test //macos:ArgUnitTest \
    --test_arg="--command_line_args=arg1,arg2,arg3" || fail "should pass"

  expect_log "Test Suite 'ArgUnitTest' passed"
}

function test_macos_unit_test_multiple_command_line_args() {
  create_runners
  create_macos_unit_argtest arg1 arg2
  do_macos_test //macos:ArgUnitTest \
    --test_arg="--command_line_args=arg1" \
    --test_arg="--command_line_args=arg2" || fail "should pass"

  expect_log "Test Suite 'ArgUnitTest' passed"
}

function test_macos_unit_other_arg() {
  create_runners
  create_macos_unit_tests
  ! do_macos_test //macos:PassingUnitTest --test_arg=invalid_arg || fail "should fail"

  expect_log "error: Unsupported argument 'invalid_arg'"
}

function test_macos_unit_test_with_multi_equal_env() {
  create_runners
  create_macos_unit_envtest ENV_KEY1 ENV_VALUE2=ENV_VALUE3
  do_macos_test --test_env=ENV_KEY1=ENV_VALUE2=ENV_VALUE3 //macos:EnvUnitTest || fail "should pass"

  expect_log "Test Suite 'EnvUnitTest' passed"
}

function test_macos_unit_test_pass_asan() {
  create_runners
  create_macos_unit_tests
  do_macos_test --features=asan //macos:PassingUnitTest || fail "should pass"

  expect_log "Test Suite 'PassingUnitTest' passed"
  expect_log "Test Suite 'PassingUnitTest.xctest' passed"
  expect_log "Executed 4 tests, with 0 failures"
}

# Tests a test execution with parallel testing enabled is successful.
function test_macos_unit_test_parallel_testing_pass() {
  create_runners
  create_macos_unit_tests
  do_macos_test \
    --test_arg=--xcodebuild_args=-parallel-testing-enabled \
    --test_arg=--xcodebuild_args=YES \
    --test_arg=--xcodebuild_args=-parallel-testing-worker-count \
    --test_arg=--xcodebuild_args=1 \
    //macos:SmallUnitTest || fail "should pass"

  expect_log "Test case '-\[SmallUnitTest1 testPass\]' passed"
  expect_log "Test case '-\[SmallUnitTest2 testPass\]' passed"
  expect_log "//macos:SmallUnitTest\s\+PASSED"
  expect_log "Executed 1 out of 1 test: 1 test passes."
}

# Tests a test execution with parallel testing enabled is failed when
# a test filter leads to no tests being run.
function test_macos_unit_test_parallel_testing_no_tests_fail() {
  create_runners
  create_macos_unit_tests
  ! do_macos_test --test_arg=--xcodebuild_args=-parallel-testing-enabled \
    --test_arg=--xcodebuild_args=YES \
    --test_arg=--xcodebuild_args=-parallel-testing-worker-count \
    --test_arg=--xcodebuild_args=1 \
    --test_filter="BadFilter" \
    //macos:SmallUnitTest || fail "should fail"

  expect_not_log "Test suite 'SmallUnitTest1' started"
  expect_not_log "Test suite 'SmallUnitTest2' started"
  expect_log "FAIL: //macos:SmallUnitTest"
  expect_log "Executed 1 out of 1 test: 1 fails locally."
}

run_suite "macos_unit_test with macos xctestrun runner bundling tests"
