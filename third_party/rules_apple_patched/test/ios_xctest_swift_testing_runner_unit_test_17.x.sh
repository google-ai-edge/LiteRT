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

MIN_OS_IOS="17.0.0"

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
    "@build_bazel_rules_apple//apple/testing/default_runner:ios_xctestrun_runner.bzl",
    "ios_xctestrun_runner"
)

ios_xctestrun_runner(
    name = "ios_x86_64_sim_runner_17",
    device_type = "iPhone 11",
    os_version = "17.2",
)
EOF
}

function create_swift_testing_tests() {
  cat > ios/passing_swift_testing_test.swift <<EOF
import Testing
import XCTest

@Suite("Simple passing swift test suite")
class SimpleSwiftTestingTest {
  @Test
  func testCaseOne() {
    let one = 1
    #expect(one == 1, "this test passes")
  }

  @Test
  func testCaseTwo() {
    let two = 2
    #expect(two == 2, "this test passes")
  }
}
EOF

cat > ios/failing_swift_testing_test.swift <<EOF
import Testing
import XCTest

@Suite("Simple failing swift test suite")
class SimpleSwiftTestingTest {
  @Test
  func testCaseOne() {

    let one = 1
    #expect(one == 2, "this test fails")
  }
}
EOF


  cat > ios/empty_swift_testing_test.swift <<EOF
import Testing
import XCTest

@Suite("Simple empty swift test suite")
class SimpleSwiftTestingTest {

}
EOF

  cat >> ios/BUILD <<EOF
swift_library(
  name = "passing_swift_testing_test_lib",
  testonly = True,
  srcs = ["passing_swift_testing_test.swift"],
)

swift_library(
  name = "failing_swift_testing_test_lib",
  testonly = True,
  srcs = ["failing_swift_testing_test.swift"],
)

swift_library(
  name = "empty_swift_testing_test_lib",
  testonly = True,
  srcs = ["empty_swift_testing_test.swift"],
)

ios_unit_test(
  name = "PassingSwiftTestingTest",
  deps = [":passing_swift_testing_test_lib"],
  minimum_os_version = "${MIN_OS_IOS}",
  runner = ":ios_x86_64_sim_runner_17",
)

ios_unit_test(
  name = "FailingSwiftTestingTest",
  deps = [":failing_swift_testing_test_lib"],
  minimum_os_version = "${MIN_OS_IOS}",
  runner = ":ios_x86_64_sim_runner_17",
)

ios_unit_test(
  name = "EmptySwiftTestingTest",
  deps = [":empty_swift_testing_test_lib"],
  minimum_os_version = "${MIN_OS_IOS}",
  runner = ":ios_x86_64_sim_runner_17",
)
EOF
}

function do_ios_test() {
  do_test ios "--test_output=all" "--spawn_strategy=local" "$@"
}

function test_ios_swift_testing_pass() {
  create_sim_runners
  create_swift_testing_tests
  do_ios_test //ios:PassingSwiftTestingTest || fail "should pass"
  expect_log "Suite \"Simple passing swift test suite\" passed"
  expect_log "Test run with 2 tests passed after"
}

function test_ios_swift_testing_fail() {
  create_sim_runners
  create_swift_testing_tests
  (! do_ios_test //ios:FailingSwiftTestingTest) || fail "should not pass"
  expect_log "Suite \"Simple failing swift test suite\" failed after"
  expect_log "Test run with 1 test failed after"
}

function test_ios_swift_testing_empty() {
  create_sim_runners
  create_swift_testing_tests
  (! do_ios_test //ios:EmptySwiftTestingTest) || fail "should not pass"
  expect_log "Suite \"Simple empty swift test suite\" passed"
  expect_log "Test run with 0 tests passed after"
  expect_log "error: no tests were executed, is the test bundle empty?"
}

run_suite "ios_unit_test with iOS 17.x test runner bundling tests"
