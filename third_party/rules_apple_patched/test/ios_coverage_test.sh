#!/bin/bash

# Copyright 2022 The Bazel Authors. All rights reserved.
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

# Integration tests for testing iOS tests with code coverage enabled.

set -euo pipefail

function set_up() {
  mkdir -p app
}

function tear_down() {
  rm -rf app
}

function create_common_files() {
  cat > app/BUILD <<EOF
load("@build_bazel_rules_apple//apple:ios.bzl", "ios_application", "ios_unit_test", "ios_ui_test")
load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

objc_library(
    name = "app_lib",
    hdrs = ["main.h"],
    srcs = ["main.m"],
)

swift_library(
    name = "coverage_app_lib",
    srcs = ["CoverageApp.swift"],
)

objc_library(
    name = "shared_logic",
    hdrs = ["SharedLogic.h"],
    srcs = ["SharedLogic.m"],
)

objc_library(
    name = "hosted_test_lib",
    srcs = ["HostedTest.m"],
    deps = [":app_lib", ":shared_logic"],
)

objc_library(
    name = "standalone_test_lib",
    srcs = ["StandaloneTest.m"],
    deps = [":shared_logic"],
)

swift_library(
    name = "coverage_ui_test_lib",
    srcs = ["PassingUITest.swift"],
    testonly = True
)
EOF

  cat > app/main.h <<EOF
int foo();
EOF

  cat > app/main.m <<EOF
#import <UIKit/UIKit.h>

int foo() {
  return 1;
}

@interface AppDelegate : UIResponder <UIApplicationDelegate>
@end

@implementation AppDelegate
@end

int main(int argc, char **argv) {
  return UIApplicationMain(argc, argv, nil, @"AppDelegate");
}
EOF

  cat > app/SharedLogic.h <<EOF
#import <Foundation/Foundation.h>

@interface SharedLogic: NSObject
- (void)doSomething;
@end
EOF

  cat > app/SharedLogic.m <<EOF
#import "app/SharedLogic.h"

@implementation SharedLogic
- (void)doSomething {}
@end
EOF

  cat > app/HostedTest.m <<EOF
#import <XCTest/XCTest.h>
#import "app/main.h"
#import "app/SharedLogic.h"
@interface HostedTest: XCTestCase
@end

@implementation HostedTest
- (void)testHostedAPI {
  [[SharedLogic new] doSomething];
  XCTAssertEqual(1, foo());
}
@end
EOF

  cat > app/StandaloneTest.m <<EOF
#import <XCTest/XCTest.h>
#import "app/SharedLogic.h"
@interface StandaloneTest: XCTestCase
@end

@implementation StandaloneTest
- (void)testAnything {
  [[SharedLogic new] doSomething];
  XCTAssert(true);
}
@end
EOF

  cat > app/Info.plist <<EOF
{
  CFBundleIdentifier = "\${PRODUCT_BUNDLE_IDENTIFIER}";
  CFBundleName = "\${PRODUCT_NAME}";
  CFBundlePackageType = "APPL";
  CFBundleShortVersionString = "1.0";
  CFBundleVersion = "1.0";
}
EOF

  cat > app/coverage_manifest.txt <<EOF
app/SharedLogic.m
EOF

cat > app/CoverageApp.swift <<EOF
import SwiftUI

@main
struct MyApp: App {
    var body: some Scene {
        WindowGroup {
            Text("Hello World")
        }
    }
}
EOF

cat > app/PassingUITest.swift <<EOF
import XCTest

class PassingUITest: XCTestCase {
    let app = XCUIApplication()

  override func setUp() {
    super.setUp()
    app.launch()
  }

  func testPass() throws {
    XCTAssertTrue(app.staticTexts["Hello World"].exists)
  }
}
EOF

  cat >> app/BUILD <<EOF
ios_application(
    name = "app",
    bundle_id = "my.bundle.id",
    families = ["iphone"],
    infoplists = ["Info.plist"],
    minimum_os_version = "${MIN_OS_IOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    deps = [":app_lib"],
)

ios_application(
    name = "coverage_app",
    bundle_id = "my.bundle.id",
    families = ["iphone"],
    infoplists = ["Info.plist"],
    minimum_os_version = "15",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    deps = [":coverage_app_lib"],
)

ios_unit_test(
    name = "hosted_test",
    deps = [":hosted_test_lib"],
    env = {"STARTUP_TIMEOUT_SEC": "300"},
    minimum_os_version = "${MIN_OS_IOS}",
    size = "large",
    test_host = ":app",
)

ios_unit_test(
    name = "standalone_test",
    deps = [":standalone_test_lib"],
    env = {"STARTUP_TIMEOUT_SEC": "300"},
    minimum_os_version = "${MIN_OS_IOS}",
    size = "large",
)

ios_unit_test(
    name = "standalone_test_new_runner",
    deps = [":standalone_test_lib"],
    minimum_os_version = "${MIN_OS_IOS}",
    runner = "@build_bazel_rules_apple//apple/testing/default_runner:ios_xctestrun_random_runner",
    size = "large",
)

ios_unit_test(
    name = "coverage_manifest_test",
    deps = [":standalone_test_lib"],
    minimum_os_version = "${MIN_OS_IOS}",
    test_coverage_manifest = "coverage_manifest.txt",
    size = "large",
)

ios_unit_test(
    name = "coverage_manifest_test_new_runner",
    deps = [":standalone_test_lib"],
    minimum_os_version = "${MIN_OS_IOS}",
    test_coverage_manifest = "coverage_manifest.txt",
    runner = "@build_bazel_rules_apple//apple/testing/default_runner:ios_xctestrun_ordered_runner",
    size = "large",
)

ios_ui_test(
    name = "test_coverage_ui_test_new_runner",
    deps = [":coverage_ui_test_lib"],
    minimum_os_version = "15",
    test_host = ":coverage_app",
    runner = "@build_bazel_rules_apple//apple/testing/default_runner:ios_xctestrun_random_runner",
    size = "large",
)
EOF
}

function test_standalone_unit_test_coverage() {
  create_common_files
  do_coverage ios --test_output=errors --ios_minimum_os=9.0 --experimental_use_llvm_covmap --zip_undeclared_test_outputs //app:standalone_test || fail "Should build"
  assert_contains "SharedLogic.m:-\[SharedLogic doSomething\]" "test-testlogs/app/standalone_test/coverage.dat"
}

function test_standalone_unit_test_coverage_new_runner() {
  create_common_files
  do_coverage ios --test_output=errors --ios_minimum_os=9.0 --experimental_use_llvm_covmap --zip_undeclared_test_outputs //app:standalone_test_new_runner || fail "Should build"
  assert_contains "SharedLogic.m:-\[SharedLogic doSomething\]" "test-testlogs/app/standalone_test/coverage.dat"
}

function test_standalone_unit_test_coverage_json() {
  create_common_files
  do_coverage ios --test_output=errors --ios_minimum_os=9.0 --experimental_use_llvm_covmap --zip_undeclared_test_outputs --test_env=COVERAGE_PRODUCE_JSON=1 //app:standalone_test || fail "Should build"
  unzip_single_file "test-testlogs/app/standalone_test/test.outputs/outputs.zip" coverage.json \
      grep -q '"name":"SharedLogic.m:-\[SharedLogic doSomething\]"'
}

function test_standalone_unit_test_coverage_coverage_manifest() {
  create_common_files
  do_coverage ios --test_output=errors --ios_minimum_os=9.0 --experimental_use_llvm_covmap --zip_undeclared_test_outputs //app:coverage_manifest_test || fail "Should build"
  assert_contains "SharedLogic.m:-\[SharedLogic doSomething\]" "test-testlogs/app/coverage_manifest_test/coverage.dat"
  assert_contains "SF:app/SharedLogic.m" "test-testlogs/app/coverage_manifest_test/coverage.dat"
  cat "test-testlogs/app/coverage_manifest_test/coverage.dat"
  (! grep "SF:" "test-testlogs/app/coverage_manifest_test/coverage.dat" | grep -v "SF:app/SharedLogic.m") || fail "Should not contain any other files"
}

function test_standalone_unit_test_coverage_coverage_manifest_new_runner() {
  create_common_files
  do_coverage ios --test_output=errors --ios_minimum_os=9.0 --experimental_use_llvm_covmap --zip_undeclared_test_outputs //app:coverage_manifest_test_new_runner || fail "Should build"
  assert_contains "SharedLogic.m:-\[SharedLogic doSomething\]" "test-testlogs/app/coverage_manifest_test_new_runner/coverage.dat"
  assert_contains "SF:app/SharedLogic.m" "test-testlogs/app/coverage_manifest_test_new_runner/coverage.dat"
  cat test-testlogs/app/coverage_manifest_test_new_runner/coverage.dat
  (! grep "SF:" "test-testlogs/app/coverage_manifest_test_new_runner/coverage.dat" | grep -v "SF:app/SharedLogic.m") || fail "Should not contain other files"
}

function test_hosted_unit_test_coverage() {
  create_common_files
  do_coverage ios --test_output=errors --ios_minimum_os=9.0 --experimental_use_llvm_covmap --zip_undeclared_test_outputs //app:hosted_test || fail "Should build"

  # Validate normal coverage is included
  assert_contains "SharedLogic.m:-\[SharedLogic doSomething\]" "test-testlogs/app/hosted_test/coverage.dat"
  # Validate coverage for the hosting binary is included
  assert_contains "FN:3,foo" "test-testlogs/app/hosted_test/coverage.dat"

  if [ -d "test-bin/app/hosted_test.runfiles/_main" ]; then
    # Bzlmod always uses '_main' for current repository
    path="test-bin/app/hosted_test.runfiles/_main/app/hosted_test.zip"
  else
    path="test-bin/app/hosted_test.runfiles/build_bazel_rules_apple_integration_tests/app/hosted_test.zip"
  fi

  # Validate that the symbol called from the hosted binary exists and is undefined
  unzip_single_file \
    "$path" \
    "hosted_test.xctest/hosted_test" \
    nm -u - | grep foo || fail "Undefined 'foo' symbol not found"
}

function test_ui_test_coverage_new_runner() {
  create_common_files
  do_coverage ios --test_output=errors --ios_minimum_os=15.0 --experimental_use_llvm_covmap --zip_undeclared_test_outputs //app:test_coverage_ui_test_new_runner || fail "Should build"
  assert_contains "DA:5,1" "test-testlogs/app/test_coverage_ui_test_new_runner/coverage.dat"
}

run_suite "ios coverage tests"
