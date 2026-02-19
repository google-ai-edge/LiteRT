#!/bin/bash

# Copyright 2024 The Bazel Authors. All rights reserved.
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

set -eu

# Integration tests for apple_metal_library.

function set_up() {
  mkdir -p app
}

function tear_down() {
  rm -rf app
}

# Creates common source and targets for apple_metal_library in an iOS app.
function create_common_files() {
  cat > app/BUILD <<EOF
load("@build_bazel_rules_apple//apple:ios.bzl", "ios_application")
load("@build_bazel_rules_apple//apple:resources.bzl", "apple_metal_library")
load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

apple_metal_library(
    name = "SampleMetal",
    hdrs = ["@build_bazel_rules_apple//test/testdata/resources:metal_hdrs"],
    srcs = ["@build_bazel_rules_apple//test/testdata/resources:metal_srcs"],
    copts = ["-frecord-sources"]
)

objc_library(
    name = "objc_lib",
    srcs = ["objc_lib.m"],
    data = [":SampleMetal"],
)

ios_application(
    name = "app_objc",
    bundle_id = "my.bundle.id",
    families = ["iphone", "ipad"],
    infoplists = ["Info.plist"],
    minimum_os_version = "${MIN_OS_IOS}",
    deps = [":objc_lib"],
)

swift_library(
    name = "swift_lib",
    srcs = ["swift_lib.swift"],
    data = [":SampleMetal"],
)

ios_application(
    name = "app_swift",
    bundle_id = "my.bundle.id",
    families = ["iphone", "ipad"],
    infoplists = ["Info.plist"],
    minimum_os_version = "${MIN_OS_IOS}",
    deps = [":swift_lib"],
)
EOF

  cat > app/objc_lib.m <<EOF
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@interface ObjcLib: NSObject
@end

@implementation ObjcLib
- (void)test {
  NSLog(@"Hello, world");
}
@end

int main(int argc, char **argv) {
  return 0;
}
EOF

  cat > app/swift_lib.swift <<EOF
import UIKit
import Metal

@UIApplicationMain
class AppDelegate: UIResponder, UIApplicationDelegate {
}
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
}

# Tests that the Info.plist in the packaged bundle has the correct content.
function test_intentdefinition_builds() {
  create_common_files
  pwd

  do_build ios //app:app_objc || fail "Should build"
  assert_zip_contains "test-bin/app/app_objc.ipa" "Payload/app_objc.app/default.metallib"

  do_build ios //app:app_swift || fail "Should build"
  assert_zip_contains "test-bin/app/app_swift.ipa" "Payload/app_swift.app/default.metallib"
}

run_suite "apple_intent_library tests"
