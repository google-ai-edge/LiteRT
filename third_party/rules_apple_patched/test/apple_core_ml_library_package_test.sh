#!/bin/bash

# Copyright 2019 The Bazel Authors. All rights reserved.
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

# Integration tests for apple_core_ml_library.

function set_up() {
  mkdir -p app
}

function tear_down() {
  rm -rf app
}

# Creates common source, targets, and basic plist for macOS loadable bundles.
function create_common_files() {
  cat > app/BUILD <<EOF
load("@build_bazel_rules_apple//apple:ios.bzl", "ios_application")
load(
    "@build_bazel_rules_apple//apple:resources.bzl",
    "apple_core_ml_library",
    "swift_apple_core_ml_library",
)
load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

objc_library(
    name = "app_lib",
    srcs = ["main.m"],
    deps = [":swift_lib", ":objc_lib"],
)

objc_library(
    name = "objc_lib",
    srcs = ["objc_lib.m"],
    deps = [":SampleCoreML"],
)

swift_library(
    name = "swift_lib",
    srcs = ["swift_lib.swift"],
    deps = [":SampleSwiftCoreML"],
)

apple_core_ml_library(
    name = "SampleCoreML",
    mlmodel = "@build_bazel_rules_apple//test/testdata/resources:sample.mlpackage",
)

swift_apple_core_ml_library(
    name = "SampleSwiftCoreML",
    mlmodel = "@build_bazel_rules_apple//test/testdata/resources:sample.mlpackage",
)

ios_application(
    name = "app",
    bundle_id = "my.bundle.id",
    families = ["iphone", "ipad"],
    infoplists = ["Info.plist"],
    minimum_os_version = "${MIN_OS_IOS}",
    deps = [":app_lib"],
)
EOF

  cat > app/main.m <<EOF
int main(int argc, char **argv) {
  return 0;
}
EOF

  cat > app/objc_lib.m <<EOF
#import <Foundation/Foundation.h>
#import "app/SampleCoreML.h"

@interface ObjcLib: NSObject
@end

@implementation ObjcLib
- (void)test:(sample *)sample {
  NSLog(@"%@", sample);
}
@end
EOF

  cat > app/swift_lib.swift <<EOF
import Foundation
import app_SampleSwiftCoreML

public struct SwiftLib {
  public var mySample = sample()
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
function test_mlmodel_builds() {
  create_common_files

  do_build ios //app:app || fail "Should build"

  assert_zip_contains "test-bin/app/app.ipa" "Payload/app.app/sample.mlmodelc/"
}

run_suite "apple_core_ml_library tests"
