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

# Integration tests for bundling CoreML models into resource bundles.

function set_up() {
  mkdir -p app
}

function tear_down() {
  rm -rf app
}

# Creates common source, targets, and basic plist for iOS applications.
function create_common_files() {
  cat > app/BUILD <<EOF
load("@build_bazel_rules_apple//apple:ios.bzl", "ios_application")
load(
    "@build_bazel_rules_apple//apple:resources.bzl",
    "apple_precompiled_resource_bundle",
  )
load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

apple_precompiled_resource_bundle(
    name = "app.resources",
    bundle_name = "App_Resources",
    resources = [
        "@build_bazel_rules_apple//test/testdata/resources:sample.mlpackage",
    ],
)

swift_library(
    name = "app_lib",
    data = ["app.resources"],
    srcs = ["AppDelegate.swift"],
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

  cat > app/AppDelegate.swift <<EOF
import UIKit

@UIApplicationMain
class AppDelegate: UIResponder, UIApplicationDelegate {
  var window: UIWindow?
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

# Tests that the mlpackage is compiled correctly into a resource bundle
function test_mlmodel_resource_bundle_builds() {
  create_common_files

  do_build ios //app:app || fail "Should build"

  assert_zip_contains "test-bin/app/app.ipa" "Payload/app.app/App_Resources.bundle/sample.mlmodelc/"
}

run_suite "coreml resource bundle tests"
