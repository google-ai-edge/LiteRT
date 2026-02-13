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

# Integration tests for testing `docc_archive` rules.

set -euo pipefail

function set_up() {
  mkdir -p app
}

function tear_down() {
  rm -rf app
}

function create_common_files() {
  cat > app/BUILD <<EOF
load("@build_bazel_rules_apple//apple:docc.bzl", "docc_archive")
load("@build_bazel_rules_apple//apple:ios.bzl", "ios_application")
load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

swift_library(
    name = "shared_logic",
    module_name = "SharedLogic",
    srcs = ["shared.swift"],
)

swift_library(
    name = "app_lib",
    srcs = ["App.swift"],
    deps = [":shared_logic"],
)

swift_library(
    name = "docc_bundle_lib",
    srcs = ["bundle.swift"],
    data = glob(["Resources/**"]),
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

docc_archive(
    name = "docc_archive_without_docc_bundle",
    dep = ":app",
    fallback_bundle_identifier = "my.bundle.id",
    fallback_bundle_version = "1.0",
    fallback_display_name = "app",
)

docc_archive(
    name = "docc_archive_with_docc_bundle",
    dep = ":docc_bundle_lib",
    fallback_bundle_identifier = "my.bundle.id.bundle_lib",
    fallback_bundle_version = "1.0",
    fallback_display_name = "bundle_lib",
)
EOF

  cat > app/App.swift <<EOF
import SharedLogic
import SwiftUI

@main
public struct MyApp: App {

    /// Initializes the app
    public init() { }

    /// The main body of the app
    public var body: some Scene {
        WindowGroup {
            Text("Hello World")
              .onAppear {
                sharedDoSomething()
              }
        }
    }
}
EOF

  cat > app/shared.swift <<EOF
import Foundation

/// Does something
public func sharedDoSomething() { }
EOF

  cat > app/bundle.swift <<EOF
import Foundation

/// Does something within the bundle
public func bundleDoSomething() { }
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

  mkdir -p app/Resources/Docs.docc
  cat > app/Resources/Docs.docc/README.md <<EOF
# My App

This is a test app.
EOF

}

function test_preview_docc_archive_with_docc_bundle() {
  create_common_files
  do_action run ios //app:docc_archive_with_docc_bundle
}

function test_preview_docc_archive_without_docc_bundle() {
  create_common_files
  do_action run ios //app:docc_without_docc_bundle
}

run_suite "docc_archive tests"
