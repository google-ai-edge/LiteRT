#!/bin/bash

# Copyright 2017 The Bazel Authors. All rights reserved.
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

# Integration tests for bundling tvOS apps with extensions.

function set_up() {
  mkdir -p app
}

function tear_down() {
  rm -rf app
}

# Creates common source, targets, and basic plist for tvOS applications.
function create_minimal_tvos_application_with_extension() {
  cat > app/BUILD <<EOF
load("@build_bazel_rules_apple//apple:tvos.bzl",
     "tvos_application",
     "tvos_extension",
    )

objc_library(
    name = "lib",
    hdrs = ["Foo.h"],
    srcs = ["main.m"],
)

tvos_application(
    name = "app",
    bundle_id = "my.bundle.id",
    extensions = [":ext"],
    infoplists = ["Info-App.plist"],
    minimum_os_version = "${MIN_OS_TVOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_tvos.mobileprovision",
    deps = [":lib"],
)

tvos_extension(
    name = "ext",
    bundle_id = "my.bundle.id.extension",
    infoplists = ["Info-Ext.plist"],
    minimum_os_version = "${MIN_OS_TVOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_tvos.mobileprovision",
    deps = [":lib"],
)
EOF

  cat > app/Foo.h <<EOF
#import <Foundation/Foundation.h>
// This dummy class is needed to generate code in the extension target,
// which does not take main() from here, rather from an SDK.
@interface Foo: NSObject
- (void)doSomething;
@end
EOF

  cat > app/main.m <<EOF
#import <Foundation/Foundation.h>
#import "app/Foo.h"
@implementation Foo
- (void)doSomething { }
@end

int main(int argc, char **argv) {
  return 0;
}
EOF

  cat > app/Info-App.plist <<EOF
{
  CFBundleIdentifier = "\${PRODUCT_BUNDLE_IDENTIFIER}";
  CFBundleName = "\${PRODUCT_NAME}";
  CFBundlePackageType = "APPL";
  CFBundleShortVersionString = "1.0";
  CFBundleVersion = "1.0";
}
EOF

  cat > app/Info-Ext.plist <<EOF
{
  CFBundleIdentifier = "\${PRODUCT_BUNDLE_IDENTIFIER}";
  CFBundleName = "\${PRODUCT_NAME}";
  CFBundlePackageType = "XPC!";
  CFBundleShortVersionString = "1.0";
  CFBundleVersion = "1.0";
  NSExtension = {
    NSExtensionPrincipalClass = "DummyValue";
    NSExtensionPointIdentifier = "com.apple.widget-extension";
  };
}
EOF
}

# Test missing the CFBundleVersion fails the build.
function test_missing_version_fails() {
  create_minimal_tvos_application_with_extension

  cat > app/Info-Ext.plist <<EOF
{
  CFBundleIdentifier = "\${PRODUCT_BUNDLE_IDENTIFIER}";
  CFBundleName = "\${PRODUCT_NAME}";
  CFBundlePackageType = "XPC!";
  CFBundleShortVersionString = "1.0";
  NSExtension = {
    NSExtensionPrincipalClass = "DummyValue";
    NSExtensionPointIdentifier = "com.apple.widget-extension";
  };
}
EOF

  ! do_build tvos //app:app \
    || fail "Should fail build"

  expect_log 'Target "@@\?//app:ext" is missing CFBundleVersion.'
}

# Test missing the CFBundleShortVersionString fails the build.
function test_missing_short_version_fails() {
  create_minimal_tvos_application_with_extension

  cat > app/Info-Ext.plist <<EOF
{
  CFBundleIdentifier = "\${PRODUCT_BUNDLE_IDENTIFIER}";
  CFBundleName = "\${PRODUCT_NAME}";
  CFBundlePackageType = "XPC!";
  CFBundleVersion = "1.0";
  NSExtension = {
    NSExtensionPrincipalClass = "DummyValue";
    NSExtensionPointIdentifier = "com.apple.widget-extension";
  };
}
EOF

  ! do_build tvos //app:app \
    || fail "Should fail build"

  expect_log 'Target "@@\?//app:ext" is missing CFBundleShortVersionString.'
}

# Tests that if an application contains an extension with a bundle ID that is
# not the app's ID followed by at least another component, the build fails.
function test_extension_with_mismatched_bundle_id_fails_to_build() {
  cat > app/BUILD <<EOF
load("@build_bazel_rules_apple//apple:tvos.bzl",
     "tvos_application",
     "tvos_extension",
    )

objc_library(
    name = "lib",
    srcs = ["main.m"],
)

tvos_application(
    name = "app",
    bundle_id = "my.bundle.id",
    extensions = [":ext"],
    infoplists = ["Info-App.plist"],
    minimum_os_version = "${MIN_OS_TVOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_tvos.mobileprovision",
    deps = [":lib"],
)

tvos_extension(
    name = "ext",
    bundle_id = "my.extension.id",
    infoplists = ["Info-Ext.plist"],
    minimum_os_version = "${MIN_OS_TVOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_tvos.mobileprovision",
    deps = [":lib"],
)
EOF

  cat > app/main.m <<EOF
#import <Foundation/Foundation.h>
// This dummy class is needed to generate code in the extension target,
// which does not take main() from here, rather from an SDK.
@interface Foo: NSObject
@end
@implementation Foo
@end

int main(int argc, char **argv) {
  return 0;
}
EOF

  cat > app/Info-App.plist <<EOF
{
  CFBundleIdentifier = "\${PRODUCT_BUNDLE_IDENTIFIER}";
  CFBundleName = "\${PRODUCT_NAME}";
  CFBundlePackageType = "APPL";
  CFBundleShortVersionString = "1.0";
  CFBundleVersion = "1.0";
}
EOF

  cat > app/Info-Ext.plist <<EOF
{
  CFBundleIdentifier = "\${PRODUCT_BUNDLE_IDENTIFIER}";
  CFBundleName = "\${PRODUCT_NAME}";
  CFBundlePackageType = "XPC!";
  CFBundleShortVersionString = "1.0";
  CFBundleVersion = "1.0";
  NSExtension = {
    NSExtensionPrincipalClass = "DummyValue";
    NSExtensionPointIdentifier = "com.apple.widget-extension";
  };
}
EOF

  ! do_build tvos //app:app || fail "Should not build"
  expect_log 'While processing target "@@\?//app:app"; the CFBundleIdentifier of the child target "@@\?//app:ext" should have "my.bundle.id." as its prefix, but found "my.extension.id".'
}

# Tests that if an application contains an extension with different
# CFBundleShortVersionString the build fails.
function test_extension_with_mismatched_short_version_fails_to_build() {
  cat > app/BUILD <<EOF
load("@build_bazel_rules_apple//apple:tvos.bzl",
     "tvos_application",
     "tvos_extension",
    )

objc_library(
    name = "lib",
    srcs = ["main.m"],
)

tvos_application(
    name = "app",
    bundle_id = "my.bundle.id",
    extensions = [":ext"],
    infoplists = ["Info-App.plist"],
    minimum_os_version = "${MIN_OS_TVOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_tvos.mobileprovision",
    deps = [":lib"],
)

tvos_extension(
    name = "ext",
    bundle_id = "my.bundle.id.extension",
    infoplists = ["Info-Ext.plist"],
    minimum_os_version = "${MIN_OS_TVOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_tvos.mobileprovision",
    deps = [":lib"],
)
EOF

  cat > app/main.m <<EOF
#import <Foundation/Foundation.h>
// This dummy class is needed to generate code in the extension target,
// which does not take main() from here, rather from an SDK.
@interface Foo: NSObject
@end
@implementation Foo
@end

int main(int argc, char **argv) {
  return 0;
}
EOF

  cat > app/Info-App.plist <<EOF
{
  CFBundleIdentifier = "\${PRODUCT_BUNDLE_IDENTIFIER}";
  CFBundleName = "\${PRODUCT_NAME}";
  CFBundlePackageType = "APPL";
  CFBundleShortVersionString = "1.0";
  CFBundleVersion = "1.0";
}
EOF

  cat > app/Info-Ext.plist <<EOF
{
  CFBundleIdentifier = "\${PRODUCT_BUNDLE_IDENTIFIER}";
  CFBundleName = "\${PRODUCT_NAME}";
  CFBundlePackageType = "XPC!";
  CFBundleShortVersionString = "1.1";
  CFBundleVersion = "1.0";
  NSExtension = {
    NSExtensionPrincipalClass = "DummyValue";
    NSExtensionPointIdentifier = "com.apple.widget-extension";
  };
}
EOF

  ! do_build tvos //app:app || fail "Should not build"
  expect_log "While processing target \"@@\?//app:app\"; the CFBundleShortVersionString of the child target \"@@\?//app:ext\" should be the same as its parent's version string \"1.0\", but found \"1.1\"."
}

# Tests that if an application contains an extension with different
# CFBundleVersion the build fails.
function test_extension_with_mismatched_version_fails_to_build() {
  cat > app/BUILD <<EOF
load("@build_bazel_rules_apple//apple:tvos.bzl",
     "tvos_application",
     "tvos_extension",
    )

objc_library(
    name = "lib",
    srcs = ["main.m"],
)

tvos_application(
    name = "app",
    bundle_id = "my.bundle.id",
    extensions = [":ext"],
    infoplists = ["Info-App.plist"],
    minimum_os_version = "${MIN_OS_TVOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_tvos.mobileprovision",
    deps = [":lib"],
)

tvos_extension(
    name = "ext",
    bundle_id = "my.bundle.id.extension",
    infoplists = ["Info-Ext.plist"],
    minimum_os_version = "${MIN_OS_TVOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_tvos.mobileprovision",
    deps = [":lib"],
)
EOF

  cat > app/main.m <<EOF
#import <Foundation/Foundation.h>
// This dummy class is needed to generate code in the extension target,
// which does not take main() from here, rather from an SDK.
@interface Foo: NSObject
@end
@implementation Foo
@end

int main(int argc, char **argv) {
  return 0;
}
EOF

  cat > app/Info-App.plist <<EOF
{
  CFBundleIdentifier = "\${PRODUCT_BUNDLE_IDENTIFIER}";
  CFBundleName = "\${PRODUCT_NAME}";
  CFBundlePackageType = "APPL";
  CFBundleShortVersionString = "1.0";
  CFBundleVersion = "1.0";
}
EOF

  cat > app/Info-Ext.plist <<EOF
{
  CFBundleIdentifier = "\${PRODUCT_BUNDLE_IDENTIFIER}";
  CFBundleName = "\${PRODUCT_NAME}";
  CFBundlePackageType = "XPC!";
  CFBundleShortVersionString = "1.0";
  CFBundleVersion = "1.1";
  NSExtension = {
    NSExtensionPrincipalClass = "DummyValue";
    NSExtensionPointIdentifier = "com.apple.widget-extension";
  };
}
EOF

  ! do_build tvos //app:app || fail "Should not build"
  expect_log "While processing target \"@@\?//app:app\"; the CFBundleVersion of the child target \"@@\?//app:ext\" should be the same as its parent's version string \"1.0\", but found \"1.1\"."
}

# Tests that an extension with legacy_entry_point=True builds successfully.
function test_legacy_entry_point_extension_builds() {
  cat > app/BUILD <<EOF
load("@build_bazel_rules_apple//apple:tvos.bzl",
     "tvos_application",
     "tvos_extension",
    )

objc_library(
    name = "lib",
    srcs = ["main.m"],
)

tvos_application(
    name = "app",
    bundle_id = "my.bundle.id",
    extensions = [":ext"],
    infoplists = ["Info-App.plist"],
    minimum_os_version = "${MIN_OS_TVOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_tvos.mobileprovision",
    deps = [":lib"],
)

tvos_extension(
    name = "ext",
    bundle_id = "my.bundle.id.extension",
    infoplists = ["Info-Ext.plist"],
    legacy_entry_point = True,
    minimum_os_version = "${MIN_OS_TVOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_tvos.mobileprovision",
    deps = [":lib"],
)
EOF

  cat > app/main.m <<EOF
#import <Foundation/Foundation.h>
// This dummy class is needed to generate code in the extension target,
// which does not take main() from here, rather from an SDK.
@interface Foo: NSObject
@end
@implementation Foo
@end

int main(int argc, char **argv) {
  return 0;
}
EOF

  cat > app/Info-App.plist <<EOF
{
  CFBundleIdentifier = "\${PRODUCT_BUNDLE_IDENTIFIER}";
  CFBundleName = "\${PRODUCT_NAME}";
  CFBundlePackageType = "APPL";
  CFBundleShortVersionString = "1.0";
  CFBundleVersion = "1.0";
}
EOF

  cat > app/Info-Ext.plist <<EOF
{
  CFBundleIdentifier = "\${PRODUCT_BUNDLE_IDENTIFIER}";
  CFBundleName = "\${PRODUCT_NAME}";
  CFBundlePackageType = "XPC!";
  CFBundleShortVersionString = "1.0";
  CFBundleVersion = "1.0";
  NSExtension = {
    NSExtensionPrincipalClass = "DummyValue";
    NSExtensionPointIdentifier = "com.apple.widget-extension";
  };
}
EOF

  do_build tvos //app:app || fail "Should build"
}

run_suite "tvos_extension bundling tests"
