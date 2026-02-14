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

# Integration tests for bundling simple watchOS applications.

function set_up() {
  mkdir -p app
}

function tear_down() {
  rm -rf app
}

# Creates the minimal companion iOS app and the support files needed to
# make a watchOS app (allowing the caller to manually add the watchOS
# targets to the BUILD file).
function create_companion_app_and_watchos_application_support_files() {
  cat > app/BUILD <<EOF
load("@build_bazel_rules_apple//apple:ios.bzl", "ios_application")
load("@build_bazel_rules_apple//apple:watchos.bzl",
     "watchos_application",
     "watchos_extension"
    )

objc_library(
    name = "lib",
    srcs = ["main.m"],
)

objc_library(
    name = "watch_lib",
    sdk_frameworks = ["WatchKit"],
)

ios_application(
    name = "app",
    bundle_id = "my.bundle.id",
    families = ["iphone"],
    infoplists = ["Info-PhoneApp.plist"],
    minimum_os_version = "${MIN_OS_IOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    watch_application = ":watch_app",
    deps = [":lib"],
)

EOF

  cat > app/main.m <<EOF
int main(int argc, char **argv) {
  return 0;
}
EOF

  cat > app/Info-PhoneApp.plist <<EOF
{
  CFBundleIdentifier = "\${PRODUCT_BUNDLE_IDENTIFIER}";
  CFBundleName = "\${PRODUCT_NAME}";
  CFBundlePackageType = "APPL";
  CFBundleShortVersionString = "1";
  CFBundleVersion = "1";
}
EOF

  cat > app/Info-WatchApp.plist <<EOF
{
  CFBundleIdentifier = "\${PRODUCT_BUNDLE_IDENTIFIER}";
  CFBundleName = "\${PRODUCT_NAME}";
  CFBundlePackageType = "APPL";
  CFBundleShortVersionString = "1";
  CFBundleVersion = "1";
  WKCompanionAppBundleIdentifier = "my.bundle.id";
  WKWatchKitApp = true;
}
EOF

  cat > app/Info-WatchExt.plist <<EOF
{
  CFBundleIdentifier = "\${PRODUCT_BUNDLE_IDENTIFIER}";
  CFBundleName = "\${PRODUCT_NAME}";
  CFBundlePackageType = "XPC!";
  CFBundleShortVersionString = "1";
  CFBundleVersion = "1";
  NSExtension = {
    NSExtensionAttributes = {
      WKAppBundleIdentifier = "my.bundle.id.watch-app";
    };
    NSExtensionPointIdentifier = "com.apple.watchkit";
  };
}
EOF

  cat > app/entitlements.entitlements <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>test-an-entitlement</key>
  <false/>
</dict>
</plist>
EOF
}

# Creates minimal watchOS application and extension targets along with a
# companion iOS app.
function create_minimal_watchos_application_with_companion() {
  create_companion_app_and_watchos_application_support_files

  cat >> app/BUILD <<EOF
watchos_application(
    name = "watch_app",
    bundle_id = "my.bundle.id.watch-app",
    entitlements = "entitlements.entitlements",
    extension = ":watch_ext",
    infoplists = ["Info-WatchApp.plist"],
    minimum_os_version = "${MIN_OS_WATCHOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
)

watchos_extension(
    name = "watch_ext",
    bundle_id = "my.bundle.id.watch-app.watch-ext",
    entitlements = "entitlements.entitlements",
    infoplists = ["Info-WatchExt.plist"],
    minimum_os_version = "${MIN_OS_WATCHOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    deps = [":watch_lib"],
)
EOF
}

# Test missing the CFBundleVersion fails the build.
function test_watch_app_missing_version_fails() {
  create_minimal_watchos_application_with_companion

  # Replace the file, but without CFBundleVersion.
  cat > app/Info-WatchApp.plist <<EOF
{
  CFBundleIdentifier = "\${PRODUCT_BUNDLE_IDENTIFIER}";
  CFBundleName = "\${PRODUCT_NAME}";
  CFBundlePackageType = "APPL";
  CFBundleShortVersionString = "1";
  WKCompanionAppBundleIdentifier = "my.bundle.id";
  WKWatchKitApp = true;
}
EOF

  ! do_build watchos //app:app \
    || fail "Should fail build"

  expect_log 'Target "@@\?//app:watch_app" is missing CFBundleVersion.'
}

# Test missing the CFBundleShortVersionString fails the build.
function test_watch_app_missing_short_version_fails() {
  create_minimal_watchos_application_with_companion

  # Replace the file, but without CFBundleShortVersionString.
  cat > app/Info-WatchApp.plist <<EOF
{
  CFBundleIdentifier = "\${PRODUCT_BUNDLE_IDENTIFIER}";
  CFBundleName = "\${PRODUCT_NAME}";
  CFBundlePackageType = "APPL";
  CFBundleVersion = "1";
  WKCompanionAppBundleIdentifier = "my.bundle.id";
  WKWatchKitApp = true;
}
EOF

  ! do_build watchos //app:app \
    || fail "Should fail build"

  expect_log 'Target "@@\?//app:watch_app" is missing CFBundleShortVersionString.'
}

# Test missing the CFBundleVersion fails the build.
function test_watch_ext_missing_version_fails() {
  create_minimal_watchos_application_with_companion

  # Replace the file, but without CFBundleVersion.
  cat > app/Info-WatchExt.plist <<EOF
{
  CFBundleIdentifier = "\${PRODUCT_BUNDLE_IDENTIFIER}";
  CFBundleName = "\${PRODUCT_NAME}";
  CFBundlePackageType = "XPC!";
  CFBundleShortVersionString = "1";
  NSExtension = {
    NSExtensionAttributes = {
      WKAppBundleIdentifier = "my.bundle.id.watch-app";
    };
    NSExtensionPointIdentifier = "com.apple.watchkit";
  };
}
EOF

  ! do_build watchos //app:app \
    || fail "Should fail build"

  expect_log 'Target "@@\?//app:watch_ext" is missing CFBundleVersion.'
}

# Test missing the CFBundleShortVersionString fails the build.
function test_watch_ext_missing_short_version_fails() {
  create_minimal_watchos_application_with_companion

  # Replace the file, but without CFBundleShortVersionString.
  cat > app/Info-WatchExt.plist <<EOF
{
  CFBundleIdentifier = "\${PRODUCT_BUNDLE_IDENTIFIER}";
  CFBundleName = "\${PRODUCT_NAME}";
  CFBundlePackageType = "XPC!";
  CFBundleVersion = "1";
  NSExtension = {
    NSExtensionAttributes = {
      WKAppBundleIdentifier = "my.bundle.id.watch-app";
    };
    NSExtensionPointIdentifier = "com.apple.watchkit";
  };
}
EOF

  ! do_build watchos //app:app \
    || fail "Should fail build"

  expect_log 'Target "@@\?//app:watch_ext" is missing CFBundleShortVersionString.'
}

# Tests that the linkmap outputs are produced when --objc_generate_linkmap is
# present.
function test_linkmaps_generated() {
  create_minimal_watchos_application_with_companion
  do_build watchos --objc_generate_linkmap \
      //app:watch_ext || fail "Should build"

  declare -a archs=( $(current_archs watchos) )
  for arch in "${archs[@]}"; do
    assert_exists "test-bin/app/watch_ext_${arch}.linkmap"
  done
}

# Tests that failures to extract from a provisioning profile are propertly
# reported (from watchOS application profile).
function test_provisioning_profile_extraction_failure_watch_application() {
  create_companion_app_and_watchos_application_support_files

  cat >> app/BUILD <<EOF
watchos_application(
    name = "watch_app",
    bundle_id = "my.bundle.id.watch-app",
    extension = ":watch_ext",
    infoplists = ["Info-WatchApp.plist"],
    minimum_os_version = "${MIN_OS_WATCHOS}",
    provisioning_profile = "bogus.mobileprovision",
)

watchos_extension(
    name = "watch_ext",
    bundle_id = "my.bundle.id.watch-app.watch-ext",
    infoplists = ["Info-WatchExt.plist"],
    minimum_os_version = "${MIN_OS_WATCHOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    deps = [":watch_lib"],
)
EOF

  cat > app/bogus.mobileprovision <<EOF
BOGUS BOGUS BOGUS BOGUS BOGUS BOGUS BOGUS BOGUS BOGUS BOGUS BOGUS BOGUS BOGUS
EOF

  if is_device_build watchos ; then
    ! do_build watchos //app:app || fail "Should fail"
    # The fact that multiple things are tried is left as an impl detail and
    # only the final message is looked for.
    expect_log 'While processing target "@@\?//app:watch_app", failed to extract from the provisioning profile "app/bogus.mobileprovision".'
  else
    # For simulator builds, entitlements are added as a Mach-O section in
    # the binary, so the build shouldn't fail.
    do_build watchos //app:app || fail "Should build"
  fi
}

# Tests that failures to extract from a provisioning profile are propertly
# reported (from watchOS extension profile).
function test_provisioning_profile_extraction_failure_watch_extension() {
  create_companion_app_and_watchos_application_support_files

  cat >> app/BUILD <<EOF
watchos_application(
    name = "watch_app",
    bundle_id = "my.bundle.id.watch-app",
    extension = ":watch_ext",
    infoplists = ["Info-WatchApp.plist"],
    minimum_os_version = "${MIN_OS_WATCHOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
)

watchos_extension(
    name = "watch_ext",
    bundle_id = "my.bundle.id.watch-app.watch-ext",
    infoplists = ["Info-WatchExt.plist"],
    minimum_os_version = "${MIN_OS_WATCHOS}",
    provisioning_profile = "bogus.mobileprovision",
    deps = [":watch_lib"],
)
EOF

  cat > app/bogus.mobileprovision <<EOF
BOGUS BOGUS BOGUS BOGUS BOGUS BOGUS BOGUS BOGUS BOGUS BOGUS BOGUS BOGUS BOGUS
EOF

  ! do_build watchos //app:app || fail "Should fail"
  # The fact that multiple things are tried is left as an impl detail and
  # only the final message is looked for.
  expect_log 'While processing target "@@\?//app:watch_ext", failed to extract from the provisioning profile "app/bogus.mobileprovision".'
}

# Test that a watchOS app with a bundle_id that isn't a prefixed by
# the iOS app fails the build.
function test_app_with_mismatched_bundle_id_fails_to_build() {
  create_companion_app_and_watchos_application_support_files

  cat >> app/BUILD <<EOF
watchos_application(
    name = "watch_app",
    bundle_id = "my.bundle2.id.watch-app",
    entitlements = "entitlements.entitlements",
    extension = ":watch_ext",
    infoplists = ["Info-WatchApp.plist"],
    minimum_os_version = "${MIN_OS_WATCHOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
)

watchos_extension(
    name = "watch_ext",
    bundle_id = "my.bundle2.id.watch-app.watch-ext",
    entitlements = "entitlements.entitlements",
    infoplists = ["Info-WatchExt.plist"],
    minimum_os_version = "${MIN_OS_WATCHOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    deps = [":watch_lib"],
)
EOF

  # The WKAppBundleIdentifier has to also match.
  cat > app/Info-WatchExt.plist <<EOF
{
  CFBundleIdentifier = "\${PRODUCT_BUNDLE_IDENTIFIER}";
  CFBundleName = "\${PRODUCT_NAME}";
  CFBundlePackageType = "XPC!";
  CFBundleShortVersionString = "1";
  CFBundleVersion = "1";
  NSExtension = {
    NSExtensionAttributes = {
      WKAppBundleIdentifier = "my.bundle2.id.watch-app";
    };
    NSExtensionPointIdentifier = "com.apple.watchkit";
  };
}
EOF

  ! do_build watchos //app:app || fail "Should not build"
  expect_log 'While processing target "@@\?//app:app"; the CFBundleIdentifier of the child target "@@\?//app:watch_app" should have "my.bundle.id." as its prefix, but found "my.bundle2.id.watch-app".'
}

# Test that a watchOS extension with a bundle_id that isn't a prefixed by
# the watchOS app fails the build.
function test_extension_with_mismatched_bundle_id_fails_to_build() {
  create_companion_app_and_watchos_application_support_files

  cat >> app/BUILD <<EOF
watchos_application(
    name = "watch_app",
    bundle_id = "my.bundle.id.watch-app",
    entitlements = "entitlements.entitlements",
    extension = ":watch_ext",
    infoplists = ["Info-WatchApp.plist"],
    minimum_os_version = "${MIN_OS_WATCHOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
)

watchos_extension(
    name = "watch_ext",
    bundle_id = "my.bundle2.id.watch-app.watch-ext",
    entitlements = "entitlements.entitlements",
    infoplists = ["Info-WatchExt.plist"],
    minimum_os_version = "${MIN_OS_WATCHOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    deps = [":watch_lib"],
)
EOF

  ! do_build watchos //app:app || fail "Should not build"
  expect_log 'While processing target "@@\?//app:watch_app"; the CFBundleIdentifier of the child target "@@\?//app:watch_ext" should have "my.bundle.id.watch-app." as its prefix, but found "my.bundle2.id.watch-app.watch-ext".'
}

# Test that a watchOS app with a different CFBundleShortVersionString than
# the iOS app fails the build.
function test_app_with_mismatched_short_version_fails_to_build() {
  create_companion_app_and_watchos_application_support_files

  cat >> app/BUILD <<EOF
watchos_application(
    name = "watch_app",
    bundle_id = "my.bundle.id.watch-app",
    entitlements = "entitlements.entitlements",
    extension = ":watch_ext",
    infoplists = ["Info-WatchApp.plist"],
    minimum_os_version = "${MIN_OS_WATCHOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
)

watchos_extension(
    name = "watch_ext",
    bundle_id = "my.bundle.id.watch-app.watch-ext",
    entitlements = "entitlements.entitlements",
    infoplists = ["Info-WatchExt.plist"],
    minimum_os_version = "${MIN_OS_WATCHOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    deps = [":watch_lib"],
)
EOF

  # Give the watch app and extension a different short version.
  cat > app/Info-WatchApp.plist <<EOF
{
  CFBundleIdentifier = "\${PRODUCT_BUNDLE_IDENTIFIER}";
  CFBundleName = "\${PRODUCT_NAME}";
  CFBundlePackageType = "APPL";
  CFBundleShortVersionString = "1.1";
  CFBundleVersion = "1";
  WKCompanionAppBundleIdentifier = "my.bundle.id";
  WKWatchKitApp = true;
}
EOF

  cat > app/Info-WatchExt.plist <<EOF
{
  CFBundleIdentifier = "\${PRODUCT_BUNDLE_IDENTIFIER}";
  CFBundleName = "\${PRODUCT_NAME}";
  CFBundlePackageType = "XPC!";
  CFBundleShortVersionString = "1.1";
  CFBundleVersion = "1";
  NSExtension = {
    NSExtensionAttributes = {
      WKAppBundleIdentifier = "my.bundle.id.watch-app";
    };
    NSExtensionPointIdentifier = "com.apple.watchkit";
  };
}
EOF

  ! do_build watchos //app:app || fail "Should not build"
  expect_log "While processing target \"@@\?//app:app\"; the CFBundleShortVersionString of the child target \"@@\?//app:watch_app\" should be the same as its parent's version string \"1\", but found \"1.1\"."
}

# Test that a watchOS extension with a different CFBundleShortVersionString
# than the watchOS app fails the build.
function test_extension_with_mismatched_short_version_fails_to_build() {
  create_companion_app_and_watchos_application_support_files

  cat >> app/BUILD <<EOF
watchos_application(
    name = "watch_app",
    bundle_id = "my.bundle.id.watch-app",
    entitlements = "entitlements.entitlements",
    extension = ":watch_ext",
    infoplists = ["Info-WatchApp.plist"],
    minimum_os_version = "${MIN_OS_WATCHOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
)

watchos_extension(
    name = "watch_ext",
    bundle_id = "my.bundle.id.watch-app.watch-ext",
    entitlements = "entitlements.entitlements",
    infoplists = ["Info-WatchExt.plist"],
    minimum_os_version = "${MIN_OS_WATCHOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    deps = [":watch_lib"],
)
EOF

  cat > app/Info-WatchExt.plist <<EOF
{
  CFBundleIdentifier = "\${PRODUCT_BUNDLE_IDENTIFIER}";
  CFBundleName = "\${PRODUCT_NAME}";
  CFBundlePackageType = "XPC!";
  CFBundleShortVersionString = "1.1";
  CFBundleVersion = "1";
  NSExtension = {
    NSExtensionAttributes = {
      WKAppBundleIdentifier = "my.bundle.id.watch-app";
    };
    NSExtensionPointIdentifier = "com.apple.watchkit";
  };
}
EOF

  ! do_build watchos //app:app || fail "Should not build"
  expect_log "While processing target \"@@\?//app:watch_app\"; the CFBundleShortVersionString of the child target \"@@\?//app:watch_ext\" should be the same as its parent's version string \"1\", but found \"1.1\"."
}

# Test that a watchOS app with a different CFBundleVersion than the iOS app
# fails the build.
function test_app_with_mismatched_version_fails_to_build() {
  create_companion_app_and_watchos_application_support_files

  cat >> app/BUILD <<EOF
watchos_application(
    name = "watch_app",
    bundle_id = "my.bundle.id.watch-app",
    entitlements = "entitlements.entitlements",
    extension = ":watch_ext",
    infoplists = ["Info-WatchApp.plist"],
    minimum_os_version = "${MIN_OS_WATCHOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
)

watchos_extension(
    name = "watch_ext",
    bundle_id = "my.bundle.id.watch-app.watch-ext",
    entitlements = "entitlements.entitlements",
    infoplists = ["Info-WatchExt.plist"],
    minimum_os_version = "${MIN_OS_WATCHOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    deps = [":watch_lib"],
)
EOF

  # Give the watch app and extension a different version.
  cat > app/Info-WatchApp.plist <<EOF
{
  CFBundleIdentifier = "\${PRODUCT_BUNDLE_IDENTIFIER}";
  CFBundleName = "\${PRODUCT_NAME}";
  CFBundlePackageType = "APPL";
  CFBundleShortVersionString = "1";
  CFBundleVersion = "1.1";
  WKCompanionAppBundleIdentifier = "my.bundle.id";
  WKWatchKitApp = true;
}
EOF

  cat > app/Info-WatchExt.plist <<EOF
{
  CFBundleIdentifier = "\${PRODUCT_BUNDLE_IDENTIFIER}";
  CFBundleName = "\${PRODUCT_NAME}";
  CFBundlePackageType = "XPC!";
  CFBundleShortVersionString = "1";
  CFBundleVersion = "1.1";
  NSExtension = {
    NSExtensionAttributes = {
      WKAppBundleIdentifier = "my.bundle.id.watch-app";
    };
    NSExtensionPointIdentifier = "com.apple.watchkit";
  };
}
EOF

  ! do_build watchos //app:app || fail "Should not build"
  expect_log "While processing target \"@@\?//app:app\"; the CFBundleVersion of the child target \"@@\?//app:watch_app\" should be the same as its parent's version string \"1\", but found \"1.1\"."
}

# Test that a watchOS extension with a different CFBundleVersion than the
# watchOS app fails the build.
function test_extension_with_mismatched_version_fails_to_build() {
  create_companion_app_and_watchos_application_support_files

  cat >> app/BUILD <<EOF
watchos_application(
    name = "watch_app",
    bundle_id = "my.bundle.id.watch-app",
    entitlements = "entitlements.entitlements",
    extension = ":watch_ext",
    infoplists = ["Info-WatchApp.plist"],
    minimum_os_version = "${MIN_OS_WATCHOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
)

watchos_extension(
    name = "watch_ext",
    bundle_id = "my.bundle.id.watch-app.watch-ext",
    entitlements = "entitlements.entitlements",
    infoplists = ["Info-WatchExt.plist"],
    minimum_os_version = "${MIN_OS_WATCHOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    deps = [":watch_lib"],
)
EOF

  cat > app/Info-WatchExt.plist <<EOF
{
  CFBundleIdentifier = "\${PRODUCT_BUNDLE_IDENTIFIER}";
  CFBundleName = "\${PRODUCT_NAME}";
  CFBundlePackageType = "XPC!";
  CFBundleShortVersionString = "1";
  CFBundleVersion = "1.1";
  NSExtension = {
    NSExtensionAttributes = {
      WKAppBundleIdentifier = "my.bundle.id.watch-app";
    };
    NSExtensionPointIdentifier = "com.apple.watchkit";
  };
}
EOF

  ! do_build watchos //app:app || fail "Should not build"
  expect_log "While processing target \"@@\?//app:watch_app\"; the CFBundleVersion of the child target \"@@\?//app:watch_ext\" should be the same as its parent's version string \"1\", but found \"1.1\"."
}

# Test that a watchOS app with the wrong bundle_id for its
# WKCompanionAppBundleIdentifier fails to build.
function test_app_wrong_WKCompanionAppBundleIdentifier_fails_to_build() {
  create_companion_app_and_watchos_application_support_files

  cat >> app/BUILD <<EOF
watchos_application(
    name = "watch_app",
    bundle_id = "my.bundle.id.watch-app",
    entitlements = "entitlements.entitlements",
    extension = ":watch_ext",
    infoplists = ["Info-WatchApp.plist"],
    minimum_os_version = "${MIN_OS_WATCHOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
)

watchos_extension(
    name = "watch_ext",
    bundle_id = "my.bundle.id.watch-app.watch-ext",
    entitlements = "entitlements.entitlements",
    infoplists = ["Info-WatchExt.plist"],
    minimum_os_version = "${MIN_OS_WATCHOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    deps = [":watch_lib"],
)
EOF

  cat > app/Info-WatchApp.plist <<EOF
{
  CFBundleIdentifier = "\${PRODUCT_BUNDLE_IDENTIFIER}";
  CFBundleName = "\${PRODUCT_NAME}";
  CFBundlePackageType = "APPL";
  CFBundleShortVersionString = "1";
  CFBundleVersion = "1";
  WKCompanionAppBundleIdentifier = "my.bundle2.id";
  WKWatchKitApp = true;
}
EOF

  ! do_build watchos //app:app || fail "Should not build"
  expect_log "While processing target \"@@\?//app:app\"; the Info.plist for child target \"@@\?//app:watch_app\" has the wrong value for \"WKCompanionAppBundleIdentifier\"; expected u\?'my.bundle.id', but found 'my.bundle2.id'."
}

# Test that a watchOS extension with the wrong bundle_id for its
# WKAppBundleIdentifier fails to build.
function test_extension_wrong_WKAppBundleIdentifier_fails_to_build() {
  create_companion_app_and_watchos_application_support_files

  cat >> app/BUILD <<EOF
watchos_application(
    name = "watch_app",
    bundle_id = "my.bundle.id.watch-app",
    entitlements = "entitlements.entitlements",
    extension = ":watch_ext",
    infoplists = ["Info-WatchApp.plist"],
    minimum_os_version = "${MIN_OS_WATCHOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
)

watchos_extension(
    name = "watch_ext",
    bundle_id = "my.bundle.id.watch-app.watch-ext",
    entitlements = "entitlements.entitlements",
    infoplists = ["Info-WatchExt.plist"],
    minimum_os_version = "${MIN_OS_WATCHOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    deps = [":watch_lib"],
)
EOF

  cat > app/Info-WatchExt.plist <<EOF
{
  CFBundleIdentifier = "\${PRODUCT_BUNDLE_IDENTIFIER}";
  CFBundleName = "\${PRODUCT_NAME}";
  CFBundlePackageType = "XPC!";
  CFBundleShortVersionString = "1";
  CFBundleVersion = "1";
  NSExtension = {
    NSExtensionAttributes = {
      WKAppBundleIdentifier = "my.bundle2.id.watch-app";
    };
    NSExtensionPointIdentifier = "com.apple.watchkit";
  };
}
EOF

  ! do_build watchos //app:app || fail "Should not build"
  expect_log "While processing target \"@@\?//app:watch_app\"; the Info.plist for child target \"@@\?//app:watch_ext\" has the wrong value for \"NSExtension:NSExtensionAttributes:WKAppBundleIdentifier\"; expected u\?'my.bundle.id.watch-app', but found 'my.bundle2.id.watch-app'."
}

run_suite "watchos_application bundling tests"
