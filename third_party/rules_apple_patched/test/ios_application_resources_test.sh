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

# Integration tests for bundling simple iOS applications with resources.

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
load("@build_bazel_rules_apple//apple:resources.bzl", "apple_resource_group")

objc_library(
    name = "lib",
    srcs = ["main.m"],
)
EOF

  cat > app/main.m <<EOF
int main(int argc, char **argv) {
  return 0;
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

function create_with_localized_unprocessed_resources() {
  create_common_files

  cat >> app/BUILD <<EOF
objc_library(
    name = "resources",
    data = [
        "@build_bazel_rules_apple//test/testdata/resources:localized_generic_resources"
    ],
)

ios_application(
    name = "app",
    bundle_id = "my.bundle.id",
    families = ["iphone"],
    infoplists = ["Info.plist"],
    minimum_os_version = "${MIN_OS_IOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    deps = [":lib", ":resources"],
)
EOF

}

# Tests that generic flattened but unprocessed resources are bundled correctly
# (preserving their .lproj directory). Structured resources do not apply here,
# because they are never treated as localizable.
function test_localized_unprocessed_resources() {
  create_with_localized_unprocessed_resources

  # Basic build, no filter
  do_build ios //app:app || fail "Should build"
  expect_not_log "Please verify apple.locales_to_include is defined properly"
  assert_zip_contains "test-bin/app/app.ipa" \
      "Payload/app.app/it.lproj/localized.txt"
}

# Should generate a warning because 'sw' doesn't match anything, but things
# were filtered, so it could have been a typo.
function test_localized_unprocessed_resources_filter_all() {
  create_with_localized_unprocessed_resources

  do_build ios //app:app --define "apple.locales_to_include=sw" || \
      fail "Should build"
  expect_log_once "Please verify apple.locales_to_include is defined properly"
  expect_log_once "\[\"sw\"\]"
  assert_zip_not_contains "test-bin/app/app.ipa" \
      "Payload/app.app/it.lproj/localized.txt"
}

# Should not generate a warning because although 'fr' doesn't match anything
# nothing was filtered away (i.e. - no harm if it was a typo).
function test_localized_unprocessed_resources_filter_mixed() {
  create_with_localized_unprocessed_resources

  do_build ios //app:app --define "apple.locales_to_include=fr,it" \
      || fail "Should build"
  expect_not_log "Please verify apple.locales_to_include is defined properly"
  assert_zip_contains "test-bin/app/app.ipa" \
      "Payload/app.app/it.lproj/localized.txt"
}

function test_localized_unprocessed_resources_filter_with_attribute() {
  create_common_files
  cat >> app/BUILD <<EOF
objc_library(
    name = "resources",
    data = [
        "@build_bazel_rules_apple//test/testdata/resources:localized_generic_resources"
    ],
)

ios_application(
    name = "app",
    bundle_id = "my.bundle.id",
    families = ["iphone"],
    infoplists = ["Info.plist"],
    locales_to_include = ["it"],
    minimum_os_version = "${MIN_OS_IOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    deps = [":lib", ":resources"],
)
EOF

  do_build ios //app:app \
      || fail "Should build"
  expect_not_log "Please verify apple.locales_to_include is defined properly"
  assert_zip_contains "test-bin/app/app.ipa" \
      "Payload/app.app/it.lproj/localized.txt"
  assert_zip_not_contains "test-bin/app/app.ipa" \
      "Payload/app.app/fr.lproj/localized.txt"
}

# Tests that the localizations in the Settings.bundle that are not in the base
# of the app are not included in the output when apple.trim_lproj_locales=1.
function test_settings_bundle_localization_strip() {
  create_common_files

  cat >> app/BUILD <<EOF
ios_application(
    name = "app",
    bundle_id = "my.bundle.id",
    families = ["iphone"],
    infoplists = ["Info.plist"],
    minimum_os_version = "${MIN_OS_IOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    settings_bundle = "@build_bazel_rules_apple//test/testdata/resources:settings_bundle_ios",
    deps = [":lib"],
)
EOF

  do_build ios //app:app --define "apple.trim_lproj_locales=1" \
      || fail "Should build"
  assert_zip_not_contains "test-bin/app/app.ipa" \
      "Payload/app.app/Settings.bundle/it.lproj/Root.strings"
  assert_zip_not_contains "test-bin/app/app.ipa" \
      "Payload/app.app/Settings.bundle/fr.lproj/Root.strings"
  assert_zip_contains "test-bin/app/app.ipa" \
      "Payload/app.app/Settings.bundle/Base.lproj/Root.strings"
}

function test_different_files_mapped_to_the_same_target_path_fails() {
  create_common_files
  cat >> app/BUILD <<EOF
objc_library(
    name = "shared_lib",
    data = [
      "shared_res/foo.txt",
    ],
)
objc_library(
    name = "app_lib",
    deps = [":lib", ":shared_lib"],
    data = [
      "app_res/foo.txt",
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

  mkdir -p app/app_res
  mkdir -p app/shared_res
  echo app_res > app/app_res/foo.txt
  echo shared_res > app/shared_res/foo.txt

  do_build ios //app:app && fail "Should fail"

  expect_log "Multiple files would be placed at \".*foo.txt\" in the bundle, which is not allowed"

}

# Tests that the bundled application contains the compiled texture atlas.
function test_texture_atlas_bundled_with_app() {
  create_common_files

  cat > app/BUILD <<EOF
load("@build_bazel_rules_apple//apple:ios.bzl", "ios_application")

objc_library(
    name = "lib",
    srcs = ["main.m"],
    data = [
        "@build_bazel_rules_apple//test/testdata/resources:star_atlas_files",
    ],
)

ios_application(
    name = "app",
    bundle_id = "my.bundle.id",
    families = ["iphone"],
    infoplists = ["Info.plist"],
    minimum_os_version = "${MIN_OS_IOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    deps = [":lib"],
)
EOF

  do_build ios //app:app || fail "Should build"

  assert_zip_contains "test-bin/app/app.ipa" \
      "Payload/app.app/star.atlasc/star.1.png"
  assert_zip_contains "test-bin/app/app.ipa" \
      "Payload/app.app/star.atlasc/star.plist"

  # Make sure those were the only files in the .atlasc.
  atlasc_count="$(zipinfo -1 "test-bin/app/app.ipa" | \
      grep "^Payload/app\.app/star\.atlasc/..*$" | wc -l | tr -d ' ')"
  assert_equals "2" "$atlasc_count"
}

# Verify that the warning about 76x76 icons doesn't get displayed.
function test_actool_hides_warning_about_76x76_icons() {
  create_common_files

  cat > app/BUILD <<EOF
load("@build_bazel_rules_apple//apple:ios.bzl", "ios_application")

objc_library(
    name = "lib",
    srcs = ["main.m"],
    data = [
        "@build_bazel_rules_apple//test/testdata/resources:app_icons_ios",
    ],
)

ios_application(
    name = "app",
    bundle_id = "my.bundle.id",
    families = ["ipad"],
    infoplists = ["Info.plist"],
    minimum_os_version = "${MIN_OS_IOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    deps = [":lib"],
)
EOF

  do_build ios //app:app || fail "Should build"

  assert_zip_contains "test-bin/app/app.ipa" \
      "Payload/app.app/Assets.car"
  assert_zip_contains "test-bin/app/app.ipa" \
      "Payload/app.app/app_icon76x76@2x~ipad.png"
  expect_not_log "76x76"
}

# Verify that we warn with unassigned children in an asset
function test_actool_errors_with_unassigned_children_in_asset() {
  create_common_files

  cat > app/BUILD <<EOF
load("@build_bazel_rules_apple//apple:ios.bzl", "ios_application")

objc_library(
    name = "lib",
    srcs = ["main.m"],
    data = [
        "@build_bazel_rules_apple//test/testdata/resources:imageset_with_unassigned_child",
    ],
)

ios_application(
    name = "app",
    bundle_id = "my.bundle.id",
    families = ["iphone"],
    infoplists = ["Info.plist"],
    minimum_os_version = "${MIN_OS_IOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    deps = [":lib"],
)
EOF

  do_build ios //app:app && fail "Should fail"

  expect_log "The image set \"star_universal\" has an unassigned child"
}

# Verify that we error with invalid json in an asset
function test_actool_errors_with_invalid_json_in_asset() {
  create_common_files

  cat > app/BUILD <<EOF
load("@build_bazel_rules_apple//apple:ios.bzl", "ios_application")

objc_library(
    name = "lib",
    srcs = ["main.m"],
    data = [
        "@build_bazel_rules_apple//test/testdata/resources:imageset_with_invalid_json",
    ],
)

ios_application(
    name = "app",
    bundle_id = "my.bundle.id",
    families = ["iphone"],
    infoplists = ["Info.plist"],
    minimum_os_version = "${MIN_OS_IOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    deps = [":lib"],
)
EOF

  do_build ios //app:app && fail "Should fail"

  expect_log "The Contents.json describing the \"star_universal.imageset\" is not valid JSON."
}
# Verify that we error with missing children in an asset
function test_actool_errors_with_missing_children_in_asset() {
  create_common_files

  cat > app/BUILD <<EOF
load("@build_bazel_rules_apple//apple:ios.bzl", "ios_application")

objc_library(
    name = "lib",
    srcs = ["main.m"],
    data = [
        "@build_bazel_rules_apple//test/testdata/resources:imageset_missing_child",
    ],
)

ios_application(
    name = "app",
    bundle_id = "my.bundle.id",
    families = ["iphone"],
    infoplists = ["Info.plist"],
    minimum_os_version = "${MIN_OS_IOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    deps = [":lib"],
)
EOF

  do_build ios //app:app && fail "Should fail"

  expect_log "error: The file \"star.png\" for the image set \"star_universal\" does not exist."
}

# Verify that we error with badly sized assets in icons.
function test_actool_errors_with_badly_sized_children_in_icon() {
  create_common_files

  cat > app/BUILD <<EOF
load("@build_bazel_rules_apple//apple:ios.bzl", "ios_application")

objc_library(
    name = "lib",
    srcs = ["main.m"],
    data = [
        "@build_bazel_rules_apple//test/testdata/resources:app_icons_ios_with_bad_size",
    ],
)

ios_application(
    name = "app",
    bundle_id = "my.bundle.id",
    families = ["iphone"],
    infoplists = ["Info.plist"],
    minimum_os_version = "${MIN_OS_IOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    deps = [":lib"],
)
EOF

  do_build ios //app:app && fail "Should fail"

  expect_log "error: app_icon.appiconset/app_icon_40pt_3x.png is 120x121 but should be 120x120."
}

# Tests that the localizations that are explictly excluded via
# --define "apple.locales_to_exclude=fr" are not included in the output bundle.
function test_localization_excludes() {
  create_common_files

  cat > app/BUILD <<EOF
load("@build_bazel_rules_apple//apple:ios.bzl", "ios_application")

objc_library(
    name = "lib",
    srcs = ["main.m"],
    data = [
        "@build_bazel_rules_apple//test/testdata/resources:localized_strings",
    ],
)

ios_application(
    name = "app",
    bundle_id = "my.bundle.id",
    families = ["iphone"],
    infoplists = ["Info.plist"],
    minimum_os_version = "${MIN_OS_IOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    deps = [":lib"],
)
EOF

  do_build ios //app:app --define "apple.locales_to_exclude=fr" \
      || fail "Should build"
  assert_zip_contains "test-bin/app/app.ipa" \
      "Payload/app.app/it.lproj/localized.strings"
  assert_zip_not_contains "test-bin/app/app.ipa" \
      "Payload/app.app/fr.lproj/localized.strings"
}

# Tests that the localizations that are explictly excluded via
# --define "apple.locales_to_exclude=fr" overrides the ones explicitly included via "apple.locales_to_include" and are not included in the output bundle.
function test_localization_excludes_includes_conflict() {
  create_common_files

  cat > app/BUILD <<EOF
load("@build_bazel_rules_apple//apple:ios.bzl", "ios_application")

objc_library(
    name = "lib",
    srcs = ["main.m"],
    data = [
        "@build_bazel_rules_apple//test/testdata/resources:localized_strings",
    ],
)

ios_application(
    name = "app",
    bundle_id = "my.bundle.id",
    families = ["iphone"],
    infoplists = ["Info.plist"],
    minimum_os_version = "${MIN_OS_IOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    deps = [":lib"],
)
EOF

  ! do_build ios //app:app --define "apple.locales_to_exclude=fr" --define "apple.locales_to_include=fr,it" \
      || fail "Should fail build"
  error_message="@\/\/app:app dropping \[\"fr\"\] as they are explicitly excluded but also explicitly included. \
Please verify apple.locales_to_include and apple.locales_to_exclude are defined properly."
  expect_log "$error_message"
}

# Tests that the bundled application contains nested resource in private_deps(swift_library) or implementation_deps(objc_library)
function test_nested_private_andimplementation_deps_bundled_with_app() {
  create_common_files
  touch "app/dummy.swift"

  cat > app/BUILD <<EOF
load("@build_bazel_rules_apple//apple:ios.bzl", "ios_application")
load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

objc_library(
    name = "res1",
    data = [
        "@build_bazel_rules_apple//test/testdata/resources:sample.png",
    ],
)

objc_library(
    name = "res2",
    data = [
        "@build_bazel_rules_apple//test/testdata/resources:view_ios.xib",
    ],
    deps = [":res1"],
)

swift_library(
    name = "swift_lib",
    srcs = ["dummy.swift"],
    private_deps = [":res2"],
)

objc_library(
    name = "lib",
    srcs = ["main.m"],
    implementation_deps = [":swift_lib"],
)

ios_application(
    name = "app",
    bundle_id = "my.bundle.id",
    families = ["iphone"],
    infoplists = ["Info.plist"],
    minimum_os_version = "${MIN_OS_IOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    deps = [":lib"],
)
EOF

  do_build ios //app:app || fail "Should build"

  assert_zip_contains "test-bin/app/app.ipa" \
      "Payload/app.app/sample.png"
  assert_zip_contains "test-bin/app/app.ipa" \
      "Payload/app.app/view_ios.nib"
}

run_suite "ios_application bundling with resources tests"
