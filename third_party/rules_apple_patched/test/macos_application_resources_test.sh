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

# Integration tests for bundling simple macOS applications with resources.

function set_up() {
  mkdir -p app
}

function tear_down() {
  rm -rf app
}

# Creates common source, targets, and basic plist for macOS applications.
function create_common_files() {
  cat > app/BUILD <<EOF
load("@build_bazel_rules_apple//apple:macos.bzl", "macos_application")
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

macos_application(
    name = "app",
    bundle_id = "my.bundle.id",
    infoplists = ["Info.plist"],
    minimum_os_version = "${MIN_OS_MACOS}",
    deps = [":lib", ":resources"],
)
EOF
}

# Should generate a warning because 'sw'/Swahili doesn't match anything, but
# that things were filtered, so it could have been a typo.
function test_localized_unprocessed_resources_filter_all() {
  create_with_localized_unprocessed_resources

  do_build macos //app:app --define "apple.locales_to_include=sw" \
      || fail "Should build"
  expect_log_once "Please verify apple.locales_to_include is defined properly"
  expect_log_once "\[\"sw\"\]"
  assert_zip_not_contains "test-bin/app/app.zip" \
      "app.app/Contents/Resources/it.lproj/localized.txt"
}

# Should not generate a warning because although 'fr' doesn't match anything
# nothing was filtered away (i.e. - no harm if it was a typo).
function test_localized_unprocessed_resources_filter_mixed() {
  create_with_localized_unprocessed_resources

  do_build macos //app:app --define "apple.locales_to_include=fr,it" \
      || fail "Should build"
  expect_not_log "Please verify apple.locales_to_include is defined properly"
  assert_zip_contains "test-bin/app/app.zip" \
      "app.app/Contents/Resources/it.lproj/localized.txt"
}

# TODO: Something like the ios_application_resources_test.sh's
# test_app_icons_and_launch_images, but for the relevant bits for macOS.

# TODO: Something like the ios_application_resources_test.sh's
# test_launch_storyboard.

# Tests that the localizations from the base of the Resource folder are used to
# strip subfolder localizations with apple.trim_lproj_locales=1.
function test_bundle_localization_strip() {
  create_common_files

  mkdir -p app/fr.lproj
  touch app/fr.lproj/localized.strings

  cat >> app/BUILD <<EOF
objc_library(
    name = "resources",
    data = [
        "@build_bazel_rules_apple//test/testdata/resources:bundle_library_macos",
        "fr.lproj/localized.strings",
    ],
)

macos_application(
    name = "app",
    bundle_id = "my.bundle.id",
    infoplists = ["Info.plist"],
    minimum_os_version = "${MIN_OS_MACOS}",
    deps = [":lib", ":resources"],
)
EOF

  do_build macos //app:app --define "apple.trim_lproj_locales=1" \
      || fail "Should build"

  # Verify the app has a `fr` localization and not an `it` localization.
  assert_zip_contains "test-bin/app/app.zip" \
      "app.app/Contents/Resources/fr.lproj/localized.strings"
  assert_zip_not_contains "test-bin/app/app.zip" \
      "app.app/Contents/Resources/it.lproj/localized.strings"

  # Verify the `it` localization from the bundle is removed.
  assert_zip_not_contains "test-bin/app/app.zip" \
      "app.app/Contents/Resources/bundle_library_macos.bundle/it.lproj/localized.strings"
  assert_zip_contains "test-bin/app/app.zip" \
      "app.app/Contents/Resources/bundle_library_macos.bundle/fr.lproj/localized.strings"
}

# Tests that the localizations that are explictly excluded via
# --define "apple.locales_to_exclude=fr" are not included in the output bundle.
function test_bundle_localization_excludes() {
  create_common_files

  mkdir -p app/fr.lproj
  touch app/fr.lproj/localized.strings

  mkdir -p app/it.lproj
  touch app/it.lproj/localized.strings

  cat >> app/BUILD <<EOF
objc_library(
    name = "resources",
    data = [
        "fr.lproj/localized.strings",
        "it.lproj/localized.strings",
    ],
)

macos_application(
    name = "app",
    bundle_id = "my.bundle.id",
    infoplists = ["Info.plist"],
    minimum_os_version = "${MIN_OS_MACOS}",
    deps = [":lib", ":resources"],
)
EOF

  do_build macos //app:app --define "apple.locales_to_exclude=fr" \
      || fail "Should build"

  # Verify the app has a `it` localization and not an `fr` localization.
  assert_zip_not_contains "test-bin/app/app.zip" \
      "app.app/Contents/Resources/fr.lproj/localized.strings"
  assert_zip_contains "test-bin/app/app.zip" \
      "app.app/Contents/Resources/it.lproj/localized.strings"
}

# Tests that the localizations that are explictly excluded via
# --define "apple.locales_to_exclude=fr" overrides the ones explicitly included
# via "apple.locales_to_include" and are not included in the output bundle.
function test_bundle_localization_excludes_includes_conflict() {
  create_common_files

  mkdir -p app/fr.lproj
  touch app/fr.lproj/localized.strings

  mkdir -p app/it.lproj
  touch app/it.lproj/localized.strings

  cat >> app/BUILD <<EOF
objc_library(
    name = "resources",
    data = [
        "fr.lproj/localized.strings",
        "it.lproj/localized.strings",
    ],
)

macos_application(
    name = "app",
    bundle_id = "my.bundle.id",
    infoplists = ["Info.plist"],
    minimum_os_version = "${MIN_OS_MACOS}",
    deps = [":lib", ":resources"],
)
EOF

  ! do_build macos //app:app --define "apple.locales_to_exclude=fr" --define "apple.locales_to_include=fr,it" \
      || fail "Should fail build"
  error_message="@\/\/app:app dropping \[\"fr\"\] as they are explicitly excluded but also explicitly included. \
Please verify apple.locales_to_include and apple.locales_to_exclude are defined properly."
  expect_log "$error_message"
}


run_suite "macos_application bundling with resources tests"
