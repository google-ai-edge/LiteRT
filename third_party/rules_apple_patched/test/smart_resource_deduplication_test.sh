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

set -eu

# Integration tests for smart resource deduplication.

function set_up() {
  mkdir -p app
}

function tear_down() {
  rm -rf app
}

function create_basic_project() {
  cat > app/main.m <<EOF
int main(int argc, char **argv) {
  return 0;
}
EOF

  cat > app/Info.plist <<EOF
{
  CFBundleIdentifier = "\${PRODUCT_BUNDLE_IDENTIFIER}";
  CFBundleName = "\${PRODUCT_NAME}";
  CFBundleShortVersionString = "1.0";
  CFBundleVersion = "1.0";
}
EOF

  cat > app/FrameworkInfo.plist <<EOF
{
  CFBundleIdentifier = "\${PRODUCT_BUNDLE_IDENTIFIER}";
  CFBundleName = "\${PRODUCT_NAME}";
  CFBundlePackageType = "FMWK";
  CFBundleShortVersionString = "1.0";
  CFBundleVersion = "1.0";
}
EOF

  cat > app/app.strings <<EOF
"my_string" = "I should be at the top level!";
EOF

  cat > app/resource_only_lib.txt <<EOF
I am referenced by a resource only objc_library;
EOF

  cat > app/BUILD <<EOF
load("@build_bazel_rules_apple//apple:ios.bzl",
     "ios_application", "ios_framework")

genrule(
    name = "gen_file",
    srcs = [],
    outs = ["gen_file.txt"],
    cmd = "echo 'this is generated' > \"\$@\"",
)

objc_library(
    name = "resource_only_lib",
    data = ["resource_only_lib.txt"]
)

objc_library(
    name = "shared_lib",
    deps = [":resource_only_lib"],
    data = [
        "@build_bazel_rules_apple//test/testdata/resources:assets",
        "@build_bazel_rules_apple//test/testdata/resources:basic_bundle",
        "gen_file.txt",
        "@build_bazel_rules_apple//test/testdata/resources:nonlocalized.plist",
        "@build_bazel_rules_apple//test/testdata/resources:sample.png",
        "@build_bazel_rules_apple//test/testdata/resources:nonlocalized.strings",
    ],
)

objc_library(
    name = "shared_lib_with_no_direct_resources",
    deps = [":resource_only_lib"],
)

objc_library(
    name = "app_lib",
    srcs = ["main.m"],
    deps = [":shared_lib", ":resource_only_lib"],
    data = [
        "@build_bazel_rules_apple//test/testdata/resources:assets",
        "@build_bazel_rules_apple//test/testdata/resources:basic_bundle",
        "@build_bazel_rules_apple//test/testdata/resources:sample.png",
    ],
)

objc_library(
    name = "app_lib_with_no_direct_resources",
    srcs = ["main.m"],
    deps = [":shared_lib_with_no_direct_resources"],
)
EOF
}

function test_resources_in_app_and_framework() {
  create_basic_project

  cat >> app/BUILD <<EOF
ios_framework(
    name = "framework",
    bundle_id = "com.framework",
    families = ["iphone"],
    infoplists = ["FrameworkInfo.plist"],
    minimum_os_version = "${MIN_OS_IOS}",
    deps = [":shared_lib"],
)

ios_application(
    name = "app",
    bundle_id = "com.app",
    families = ["iphone"],
    frameworks = [":framework"],
    infoplists = ["Info.plist"],
    minimum_os_version = "${MIN_OS_IOS_NPLUS1}",
    strings = ["app.strings"],
    deps = [":app_lib"],
)
EOF

  do_build ios //app:app || fail "Should build"

  # Verify framework has resources
  assert_assets_contains "test-bin/app/app.ipa" \
      "Payload/app.app/Frameworks/framework.framework/Assets.car" "star.png"
  assert_zip_contains "test-bin/app/app.ipa" \
      "Payload/app.app/Frameworks/framework.framework/nonlocalized.plist"
  assert_zip_contains "test-bin/app/app.ipa" \
      "Payload/app.app/Frameworks/framework.framework/nonlocalized.strings"
  assert_zip_contains "test-bin/app/app.ipa" \
      "Payload/app.app/Frameworks/framework.framework/sample.png"
  assert_zip_contains "test-bin/app/app.ipa" \
      "Payload/app.app/Frameworks/framework.framework/basic.bundle/basic_bundle.txt"
  assert_zip_contains "test-bin/app/app.ipa" \
      "Payload/app.app/Frameworks/framework.framework/resource_only_lib.txt"
  assert_zip_contains "test-bin/app/app.ipa" \
      "Payload/app.app/Frameworks/framework.framework/gen_file.txt"

  # Because app_lib directly references these assets, smart dedupe ensures that
  # they are present in the same bundle as the binary that has app_lib, which
  # in this case it's app.app.
  assert_assets_contains "test-bin/app/app.ipa" \
      "Payload/app.app/Assets.car" "star.png"
  assert_zip_contains "test-bin/app/app.ipa" \
      "Payload/app.app/sample.png"
  assert_zip_contains "test-bin/app/app.ipa" \
      "Payload/app.app/basic.bundle/basic_bundle.txt"

  # These resources are not referenced by app_lib, so they should not appear in
  # the app bundle
  assert_zip_not_contains "test-bin/app/app.ipa" \
      "Payload/app.app/nonlocalized.plist"
  assert_zip_not_contains "test-bin/app/app.ipa" \
      "Payload/app.app/nonlocalized.strings"
  assert_zip_not_contains "test-bin/app/app.ipa" \
      "Payload/app.app/gen_file.txt"

  # This file is added by the top level bundling target, so it should be present
  # at the top level bundle.
  assert_zip_contains "test-bin/app/app.ipa" \
      "Payload/app.app/app.strings"

  # This resource is only depended on by the :resource_only_lib target, but that
  # target doesn't have any sources, therefore shouldn't claim ownership of the
  # resource. Without accounting for the lack of sources, this file would be
  # deduplicated as the :resource_only_lib would be the only owner _and_ present
  # in both the framework and the app. But when accounting for the lack of
  # sources in :resource_only_lib, the resource is bundled in the app as well.
  assert_zip_contains "test-bin/app/app.ipa" \
      "Payload/app.app/resource_only_lib.txt"
}

function test_shared_resource_deduplicated_when_not_referenced_by_app_only_lib() {
  create_basic_project

  cat >> app/BUILD <<EOF
ios_framework(
    name = "framework",
    bundle_id = "com.framework",
    families = ["iphone"],
    infoplists = ["FrameworkInfo.plist"],
    minimum_os_version = "${MIN_OS_IOS}",
    deps = [":shared_lib_with_no_direct_resources"],
)

ios_application(
    name = "app",
    bundle_id = "com.app",
    families = ["iphone"],
    frameworks = [":framework"],
    infoplists = ["Info.plist"],
    minimum_os_version = "${MIN_OS_IOS_NPLUS1}",
    strings = ["app.strings"],
    deps = [":app_lib_with_no_direct_resources"],
)
EOF

  do_build ios //app:app || fail "Should build"

  # This is a tricky corner case, in which a resource only lib is depended by
  # dependency chains that contain no other resources. In this very specific
  # scenario, the resource might not be correctly deduplicated.
  assert_zip_not_contains "test-bin/app/app.ipa" \
      "Payload/app.app/resource_only_lib.txt"
}

function test_resources_added_directly_into_apps() {
  create_basic_project

  cat >> app/BUILD <<EOF
ios_framework(
    name = "framework",
    bundle_id = "com.framework",
    families = ["iphone"],
    infoplists = ["FrameworkInfo.plist"],
    minimum_os_version = "${MIN_OS_IOS}",
    deps = [":shared_lib"],
)

objc_library(
    name = "only_main_lib",
    srcs = ["main.m"],
)

ios_application(
    name = "app",
    bundle_id = "com.app",
    families = ["iphone"],
    frameworks = [":framework"],
    infoplists = ["Info.plist"],
    minimum_os_version = "${MIN_OS_IOS}",
    deps = [":resource_only_lib", ":shared_lib", ":only_main_lib"],
)
EOF

  do_build ios //app:app || fail "Should build"

  # Verify that the resource is in the framework.
  assert_zip_contains "test-bin/app/app.ipa" \
      "Payload/app.app/Frameworks/framework.framework/resource_only_lib.txt"

  # Even though there is no app specific library that declares ownership of
  # this file (shared_lib is also present in the framework), this file should be
  # present in the app as it is added as a direct dependency on the application.
  assert_zip_contains "test-bin/app/app.ipa" \
      "Payload/app.app/resource_only_lib.txt"
}

run_suite "smart resource deduplication tests"
