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

# Integration tests for bundling simple iOS applications.

function set_up() {
  mkdir -p app
}

function tear_down() {
  rm -rf app
}

# Creates common source, targets, and basic plist for iOS applications.
function create_common_files() {
  cat > app/BUILD <<EOF
load("@build_bazel_rules_apple//apple:ios.bzl",
     "ios_application"
    )
load("@build_bazel_rules_apple//apple:resources.bzl",
     "apple_resource_bundle",
    )

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
  CFBundleVersion = "1.0.0";
  CFBundleShortVersionString = "1.0";
}
EOF
}

# Usage: create_minimal_ios_application [product type]
#
# Creates a minimal iOS application target. The optional product type is
# the Starlark constant that should be set on the application using the
# `product_type` attribute.
function create_minimal_ios_application() {
  if [[ ! -f app/BUILD ]]; then
    fail "create_common_files must be called first."
  fi

  product_type="${1:-}"

  cat >> app/BUILD <<EOF
ios_application(
    name = "app",
    bundle_id = "my.bundle.id",
    families = ["iphone"],
    infoplists = ["Info.plist"],
    minimum_os_version = "${MIN_OS_IOS}",
EOF

  if [[ -n "$product_type" ]]; then
  cat >> app/BUILD <<EOF
    product_type = $product_type,
EOF
  fi

  cat >> app/BUILD <<EOF
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    deps = [":lib"],
)
EOF
}

# Test missing the CFBundleVersion fails the build.
function test_missing_version_fails() {
  create_common_files
  create_minimal_ios_application

  # Replace the file, but without CFBundleVersion.
  cat > app/Info.plist <<EOF
{
  CFBundleIdentifier = "\${PRODUCT_BUNDLE_IDENTIFIER}";
  CFBundleName = "\${PRODUCT_NAME}";
  CFBundlePackageType = "APPL";
  CFBundleShortVersionString = "1.0";
}
EOF

  ! do_build ios //app:app \
    || fail "Should fail build"

  expect_log 'Target "@@\?//app:app" is missing CFBundleVersion.'
}

# Test missing the CFBundleShortVersionString fails the build.
function test_missing_short_version_fails() {
  create_common_files
  create_minimal_ios_application

  # Replace the file, but without CFBundleShortVersionString.
  cat > app/Info.plist <<EOF
{
  CFBundleIdentifier = "\${PRODUCT_BUNDLE_IDENTIFIER}";
  CFBundleName = "\${PRODUCT_NAME}";
  CFBundlePackageType = "APPL";
  CFBundleVersion = "1.0.0";
}
EOF

  ! do_build ios //app:app \
    || fail "Should fail build"

  expect_log 'Target "@@\?//app:app" is missing CFBundleShortVersionString.'
}

# Tests that the IPA post-processor is executed and can modify the bundle.
function test_ipa_post_processor() {
  create_common_files

  cat >> app/BUILD <<EOF
ios_application(
    name = "app",
    bundle_id = "my.bundle.id",
    families = ["iphone"],
    infoplists = ["Info.plist"],
    ipa_post_processor = "post_processor.sh",
    minimum_os_version = "${MIN_OS_IOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    deps = [":lib"],
)
EOF

  cat > app/post_processor.sh <<EOF
#!/bin/bash
WORKDIR="\$1"
echo "foo" > "\$WORKDIR/Payload/app.app/inserted_by_post_processor.txt"
EOF
  chmod +x app/post_processor.sh

  do_build ios //app:app || fail "Should build"
  assert_equals "foo" "$(unzip_single_file "test-bin/app/app.ipa" \
      "Payload/app.app/inserted_by_post_processor.txt")"
}

# Tests that linkopts get passed to the underlying apple_binary target.
function test_linkopts_passed_to_binary() {
  create_common_files

  cat >> app/BUILD <<EOF
ios_application(
    name = "app",
    bundle_id = "my.bundle.id",
    families = ["iphone"],
    infoplists = ["Info.plist"],
    linkopts = ["-alias", "_main", "_linkopts_test_main", "-u", "_linkopts_test_main"],
    minimum_os_version = "${MIN_OS_IOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    deps = [":lib"],
)
EOF

  do_build ios //app:app || fail "Should build"

  unzip_single_file "test-bin/app/app.ipa" "Payload/app.app/app" |
      nm -j - | grep _linkopts_test_main  > /dev/null \
      || fail "Could not find -alias symbol in binary; " \
              "linkopts may have not propagated"
}

# Tests additional_linker_inputs and $(location) expansion in linker argument.
function test_additional_linker_inputs_expansion() {
  create_common_files

  cat >> app/BUILD <<EOF
genrule(name = "linker_input", cmd="touch \$@", outs=["a.lds"])

ios_application(
    name = "app",
    additional_linker_inputs = [":linker_input"],
    bundle_id = "my.bundle.id",
    families = ["iphone"],
    infoplists = ["Info.plist"],
EOF

  # Using `<<'EOF'` to suppress variable expansion.
  cat >> app/BUILD <<'EOF'
    linkopts = ["-order_file", "$(location :linker_input)"],
EOF

  cat >> app/BUILD <<EOF
    minimum_os_version = "${MIN_OS_IOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    deps = [":lib"],
)
EOF

  do_build ios //app:app || fail "Should build"
}

# Tests that the PkgInfo file exists in the bundle and has the expected
# content.
function test_pkginfo_contents() {
  create_common_files
  create_minimal_ios_application
  do_build ios //app:app || fail "Should build"

  assert_equals "APPL????" "$(unzip_single_file "test-bin/app/app.ipa" \
      "Payload/app.app/PkgInfo")"
}

# Helper to test different values if a build adds the debugger entitlement.
# First arg is "y|n" if provisioning profile should contain debugger entitlement
# Second arg is "y|n" if debugger entitlement should be contained on signed app
# Third arg is "y|n" if `_include_debug_entitlements` is `True` (mainly `--define=apple.add_debugger_entitlement=yes`)
# Any other args are passed to `do_build`.
function verify_debugger_entitlements_with_params() {
  readonly INCLUDE_DEBUGGER=$1; shift
  readonly SHOULD_CONTAIN=$1; shift
  readonly FORCED_DEBUGGER=$1; shift

  create_common_files

  cp $(rlocation rules_apple/test/testdata/provisioning/integration_testing_ios.mobileprovision) \
    app/profile.mobileprovision
  if [[ "${INCLUDE_DEBUGGER}" == "n" ]]; then
    sed -i'.original' -e '/get-task-allow/,+1 d' app/profile.mobileprovision
  fi

  cat >> app/BUILD <<EOF
ios_application(
    name = "app",
    bundle_id = "my.bundle.id",
    entitlements = "entitlements.plist",
    families = ["iphone"],
    infoplists = ["Info.plist"],
    minimum_os_version = "${MIN_OS_IOS}",
    provisioning_profile = "profile.mobileprovision",
    deps = [":lib"],
)
EOF

  # Use a local entitlements file so the default isn't extracted from the
  # provisioning profile (which likely has get-task-allow).
  cat > app/entitlements.plist <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>keychain-access-groups</key>
  <array>
    <string>$(AppIdentifierPrefix)$(CFBundleIdentifier)</string>
  </array>
</dict>
</plist>
EOF

  create_dump_codesign "//app:app" "Payload/app.app" -d --entitlements :-
  readonly CODESIGN_OUTPUT="test-bin/app/codesign_output"

  if is_device_build ios ; then
    # For device builds, entitlements are in the codesign output.
    do_build ios "$@" //app:dump_codesign || fail "Should build"

    readonly FILE_TO_CHECK="${CODESIGN_OUTPUT}"
  else
    # For simulator builds, entitlements are added as a Mach-O section in
    # the binary.
    do_build ios "$@" //app:app || fail "Should build"
    unzip_single_file "test-bin/app/app.ipa" "Payload/app.app/app" > "${TEST_TMPDIR}/binary"
    print_debug_entitlements "${TEST_TMPDIR}/binary" "${TEST_TMPDIR}/dumped_entitlements"

    readonly FILE_TO_CHECK="${TEST_TMPDIR}/dumped_entitlements"

    # Simulator builds also have entitlements in the codesign output,
    # but only `com.apple.security.get-task-allow` and nothing else
    do_build ios "$@" //app:dump_codesign || fail "Should build"

    if [[ "${FORCED_DEBUGGER}" == "y" ]] ; then
      assert_contains "<key>com.apple.security.get-task-allow</key>" "${CODESIGN_OUTPUT}"
      assert_not_contains "<key>keychain-access-groups</key>" "${CODESIGN_OUTPUT}"
    else
      assert_not_contains "<key>com.apple.security.get-task-allow</key>" "${CODESIGN_OUTPUT}"
      assert_not_contains "<key>keychain-access-groups</key>" "${CODESIGN_OUTPUT}"
    fi
  fi

  if [[ "${SHOULD_CONTAIN}" == "y" ]] ; then
    assert_contains "<key>get-task-allow</key>" "${FILE_TO_CHECK}"
    assert_contains "<key>keychain-access-groups</key>" "${FILE_TO_CHECK}"
  else
    assert_not_contains "<key>get-task-allow</key>" "${FILE_TO_CHECK}"
    assert_contains "<key>keychain-access-groups</key>" "${FILE_TO_CHECK}"
  fi
}

# Tests that debugger entitlement is not auto-added to the application correctly
# if it's not included on provisioning profile.
function test_debugger_entitlements_default() {
  # For default builds, configuration.bzl also forces -c opt, so there will be
  #   no debug entitlements.
  verify_debugger_entitlements_with_params n n n
}

# Tests that debugger entitlement is auto-added to the application correctly
# if it's included on provisioning profile.
function test_debugger_entitlements_from_provisioning_profile() {
  verify_debugger_entitlements_with_params y y n
}

# Test the different values for apple.add_debugger_entitlement.
function test_debugger_entitlements_forced_false() {
  verify_debugger_entitlements_with_params n n n \
      --define=apple.add_debugger_entitlement=false
}
function test_debugger_entitlements_forced_no() {
  verify_debugger_entitlements_with_params n n n \
      --define=apple.add_debugger_entitlement=no
}
function test_debugger_entitlements_forced_yes() {
  verify_debugger_entitlements_with_params n y y \
      --define=apple.add_debugger_entitlement=YES
}
function test_debugger_entitlements_forced_true() {
  verify_debugger_entitlements_with_params n y y \
      --define=apple.add_debugger_entitlement=True
}

# Tests that the target name is sanitized before it is used as the symbol name
# for embedded debug entitlements.
function test_target_name_sanitized_for_entitlements() {
  create_common_files

  cat >> app/BUILD <<EOF
ios_application(
    name = "app-with-hyphen",
    bundle_id = "my.bundle.id",
    entitlements = "entitlements.plist",
    families = ["iphone"],
    infoplists = ["Info.plist"],
    minimum_os_version = "${MIN_OS_IOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    deps = [":lib"],
)
EOF

  cat > app/entitlements.plist <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>test-an-entitlement</key>
  <false/>
</dict>
</plist>
EOF

  if ! is_device_build ios ; then
    do_build ios //app:app-with-hyphen || fail "Should build"

    unzip_single_file "test-bin/app/app-with-hyphen.ipa" \
        "Payload/app-with-hyphen.app/app-with-hyphen" > "${TEST_TMPDIR}/binary"
    print_debug_entitlements "${TEST_TMPDIR}/binary" "${TEST_TMPDIR}/dumped_entitlements"
    grep -sq "<key>test-an-entitlement</key>" "${TEST_TMPDIR}/dumped_entitlements" || \
        fail "Failed to find custom entitlement"
  fi
}

# Tests that failures to extract from a provisioning profile are propertly
# reported.
function test_provisioning_profile_extraction_failure() {
  create_common_files

  cat >> app/BUILD <<EOF
ios_application(
    name = "app",
    bundle_id = "my.bundle.id",
    families = ["iphone"],
    infoplists = ["Info.plist"],
    minimum_os_version = "${MIN_OS_IOS}",
    provisioning_profile = "bogus.mobileprovision",
    deps = [":lib"],
)
EOF

  cat > app/bogus.mobileprovision <<EOF
BOGUS BOGUS BOGUS BOGUS BOGUS BOGUS BOGUS BOGUS BOGUS BOGUS BOGUS BOGUS BOGUS
EOF

  ! do_build ios //app:app || fail "Should fail"
  # The fact that multiple things are tried is left as an impl detail and
  # only the final message is looked for.
  expect_log 'While processing target "@@\?//app:app", failed to extract from the provisioning profile "app/bogus.mobileprovision".'
}

# Tests that applications can transitively depend on apple_resource_bundle, and
# that the bundle library resources for the appropriate architecture are
# used in a multi-arch build.
function test_bundle_library_dependency() {
  create_common_files

  cat >> app/BUILD <<EOF
ios_application(
    name = "app",
    bundle_id = "my.bundle.id",
    families = ["iphone"],
    infoplists = ["Info.plist"],
    minimum_os_version = "${MIN_OS_IOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    deps = [
        ":lib",
        ":resLib",
    ],
)

objc_library(
    name = "resLib",
    data = [":appResources"],
)

apple_resource_bundle(
    name = "appResources",
    resources = select({
        "@build_bazel_rules_apple//apple:ios_x86_64": ["foo_sim.txt"],
        "@build_bazel_rules_apple//apple:ios_arm64": ["foo_device.txt"],
        "@build_bazel_rules_apple//apple:ios_arm64e": ["foo_device.txt"],
        "@build_bazel_rules_apple//apple:ios_sim_arm64": ["foo_sim.txt"],
    }),
)
EOF

  cat > app/foo_sim.txt <<EOF
foo_sim
EOF
  cat > app/foo_device.txt <<EOF
foo_device
EOF

  do_build ios //app:app || fail "Should build"

  if is_device_build ios ; then
    assert_zip_contains "test-bin/app/app.ipa" \
        "Payload/app.app/appResources.bundle/foo_device.txt"
  else
    assert_zip_contains "test-bin/app/app.ipa" \
        "Payload/app.app/appResources.bundle/foo_sim.txt"
  fi
}

# Tests that the bundle name can be overridden to differ from the target name.
function test_bundle_name_can_differ_from_target() {
  create_common_files

  cat >> app/BUILD <<EOF
ios_application(
    name = "app",
    bundle_id = "my.bundle.id",
    bundle_name = "different",
    families = ["iphone"],
    infoplists = ["Info.plist"],
    minimum_os_version = "${MIN_OS_IOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    deps = [":lib"],
)
EOF

  do_build ios //app:app || fail "Should build"

  # Both the bundle name and the executable name should correspond to
  # bundle_name.
  assert_zip_contains "test-bin/app/app.ipa" "Payload/different.app/"
  assert_zip_contains "test-bin/app/app.ipa" "Payload/different.app/different"
}

# Tests that the label passed to the version attribute overwrites the version
# information already in the plist without error.
function test_version_attr_overrides_plist_contents() {
  create_common_files

  cat >> app/BUILD <<EOF
load("@build_bazel_rules_apple//apple:versioning.bzl",
     "apple_bundle_version",
    )

ios_application(
    name = "app",
    bundle_id = "my.bundle.id",
    families = ["iphone"],
    infoplists = ["Info.plist"],
    minimum_os_version = "${MIN_OS_IOS}",
    provisioning_profile = "@build_bazel_rules_apple//test/testdata/provisioning:integration_testing_ios.mobileprovision",
    version = ":app_version",
    deps = [":lib"],
)

apple_bundle_version(
    name = "app_version",
    build_version = "9.8.7",
    short_version_string = "6.5",
)
EOF

  create_dump_plist "//app:app" "Payload/app.app/Info.plist" \
      CFBundleVersion \
      CFBundleShortVersionString
  do_build ios //app:dump_plist || fail "Should build"

  assert_equals "9.8.7" "$(cat "test-bin/app/CFBundleVersion")"
  assert_equals "6.5" "$(cat "test-bin/app/CFBundleShortVersionString")"
}

# Tests tree artifacts builds and disable codesigning for simulator play along.
function test_tree_artifacts_and_disable_simulator_codesigning() {
  create_common_files
  create_minimal_ios_application
  do_build ios //app:app \
      --define=apple.experimental.tree_artifact_outputs=yes \
      --features=apple.skip_codesign_simulator_bundles || fail "Should build"
}

run_suite "ios_application bundling tests"
