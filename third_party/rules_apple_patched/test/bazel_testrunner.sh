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

# Test runner that sets up the environment for Apple shell integration tests.
#
# Use the `apple_shell_test` rule in //test:test_rules.bzl to spawn this.
#
# Usage:
#   bazel_testrunner.sh <test_script>
#
# test_script: The name of the test script to execute inside the test
#     directory.

# --- begin runfiles.bash initialization v3 ---
# Copy-pasted from the Bazel Bash runfiles library v3.
set -uo pipefail; f=bazel_tools/tools/bash/runfiles/runfiles.bash
source "${RUNFILES_DIR:-/dev/null}/$f" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "${RUNFILES_MANIFEST_FILE:-/dev/null}" | cut -f2- -d' ')" 2>/dev/null || \
  source "$0.runfiles/$f" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "$0.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "$0.exe.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
  { echo>&2 "ERROR: cannot find $f"; exit 1; }; f=;
# --- end runfiles.bash initialization v3 ---

test_script="$1"; shift

# Use the image's default Xcode version when running tests to avoid flakes
# because of multiple Xcodes running at the same time.
export XCODE_VERSION_FOR_TESTS="$(xcodebuild -version | sed -nE 's/Xcode (.+)/\1/p')"

function print_message_and_exit() {
  echo "$1" >&2; exit 1;
}

# Location of the external dependencies linked to the test through @workspace
# links.
EXTERNAL_DIR="$(pwd)/external"

CURRENT_SCRIPT="${BASH_SOURCE[0]}"
# Go to the directory where the script is running
cd "$(dirname ${CURRENT_SCRIPT})" \
  || print_message_and_exit "Unable to access "$(dirname ${CURRENT_SCRIPT})""

DIR=$(pwd)
# Load the unit test framework
source "$DIR/unittest.bash" || print_message_and_exit "unittest.bash not found!"

function resolve_external_repository() {
  dirname "$(perl -MCwd -e 'print Cwd::abs_path shift' "$(rlocation "$1/BUILD")")"
}

# Load the test environment
function create_new_workspace() {
  new_workspace_dir="${1:-$(mktemp -d ${TEST_TMPDIR}/workspace.XXXXXXXX)}"
  rm -fr "${new_workspace_dir}"
  mkdir -p "${new_workspace_dir}"
  cd "${new_workspace_dir}"

  rules_apple_path=$(resolve_external_repository rules_apple)

  touch MODULE.bazel
  cat > MODULE.bazel <<EOF
module(name = "build_bazel_rules_apple_integration_tests", version = "0")

# Specify oldest possible bzlmod versions and let rules_apple versions take precedence
bazel_dep(name = "apple_support", version = "0.11.0", repo_name = "build_bazel_apple_support")
bazel_dep(name = "rules_swift", version = "2.0.0", repo_name = "build_bazel_rules_swift")
bazel_dep(name = "rules_apple", version = "0", repo_name = "build_bazel_rules_apple")
bazel_dep(name = "rules_shell", version = "0.3.0")

xcode_configure = use_extension("@bazel_tools//tools/osx:xcode_configure.bzl", "xcode_configure_extension")
use_repo(xcode_configure, "local_config_xcode")

apple_cc_configure = use_extension("@build_bazel_apple_support//crosstool:setup.bzl", "apple_cc_configure_extension")
use_repo(apple_cc_configure, "local_config_apple_cc")

local_path_override(
    module_name = "rules_apple",
    path = "$rules_apple_path",
)
EOF

  touch WORKSPACE
}

# Set-up a clean default workspace.
function setup_clean_workspace() {
  export WORKSPACE_DIR="${TEST_TMPDIR}/workspace"
  echo "setting up client in ${WORKSPACE_DIR}" > "$TEST_log"
  rm -fr "${WORKSPACE_DIR}"
  create_new_workspace "${WORKSPACE_DIR}"
  [ "${new_workspace_dir}" = "${WORKSPACE_DIR}" ] || \
    { echo "Failed to create workspace" >&2; exit 1; }
  export BAZEL_INSTALL_BASE=$(bazel info install_base)
  export BAZEL_GENFILES=$(bazel info bazel-genfiles "${EXTRA_BUILD_OPTIONS[@]:-}")
  export BAZEL_BIN=$(bazel info bazel-bin "${EXTRA_BUILD_OPTIONS[@]:-}")
}

# Any remaining arguments are passed to every `bazel build` invocation in the
# subsequent tests (see `do_build` in apple_shell_testutils.sh).
export EXTRA_BUILD_OPTIONS=( "$@" ); shift $#

echo "Applying extra options to each build: ${EXTRA_BUILD_OPTIONS[*]:-}" > "$TEST_log"

setup_clean_workspace

# Try to find the desired version of Xcode installed on the system. If it's not
# present, fallback to the most recent version currently installed and warn the
# user that results might be affected by this. (This makes it easier to support
# local test runs without having to change the version above from the CI
# default.)
readonly XCODE_QUERY=$(bazel query \
    "attr(aliases, $XCODE_VERSION_FOR_TESTS, " \
    "labels(versions, @local_config_xcode//:host_xcodes))" | \
    head -n 1)
if [[ -z "$XCODE_QUERY" ]]; then
  readonly OLD_XCODE_VERSION="$XCODE_VERSION_FOR_TESTS"
  XCODE_VERSION_FOR_TESTS=$(bazel query \
      "labels(versions, @local_config_xcode//:host_xcodes)" | \
      head -n 1 | \
      sed s#@local_config_xcode//:version## | \
      sed s#_#.#g)

  printf "WARN: The desired version of Xcode ($OLD_XCODE_VERSION) was not " >> "$TEST_log"
  printf "installed; using the highest version currently installed instead " >> "$TEST_log"
  printf "($XCODE_VERSION_FOR_TESTS). Note that this may produce unpredictable " >> "$TEST_log"
  printf "results in tests that depend on the behavior of a specific version " >> "$TEST_log"
  printf "of Xcode.\n" >> "$TEST_log"
fi

source "$(rlocation rules_apple/test/apple_shell_testutils.sh)"
source "$(rlocation rules_apple/test/${test_script})"
