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

# Integration tests for the Apple bundle versioning rule.

function set_up() {
  mkdir -p pkg

  cat > pkg/saver.bzl <<EOF
load("@build_bazel_rules_apple//apple:providers.bzl",
     "AppleBundleVersionInfo")

def _saver_impl(ctx):
  infile = ctx.attr.bundle_version[AppleBundleVersionInfo].version_file
  outfile = ctx.outputs.version_file
  ctx.actions.run_shell(
      inputs=[infile],
      outputs=[outfile],
      command="cp %s %s" % (infile.path, outfile.path),
  )

saver = rule(
    _saver_impl,
    attrs={
        "bundle_version": attr.label(
            providers=[[AppleBundleVersionInfo]],
        ),
    },
    outputs = {
        "version_file": "%{name}.txt",
    },
)
EOF

  cat > pkg/BUILD <<EOF
load("@build_bazel_rules_apple//apple:versioning.bzl",
     "apple_bundle_version")
load(":saver.bzl", "saver")

saver(
    name = "saved_version",
    bundle_version = ":bundle_version",
)
EOF
}

function tear_down() {
  rm -rf pkg
}

# Test that the build label passed via --embed_label can be parsed out
# correctly.
function test_build_label_substitution() {
  cat >> pkg/BUILD <<'EOF'
apple_bundle_version(
    name = "bundle_version",
    build_label_pattern = "MyApp_{version}_RC0*{candidate}",
    build_version = "{version}.{candidate}",
    short_version_string = "{version}",
    capture_groups = {
        "version": r"\d+\.\d+",
        "candidate": r"\d+",
    },
)
EOF

  do_build ios //pkg:saved_version --embed_label=MyApp_1.2_RC03 || \
      fail "Should build"
  assert_contains "\"build_version\": \"1.2.3\"" test-bin/pkg/saved_version.txt
  assert_contains "\"short_version_string\": \"1.2\"" \
      test-bin/pkg/saved_version.txt
}

# Test that the fallback_build_label is *not* used when --embed_label *is*
# passed on the build.
function test_build_label_substitution_ignores_fallback_label() {
  cat >> pkg/BUILD <<'EOF'
apple_bundle_version(
    name = "bundle_version",
    build_label_pattern = "MyApp_{version}_RC0*{candidate}",
    build_version = "{version}.{candidate}",
    fallback_build_label = "MyApp_99.99_RC99",
    short_version_string = "{version}",
    capture_groups = {
        "version": r"\d+\.\d+",
        "candidate": r"\d+",
    },
)
EOF

  do_build ios //pkg:saved_version --embed_label=MyApp_1.2_RC03 || \
      fail "Should build"
  assert_contains "\"build_version\": \"1.2.3\"" test-bin/pkg/saved_version.txt
  assert_contains "\"short_version_string\": \"1.2\"" \
      test-bin/pkg/saved_version.txt
}


# Test that the build fails if the build label does not match the regular
# expression that is built after substituting the regex groups for the
# placeholders. This fails during tool execution.
function test_build_label_that_does_not_match_regex_fails() {
  cat >> pkg/BUILD <<'EOF'
apple_bundle_version(
    name = "bundle_version",
    build_label_pattern = "MyApp_{version}_RC0*{candidate}",
    build_version = "{version}.{candidate}",
    short_version_string = "{version}",
    capture_groups = {
        "version": r"\d+\.\d+\.\d+",
        "candidate": r"\d+",
    },
)
EOF

  ! do_build ios //pkg:saved_version --embed_label=MyApp_1.2_RC03 || \
      fail "Should fail"
  expect_log "The build label (\"MyApp_1.2_RC03\") did not match the pattern"
}


run_suite "apple_bundle_version tests"
