#!/usr/bin/env python3
# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fail CI when OpenVINO versions drift between setup.py and openvino.bzl.

The Intel OV compiler plugin and dispatch library are built against the
OpenVINO SDK pinned in third_party/intel_openvino/openvino.bzl. The NPU
compiler shared library and the `openvino` PyPI wheel pinned by
install_requires are both wired up in ci/tools/python/vendor_sdk/intel/
setup.py. All three must pin the same OpenVINO build; the build number
(the numeric segment in `2026.2.0-21820-<commit>/`) is the unique
identifier — it appears both in the toolkit archive directory and as the
PyPI wheel filename's build tag.
"""

import pathlib
import re
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
BZL = REPO_ROOT / "third_party/intel_openvino/openvino.bzl"
SETUP = REPO_ROOT / "ci/tools/python/vendor_sdk/intel/setup.py"

# Nightly archive paths look like:
#   .../packages/nightly/2026.2.0-21820-9a25caa5a15/openvino_toolkit_*.tgz
# We extract the build number (21820) as the canonical sync key.
_BZL_BUILD_RE = re.compile(
    r"/nightly/\d{4}\.\d+\.\d+-(\d+)-[0-9a-f]{7,40}/"
)
_SETUP_BUILD_RE = re.compile(r"_OV_BUILD_NUMBER\s*=\s*'(\d+)'")


def main() -> int:
  bzl_builds = set(_BZL_BUILD_RE.findall(BZL.read_text()))
  m = _SETUP_BUILD_RE.search(SETUP.read_text())
  if not m:
    print(
        f"ERROR: could not find _OV_BUILD_NUMBER in {SETUP}", file=sys.stderr
    )
    return 1
  setup_build = m.group(1)

  if not bzl_builds:
    print(
        f"ERROR: {BZL} contains no nightly OpenVINO URLs matching the"
        " expected `/nightly/<version>-<build>-<commit>/` schema.",
        file=sys.stderr,
    )
    return 1
  if len(bzl_builds) != 1:
    print(
        f"ERROR: {BZL} references multiple OpenVINO builds:"
        f" {sorted(bzl_builds)}",
        file=sys.stderr,
    )
    return 1
  bzl_build = next(iter(bzl_builds))

  if setup_build != bzl_build:
    print(
        "ERROR: OpenVINO build-number pin drift:\n"
        f"  {BZL}: {bzl_build}\n"
        f"  {SETUP} _OV_BUILD_NUMBER: {setup_build}\n"
        "Both must pin the same OpenVINO build.",
        file=sys.stderr,
    )
    return 1

  print(f"OK: OpenVINO build {bzl_build} matches in both files.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
