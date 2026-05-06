#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Fail CI when OpenVINO versions drift between setup.py and openvino.bzl.

The Intel OV compiler plugin and dispatch library are built against the
OpenVINO SDK pinned in third_party/intel_openvino/openvino.bzl. The NPU
compiler shared library is fetched by ci/tools/python/vendor_sdk/intel/setup.py
at pip install time. Both must pin the same OpenVINO build.
"""

import pathlib
import re
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
BZL = REPO_ROOT / "third_party/intel_openvino/openvino.bzl"
SETUP = REPO_ROOT / "ci/tools/python/vendor_sdk/intel/setup.py"

# OpenVINO build identifier: "<year>.<minor>.<patch>.<build>.<hash>", e.g.
# "2026.1.0.21367.63e31528c62". The trailing commit hash is 7-40 hex chars.
# The surrounding context in openvino.bzl URLs ("..._x86_64.tgz") is not a
# word boundary for Python's \b, so we match a 5-component dotted identifier
# ending with the hash and rely on the preceding "/" or "_" being non-alnum.
_BUILD_RE = re.compile(r"\d{4}\.\d+\.\d+\.\d+\.[0-9a-f]{7,40}")
_SETUP_BUILD_RE = re.compile(r"_OV_BUILD\s*=\s*'([^']+)'")


def main() -> int:
  bzl_builds = set(_BUILD_RE.findall(BZL.read_text()))
  m = _SETUP_BUILD_RE.search(SETUP.read_text())
  if not m:
    print(f"ERROR: could not find _OV_BUILD in {SETUP}", file=sys.stderr)
    return 1
  setup_build = m.group(1)

  if len(bzl_builds) != 1:
    print(
        f"ERROR: {BZL} references multiple OpenVINO builds: {sorted(bzl_builds)}",
        file=sys.stderr,
    )
    return 1
  bzl_build = next(iter(bzl_builds))

  if setup_build != bzl_build:
    print(
        "ERROR: OpenVINO build pin drift:\n"
        f"  {BZL}: {bzl_build}\n"
        f"  {SETUP}: {setup_build}\n"
        "Both must pin the same OpenVINO release.",
        file=sys.stderr,
    )
    return 1

  print(f"OK: OpenVINO build {bzl_build} matches in both files.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
