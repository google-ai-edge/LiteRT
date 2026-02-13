# Copyright 2020 The Bazel Authors. All rights reserved.
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
#

import argparse
import glob
import os
import re
import shutil
import sys
import tempfile

from tools.bitcode_strip import bitcode_strip
from tools.wrapper_common import execute
from tools.wrapper_common import lipo

_OTOOL_MINIMUM_OS_VERSION_RE = re.compile(
    r"""
(
    cmd\ LC_VERSION_MIN_(?P<lc_version_min_platform>[^\n]+)\n
    .*?
    version\ (?P<lc_version_min_version>\d+\.\d+)
  |
    cmd\ LC_BUILD_VERSION
    .*?
    platform\ (?P<lc_build_version_platform>[^\n]+)\n
    .*?
    minos\ (?P<lc_build_version_minos>\d+\.\d+)
)
""", re.VERBOSE | re.MULTILINE | re.DOTALL)

# Minimum OS versions after which the Swift runtime is packaged with the OS. If
# the deployment target of a binary is greater than or equal to the versions
# defined here it does not need to bundle the Swift runtime.
_MIN_OS_PLATFORM_SWIFT_PRESENCE = {
    "ios": (12, 2),
    "iphoneos": (12, 2),
    "macos": (10, 14, 4),
    "macosx": (10, 14, 4),
    "tvos": (12, 2),
    "watchos": (5, 2),
}


def _deployment_target_requires_bundled_swift_runtime(platform, version):
  """Returns true if the given deployment target requires a bundled copy of the Swift runtime."""

  platform = platform.lower().replace("simulator", "")
  version = tuple(int(component) for component in version.split("."))

  return version < _MIN_OS_PLATFORM_SWIFT_PRESENCE.get(platform, (0, 0))


def _binary_requires_bundled_swift_runtime(binary):
  """Returns true if the deployment target of the given binary requires a bundled copy of the Swift runtime."""

  cmd = ["otool", "-lV", "-arch", "all", binary]
  _, stdout, stderr = execute.execute_and_filter_output(
      cmd, raise_on_failure=True)
  if stderr:
    print(stderr)

  # Loop to ensure we process all architectures within the binary. Different
  # architectures may have different deployment targets.
  while True:
    match = _OTOOL_MINIMUM_OS_VERSION_RE.search(stdout)
    if not match:
      return False

    groups = match.groupdict()
    # Only one of each alternative of platform and version can be set.
    platform = groups["lc_version_min_platform"] or groups[
        "lc_build_version_platform"]
    version = groups["lc_version_min_version"] or groups[
        "lc_build_version_minos"]

    if _deployment_target_requires_bundled_swift_runtime(platform, version):
      return True

    stdout = stdout[match.endpos:]


def _copy_swift_stdlibs(binaries_to_scan, sdk_platform, destination_path):
  """Copies the Swift stdlibs required by the binaries to the destination."""
  # Rely on the swift-stdlib-tool to determine the subset of Swift stdlibs that
  # these binaries require.
  developer_dir = os.environ["DEVELOPER_DIR"]
  swift_dylibs_root = "Toolchains/XcodeDefault.xctoolchain/usr/lib/swift-*"
  swift_library_dir_pattern = os.path.join(developer_dir, swift_dylibs_root,
                                           sdk_platform)
  swift_library_dirs = glob.glob(swift_library_dir_pattern)

  cmd = [
      "xcrun", "swift-stdlib-tool", "--copy", "--platform", sdk_platform,
      "--destination", destination_path
  ]
  for swift_library_dir in swift_library_dirs:
    cmd.extend(["--source-libraries", swift_library_dir])
  for binary_to_scan in binaries_to_scan:
    cmd.extend(["--scan-executable", binary_to_scan])

  _, stdout, stderr = execute.execute_and_filter_output(cmd,
                                                        raise_on_failure=True)
  if stderr:
    print(stderr)
  if stdout:
    print(stdout)

  # swift-stdlib-tool currently bundles an unnecessary copy of the Swift runtime
  # whenever it bundles the back-deploy version of the Swift concurrency
  # runtime. This is because the back-deploy version of the Swift concurrency
  # runtime contains an `@rpath`-relative reference to the Swift runtime due to
  # being built with a deployment target that predates the Swift runtime being
  # shipped with operating system.
  # The Swift runtime only needs to be bundled if the binary's deployment target
  # is old enough that it may run on OS versions that lack the Swift runtime,
  # so we detect this scenario and remove the Swift runtime from the output
  # path.
  if not any(
      _binary_requires_bundled_swift_runtime(binary)
      for binary in binaries_to_scan):
    libswiftcore_path = os.path.join(destination_path, "libswiftCore.dylib")
    if os.path.exists(libswiftcore_path):
      os.remove(libswiftcore_path)


def _lipo_exec_files(exec_files, target_archs, strip_bitcode, source_path,
                     destination_path):
  """Strips executable files if needed and copies them to the destination."""
  # Find all architectures from the set of files we might have to lipo.
  _, exec_archs = lipo.find_archs_for_binaries(
      [os.path.join(source_path, f) for f in exec_files]
  )

  # Copy or lipo each file as needed, from source to destination.
  for exec_file in exec_files:
    exec_file_source_path = os.path.join(source_path, exec_file)
    exec_file_destination_path = os.path.join(destination_path, exec_file)
    file_archs = exec_archs[exec_file_source_path]

    archs_to_keep = target_archs & file_archs

    # On M1 hardware, thin x86_64 libraries do not need lipo when archs_to_keep
    # is empty.
    if len(file_archs) == 1 or archs_to_keep == file_archs or not archs_to_keep:
      # If there is no need to lipo, copy and mark as executable.
      shutil.copy(exec_file_source_path, exec_file_destination_path)
      os.chmod(exec_file_destination_path, 0o755)
    else:
      lipo.invoke_lipo(
          exec_file_source_path, archs_to_keep, exec_file_destination_path
      )
    if strip_bitcode:
      bitcode_strip.invoke(exec_file_destination_path, exec_file_destination_path)


def main():
  parser = argparse.ArgumentParser(description="swift stdlib tool")
  parser.add_argument(
      "--binary", type=str, required=True, action="append",
      help="path to a binary file which will be the basis for Swift stdlib tool"
      " operations"
  )
  parser.add_argument(
      "--platform", type=str, required=True, help="the target platform, e.g. "
      "'iphoneos'"
  )
  parser.add_argument(
      "--strip_bitcode", action="store_true", default=False, help="strip "
      "bitcode from the Swift support libraries"
  )
  parser.add_argument(
      "--output_path", type=str, required=True, help="path to save the Swift "
      "support libraries to"
  )
  args = parser.parse_args()

  # Create a temporary location for the unstripped Swift stdlibs.
  temp_path = tempfile.mkdtemp(prefix="swift_stdlib_tool.XXXXXX")

  # Use the binaries to copy only the Swift stdlibs we need for this app.
  _copy_swift_stdlibs(args.binary, args.platform, temp_path)

  # Determine the binary slices we need to strip with lipo.
  target_archs, _ = lipo.find_archs_for_binaries(args.binary)

  # Select all of the files in this temp directory, which are our Swift stdlibs.
  stdlib_files = [
      f for f in os.listdir(temp_path) if os.path.isfile(
          os.path.join(temp_path, f)
      )
  ]

  destination_path = args.output_path
  # Ensure directory exists for remote execution.
  os.makedirs(destination_path, exist_ok=True)

  # Copy or use lipo to strip the executable Swift stdlibs to their destination.
  _lipo_exec_files(stdlib_files, target_archs, args.strip_bitcode, temp_path,
                   destination_path)

  shutil.rmtree(temp_path)


if __name__ == "__main__":
  sys.exit(main())
