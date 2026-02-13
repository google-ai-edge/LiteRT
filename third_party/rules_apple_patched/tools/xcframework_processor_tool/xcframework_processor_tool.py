# Copyright 2022 The Bazel Authors. All rights reserved.
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

"""Execution-phase XCFramework processing tool.

The XCFramework processor tool is an execution-phase time tool inspecting an
XCFramework bundle defined by the embedded Info.plist file containing
XCFramework library definitions.

This script takes an analysis time defined target triplet (platform,
architecture, and environment), previously classified XCFramework file paths
(i.e. headers, module maps, binaries), and Bazel declared files and directories
to copy the effective XCFramework library files to a desired output location.

An XCFramework Info.plist file defines each XCFramework library with the
following information:

  - LibraryIdentifier:
      References relative directory containing an Apple framework or library.
  - LibraryPath:
      References relative path to a framework bundle, or library.
  - HeadersPath:
      References (if available) a relative directory containing
      Objective-C(++)/Swift interface files.
  - DebugSymbolsPath:
      References (if available) a relative directory containing dSYM bundles.
  - BitcodeSymbolMapsPath:
      References (if available) a relative directory containing bitcode symbol
      map files.
  - SupportedArchitectures:
      List of supported architectures by library.
  - SupportedPlatforms:
      List of supported platforms by the library.
  - SupportedPlatformVariant:
      List of supported platform variants (e.g. simulator) if available.
"""

import argparse
import os
import plistlib
import shutil
import sys
from typing import Any, Dict, List


def _create_args_parser() -> argparse.ArgumentParser:
  """Create parser and return parsed arguments."""
  parser = argparse.ArgumentParser(description="xcframework tool")

  value_args = {
      "architecture": "Target Apple architecture (e.g. x86_64, arm64).",
      "binary": "Bazel declared file for XCFramework binary file.",
      "bundle_name": "The XCFramework bundle name (i.e. name.xcframework).",
      "environment": "Target Apple environment (e.g. device, simulator).",
      "library_dir": "Bazel declared directory for copied XCFramework files.",
      "framework_imports_dir":
          "Bazel declared directory for XCFramework bundle files.",
      "info_plist": "XCFramework Info.plist file.",
      "platform": "Target Apple platform (e.g. macos, ios).",
  }
  for arg_name, arg_help in value_args.items():
    parser.add_argument(
        f"--{arg_name}",
        type=str,
        required=True,
        action="store",
        help=arg_help)


  optional_value_args = {
      "dsym_imports_dir":
          "Bazel declared directory for imported dSYM bundle file.",
  }
  for arg_name, arg_help in optional_value_args.items():
    parser.add_argument(
        f"--{arg_name}",
        type=str,
        action="store",
        help=arg_help)

  list_args = {
      "binary_file": "Imported XCFramework binary file path.",
      "header_file": "Imported XCFramework header file path.",
  }
  for arg_name, arg_help in list_args.items():
    parser.add_argument(
        f"--{arg_name}",
        type=str,
        required=True,
        action="append",
        help=arg_help,
        dest=f"{arg_name}s",
    )

  optional_list_args = {
      "bundle_file": "Imported XCFramework bundle file path.",
      "modulemap_file": "Imported XCFramework modulemap file path.",
      "swiftinterface_file": "Imported XCFramework Swift module file path.",
      "dsym": "Imported XCFramework dSYM bundle path.",
  }
  for arg_name, arg_help in optional_list_args.items():
    parser.add_argument(
        f"--{arg_name}",
        type=str,
        required=False,
        action="append",
        help=arg_help,
        dest=f"{arg_name}s",
    )

  return parser


def _get_plist_dict(info_plist_path: str) -> Dict[str, Any]:
  """Returns dictionary from XCFramework Info.plist file content.

  Args:
    info_plist_path: XCFramework Info.plist file path to read from.
  Returns:
    Dictionary of parsed Info.plist contents.
  Raises:
    ValueError - if file does not contain XCFrameworkFormatVersion key.
  """
  with open(info_plist_path, "rb") as info_plist_file:
    info_plist_dict = plistlib.load(info_plist_file)
    if "XCFrameworkFormatVersion" not in info_plist_dict:
      raise ValueError(f"""
Info.plist file does not contain key: 'XCFrameworkFormatVersion'. Contents:

{info_plist_dict}

Is it an XCFramework Info.plist file?
""")

    return info_plist_dict


def _get_library_from_plist(
    *,
    architecture: str,
    environment: str,
    info_plist: Dict[str, Any],
    platform: str) -> Dict[str, Any]:
  """Returns an XCFramework library definition from XCFramework Info.plist file.

  Traverse XCFramework Info.plist libraries definitions to find a supported
  library for the given target triplet (platform, architecture, environment).

  Args:
    architecture: Target Apple architecture (e.g. x86_64, arm64).
    environment: Target Apple environment (e.g. device, simulator).
    info_plist: Parsed XCFramework Info.plist file contents.
    platform: Target Apple platform (e.g. macos, ios).
  Returns:
    Info.plist dictionary referencing target XCFramework library.
  Raises:
    ValueError - if no XCFramework library supporting target triplet is found.
  """
  available_libraries = info_plist.get("AvailableLibraries", [])
  for library in available_libraries:
    supported_platform = library.get("SupportedPlatform", [])
    if platform != supported_platform:
      continue

    supported_environment = library.get("SupportedPlatformVariant", [])
    if environment == "device" and supported_environment:
      continue
    if (environment != "device"
        and environment != supported_environment):
      continue

    supported_architectures = library.get("SupportedArchitectures", [])
    if architecture not in supported_architectures:
      continue

    return library

  library_identifiers = [
      library.get("LibraryIdentifier")
      for library in available_libraries
  ]
  raise ValueError(f"""
Imported XCFramework does not support the following platform:
  - platform: {platform}
  - architecture: {architecture}
  - environment: {environment}

Supported platforms: {library_identifiers}
""")


def _relpath_from_subdirectory(*, absolute_path, subdirectory):
  """Returns a relative path from the root of a given subdirectory.

  Args:
    absolute_path: String. An absolute path to search within.
    subdirectory: String. The name of the subdirectory to search for within the
      absolute path given.
  Returns:
    The relative path from the point where the subdirectory was found to the
    file or directory referenced by the provided absolute path.
  Raises:
    ValueError - if no path could be found.
  """
  found_dir = None
  parent_dir = os.path.dirname(absolute_path)
  while parent_dir != "" and parent_dir != "/" and found_dir is None:
    if parent_dir.endswith(os.sep + subdirectory):
      found_dir = parent_dir
    else:
      parent_dir = os.path.dirname(parent_dir)

  if parent_dir == "/" or parent_dir == "":
    raise ValueError(f"""
Internal Error: Could not find {subdirectory} in path: {absolute_path}
""")

  return os.path.relpath(absolute_path, start=found_dir)


def _copy_xcframework_files(
    *,
    executable: bool = False,
    library_identifier: str,
    output_directories: List[str],
    copy_from_subdirectory: str = "",
    xcframework_files: List[str]) -> None:
  """Copies XCFramework files filtered by library identifier to a directory.

  Args:
    executable: Indicates whether or not the file(s) should be made executable.
    library_identifier: XCFramework library identifier to filter files with.
    output_directories: List of directory paths to copy files to.
    copy_from_subdirectory: String. The name of a subdirectory that will be used
      to generate relative paths to copy files from, preserving subdirectory
      paths in between. If declared, this overrides the default behavior to only
      preserve paths from the common path of all provided library_files.
    xcframework_files: List of XCFramework files to filter by library identifier
      and copy files from.
  """
  if not xcframework_files:
    return

  library_files = [
      f
      for f in xcframework_files
      if f"/{library_identifier}/" in f
  ]

  # This path will reference the following path:
  #   a) For framework based XCFramework -> the .framework directory.
  #   b) For library based XCFramework -> the library identifier directory.
  if not copy_from_subdirectory:
    library_dir_path = os.path.commonpath(library_files)

  for library_file in library_files:
    if copy_from_subdirectory:
      rel_path = _relpath_from_subdirectory(
          absolute_path=library_file,
          subdirectory=copy_from_subdirectory)
    else:
      rel_path = os.path.relpath(library_file, start=library_dir_path)

    for output_directory in output_directories:
      dest_path = os.path.join(output_directory, rel_path)
      os.makedirs(os.path.dirname(dest_path), exist_ok=True)
      dest_file_path = shutil.copy2(library_file, dest_path)
      os.chmod(dest_file_path, 0o755 if executable else 0o644)


def main() -> int:
  args_parser = _create_args_parser()
  args = args_parser.parse_args()

  bundle_name = args.bundle_name
  info_plist = _get_plist_dict(args.info_plist)
  xcframework_library = _get_library_from_plist(
      architecture=args.architecture,
      environment=args.environment,
      info_plist=info_plist,
      platform=args.platform,
  )

  library_identifier = xcframework_library.get("LibraryIdentifier")
  assert library_identifier is not None, "XCFramework is missing a LibraryIdentifier field for the matched target triple."

  # Bundle files are copied to two different directories due the following:
  #
  # - library_dir: This directory will hold the inferred Apple framework, and
  #     it's propagated to the AppleDynamicFrameworkInfo provider. This
  #     provider is responsible of propagation linking information to
  #     top-level targets (e.g. ios_application, objc_library).
  #
  # - framework_imports_dir: This directory will hold Apple framework files
  #     that are bundable at top-level targets (e.g. ios_application).
  #     This directory is propagated to the AppleFrameworkImportInfo provider
  #     as a tree-artifact containing all files to be bundled, and gets
  #     processed by the framework_import partial.
  _copy_xcframework_files(
      library_identifier=library_identifier,
      output_directories=[args.library_dir, args.framework_imports_dir],
      xcframework_files=args.bundle_files)

  _copy_xcframework_files(
      executable=True,
      library_identifier=library_identifier,
      output_directories=[args.library_dir],
      xcframework_files=args.binary_files)

  headers_dir = os.path.join(args.library_dir, "Headers")
  _copy_xcframework_files(
      library_identifier=library_identifier,
      output_directories=[headers_dir],
      copy_from_subdirectory="Headers",
      xcframework_files=args.header_files)

  modules_dir = os.path.join(args.library_dir, "Modules")
  _copy_xcframework_files(
      library_identifier=library_identifier,
      output_directories=[modules_dir],
      xcframework_files=args.modulemap_files)

  swiftmodule_dir = os.path.join(modules_dir, f"{bundle_name}.swiftmodule")
  _copy_xcframework_files(
      library_identifier=library_identifier,
      output_directories=[swiftmodule_dir],
      xcframework_files=args.swiftinterface_files)
  
  _copy_xcframework_files(
      library_identifier=library_identifier,
      output_directories=[args.dsym_imports_dir],
      xcframework_files=args.dsyms)

  return 0

if __name__ == "__main__":
  sys.exit(main())
