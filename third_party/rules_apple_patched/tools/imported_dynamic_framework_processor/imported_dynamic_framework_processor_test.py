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
"""Tests for xcframework_processor_tool."""

import os
import unittest
from unittest import mock

from tools.imported_dynamic_framework_processor import imported_dynamic_framework_processor
from tools.wrapper_common import execute
from tools.wrapper_common import lipo


class ImportedDynamicFrameworkProcessorTest(unittest.TestCase):

  @mock.patch.object(execute, "execute_and_filter_output")
  def test_get_install_path_for_binary(self, mock_execute):
    with self.assertRaisesRegex(
        ValueError, r"Could not find framework binary.*"):
      mock_execute.return_value = (None, "no rpath", None)
      imported_dynamic_framework_processor._get_install_path_for_binary(None)

    mock_execute.return_value = (
        None, "@rpath/MyFramework.framework/MyFramework", None)
    result = imported_dynamic_framework_processor._get_install_path_for_binary(
        None)
    self.assertEqual(result, "@rpath/MyFramework.framework/MyFramework")

  @mock.patch.object(
      imported_dynamic_framework_processor, "_get_install_path_for_binary")
  def test_get_version_from_install_path_fails(self, mock_install_path):
    with self.assertRaisesRegex(
        ValueError, r"Framework binary install path does not match.*"):
      mock_install_path.return_value = "@rpath/libMyAwesomeLibrary"
      (imported_dynamic_framework_processor
       ._get_framework_version_from_install_path(None))

    with self.assertRaisesRegex(
        ValueError, r"Framework binary install path does not match.*"):
      mock_install_path.return_value = "@rpath/MyFramework.framework/MyFramework"
      (imported_dynamic_framework_processor
       ._get_framework_version_from_install_path(None))

  @mock.patch.object(
      imported_dynamic_framework_processor, "_get_install_path_for_binary")
  def test_get_version_from_install_path_parse_version(self, mock_install_path):
    mock_install_path.return_value = (
        "@rpath/MyFramework.framework/Versions/A/MyFramework")
    actual_version = (
        imported_dynamic_framework_processor
        ._get_framework_version_from_install_path(None))
    self.assertEqual(actual_version, "A")

    mock_install_path.return_value = (
        "@rpath/MyFramework.framework/Versions/105.0.5195.102/MyFramework")
    actual_version = (
        imported_dynamic_framework_processor
        ._get_framework_version_from_install_path(None))
    self.assertEqual(actual_version, "105.0.5195.102")

    mock_install_path.return_value = (
        "@rpath/MyFramework.framework/Versions/A/Resources.bundle/Info.plist"
    )
    actual_version = (
        imported_dynamic_framework_processor
        ._get_framework_version_from_install_path(None))
    self.assertEqual(actual_version, "A")

  @mock.patch.object(os, "listdir")
  def test_get_version_from_structure(self, mock_listdir):
    mock_listdir.return_value = ["A", "B", "Current"]
    result = imported_dynamic_framework_processor._try_get_framework_version_from_structure("<framework>")
    self.assertIsNone(result)

    mock_listdir.return_value = ["A", "Current"]
    result = imported_dynamic_framework_processor._try_get_framework_version_from_structure("<framework>")
    self.assertEqual(result, "A")

  @mock.patch.object(lipo, "find_archs_for_binaries")
  def test_strip_or_copy_binary_fails_with_no_binary_archs(
      self, mock_lipo):
    with self.assertRaisesRegex(
        ValueError,
        "Could not find binary architectures for binaries using lipo.*"):
      mock_lipo.return_value = (None, None)
      imported_dynamic_framework_processor._strip_or_copy_binary(
          framework_binary="/tmp/path/to/fake/binary",
          output_path="/tmp/path/to/outputs",
          strip_bitcode=False,
          requested_archs=["x86_64"])

  @mock.patch.object(lipo, "find_archs_for_binaries")
  def test_strip_or_copy_binary_fails_with_no_matching_archs(
      self, mock_lipo):
    with self.assertRaisesRegex(
        ValueError,
        ".*Precompiled framework does not share any binary architecture.*"):
      mock_lipo.return_value = (set(["x86_64"]), None)
      imported_dynamic_framework_processor._strip_or_copy_binary(
          framework_binary="/tmp/path/to/fake/binary",
          output_path="/tmp/path/to/outputs",
          strip_bitcode=False,
          requested_archs=["arm64"])

  @mock.patch.object(lipo, "find_archs_for_binaries")
  @mock.patch.object(
      imported_dynamic_framework_processor, "_copy_framework_file")
  @mock.patch.object(
      imported_dynamic_framework_processor, "_strip_framework_binary")
  def test_strip_or_copy_binary_thins_framework_binary(
      self, mock_strip_framework_binary, mock_copy_framework_file, mock_lipo):
    mock_lipo.return_value = (set(["x86_64", "arm64"]), None)
    imported_dynamic_framework_processor._strip_or_copy_binary(
        framework_binary="/tmp/path/to/fake/binary",
        output_path="/tmp/path/to/outputs",
        strip_bitcode=False,
        requested_archs=["arm64"])

    mock_copy_framework_file.assert_not_called()
    mock_strip_framework_binary.assert_called_with(
        "/tmp/path/to/fake/binary",
        "/tmp/path/to/outputs",
        set(["arm64"]))

  @mock.patch.object(lipo, "find_archs_for_binaries")
  @mock.patch.object(
      imported_dynamic_framework_processor, "_copy_framework_file")
  @mock.patch.object(
      imported_dynamic_framework_processor, "_strip_framework_binary")
  def test_strip_or_copy_binary_skips_lipo_with_single_arch_binary(
      self, mock_strip_framework_binary, mock_copy_framework_file, mock_lipo):
    mock_lipo.return_value = (set(["arm64"]), None)
    imported_dynamic_framework_processor._strip_or_copy_binary(
        framework_binary="/tmp/path/to/fake/binary",
        output_path="/tmp/path/to/outputs",
        strip_bitcode=False,
        requested_archs=["arm64"])

    mock_strip_framework_binary.assert_not_called()
    mock_copy_framework_file.assert_called_with(
        "/tmp/path/to/fake/binary",
        executable=True,
        output_path="/tmp/path/to/outputs")

  @mock.patch.object(lipo, "find_archs_for_binaries")
  @mock.patch.object(
      imported_dynamic_framework_processor, "_copy_framework_file")
  @mock.patch.object(
      imported_dynamic_framework_processor, "_strip_framework_binary")
  def test_strip_or_copy_binary_skips_lipo_with_matching_archs_bin(
      self, mock_strip_framework_binary, mock_copy_framework_file, mock_lipo):
    mock_lipo.return_value = (set(["x86_64", "arm64"]), None)
    imported_dynamic_framework_processor._strip_or_copy_binary(
        framework_binary="/tmp/path/to/fake/binary",
        output_path="/tmp/path/to/outputs",
        strip_bitcode=False,
        requested_archs=["x86_64", "arm64"])

    mock_strip_framework_binary.assert_not_called()
    mock_copy_framework_file.assert_called_with(
        "/tmp/path/to/fake/binary",
        executable=True,
        output_path="/tmp/path/to/outputs")

if __name__ == "__main__":
  unittest.main()
