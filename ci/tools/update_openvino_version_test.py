#!/usr/bin/env python3
# Copyright 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for OpenVINO archive URL and version derivation."""

import unittest

import update_openvino_version as updater


class OpenvinoVersionTest(unittest.TestCase):

  def test_channels_and_versions(self):
    base_url = (
        "https://storage.openvinotoolkit.org/repositories/openvino/packages/"
    )
    cases = (
        (
            "2026.4.1/windows/",
            "2026.4.1.22982.07f9c262b05",
            "release",
            "2026.4.1",
        ),
        (
            "nightly/2026.5.0/",
            "2026.5.0.dev20261106",
            "nightly",
            "2026.5.0.dev20261106",
        ),
        (
            "pre-release/2026.5.0rc1/",
            "2026.5.0.dev20261106",
            "pre-release",
            "2026.5.0.dev20261106",
        ),
    )
    for directory, build, expected_channel, expected_version in cases:
      with self.subTest(channel=expected_channel):
        windows_url = (
            f"{base_url}{directory}openvino_toolkit_windows_{build}_x86_64.zip"
        )
        updater._validate_windows_url(windows_url)
        channel, urls = updater._derive_urls(windows_url)
        self.assertEqual(channel, expected_channel)
        self.assertEqual(
            updater._pep440_version(channel, urls), expected_version
        )
        for os_key in ("ubuntu24", "ubuntu22", "android"):
          expected_directory = directory.replace("/windows/", "/linux/")
          self.assertEqual(
              urls[os_key],
              f"{base_url}{expected_directory}"
              f"openvino_toolkit_{os_key}_{build}_x86_64.tgz",
          )
          self.assertEqual(
              updater._dir_for(urls[os_key]),
              f"openvino_toolkit_{os_key}_{build}_x86_64",
          )

  def test_invalid_development_version(self):
    for channel in ("nightly", "pre-release"):
      with self.subTest(channel=channel):
        with self.assertRaisesRegex(ValueError, channel):
          updater._pep440_version(
              channel,
              {"windows": "openvino_toolkit_windows_invalid_x86_64.zip"},
          )


if __name__ == "__main__":
  unittest.main()