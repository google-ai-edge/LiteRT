# Copyright 2025 The LiteRT Authors. All Rights Reserved.
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
# ==============================================================================

"""Qualcomm SDK for AI Edge LiteRT."""

__version__ = "{{ PACKAGE_VERSION }}"

import os
import pathlib
import sys

_SDK_FILES_SUBDIR = "data"


def get_sdk_path() -> pathlib.Path | None:
  """Returns the absolute path to the root of the downloaded SDK files."""
  try:
    package_dir = pathlib.Path(__file__).parent.resolve()
    sdk_path = package_dir / _SDK_FILES_SUBDIR
    if sdk_path.is_dir():
      return sdk_path
    else:
      print(
          f"Warning: SDK files directory not found at {sdk_path}",
          file=sys.stderr,
      )
      return None
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f"Error determining SDK path: {e}", file=sys.stderr)
    return None
