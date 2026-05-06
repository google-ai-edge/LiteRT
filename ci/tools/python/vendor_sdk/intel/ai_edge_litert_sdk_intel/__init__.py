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

"""Intel OpenVINO SDK for AI Edge LiteRT.

Ships `libopenvino_intel_npu_compiler.{so,dll}` fetched from the OpenVINO
toolkit archive at install time (see setup.py). Level Zero loader, NPU
firmware, and the NPU UMD (intel-level-zero-npu / intel-fw-npu /
intel-driver-compiler-npu) remain a user-installed prerequisite.
"""

import os
import pathlib
import sys
from typing import Optional

__version__ = "{{ PACKAGE_VERSION }}"

_SDK_FILES_SUBDIR = "data"


def get_sdk_path() -> Optional[pathlib.Path]:
  """Returns the absolute path to the downloaded SDK data directory, or None.

  Returns None when the directory does not exist (e.g. when the archive
  download was skipped at install time).
  """
  sdk_path = pathlib.Path(__file__).parent.resolve() / _SDK_FILES_SUBDIR
  if sdk_path.is_dir():
    return sdk_path
  return None


def path_to_sdk_libs() -> Optional[pathlib.Path]:
  """Returns the directory callers should prepend to LD_LIBRARY_PATH / PATH."""
  return get_sdk_path()


# On Windows, register the data dir with the process DLL search path so that
# in-process consumers (e.g. the LiteRT dispatch DLL loading the NPU compiler)
# can resolve openvino_intel_npu_compiler.dll without the caller manually
# tweaking PATH. No-op on other platforms.
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
  _sdk_path = get_sdk_path()
  if _sdk_path is not None:
    try:
      os.add_dll_directory(str(_sdk_path))
    except OSError:
      pass
