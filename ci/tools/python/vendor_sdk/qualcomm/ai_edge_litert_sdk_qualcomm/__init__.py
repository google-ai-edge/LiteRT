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

from __future__ import annotations

__version__ = "{{ PACKAGE_VERSION }}"

import os
import pathlib
import platform
import sys
from typing import Optional

_SDK_FILES_SUBDIR = "data"

_X86_64_HOST_LIB_SUBDIR = "x86_64-linux-clang"

# QAIRT Linux aarch64 host library builds as (minimum glibc, subdir), newest
# first. The ubuntu-gcc9.4 build only supports HTP v68.
_AARCH64_HOST_LIB_SUBDIRS = (
    ((2, 34), "aarch64-oe-linux-gcc11.2"),
    ((2, 29), "aarch64-ubuntu-gcc9.4"),
    ((), "aarch64-oe-linux-gcc9.3"),  # Any glibc, including unknown.
)

# QAIRT Windows on Snapdragon (ARM64) runtime library build.
_WINDOWS_ARM64_LIB_SUBDIR = "aarch64-windows-msvc"

# Present when the kernel FastRPC driver exposes a compute DSP.
_FASTRPC_CDSP_DEVICE = "/dev/fastrpc-cdsp"

# Substrings of the Windows processor name of Qualcomm SoCs, which have a
# Hexagon NPU (e.g. "Snapdragon(R) X Elite - X1E80100 - Qualcomm(R) Oryon(TM)").
_WINDOWS_QUALCOMM_PROCESSOR_NAMES = ("snapdragon", "qualcomm")


def _is_linux_aarch64() -> bool:
  return sys.platform == "linux" and platform.machine() in ("aarch64", "arm64")


def _is_windows_arm64() -> bool:
  return sys.platform == "win32" and platform.machine().lower() in (
      "arm64",
      "aarch64",
  )


def _glibc_version() -> tuple[int, ...]:
  """Returns the host glibc version, e.g. (2, 39), or () if unknown."""
  lib, version = platform.libc_ver()
  if lib != "glibc" or not version:
    return ()
  return tuple(int(part) for part in version.split(".")[:2])


def _windows_processor_name() -> str:
  """Returns the Windows processor brand string, or "" if unavailable."""
  try:
    import winreg  # pylint: disable=g-import-not-at-top

    with winreg.OpenKey(
        winreg.HKEY_LOCAL_MACHINE,
        r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
    ) as key:
      value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
      return str(value)
  except (ImportError, OSError):
    return platform.processor() or ""


def _is_qualcomm_windows_device() -> bool:
  """Returns whether this is a Windows on Snapdragon device."""
  if not _is_windows_arm64():
    return False
  name = _windows_processor_name().lower()
  return any(n in name for n in _WINDOWS_QUALCOMM_PROCESSOR_NAMES)


def path_to_sdk_libs() -> Optional[pathlib.Path]:
  """Returns the QNN host library directory for the current host.

  On Linux aarch64 this is the newest build the host glibc supports. On Windows
  ARM64 this is the `aarch64-windows-msvc` runtime build.

  Returns:
    The host library directory, or None if the SDK files (or, on aarch64, a
    build compatible with the host glibc) are not present.
  """
  sdk_path = get_sdk_path()
  if not sdk_path:
    return None
  if _is_windows_arm64():
    lib_dir = sdk_path / "lib" / _WINDOWS_ARM64_LIB_SUBDIR
    return lib_dir if lib_dir.is_dir() else None
  if platform.machine() not in ("aarch64", "arm64"):
    return sdk_path / "lib" / _X86_64_HOST_LIB_SUBDIR
  glibc = _glibc_version()
  for min_glibc, subdir in _AARCH64_HOST_LIB_SUBDIRS:
    if glibc >= min_glibc and (sdk_path / "lib" / subdir).is_dir():
      return sdk_path / "lib" / subdir
  return None


def get_sdk_path() -> Optional[pathlib.Path]:
  """Returns the absolute path to the root of the downloaded SDK files."""
  is_linux = sys.platform == "linux"
  is_supported_arch = platform.machine() in (
      "x86_64",
      "i386",
      "i686",
      "aarch64",
      "arm64",
  )
  if not ((is_linux and is_supported_arch) or _is_windows_arm64()):
    raise NotImplementedError(
        "Currently LiteRT NPU for Qualcomm is only supported on Linux x86_64,"
        " Linux aarch64 and Windows ARM64."
    )
  try:
    package_dir = pathlib.Path(__file__).parent.resolve()
    sdk_path = package_dir / _SDK_FILES_SUBDIR
    if sdk_path.is_dir():
      return sdk_path
    else:
      return None
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f"Error determining SDK path: {e}", file=sys.stderr)
    return None


_WINDOWS_DLL_DIR_HANDLES: dict[str, object] = {}


def _add_windows_dll_directory(directory: pathlib.Path) -> None:
  """Makes DLLs in `directory` loadable by name in this process.

  `QnnHtp.dll` is loaded from `qnn_lib_dir`, but it loads its own dependencies
  (e.g. `QnnHtpV73Stub.dll`) by name, so the directory must be on the DLL
  search path: `os.add_dll_directory` for `LoadLibraryEx` and `PATH` for plain
  `LoadLibrary` (and child processes such as `apply_plugin_main.exe`).

  Args:
    directory: Directory containing DLLs to add to the search path.
  """
  dir_str = str(directory)
  dir_key = dir_str.lower()
  if (
      hasattr(os, "add_dll_directory")
      and dir_key not in _WINDOWS_DLL_DIR_HANDLES
  ):
    try:
      _WINDOWS_DLL_DIR_HANDLES[dir_key] = os.add_dll_directory(dir_str)
    except OSError:
      pass
  path = os.environ.get("PATH", "")
  parts = path.split(os.pathsep) if path else []
  if dir_key not in {p.lower() for p in parts}:
    os.environ["PATH"] = os.pathsep.join([dir_str] + parts)


def default_qualcomm_paths() -> Optional[tuple[pathlib.Path, str]]:
  """Returns `(qnn_lib_dir, dsp_skel_dir)` for running QNN HTP on this device.

  These are the values for `QualcommOptions.qnn_lib_dir` / `dsp_skel_dir`.
  `dsp_skel_dir` lists every `hexagon-<arch>/unsigned` directory, separated
  by `;` as in ADSP_LIBRARY_PATH: the QNN stub loads the skel named for the
  device's Hexagon arch (e.g. `libQnnHtpV75Skel.so`), so no SoC lookup is
  needed.

  On Linux, the QNN host libraries also need the system FastRPC client library
  (`libcdsprpc.so`) and `libatomic.so.1`; see the device setup documentation.
  On Windows ARM64, `qnn_lib_dir` is also added to the process DLL search path.

  Returns:
    The two values on a Linux aarch64 device with a compute DSP or a Windows on
    Snapdragon device, with the SDK files present; otherwise None.
  """
  if _is_linux_aarch64():
    if not os.path.exists(_FASTRPC_CDSP_DEVICE):
      return None
  elif not _is_qualcomm_windows_device():
    return None
  lib_dir = path_to_sdk_libs()
  if not lib_dir:
    return None
  skel_dirs = sorted(lib_dir.parent.glob("hexagon-v*/unsigned"))
  if not skel_dirs:
    return None
  if _is_windows_arm64():
    _add_windows_dll_directory(lib_dir)
  return lib_dir, ";".join(map(str, skel_dirs))
