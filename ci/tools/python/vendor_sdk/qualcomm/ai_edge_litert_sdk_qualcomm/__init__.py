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

import functools
import os
import pathlib
import platform
import re
import sys
from typing import Optional
import warnings

_SDK_FILES_SUBDIR = "data"

# Environment variable that overrides Hexagon arch auto-detection, e.g. "v75".
HEXAGON_ARCH_ENV_VAR = "LITERT_QNN_HEXAGON_ARCH"

_X86_64_HOST_LIB_SUBDIR = "x86_64-linux-clang"

# QAIRT ships several Linux aarch64 host library builds. Each entry is
# (minimum glibc version, subdir); the first one the host glibc satisfies and
# that exists in the SDK is used. The gcc11.2 build needs glibc >= 2.34
# (e.g. Ubuntu 22.04+, Qualcomm Linux) and ships HTP stubs for every arch. The
# gcc9.4 build targets Ubuntu 20.04 but only supports HTP v68.
_AARCH64_HOST_LIB_SUBDIRS = (
    ((2, 34), "aarch64-oe-linux-gcc11.2"),
    ((2, 29), "aarch64-ubuntu-gcc9.4"),
    ((0, 0), "aarch64-oe-linux-gcc9.3"),
)

# Hexagon DSP architecture per Qualcomm SoC (SoCs without an HTP are omitted).
# LINT.IfChange(soc_hexagon_arch)
_SOC_TO_HEXAGON_ARCH = {
    "SW6100": "v81",
    "SM8850": "v81",
    "SM8845": "v81",
    "SM8845P": "v81",
    "SC8480XP": "v81",
    "SAR2230P": "v81",
    "SM8750": "v79",
    "SXR2330P": "v79",
    "QCS8625": "v75",
    "SM8650": "v75",
    "QCS8300": "v75",
    "QCS8275": "v75",
    "QCM6690": "v73",
    "QCS6690": "v73",
    "QCS9100": "v73",
    "QCS9075": "v73",
    "SSG2125P": "v73",
    "SSG2115P": "v73",
    "SAR2130P": "v73",
    "SM8635": "v73",
    "SM8550": "v73",
    "SM7675": "v73",
    "SM7550": "v73",
    "SM7435": "v73",
    "SM6450": "v73",
    "QCM8550LA": "v73",
    "QCM8550LE": "v73",
    "SC8380XP": "v73",
    "SXR1230P": "v73",
    "SA8255": "v73",
    "SM8475": "v69",
    "SM8450": "v69",
    "SM7475": "v69",
    "SM7450": "v69",
    "SM7425": "v69",
    "SXR2230P": "v69",
    "SXR2250P": "v69",
    "SM8350": "v68",
    "SM8350P": "v68",
    "SM7350": "v68",
    "SM7325": "v68",
    "SM7315": "v68",
    "QCM6490": "v68",
    "SA8295": "v68",
    "SDM865": "v66",
    "SM8250": "v66",
    "SM8150": "v66",
    "SM7250": "v66",
    "SM7225": "v66",
    "SM7125": "v66",
    "SM6350": "v66",
    "SM6225": "v66",
    "SM6150": "v66",
    "SM6125": "v66",
    "SM4350": "v66",
    "QRB5165U": "v66",
    "QRB5165LE": "v66",
    "QCS7230LA": "v66",
    "QCS7230LE": "v66",
    "SM6375": "v66",
    "SM7150": "v65",
    "SDM845": "v65",
}
# LINT.ThenChange(../../../../../../../litert/vendors/qualcomm/supported_soc.csv)

_SOC0_MACHINE_PATH = "/sys/devices/soc0/machine"
_DEVICE_TREE_COMPATIBLE_PATH = "/proc/device-tree/compatible"

_PRELOADED = False


def _is_linux_aarch64() -> bool:
  return sys.platform == "linux" and platform.machine() in ("aarch64", "arm64")


def _glibc_version() -> tuple[int, int]:
  """Returns the host glibc (major, minor), or (0, 0) if unknown."""
  try:
    libc = os.confstr("CS_GNU_LIBC_VERSION") or ""  # e.g. "glibc 2.39"
  except (AttributeError, ValueError, OSError):
    return (0, 0)
  match = re.search(r"(\d+)\.(\d+)", libc)
  if not match:
    return (0, 0)
  return (int(match.group(1)), int(match.group(2)))


def path_to_sdk_libs() -> Optional[pathlib.Path]:
  """Returns the QNN host library directory matching the current host.

  On Linux aarch64 the build is picked by the host glibc version (see
  `_AARCH64_HOST_LIB_SUBDIRS`); on x86_64 it is `lib/x86_64-linux-clang`.

  Returns:
    The host library directory, or None if the SDK files are not present.
  """
  sdk_path = get_sdk_path()
  if not sdk_path:
    return None
  if platform.machine() not in ("aarch64", "arm64"):
    return sdk_path / "lib" / _X86_64_HOST_LIB_SUBDIR
  glibc = _glibc_version()
  candidates = [subdir for _, subdir in _AARCH64_HOST_LIB_SUBDIRS]
  compatible = [
      subdir for min_glibc, subdir in _AARCH64_HOST_LIB_SUBDIRS
      if glibc >= min_glibc
  ]
  for subdir in compatible + candidates:
    if (sdk_path / "lib" / subdir).is_dir():
      return sdk_path / "lib" / subdir
  # Nothing present; report the preferred location for diagnostics.
  return sdk_path / "lib" / (compatible or candidates)[0]


def path_to_dsp_skel(hexagon_arch: str) -> Optional[pathlib.Path]:
  """Returns the unsigned Hexagon DSP skel library directory.

  Args:
    hexagon_arch: Hexagon DSP architecture of the target SoC, e.g. "v75" for
      QCS8275 / SM8650. See `hexagon_arch_for_soc`.

  Returns:
    `<sdk>/lib/hexagon-<hexagon_arch>/unsigned`, or None if the SDK files are
    not present.
  """
  sdk_path = get_sdk_path()
  if not sdk_path:
    return None
  return sdk_path / f"lib/hexagon-{hexagon_arch}/unsigned"


def hexagon_arch_for_soc(soc_model: str) -> Optional[str]:
  """Returns the Hexagon arch (e.g. "v75") of a Qualcomm SoC, or None."""
  return _SOC_TO_HEXAGON_ARCH.get(soc_model.strip().upper())


def _read_text(path: str) -> str:
  try:
    with open(path, "rb") as f:
      return f.read().decode("utf-8", errors="ignore")
  except OSError:
    return ""


def _soc_name_candidates() -> list[str]:
  """Returns SoC name candidates for the current device, most specific first."""
  candidates = []
  machine = _read_text(_SOC0_MACHINE_PATH).strip().upper()
  if machine:
    candidates.append(machine)
  # e.g. "qcom,qcs8300-ride\0qcom,qcs8300\0" -> ["QCS8300", "QCS8300"].
  for entry in _read_text(_DEVICE_TREE_COMPATIBLE_PATH).split("\0"):
    vendor, _, model = entry.partition(",")
    if vendor == "qcom" and model:
      candidates.append(model.split("-")[0].upper())
  # Also try without a trailing letter suffix, e.g. "SA8775P" -> "SA8775".
  candidates += [re.sub(r"[A-Z]+$", "", c) for c in candidates]
  return candidates


# functools.cache needs Python 3.9; this package still supports 3.8.
@functools.lru_cache(maxsize=None)  # pylint: disable=g-use-functools-cache
def detect_soc_model() -> Optional[str]:
  """Best-effort name of the Qualcomm SoC this process runs on.

  Reads `/sys/devices/soc0/machine` and the device tree `compatible` string.

  Returns:
    A SoC name known to this package (e.g. "QCS8275"), or None.
  """
  if sys.platform != "linux":
    return None
  for candidate in _soc_name_candidates():
    if candidate in _SOC_TO_HEXAGON_ARCH:
      return candidate
  return None


def detect_hexagon_arch() -> Optional[str]:
  """Returns the Hexagon arch of the current device, or None if unknown.

  `$LITERT_QNN_HEXAGON_ARCH` (e.g. "v75") takes precedence over SoC
  auto-detection.
  """
  override = os.environ.get(HEXAGON_ARCH_ENV_VAR, "").strip().lower()
  if override:
    return override
  soc_model = detect_soc_model()
  return hexagon_arch_for_soc(soc_model) if soc_model else None


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
  if not (is_linux and is_supported_arch):
    raise NotImplementedError(
        "Currently LiteRT NPU AOT for Qualcomm is only supported on Linux"
        " x86_64 and aarch64 architectures."
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


def _preload_fastrpc_dependencies() -> None:
  """Preloads libatomic and the first loadable libcdsprpc with RTLD_GLOBAL."""
  global _PRELOADED
  if _PRELOADED:
    return
  _PRELOADED = True
  import ctypes  # pylint: disable=g-import-not-at-top

  def _try_preload(candidates: tuple[str, ...]) -> None:
    for candidate in candidates:
      try:
        ctypes.CDLL(candidate, mode=os.RTLD_LAZY | ctypes.RTLD_GLOBAL)
        return
      except OSError:
        continue

  # libatomic.so.1 is a system dependency of libQnnSystem.so (Ubuntu package
  # libatomic1); libcdsprpc.so comes from the device BSP. Neither is bundled.
  # Only the first loadable libcdsprpc is preloaded to avoid two copies.
  _try_preload(("libatomic.so.1",))
  _try_preload((
      "/usr/lib/aarch64-linux-gnu/libcdsprpc.so.1",
      "/usr/lib/libcdsprpc.so.1",
      "/usr/lib/libcdsprpc.so",
      "libcdsprpc.so.1",
      "libcdsprpc.so",
  ))


def configure_runtime(
    hexagon_arch: Optional[str] = None, *, warn_if_undetected: bool = True
) -> Optional[str]:
  """Prepares the process for on-device QNN HTP execution on Linux aarch64.

  Called lazily by ai_edge_litert right before an NPU CompiledModel is created
  or an on-device AOT compilation is started; safe to call repeatedly. It
  preloads FastRPC dependencies once and appends the Hexagon skel directory of
  `hexagon_arch` to ADSP_LIBRARY_PATH (never prepending, so user-provided
  entries keep precedence). It is a no-op on other platforms.

  Args:
    hexagon_arch: Hexagon DSP architecture (e.g. "v75"). If None, it is taken
      from `$LITERT_QNN_HEXAGON_ARCH` or detected from the device SoC.
    warn_if_undetected: Whether to emit a RuntimeWarning when the arch cannot
      be determined. Callers that already have an explicit skel directory can
      pass False.

  Returns:
    The Hexagon arch whose skel directory is on ADSP_LIBRARY_PATH, or None if
    nothing was configured.
  """
  if not _is_linux_aarch64():
    return None
  _preload_fastrpc_dependencies()
  arch = hexagon_arch or detect_hexagon_arch()
  if not arch:
    if warn_if_undetected:
      warnings.warn(
          "Could not detect the Hexagon DSP architecture of this device; set"
          f" ${HEXAGON_ARCH_ENV_VAR} (e.g. 'v75') or"
          " QualcommOptions.dsp_skel_dir.",
          RuntimeWarning,
          stacklevel=2,
      )
    return None
  try:
    skel_dir = path_to_dsp_skel(arch)
  except NotImplementedError:
    return None
  if not skel_dir or not skel_dir.is_dir():
    warnings.warn(
        f"Hexagon skel directory for {arch!r} not found ({skel_dir}); NPU"
        " execution may fail. Check the Hexagon arch or reinstall"
        " ai-edge-litert-sdk-qualcomm.",
        RuntimeWarning,
        stacklevel=2,
    )
    return None
  skel_str = str(skel_dir)
  entries = [e for e in os.environ.get("ADSP_LIBRARY_PATH", "").split(";") if e]
  if skel_str not in entries:
    os.environ["ADSP_LIBRARY_PATH"] = ";".join(entries + [skel_str])
  return arch
