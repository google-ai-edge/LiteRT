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

"""Python wrapper for LiteRT environments."""

import dataclasses
import glob
import os
import sys
from typing import Optional, Union

# pylint: disable=g-import-not-at-top
if not os.path.splitext(__file__)[0].endswith(
    os.path.join("ai_edge_litert", "environment")
):
  from litert.python.litert_wrapper.environment_wrapper import (
      _pywrap_litert_environment_wrapper as _env,
  )
else:
  from ai_edge_litert import _pywrap_litert_environment_wrapper as _env
# pylint: enable=g-import-not-at-top


def _vendors_dir() -> str:
  """Returns `<ai_edge_litert>/vendors`, or "" when the package is missing."""
  try:
    import ai_edge_litert  # pytype: disable=import-error  # pylint: disable=g-import-not-at-top
  except ImportError:
    return ""
  path = os.path.join(
      os.path.dirname(os.path.abspath(ai_edge_litert.__file__)), "vendors"
  )
  return path if os.path.isdir(path) else ""


def _vendor_subdir_if_has_lib(
    vendor: str, subdir: str, win_glob: str, nix_glob: str
) -> str:
  """Returns `<vendors>/<vendor>/<subdir>/` iff it has a matching library."""
  base = _vendors_dir()
  if not base:
    return ""
  lib_glob = win_glob if sys.platform == "win32" else nix_glob
  path = os.path.join(base, vendor, subdir)
  if os.path.isdir(path) and glob.glob(os.path.join(path, lib_glob)):
    return path
  return ""


def _find_first_vendor_with_library(
    subdir: str, win_glob: str, nix_glob: str
) -> str:
  """Returns the vendor name of the first `<vendors>/<vendor>/<subdir>/` dir
  that contains a matching library, or "" when none is found."""
  base = _vendors_dir()
  if not base:
    return ""
  try:
    entries = sorted(os.listdir(base))
  except OSError:
    return ""
  for entry in entries:
    if _vendor_subdir_if_has_lib(entry, subdir, win_glob, nix_glob):
      return entry
  return ""


def _autodiscover_dispatch_library_path() -> str:
  """Returns a vendor dispatch directory under the installed ai_edge_litert."""
  vendor = _find_first_vendor_with_library(
      "dispatch", "LiteRtDispatch*.dll", "libLiteRtDispatch_*.so"
  )
  if not vendor:
    return ""
  return _vendor_subdir_if_has_lib(
      vendor, "dispatch", "LiteRtDispatch*.dll", "libLiteRtDispatch_*.so"
  )


def _autodiscover_compiler_plugin_path(preferred_vendor: str = "") -> str:
  """Returns a vendor compiler-plugin directory under the installed
  ai_edge_litert.

  When `preferred_vendor` is set, prefer that vendor's compiler plugin so the
  compiler and dispatch libraries come from the same vendor. Falls back to the
  first vendor with a compiler plugin otherwise.
  """
  compiler_win = "LiteRtCompilerPlugin*.dll"
  compiler_nix = "libLiteRtCompilerPlugin_*.so"
  if preferred_vendor:
    path = _vendor_subdir_if_has_lib(
        preferred_vendor, "compiler", compiler_win, compiler_nix
    )
    if path:
      return path
  vendor = _find_first_vendor_with_library(
      "compiler", compiler_win, compiler_nix
  )
  if not vendor:
    return ""
  return _vendor_subdir_if_has_lib(
      vendor, "compiler", compiler_win, compiler_nix
  )


@dataclasses.dataclass(frozen=True)
class EnvironmentOptions:
  """Environment-level options for a shared LiteRT environment."""

  runtime_path: Optional[str] = None
  compiler_plugin_path: str = ""
  dispatch_library_path: str = ""


class Environment:
  """Python wrapper for a shared LiteRT environment."""

  def __init__(
      self,
      capsule,
      options: EnvironmentOptions,
  ):
    self._capsule = capsule
    self._options = options

  @classmethod
  def create(
      cls,
      *,
      options: Optional[Union[EnvironmentOptions, str]] = None,
      runtime_path: Optional[str] = None,
      compiler_plugin_path: str = "",
      dispatch_library_path: str = "",
  ) -> "Environment":
    """Creates a reusable LiteRT environment.

    Args:
      options: Optional grouped environment options. Use this to mirror the
        native C++ EnvironmentOptions API.
      runtime_path: Optional path to the LiteRT runtime library directory. This
        shortcut is mutually exclusive with options.
      compiler_plugin_path: Optional path to compiler plugin libraries. This
        shortcut is mutually exclusive with options. When omitted or empty,
        auto-discovers a vendor compiler-plugin directory under the installed
        `ai_edge_litert` package (the first `vendors/<vendor>/compiler/` dir
        that contains a compiler-plugin shared library). Needed for JIT
        compilation of raw .tflite models.
      dispatch_library_path: Optional path to dispatch libraries. This shortcut
        is mutually exclusive with options. When omitted or empty, auto-
        discovers a vendor dispatch directory under the installed
        `ai_edge_litert` package (the first `vendors/<vendor>/dispatch/` dir
        that contains a dispatch shared library).

    Returns:
      A new Environment instance.
    """
    provided_runtime_path = runtime_path
    if isinstance(options, str):
      if provided_runtime_path is not None:
        raise ValueError("runtime_path was provided twice.")
      provided_runtime_path = options
      options = None
    if options is not None and (
        provided_runtime_path is not None
        or compiler_plugin_path
        or dispatch_library_path
    ):
      raise ValueError("Pass either options or environment path shortcuts.")
    if options is None:
      options = EnvironmentOptions(
          runtime_path=provided_runtime_path,
          compiler_plugin_path=compiler_plugin_path,
          dispatch_library_path=dispatch_library_path,
      )

    # Auto-discover a vendor dispatch directory when the caller did not set
    # one, so NPU accelerator registration succeeds out of the box.
    if not options.dispatch_library_path:
      discovered = _autodiscover_dispatch_library_path()
      if discovered:
        options = dataclasses.replace(options, dispatch_library_path=discovered)

    # Auto-discover the compiler-plugin directory too, so JIT compilation
    # for raw .tflite models works out of the box. Without this, dispatch
    # loads and silently returns a non-partitioned model that the TFLite
    # fallback interpreter runs on CPU while the benchmark still reports
    # "Fully accelerated: True". Pair the compiler plugin with whichever
    # vendor's dispatch library was selected, so mixed wheels (e.g.
    # google_tensor + intel_openvino) do not cross-wire an Intel compile
    # into a google_tensor dispatch path.
    if not options.compiler_plugin_path:
      preferred_vendor = ""
      if options.dispatch_library_path:
        # dispatch path looks like <vendors>/<vendor>/dispatch/ — extract
        # <vendor>.
        parent = os.path.dirname(options.dispatch_library_path.rstrip(os.sep))
        preferred_vendor = os.path.basename(parent)
      discovered = _autodiscover_compiler_plugin_path(preferred_vendor)
      if discovered:
        options = dataclasses.replace(options, compiler_plugin_path=discovered)

    # Determine the final runtime path to be used.
    final_runtime_path = (
        provided_runtime_path
        or options.runtime_path
        or os.path.dirname(os.path.abspath(__file__))
    )

    capsule = _env.CreateEnvironment(
        runtime_path=final_runtime_path,
        compiler_plugin_path=options.compiler_plugin_path,
        dispatch_library_path=options.dispatch_library_path,
    )
    return cls(
        capsule, dataclasses.replace(options, runtime_path=final_runtime_path)
    )

  @property
  def options(self) -> EnvironmentOptions:
    """Returns the environment options used to create this environment."""
    return self._options

  @property
  def capsule(self):
    if self._capsule is None:
      raise ValueError("Environment is no longer valid")
    return self._capsule
