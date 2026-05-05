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
import os
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
        shortcut is mutually exclusive with options.
      dispatch_library_path: Optional path to dispatch libraries. This shortcut
        is mutually exclusive with options.

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
