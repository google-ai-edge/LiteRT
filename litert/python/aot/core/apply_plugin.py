# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Wrapper for calling the apply plugin tooling.

Compiler plugins are normally native shared libraries applied by the
`apply_plugin_main` binary. A SoC manufacturer may instead register a compiler
plugin implemented in Python with `register_python_plugin`; `ApplyPlugin` then
runs it in the calling process instead of launching the binary.
"""

from collections.abc import Sequence
import os
import pathlib
import re
import subprocess
import sys
import tempfile
from typing import Any, Protocol

from litert.python.aot.core import aot_types
from litert.python.aot.core import common
from litert.python.aot.core import components

_BINARY = pathlib.Path(
    "tools/apply_plugin_main.exe"
    if sys.platform == "win32"
    else "tools/apply_plugin_main"
)

_RE_PARTITION_STATS = re.compile(
    r"Partitioned subgraph<(\d+)>, selected (\d+) ops, from a total of "
    r"(\d+) ops. resulted in (\d+) partitions."
)

# Flatbuffer file identifier of TFLite models, stored at bytes 4-7.
_TFLITE_FILE_IDENTIFIER = b"TFL3"


def _is_tflite_model(model: aot_types.Model) -> bool:
  """Returns whether `model`, in memory or on disk, is a TFLite flatbuffer."""
  if model.in_memory:
    header = model.model_bytes[:8]
  elif common.is_tflite(model.path):
    with model.path.open("rb") as f:
      header = f.read(8)
  else:
    return False
  return header[4:8] == _TFLITE_FILE_IDENTIFIER


class PythonPlugin(Protocol):
  """A compiler plugin implemented in Python and run in the calling process.

  The plugin compiles `input_model` for `soc_model` and writes the compiled
  model to `output_model` (either to `output_model.path` or with
  `output_model.set_bytes`).

  Unlike native plugins, which `apply_plugin_main` runs in a subprocess:
    * `input_model` may be in memory (`input_model.in_memory`); it is not
      written to a temporary file first.
    * There is no crash isolation: a crash or hang of the plugin takes down or
      blocks the calling process.
  """

  def __call__(
      self,
      input_model: aot_types.Model,
      output_model: aot_types.Model,
      soc_model: str,
      *,
      subgraphs_to_compile: Sequence[int] | None,
      **kwargs: Any,
  ) -> aot_types.PartitionStats:
    """Compiles the model.

    Args:
      input_model: The model to compile, on disk or in memory.
      output_model: Where to write the compiled model.
      soc_model: The SoC model to compile for.
      subgraphs_to_compile: Indices of the subgraphs to compile, or `None` to
        let the plugin compile every subgraph it supports. Never empty.
      **kwargs: Backend-specific arguments passed to `ApplyPlugin`, plus
        `sdk_libs_path` if one was given.

    Returns:
      The partition statistics of the compiled model.
    """
    ...


_PYTHON_PLUGINS: dict[str, PythonPlugin] = {}


def register_python_plugin(soc_manufacturer: str, plugin: PythonPlugin) -> None:
  """Registers the Python compiler plugin of a SoC manufacturer.

  Registering the same plugin again is a no-op.

  Args:
    soc_manufacturer: The SoC manufacturer, as passed to `ApplyPlugin`.
    plugin: The plugin.

  Raises:
    ValueError: If a different plugin is already registered for the SoC
      manufacturer.
  """
  registered = _PYTHON_PLUGINS.get(soc_manufacturer)
  if registered is not None and registered is not plugin:
    raise ValueError(
        "A different Python compiler plugin is already registered for"
        f" {soc_manufacturer}."
    )
  _PYTHON_PLUGINS[soc_manufacturer] = plugin


def unregister_python_plugin(soc_manufacturer: str) -> None:
  """Removes the Python compiler plugin of a SoC manufacturer, if any."""
  _PYTHON_PLUGINS.pop(soc_manufacturer, None)


def find_python_plugin(soc_manufacturer: str) -> PythonPlugin | None:
  """Returns the Python compiler plugin of a SoC manufacturer, if any."""
  return _PYTHON_PLUGINS.get(soc_manufacturer)


class ApplyPlugin(components.ApplyPluginT):
  """Wrapper for calling the apply plugin tooling."""

  def __init__(
      self,
      experimental_capture_stderr: bool = False,
      subgraphs_to_compile: list[int] | None = None,
  ):
    self._experimental_capture_stderr = experimental_capture_stderr
    self._subgraphs_to_compile = subgraphs_to_compile

  @property
  def default_err(self) -> str:
    # NOTE: Capture stderr from underlying binary.
    return "--"

  @property
  def component_name(self) -> str:
    return "apply_plugin"

  def _apply_python_plugin(
      self,
      plugin: PythonPlugin,
      input_model: aot_types.Model,
      output_model: aot_types.Model,
      soc_model: str,
      sdk_libs_path: str | None,
      **kwargs: Any,
  ) -> None:
    """Runs a Python compiler plugin and records its partition statistics."""
    if sdk_libs_path is not None:
      kwargs["sdk_libs_path"] = sdk_libs_path
    partition_stats = plugin(
        input_model,
        output_model,
        soc_model,
        subgraphs_to_compile=self._subgraphs_to_compile or None,
        **kwargs,
    )
    if not isinstance(partition_stats, aot_types.PartitionStats):
      raise TypeError(
          f"{self.component_name}: the Python compiler plugin returned"
          f" {type(partition_stats).__name__}, expected PartitionStats."
      )
    if not _is_tflite_model(output_model):
      raise ValueError(
          f"{self.component_name}: the Python compiler plugin did not produce"
          " a TFLite model."
      )
    output_model.partition_stats = partition_stats

  def __call__(
      self,
      input_model: aot_types.Model,
      output_model: aot_types.Model,
      soc_manufacturer: str,
      soc_model: str,
      sdk_libs_path: str | None = None,
      **kwargs,
  ):
    """Applies a plugin to the input model.

    If a Python compiler plugin is registered for `soc_manufacturer` (see
    `register_python_plugin`), it is run in this process. Otherwise the native
    plugin is applied by the `apply_plugin_main` binary.

    Args:
      input_model: The path to the input model.
      output_model: The path to the output model.
      soc_manufacturer: The SOC manufacturer of the plugin.
      soc_model: The SOC model of the plugin.
      sdk_libs_path: The path to the SDK libs. If not provided, the default SDK
        path will be used.
      **kwargs: Additional arguments to pass to the underlying binary, or to the
        Python compiler plugin.

    Returns:
      The output model.

    Raises:
      ValueError: If no tflite model was created by the underlying binary or by
        the Python compiler plugin.
      TypeError: If the Python compiler plugin does not return
        `PartitionStats`.
    """
    python_plugin = find_python_plugin(soc_manufacturer)
    if python_plugin is not None:
      self._apply_python_plugin(
          python_plugin,
          input_model,
          output_model,
          soc_model,
          sdk_libs_path=sdk_libs_path,
          **kwargs,
      )
      return

    if input_model.in_memory:
      tmp_file = tempfile.NamedTemporaryFile(mode="wb", delete=False)
      input_model.save(tmp_file.name)
      tmp_file.close()
    else:
      tmp_file = None

    try:
      try:
        binary = common.get_resource(_BINARY)
      except FileNotFoundError as e:
        raise FileNotFoundError(
            "Failed to find apply plugin binary. AOT might not be available on"
            " your platform."
        ) from e
      args = [
          str(binary),
          "--cmd=apply",
          f"--model={str(input_model.path)}",
          f"--o={str(output_model.path)}",
          f"--soc_manufacturer={soc_manufacturer}",
          f"--soc_model={soc_model}",
          f"--err={self.default_err}",
      ]
      extra_args = [f"--{key}={value}" for key, value in kwargs.items()]
      args.extend(extra_args)
      if self._subgraphs_to_compile:
        subgraphs_to_compile = ",".join(
            str(s) for s in self._subgraphs_to_compile
        )
        args.append(f"--subgraphs={subgraphs_to_compile}")
      env = os.environ.copy()
      lib_search_path = common.construct_ld_library_path()
      if sdk_libs_path:
        lib_search_path = (
            f"{sdk_libs_path}{os.pathsep}{lib_search_path}"
            if lib_search_path
            else sdk_libs_path
        )
      if lib_search_path:
        if sys.platform == "win32":
          env["PATH"] = lib_search_path + os.pathsep + env.get("PATH", "")
        else:
          env["LD_LIBRARY_PATH"] = lib_search_path

      result = subprocess.run(
          args,
          check=False,
          text=True,
          stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT,
          env=env,
      )
      if result.returncode:
        log_file = tempfile.NamedTemporaryFile(
            suffix=".error", mode="w", delete=False
        )
        log_file.write(result.stdout)
        log_file.close()
        raise ValueError(
            f"{self.component_name} failed to apply plugin. See"
            f" {log_file.name} for details."
        )

      if not common.is_tflite(output_model.path):
        raise ValueError(f"{output_model.path} is not a TFLite model.")

      partition_stats = _RE_PARTITION_STATS.findall(result.stdout)
      output_model.partition_stats = aot_types.PartitionStats(
          subgraph_stats=[
              aot_types.SubgraphPartitionStats(
                  subgraph_index=int(s[0]),
                  num_ops_offloaded=int(s[1]),
                  num_total_ops=int(s[2]),
                  num_partitions_offloaded=int(s[3]),
              )
              for s in partition_stats
          ]
      )
    finally:
      if tmp_file is not None:
        try:
          os.unlink(tmp_file.name)
        except OSError:
          pass
