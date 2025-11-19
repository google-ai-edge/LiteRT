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

"""Wrapper for calling the apply plugin tooling."""


import os
import pathlib
import re
import subprocess
import tempfile

from litert.python.aot.core import common
from litert.python.aot.core import components
from litert.python.aot.core import types

_BINARY = pathlib.Path("tools/apply_plugin_main")

_RE_PARTITION_STATS = re.compile(
    r"PartitionSubgraph: (\d+), selected num ops: (\d+), from totoal ops:"
    r" (\d+), num partitions: (\d+)"
)


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

  def __call__(
      self,
      input_model: types.Model,
      output_model: types.Model,
      soc_manufacturer: str,
      soc_model: str,
      sdk_libs_path: str | None = None,
      **kwargs,
  ):
    """Applies a plugin to the input model.

    Args:
      input_model: The path to the input model.
      output_model: The path to the output model.
      soc_manufacturer: The SOC manufacturer of the plugin.
      soc_model: The SOC model of the plugin.
      sdk_libs_path: The path to the SDK libs. If not provided,
        the default SDK path will be used.
      **kwargs: Additional arguments to pass to the underlying binary.

    Returns:
      The output model.

    Raises:
      ValueError: If no tflite model was created by the underying binary.
    """
    if input_model.in_memory:
      tmp_file = tempfile.NamedTemporaryFile(mode="wb")
      input_model.save(tmp_file.name)
    else:
      tmp_file = None

    binary = common.get_resource(_BINARY)
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
    ld_library_path = common.construct_ld_library_path()
    if ld_library_path:
      if sdk_libs_path:
        ld_library_path = f"{sdk_libs_path}{os.pathsep}{ld_library_path}"
      env["LD_LIBRARY_PATH"] = ld_library_path

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
    output_model.partition_stats = types.PartitionStats(
        subgraph_stats=[
            types.SubgraphPartitionStats(
                subgraph_index=int(s[0]),
                num_ops_offloaded=int(s[1]),
                num_total_ops=int(s[2]),
                num_partitions_offloaded=int(s[3]),
            )
            for s in partition_stats
        ]
    )
    if tmp_file is not None:
      tmp_file.close()
