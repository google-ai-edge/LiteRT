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


import enum
import io
import os
import pathlib
import re
import subprocess
import tempfile

from litert.python.aot.core import common
from litert.python.aot.core import components
from litert.python.aot.core import types


# Defines how much of the apply_plugin_main logging the user cares about.
class Logs(enum.Enum):
  # Log everything.
  ALL = 0
  # Only log output from a failed apply plugin run.
  ERRORS = 1
  # Silent.
  NONE = 2


_BINARY = pathlib.Path("tools/apply_plugin_main")

_RE_PARTITION_STATS = re.compile(
    r"PartitionSubgraph: (\d+), selected num ops: (\d+), from totoal ops:"
    r" (\d+), num partitions: (\d+)"
)


class ApplyPlugin(components.ApplyPluginT):
  """Wrapper for calling the apply plugin tooling."""

  def __init__(
      self,
      *,
      log_dest: io.TextIOBase | None = None,
      log_level: Logs = Logs.NONE,
      subgraphs_to_compile: list[int] | None = None,
  ):
    self._subgraphs_to_compile = subgraphs_to_compile
    self._log_level = log_level
    self._log_dest = log_dest

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
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    if result.returncode:
      if self._log_level in [Logs.ERRORS, Logs.ALL] and self._log_dest:
        self._log_dest.write(result.stdout)
      raise ValueError(f"{self.component_name} failed to apply plugin.")

    if not common.is_tflite(output_model.path):
      raise ValueError(f"{output_model.path} is not a TFLite model.")

    # TODO(b/405256024): Use proper dataclass for passing stats
    # instead of parsing.
    if self._log_level == Logs.ALL and self._log_dest:
      self._log_dest.write(result.stdout)
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
