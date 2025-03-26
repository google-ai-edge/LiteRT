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


import pathlib
import re
import subprocess
import tempfile

from ai_edge_litert.aot.core import common
from ai_edge_litert.aot.core import components
from ai_edge_litert.aot.core import types

_BINARY = pathlib.Path("tools/apply_plugin_main")

_RE_PARTITION_STATS = re.compile(
    r"PartitionSubgraph: (\d+), selected num ops: (\d+), from totoal ops:"
    r" (\d+), num partitions: (\d+)"
)


class ApplyPlugin(components.ApplyPluginT):
  """Wrapper for calling the apply plugin tooling."""

  def __init__(self, experimental_capture_stderr: bool = False):
    self._experimental_capture_stderr = experimental_capture_stderr

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
  ):
    """Applies a plugin to the input model.

    Args:
      input_model: The path to the input model.
      output_model: The path to the output model.
      soc_manufacturer: The SOC manufacturer of the plugin.
      soc_model: The SOC model of the plugin.

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
        "apply",
        f"--model={str(input_model.path)}",
        f"--o={str(output_model.path)}",
        f"--soc_man={soc_manufacturer}",
        f"--soc_model={soc_model}",
        f"--err={self.default_err}",
    ]
    try:
      result = subprocess.run(
          args,
          check=True,
          capture_output=self._experimental_capture_stderr,
          text=True,
      )
    except subprocess.CalledProcessError as e:
      tmp_file = tempfile.NamedTemporaryFile(mode="wb", delete=False)
      tmp_file.write(e.stderr)
      tmp_file.close()
      raise ValueError(
          f"{self.component_name} failed to apply plugin. See"
          f" {tmp_file.name} for details."
      ) from e

    if not common.is_tflite(output_model.path):
      raise ValueError(f"{output_model.path} is not a TFLite model.")
    if tmp_file is not None:
      tmp_file.close()

    # TODO(b/405256024): Use proper dataclass for passing stats
    # instead of parsing.
    if self._experimental_capture_stderr:
      partition_stats = _RE_PARTITION_STATS.findall(result.stderr)
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
