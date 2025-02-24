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
import subprocess

from google3.third_party.odml.litert.litert.python.google.core import common
from google3.third_party.odml.litert.litert.python.google.core import components
from google3.third_party.odml.litert.litert.python.google.core import types

_BINARY = pathlib.Path("tools/apply_plugin_main")


class ApplyPlugin(components.ApplyPluginT):
  """Wrapper for calling the apply plugin tooling."""

  @property
  def default_err(self) -> str:
    # NOTE: Capture stderr from underlying binary.
    return "none"

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

    binary = common.get_resource(_BINARY)
    args = [
        str(binary),
        f"--model={str(input_model)}",
        f"--o={str(output_model)}",
        f"--soc_man={soc_manufacturer}",
        f"--soc_model={soc_model}",
        f"--err={self.default_err}",
    ]
    subprocess.run(args, check=True)
    if not common.is_tflite(output_model):
      raise ValueError(f"{output_model} is not a TFLite model.")
