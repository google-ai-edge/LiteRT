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

"""Implementations for the main public API functionalities."""

import pathlib

from google3.third_party.odml.litert.litert.python.google.core import common
from google3.third_party.odml.litert.litert.python.google.core import components
from google3.third_party.odml.litert.litert.python.google.core import types


def prepare_for_npu(
    flatbuffer: types.Model,
    output_dir: pathlib.Path,
    backend_class: types.BackendT,
    config: types.Config,
    plugin: components.ApplyPluginT,
    transforms: components.MlirTransformsT | None = None,
    quantizer: components.AieQuantizerT | None = None,
) -> types.Model:
  """Prepares a TFLite model for NPU execution.

  High level command that erforms various backend specific pre-processing steps
  and then applies an NPU compiler to the given model.

  Args:
      flatbuffer: Path to the input flatbuffer file.
      output_dir: Directory to write the output flatbuffer file.
      backend_class: The backend to prepare the model for.
      config: The configuration for the backend.
      plugin: The plugin to apply to the model.
      transforms: The transforms to apply to the model.
      quantizer: The quantizer to apply to the model.

  Returns:
      Path to the output flatbuffer file.

  Raises:
      ValueError: If the given path is not a valid flatbuffer file.
  """

  backend = backend_class.create(config)

  pipeline: list[types.Component] = [
      c for c in [transforms, quantizer, plugin] if c is not None
  ]

  input_name_pref = flatbuffer.name.removesuffix(common.DOT_TFLITE)

  for component in pipeline:
    output_model = (
        output_dir
        / f"{input_name_pref}_{component.component_name}{common.DOT_TFLITE}"
    )
    backend.call_component(flatbuffer, output_model, component)
    flatbuffer = output_model
    if not common.is_tflite(flatbuffer):
      raise ValueError(
          f"{component.component_name} failed to produce a TFLite model."
      )

  return flatbuffer
