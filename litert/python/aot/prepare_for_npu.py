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
from typing import cast

# pylint: disable=g-import-not-at-top
# pytype: disable=import-error
try:
  from tqdm import auto as autotqdm
except ImportError:
  from tqdm.tqdm import auto as autotqdm
# pytype: enable=import-error

from litert.python.aot.core import common
from litert.python.aot.core import components
from litert.python.aot.core import types
from litert.python.aot.vendors import import_vendor

# pylint: enable=g-import-not-at-top


def resolve_backend(config: types.Config) -> types.BackendT:
  # Import the backend based on the ID.
  backend_id = config.get("backend_id", None)
  if backend_id is None:
    raise ValueError("Backend ID is required.")
  return import_vendor.import_vendor(backend_id)


def prepare_for_npu_multiple_configs(
    flatbuffer: types.Model,
    output_dir: pathlib.Path,
    configs: list[tuple[types.BackendT, types.Config]],
    plugin: components.ApplyPluginT,
    transforms: components.MlirTransformsT | None = None,
    quantizer: components.AieQuantizerT | None = None,
    keep_going: bool = False,
) -> types.CompilationResult:
  """Prepares a TFLite model for NPU execution."""
  backends = []
  for backend_class, config in configs:
    backend = backend_class.create(config)
    backends += list(backend.specialize())

  pipeline: list[types.Component] = [
      c for c in [transforms, quantizer, plugin] if c is not None
  ]
  return compile_model(flatbuffer, output_dir, backends, pipeline, keep_going)


def prepare_for_npu(
    flatbuffer: types.Model,
    output_dir: pathlib.Path,
    backend_class: types.BackendT,
    config: types.Config,
    plugin: components.ApplyPluginT,
    transforms: components.MlirTransformsT | None = None,
    quantizer: components.AieQuantizerT | None = None,
    keep_going: bool = False,
) -> types.CompilationResult:
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
      keep_going: Whether to keep going if some backends fail.

  Returns:
      List of the paths to the output flatbuffer file.

  Raises:
      ValueError: If the given path is not a valid flatbuffer file.
  """

  backend = backend_class.create(config)

  pipeline: list[types.Component] = [
      c for c in [transforms, quantizer, plugin] if c is not None
  ]
  backends = list(backend.specialize())
  return compile_model(flatbuffer, output_dir, backends, pipeline, keep_going)


def compile_model(
    flatbuffer: types.Model,
    output_dir: pathlib.Path,
    backends: list[types.Backend],
    pipeline: list[types.Component],
    keep_going: bool = False,
) -> types.CompilationResult:
  """Compiles a TFLite model for NPU execution."""
  if flatbuffer.in_memory:
    base_name = "model"
  else:
    base_name = flatbuffer.path.name.removesuffix(common.DOT_TFLITE)
  compile_models = types.CompilationResult()
  with autotqdm.tqdm(backends, desc="Backend") as t_backends:
    for backend in t_backends:
      component_input = flatbuffer
      backend = cast(types.Backend, backend)
      input_name_pref = base_name + backend.target_id_suffix
      t_backends.set_description(f"Compiling {backend.target_id}")
      try:
        for component in pipeline:
          component = cast(types.Component, component)
          t_backends.set_description(
              f"Compiling {backend.target_id}: {component.component_name}"
          )
          component_output = types.Model.create_from_path(
              output_dir
              / f"{input_name_pref}_{component.component_name}{common.DOT_TFLITE}"
          )
          backend.call_component(component_input, component_output, component)
          if not component_output.in_memory and not common.is_tflite(
              component_output.path
          ):
            raise ValueError(
                f"{component.component_name} failed to produce a TFLite model."
            )
          component_input = component_output
        compile_models.models_with_backend.append((backend, component_input))
      except ValueError as e:
        if keep_going:
          print(f"Skipping failed compilation for {backend.target}. Error: {e}")
          compile_models.failed_backends.append((backend, str(e)))
        else:
          raise

  return compile_models
