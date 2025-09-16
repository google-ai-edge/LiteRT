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

"""AOT Compilation for LiteRT model."""
import pathlib
import tempfile

from litert.python.aot import prepare_for_npu as core
from litert.python.aot.core import apply_plugin
from litert.python.aot.core import components
from litert.python.aot.core import mlir_transforms
from litert.python.aot.core import types
from litert.python.aot.vendors import import_vendor


def aot_compile(
    input_model: types.Model | str,
    output_dir: str | pathlib.Path | None = None,
    target: types.Target | list[types.Target] | None = None,
    config: (
        types.CompilationConfig | list[types.CompilationConfig] | None
    ) = None,
    quantizer: components.AieQuantizerT | None = None,
    keep_going: bool = True,
    subgraphs_to_compile: list[int] | None = None,
    **kwargs,
) -> types.CompilationResult:
  """Prepares a TFLite model for NPU execution.

  High level command that erforms various backend specific pre-processing steps
  and then applies an NPU compiler to the given model.

  Args:
    input_model: The input model to compile.
    output_dir: Directory to write the output files to. If not specified, the
      output files will be written to the same directory as the input file.
    target: The target to compile for. If not specified, will compile to all
      registered targets.
    config: The compilation config(s). Cannot be specified with target.
    quantizer: The quantizer to use for quantization.
    keep_going: Whether to keep going if some backends fail. If False, fail fast
      on the first error and raise an exception.
    subgraphs_to_compile: The subgraph index list to compile to NPU. If None,
      compile all subgraphs.
    **kwargs: Additional arguments to pass to the backend.

  Returns:
    Compiled models.
  """
  # Only one of target or config is needed.
  if target and config:
    raise ValueError("Cannot specify both target and config.")

  if config is None:
    if target is None:
      target = import_vendor.AllRegisteredTarget()
    if isinstance(target, types.Target):
      config = types.CompilationConfig(target=target)
    elif isinstance(target, list):
      config = [types.CompilationConfig(target=t) for t in target]
    else:
      raise ValueError("Unsupported target type.")

  if isinstance(input_model, str):
    input_path = pathlib.Path(input_model)
    input_model = types.Model.create_from_path(input_path)

  # Resolve output paths.
  temp_dir = None
  if not output_dir:
    if input_model.in_memory:
      # Use a temp dir for in-memory models.
      # The temp dir will be cleaned up after the models are compiled and loaded
      # back to memory (i.e. function returns).
      temp_dir = tempfile.TemporaryDirectory()
      output_dir = temp_dir.name
    else:
      input_path = input_model.path
      output_dir = input_path.parent / "_compiled_models"
      output_dir.mkdir(parents=True, exist_ok=True)
      output_dir = str(output_dir)
  output_dir_path = pathlib.Path(output_dir)
  output_dir_path.mkdir(parents=True, exist_ok=True)

  if isinstance(config, types.CompilationConfig) or not config:
    if config:
      # Make pytype happy.
      assert isinstance(config, types.CompilationConfig)
      kw_config = config.to_dict() | kwargs
    else:
      kw_config = kwargs

    backend_class = core.resolve_backend(kw_config)

    quant_recipe = kw_config.get("quantize_recipe", None)
    if quant_recipe:
      assert quantizer is not None, "Quantizer is required for quantization."

    results = core.prepare_for_npu(
        input_model,
        output_dir_path,
        backend_class,
        kw_config,
        transforms=mlir_transforms.MlirTransforms(),
        quantizer=quantizer,
        plugin=apply_plugin.ApplyPlugin(
            experimental_capture_stderr=True,
            subgraphs_to_compile=subgraphs_to_compile,
        ),
        keep_going=keep_going,
    )
  elif isinstance(config, list):
    kw_configs = [c.to_dict() | kwargs for c in config]

    configs_with_backend = [(core.resolve_backend(c), c) for c in kw_configs]
    requires_quantizer = any("quantize_recipe" in c for c in kw_configs)
    if requires_quantizer and quantizer is None:
      raise ValueError("Quantizer is required for quantization.")

    results = core.prepare_for_npu_multiple_configs(
        input_model,
        output_dir_path,
        configs_with_backend,
        transforms=mlir_transforms.MlirTransforms(),
        quantizer=quantizer,
        plugin=apply_plugin.ApplyPlugin(
            experimental_capture_stderr=True,
            subgraphs_to_compile=subgraphs_to_compile,
        ),
        keep_going=keep_going,
    )
  else:
    # Should not reach here.
    raise ValueError("Unsupported config type.")

  if temp_dir:
    # Load the models to memory before cleaning up the temp dir.
    results.load()
    temp_dir.cleanup()

  return results
