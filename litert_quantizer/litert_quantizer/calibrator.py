# Copyright 2024 Google LLC
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

"""Quantization Calibration."""

from collections.abc import Iterable, Mapping
import copy
import enum
import json
from typing import Any, Callable, cast

import numpy as np
from typing_extensions import override

import os
import io
from litert_quantizer import algorithm_manager
from litert_quantizer import default_policy as policy
from litert_quantizer import qtyping
from litert_quantizer import recipe
from litert_quantizer import recipe_manager
from litert_quantizer.utils import calibration_utils
from litert_quantizer.utils import progress_utils
from litert_quantizer.utils import qsv_utils
from litert_quantizer.utils import tfl_flatbuffer_utils
from litert_quantizer.utils import tfl_interpreter_utils
from ai_edge_litert import _pywrap_tfl_calibration  # pylint: disable=g-direct-tensorflow-import
from ai_edge_litert import interpreter as tfl  # pylint: disable=g-direct-tensorflow-import


class CalibrationMode(enum.Enum):
  """Calibration modes for the Calibrator.

  Attributes:
    INFERENCE: Inference only mode. No calibration is performed.
    CALIBRATION_PRESERVE_ALL_TENSORS: (Default) Calibration with XNNPACK enabled
      that preserves all intermediate tensors, allowing Python to inspect their
      values.
    CALIBRATION: Keeping this for backward compatibility. Please use
      CALIBRATION_PRESERVE_ALL_TENSORS instead of this.
    CALIBRATION_PROFILER_BASED: Use TFLite's internal C++ profiler-based
      calibration. It is faster than CALIBRATION_PRESERVE_ALL_TENSORS but offers
      less flexibility in terms of what statistics can be collected.
  """
  INFERENCE = 1
  CALIBRATION_PRESERVE_ALL_TENSORS = 2
  CALIBRATION = 2
  CALIBRATION_PROFILER_BASED = 3


_SignatureInput = dict[str, Any]  # input_argument_name -> tensor_value.
_SignatureOutput = dict[
    str, np.ndarray
]  # output_argument_name -> tensor_value.
_MISSING_FUNC = cast(
    Callable[[qtyping.QSV, qtyping.QSV], qtyping.QSV], object()
)


class CalibrationInterpreter:
  """A TFL interpreter-like interface for model calibration.

  This is a wrapper around Calibrator that replaces the TFL Interpreter to
  enable calibration. If mode is CALIBRATION, it runs calibration, otherwise it
  acts as a regular TFL interpreter for inference. When in CALIBRATION mode,
  each invocation of a signature runner will update the calibration statistics
  in self._calibrator. Calibrator is needed in both modes because it contains
  tfl interpreter instance to run the model.
  """

  def __init__(
      self,
      model_path: str,
      mode: CalibrationMode = CalibrationMode.INFERENCE,
      qsv_update_func: Callable[
          [qtyping.QSV, qtyping.QSV],
          qtyping.QSV,
      ] = _MISSING_FUNC,
  ):
    """Initializes the CalibrationInterpreter.

    Args:
      model_path: The path to the TFLite model.
      mode: The mode of the interpreter. If CALIBRATION, the interpreter will
        preserve all tensors for calibration purposes.
      qsv_update_func: The function to update the QSVs across calibration steps.
    """
    self._calibrator = Calibrator(
        model_path,
        mode=mode,
        qsv_update_func=qsv_update_func,
    )
    self._mode = mode

  def get_signature_runner(self, signature_key: str | None = None):
    """Returns the signature runner."""
    return CalibrationSignatureRunner(
        self._calibrator, signature_key, self._mode
    )

  def get_calibration_results(self):
    """Returns the calibration results."""
    if self._mode == CalibrationMode.INFERENCE:
      raise ValueError(
          "Calibration results are not available in INFERENCE mode."
      )
    return self._calibrator.get_model_qsvs()

  def save_calibration_result(
      self, output_path: str, extra_metadata: Mapping[str, str] | None = None
  ) -> None:
    """Saves the calibration results."""
    if self._mode == CalibrationMode.INFERENCE:
      raise ValueError(
          "Calibration results are not available in INFERENCE mode."
      )
    self._calibrator.save_calibration_result(output_path, extra_metadata)

  def get_signature_list(self) -> list[str]:
    """Returns the signature list."""
    return self._calibrator.get_signature_list()


class CalibrationSignatureRunner:
  """Wrapper around TFL signature runner to enable calibration."""

  def __init__(
      self,
      calibrator_obj: "Calibrator",
      signature_key: str | None = None,
      mode: CalibrationMode = CalibrationMode.INFERENCE,
      quantization_recipe: qtyping.ModelQuantizationRecipe = recipe.static_wi8_ai8(),
  ):
    """Initializes the CalibrationSignatureRunner.

    Args:
      calibrator_obj: The Calibrator instance.
      signature_key: The key of the signature to run. If None, the default
        signature is used.
      mode: The mode of the runner. If CALIBRATION, invoking the runner will
        update 'calibrator_obj' with new quantization statistics values. If
        INFERENCE, the runner behaves like a standard signature runner.
      quantization_recipe: The quantization recipe to use for calibration.
        Defaults to static_wi8_ai8.
    """
    self._calibrator = calibrator_obj
    self._signature_key = signature_key
    self._mode = mode
    self._recipe_manager = recipe_manager.RecipeManager()
    self._recipe_manager.load_quantization_recipe(quantization_recipe)
    self._signature_runner = (
        self._calibrator._tfl_interpreter.get_signature_runner(
            self._signature_key
        )
    )

  def __call__(self, **kwargs):
    if self._mode == CalibrationMode.INFERENCE:
      return self._signature_runner(**kwargs)
    self._calibrator.calibrate(
        calibration_dataset={self._signature_key: [kwargs]},  # pyrefly: ignore[bad-argument-type, bad-assignment]
        model_recipe_manager=self._recipe_manager,
        cache_output=True,
    )
    outputs = self._calibrator.get_cached_output()
    assert len(outputs) == 1
    self._calibrator.clear_cached_output()
    return outputs[0]

  def get_input_details(self):
    """Returns the input details of the model."""
    return self._signature_runner.get_input_details()

  def get_output_details(self):
    """Returns the output details of the model."""
    return self._signature_runner.get_output_details()


class Calibrator:
  """Base class and factory for TFLite model calibrators.

  This class acts both as a factory for instantiating specific algorithm
  subclasses (via __new__) and as the shared interface for them.

  Notes on Design Architecture:

    1. Calibrator is not an abstract class because it uses the factory pattern
    via __new__ to return instances of its subclasses. Making it abstract would
    cause static type checkers (like pytype) to flag direct instantiations
    as errors. Instead, we raise NotImplementedError in methods that must be
    overridden by subclasses (e.g., _create_interpreter, _calibrate_step).

    2. We prohibit direct instantiation of any subclass of Calibrator to enforce
    the use of the factory pattern implemented in __new__. Meaning no custom
    wrapper classes are allowed to inherit directly from Calibrator.
  """

  def __new__(
      cls,
      float_tflite: str | bytes,
      num_threads: int = 16,
      mode: CalibrationMode = CalibrationMode.CALIBRATION_PRESERVE_ALL_TENSORS,
      qsv_update_func: Callable[
          [qtyping.QSV, qtyping.QSV],
          qtyping.QSV,
      ] = _MISSING_FUNC,
  ) -> "Calibrator":
    """Creates a new instance of the appropriate calibrator subclass.

    Args:
      float_tflite: The path to the TFLite model or the model content as bytes.
      num_threads: The number of threads to use for the TFLite interpreter.
      mode: The mode of the calibrator. Check the docstring of CalibrationMode
        for more details.
      qsv_update_func: The function to update the QSVs across calibration steps.

    Returns:
      An instance of the appropriate calibrator subclass based on the mode.

    Raises:
      TypeError: If trying to instantiate a subclass of Calibrator directly.
    """
    if cls is not Calibrator:
      raise TypeError(
          f"Direct instantiation of {cls.__name__} is not allowed. "
          "Please use the Calibrator factory class instead."
      )

    del float_tflite, num_threads, qsv_update_func

    if mode == CalibrationMode.INFERENCE:
      return super().__new__(_InferenceOnlyCalibrator)  # pyrefly: ignore[bad-argument-type]
    elif mode == CalibrationMode.CALIBRATION_PRESERVE_ALL_TENSORS:
      return super().__new__(_PreserveAllTensorsCalibrator)  # pyrefly: ignore[bad-argument-type]
    elif mode == CalibrationMode.CALIBRATION_PROFILER_BASED:
      return super().__new__(_ProfilerBasedCalibrator)  # pyrefly: ignore[bad-argument-type]
    else:
      raise ValueError(f"Unsupported calibration mode: {mode}")

  def __init__(
      self,
      float_tflite: str | bytes,
      num_threads: int = 16,
      mode: CalibrationMode = CalibrationMode.CALIBRATION_PRESERVE_ALL_TENSORS,
      qsv_update_func: Callable[
          [qtyping.QSV, qtyping.QSV],
          qtyping.QSV,
      ] = _MISSING_FUNC,
  ):
    """Initializes the Calibrator. Check details in docstring of __new__."""
    del mode  # Used only in __new__ and passed to __init__ automatically.

    self._flatbuffer_model = tfl_flatbuffer_utils.read_model(float_tflite)
    self._tfl_interpreter = self._create_interpreter(float_tflite, num_threads)
    if qsv_update_func is _MISSING_FUNC:
      self._qsv_update_func = qsv_utils.moving_average_update
      self._is_custom_qsv_update_func = False
    else:
      self._qsv_update_func = qsv_update_func
      self._is_custom_qsv_update_func = True
    # Tensor name to tensor content.
    self._tensor_content_map: dict[str, Any] = {}
    # QSV of all the tensors in the model.
    self._model_qsvs: dict[str, qtyping.QSV] = {}
    # Cached output of the model.
    self._cached_output: list[_SignatureOutput] = []
    # Metadata for the calibration result.
    self._metadata: dict[str, Any] = {"num_samples_calibrated": 0}

  def _create_interpreter(
      self,
      float_tflite: str | bytes,
      num_threads: int,
  ) -> tfl.Interpreter:
    """Creates the TFLite interpreter."""
    raise NotImplementedError(
        "Subclasses must implement _create_interpreter()."
    )

  def _calibrate_step(
      self,
      signature_key: str | None,
      data: _SignatureInput,
      model_recipe_manager: recipe_manager.RecipeManager,
      cache_output: bool,
      pbar: progress_utils.ProgressBar,
  ) -> None:
    """Performs a single calibration step for a data sample.

    Args:
      signature_key: The signature key for the data sample.
      data: The input data for the signature.
      model_recipe_manager: The recipe manager that contains the quantization
        recipes.
      cache_output: Whether to cache the output of the model during calibration.
      pbar: The progress bar to update.
    """
    raise NotImplementedError("Subclasses must implement _calibrate_step().")

  def calibrate(
      self,
      calibration_dataset: Mapping[str, Iterable[_SignatureInput]],
      model_recipe_manager: recipe_manager.RecipeManager,
      cache_output: bool = False,
  ) -> None:
    """Calibrates the model."""
    total_ops = self._get_total_operations(calibration_dataset)
    with progress_utils.ProgressBar(
        total_steps=total_ops,
        description="Running Calibration:",
        disappear_on_finish=True,
    ) as pbar:
      for signature_key, dataset in calibration_dataset.items():
        for data in dataset:
          self._metadata["num_samples_calibrated"] += 1
          self._calibrate_step(
              signature_key, data, model_recipe_manager, cache_output, pbar
          )
        self._tfl_interpreter.reset_all_variables()

  def get_model_qsvs(self) -> dict[str, qtyping.QSV]:
    """Get the model qsvs.

    Returns:
      A dictionary of tensor name to QSV.
    """
    return self._model_qsvs

  def get_cached_output(self) -> list[_SignatureOutput]:
    """Get the cached output of the model."""
    return self._cached_output

  def clear_cached_output(self) -> None:
    """Clear the cached output of the model."""
    self._cached_output = []

  def reset_model_qsvs(self) -> None:
    """Reset the model qsvs."""
    self._model_qsvs = {}
    self._metadata = {"num_samples_calibrated": 0}

  def load_model_qsvs(self, model_qsvs: str | dict[str, qtyping.QSV]) -> None:
    """Load the model qsvs.

    Args:
      model_qsvs: A dictionary of tensor name to QSV or a path to a JSON file
        that contains the model qsvs (i.e., from save_calibration_result).
    """

    if isinstance(model_qsvs, str):
      self._model_qsvs, self._metadata = (
          calibration_utils.load_calibration_results(model_qsvs)
      )
      self._metadata["num_samples_calibrated"] = self._metadata.get(
          "num_samples_calibrated", 0
      )
    else:
      self._model_qsvs = copy.deepcopy(model_qsvs)

  def save_calibration_result(
      self, file_path: str, extra_metadata: Mapping[str, str] | None = None
  ) -> None:
    """Saves the calibration result to a json file.

    Args:
      file_path: Path to save the calibration result.
      extra_metadata: Extra metadata to save.
    """
    with open(file_path, "w") as f:
      json.dump(
          {
              "model_qsvs": self._model_qsvs,
              "metadata": {**self._metadata, **(extra_metadata or {})},
          },
          f,
          cls=calibration_utils.NumpyEncoder,
      )

  def get_signature_list(self) -> list[str]:
    """Get the signature list of the model."""
    return self._tfl_interpreter.get_signature_list()

  def _update_qsvs(
      self,
      op_qsvs: dict[str, qtyping.QSV],
      ignore_tensor_names: set[str],
      qsv_update_func: Callable[[qtyping.QSV, qtyping.QSV], qtyping.QSV],
  ) -> set[str]:
    """Update the model qsvs with the new values.

    Args:
      op_qsvs: A dictionary of tensor name to QSV.
      ignore_tensor_names: A set of tensor names to ignore.
      qsv_update_func: The function to update the QSVs across calibration steps.

    Returns:
      A set of tensor names that are updated.
    """
    updated_tensor_names = set()
    for tensor_name, qsv in op_qsvs.items():
      if tensor_name in ignore_tensor_names:
        continue
      if tensor_name not in self._model_qsvs:
        self._model_qsvs[tensor_name] = qsv
      else:
        updated_qsv = qsv_update_func(self._model_qsvs[tensor_name], qsv)
        self._model_qsvs[tensor_name] = updated_qsv
      updated_tensor_names.add(tensor_name)
    return updated_tensor_names

  def _get_total_operations(self, calibration_dataset) -> int:
    """Get the total number of OPs to go through while calibrating the model."""
    data_sizes = {
        key: len(list(dataset)) for key, dataset in calibration_dataset.items()
    }
    total_ops = 0
    for key, value in data_sizes.items():
      subgraph_idx = tfl_interpreter_utils.get_signature_main_subgraph_index(
          self._tfl_interpreter, key
      )
      subgraph_inds = [subgraph_idx]
      total_operators = 0
      while subgraph_inds:
        subgraph_idx = subgraph_inds.pop()
        subgraph = self._flatbuffer_model.subgraphs[subgraph_idx]
        subgraph_operators = subgraph.operators
        if not any(
            isinstance(op, qtyping.IOOperator) for op in subgraph.operators
        ):
          subgraph_operators += (
              tfl_flatbuffer_utils.get_subgraph_input_output_operators(subgraph)
          )
        total_operators += len(subgraph_operators)
        for op in subgraph_operators:
          subgraph_inds.extend(
              tfl_flatbuffer_utils.get_op_side_effect_subgraphs(op)
          )
      total_ops += value * total_operators
    return total_ops


class _InferenceOnlyCalibrator(Calibrator):
  """Calibrator that only supports inference and not calibration."""

  @override
  def _create_interpreter(
      self,
      float_tflite: str | bytes,
      num_threads: int,
  ) -> tfl.Interpreter:
    return tfl_interpreter_utils.create_tfl_interpreter(
        float_tflite,
        use_xnnpack=True,
        num_threads=num_threads,
        preserve_all_tensors=False,
    )

  @override
  def _calibrate_step(
      self,
      signature_key: str | None,
      data: _SignatureInput,
      model_recipe_manager: recipe_manager.RecipeManager,
      cache_output: bool,
      pbar: progress_utils.ProgressBar,
  ) -> None:
    raise NotImplementedError(
        "InferenceOnlyCalibrator does not support calibration."
    )


class _PreserveAllTensorsCalibrator(Calibrator):
  """Calibrator using the preserve_all_tensors interpreter mode."""

  @override
  def _create_interpreter(
      self,
      float_tflite: str | bytes,
      num_threads: int,
  ) -> tfl.Interpreter:
    return tfl_interpreter_utils.create_tfl_interpreter(
        float_tflite,
        use_xnnpack=True,
        num_threads=num_threads,
        preserve_all_tensors=True,
    )

  @override
  def _calibrate_step(
      self,
      signature_key: str | None,
      data: _SignatureInput,
      model_recipe_manager: recipe_manager.RecipeManager,
      cache_output: bool,
      pbar: progress_utils.ProgressBar,
  ) -> None:
    op_codes = self._flatbuffer_model.operatorCodes
    # Step0: get subgraph index.
    subgraph_idx = tfl_interpreter_utils.get_signature_main_subgraph_index(
        self._tfl_interpreter, signature_key
    )
    # Initialize tensor names updated in this round of calibration.
    updated_tensor_names = set()

    # Step1: run tfl interpreter on subgraph to get tensor content.
    signature_output = tfl_interpreter_utils.invoke_interpreter_signature(
        self._tfl_interpreter, data, signature_key
    )
    if cache_output:
      self._cached_output.append(signature_output)

    # Step2: go through each op in subgraph to update quantization
    # statistic values.
    subgraphs_inds = [subgraph_idx]
    while subgraphs_inds:
      subgraph_ind = subgraphs_inds.pop()
      self._tensor_content_map.update(
          tfl_interpreter_utils.get_tensor_name_to_content_map(
              self._tfl_interpreter, subgraph_ind
          )
      )
      subgraph = self._flatbuffer_model.subgraphs[subgraph_ind]
      graph_info = qtyping.GraphInfo(
          subgraph.tensors, self._flatbuffer_model.buffers
      )
      # Add input/output operators if they are not in the subgraph.
      if not any(
          isinstance(op, qtyping.IOOperator) for op in subgraph.operators
      ):
        subgraph.operators += (
            tfl_flatbuffer_utils.get_subgraph_input_output_operators(subgraph)
        )
      for op in subgraph.operators:
        pbar.update_single_step()
        if isinstance(op, qtyping.IOOperator):
          op_key = op.op_key
        else:
          op_code = op_codes[op.opcodeIndex].builtinCode
          if op_code not in tfl_flatbuffer_utils.TFL_OP_CODE_TO_NAME:
            continue
          op_key = tfl_flatbuffer_utils.TFL_OP_CODE_TO_NAME[op_code]
        # Step2.1: query the quantization_recipe to get op quantization
        # settings.
        op_scope = tfl_flatbuffer_utils.get_op_scope(op, subgraph.tensors)
        algorithm_name, _ = model_recipe_manager.get_quantization_configs(
            op_key, op_scope
        )
        if algorithm_name == algorithm_manager.AlgorithmName.NO_QUANTIZE:
          continue
        if policy.is_non_quantizable_composite_op(op):
          continue

        # Step2.2: query algorithm_manager to get/call the related
        # calibration function.
        calibrate_func = algorithm_manager.get_quantization_func(
            algorithm_name, op_key, qtyping.QuantizeMode.CALIBRATE
        )
        op_qsvs = calibrate_func(op, graph_info, self._tensor_content_map)
        # Step3: Update tensor qsvs with the new values. Ignore the tensor
        # names that are already updated in this round of calibration.
        if not self._is_custom_qsv_update_func:
          qsvs_update_func = algorithm_manager.get_update_qsv_func(
              algorithm_name, op_key
          )
        else:
          qsvs_update_func = self._qsv_update_func
        op_updated_tensor_name = self._update_qsvs(
            op_qsvs, updated_tensor_names, qsvs_update_func
        )
        updated_tensor_names.update(op_updated_tensor_name)

        # Step4: Invoke any subgraphs invoked as a side effect of the op.
        subgraphs_inds.extend(
            tfl_flatbuffer_utils.get_op_side_effect_subgraphs(op)
        )


class _ProfilerBasedCalibrator(Calibrator):
  """Calibrator using the profiler-based calibration mode.

  1. It runs the model using _pywrap_tfl_calibration.InvokeWithCalibration()
  2. The C++ engine tracks the min/max values internally during execution.
  3. After execution, the python wrapper retrieves the statistics (tensor_stats)
  and updates self._model_qsvs.
  4. This mode only supports min/max tracking and cannot be used for algorithms
  that require custom statistics.
  """

  @override
  def _create_interpreter(
      self,
      float_tflite: str | bytes,
      num_threads: int,
  ) -> tfl.Interpreter:
    return tfl_interpreter_utils.create_tfl_interpreter(
        float_tflite,
        use_xnnpack=True,
        num_threads=num_threads,
        preserve_all_tensors=False,
        disable_delegate_node_fusion=True,
        force_delegate_node_profiling=True,
    )

  @override
  def _calibrate_step(
      self,
      signature_key: str | None,
      data: _SignatureInput,
      model_recipe_manager: recipe_manager.RecipeManager,
      cache_output: bool,
      pbar: progress_utils.ProgressBar,
  ) -> None:
    del model_recipe_manager  # Unused.
    del pbar  # Can't do per-op progress bar in this mode.

    signature_defs = self._tfl_interpreter._get_full_signature_list()  # pylint: disable=protected-access
    if signature_key is None:
      if not signature_defs:
        raise ValueError(
            "No signature key provided and no signature list found."
        )
      # Using the default signature.
      signature_key = next(iter(signature_defs))

    cpp_interpreter = self._tfl_interpreter._interpreter  # pylint: disable=protected-access
    subgraph_index = cpp_interpreter.GetSubgraphIndexFromSignature(
        signature_key
    )
    signature_def = signature_defs[signature_key]
    inputs = signature_def["inputs"]
    outputs = signature_def["outputs"]

    # Resize and allocate input tensors.
    for input_name, value in data.items():
      input_tensor_index = inputs.get(input_name)
      if input_tensor_index is None:
        raise ValueError(f"Invalid Input name ({input_name}) for SignatureDef")
      cpp_interpreter.ResizeInputTensor(
          input_tensor_index,
          np.array(value.shape, dtype=np.int32),
          strict=False,
          subgraph_index=subgraph_index,
      )
    cpp_interpreter.AllocateTensors(subgraph_index)
    for input_name, value in data.items():
      cpp_interpreter.SetTensor(inputs[input_name], value, subgraph_index)

    # Invoke the interpreter with calibration.
    tensor_stats = _pywrap_tfl_calibration.InvokeWithCalibration(
        self._tfl_interpreter._native_handle(),  # pylint: disable=protected-access
        subgraph_index,
    )

    # Get inference result.
    inference_result = {
        output_name: cpp_interpreter.GetTensor(output_index, subgraph_index)
        for output_name, output_index in outputs.items()
    }
    if cache_output:
      self._cached_output.append(inference_result)

    # Update model QSVs with tensor_stats.
    for tensor_name, stats in tensor_stats.items():
      # Wrap scalar values in numpy arrays to support .shape attribute.
      qsv = {
          "min": np.array([stats.min]),
          "max": np.array([stats.max]),
      }
      existing_qsv = self._model_qsvs.get(tensor_name)
      if existing_qsv is None:
        self._model_qsvs[tensor_name] = qsv
      else:
        self._model_qsvs[tensor_name] = self._qsv_update_func(existing_qsv, qsv)
