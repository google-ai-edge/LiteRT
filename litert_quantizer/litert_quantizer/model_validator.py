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

"""function for validating output models."""

from collections.abc import Callable, Iterable, Sequence
import dataclasses
import json
import math
import pathlib
from typing import Any, Optional, Union

import numpy as np

import os
import io
from litert_quantizer.utils import tfl_interpreter_utils as utils
from litert_quantizer.utils import validation_utils


_DEFAULT_SIGNATURE_KEY = utils.DEFAULT_SIGNATURE_KEY


@dataclasses.dataclass(frozen=True)
class SingleSignatureComparisonResult:
  """Comparison result for a single signature.

  Attributes:
    error_metrics: The names of the error metrics used for comparison.
    input_tensors: A dictionary of input tensor name and its values.
    output_tensors: A dictionary of output tensor name and its values.
    constant_tensors: A dictionary of constant tensor name and its values.
    intermediate_tensors: A dictionary of intermediate tensor name and its
      values.
  """

  error_metrics: Sequence[validation_utils.ValidationErrorMetric]
  input_tensors: dict[str, dict[str, float]]
  output_tensors: dict[str, dict[str, float]]
  constant_tensors: dict[str, dict[str, float]]
  intermediate_tensors: dict[str, dict[str, float]]


class ComparisonResult:
  """Comparison result for a model.

  Attributes:
    comparison_results: A dictionary of signature key and its comparison result.
  """

  def __init__(self, reference_model: bytes, target_model: bytes):
    """Initialize the ComparisonResult object.

    Args:
      reference_model: Model which will be used as the reference.
      target_model: Target model which will be compared against the reference.
        We expect target_model and reference_model to have the same graph
        structure.
    """
    self._reference_model = reference_model
    self._target_model = target_model
    self._comparison_results: dict[str, SingleSignatureComparisonResult] = {}

  def get_signature_comparison_result(
      self, signature_key: str = _DEFAULT_SIGNATURE_KEY
  ) -> SingleSignatureComparisonResult:
    """Get the comparison result for a signature.

    Args:
      signature_key: The signature key to be used for invoking the models.

    Returns:
      A SingleSignatureComparisonResult object.
    """
    if signature_key not in self._comparison_results:
      raise ValueError(
          f'{signature_key} is not in the comparison_results. Available'
          f' signature keys are: {self.available_signature_keys()}'
      )
    return self._comparison_results[signature_key]

  def available_signature_keys(self) -> list[str]:
    """Get the available signature keys in the comparison result."""
    return list(self._comparison_results.keys())

  def add_new_signature_results(
      self,
      error_metrics: Sequence[validation_utils.ValidationErrorMetric],
      comparison_result: dict[str, dict[str, float]],
      signature_key: str = _DEFAULT_SIGNATURE_KEY,
      validate_output_tensors_only: bool = False,
  ) -> None:
    """Add a new signature result to the comparison result.

    Args:
      error_metrics: The list of error metrics used for comparison.
      comparison_result: A dictionary of tensor name and its metric values.
      signature_key: The model signature that the comparison_result belongs to.
      validate_output_tensors_only: If True, only compare output tensors.
        Otherwise, compare all tensors.

    Raises:
      ValueError: If the signature_key is already in the comparison_results.
    """
    if signature_key in self._comparison_results:
      raise ValueError(f'{signature_key} is already in the comparison_results.')

    result = {
        key: {k: float(v) for k, v in value.items()}
        for key, value in comparison_result.items()
    }

    output_tensor_results = {}
    for name in utils.get_output_tensor_names(
        self._reference_model, signature_key
    ):
      output_tensor_results[name] = result.pop(name)

    input_tensor_results = {}
    constant_tensor_results = {}
    if validate_output_tensors_only:
      # No intermediate tensors are validated when validate_output_tensors_only
      # is True.
      result = {}
    else:
      for name in utils.get_input_tensor_names(
          self._reference_model, signature_key
      ):
        if name in result:
          input_tensor_results[name] = result.pop(name)

      # Only get constant tensors from the main subgraph of the signature.
      subgraph_index = utils.get_signature_main_subgraph_index(
          utils.create_tfl_interpreter(self._reference_model),
          signature_key,
      )
      for name in utils.get_constant_tensor_names(
          self._reference_model,
          subgraph_index,
      ):
        if name in result:
          constant_tensor_results[name] = result.pop(name)

    self._comparison_results[signature_key] = SingleSignatureComparisonResult(
        error_metrics=error_metrics,
        input_tensors=input_tensor_results,
        output_tensors=output_tensor_results,
        constant_tensors=constant_tensor_results,
        intermediate_tensors=result,
    )

  def get_all_tensor_results(self) -> dict[str, Any]:
    """Get all the tensor results in a single dictionary.

    Returns:
      A dictionary of tensor name and its value.
    """
    result = {}
    for _, signature_comparison_result in self._comparison_results.items():
      result.update(signature_comparison_result.input_tensors)
      result.update(signature_comparison_result.output_tensors)
      result.update(signature_comparison_result.constant_tensors)
      result.update(signature_comparison_result.intermediate_tensors)
    return result

  def get_model_size_reduction(self) -> tuple[int, float]:
    """Get the model size reduction in bytes and percentage."""
    reduced_model_size = len(self._reference_model) - len(self._target_model)
    reduction_perc = reduced_model_size / len(self._reference_model) * 100
    return reduced_model_size, reduction_perc

  def save(self, save_folder: str, model_name: str) -> None:
    """Saves the model comparison result.

    Args:
      save_folder: Path to the folder to save the comparison result.
      model_name: Name of the model.

    Raises:
      RuntimeError: If no quantized model is available.
    """
    reduced_model_size, reduction_ratio = self.get_model_size_reduction()

    result = {
        'reduced_size_bytes': reduced_model_size,
        'reduced_size_percentage': reduction_ratio,
    }

    error_metrics_seen = set()
    for signature, comparison_result in self._comparison_results.items():
      for metric in comparison_result.error_metrics:
        error_metrics_seen.add(metric)
      result[str(signature)] = {  # pyrefly: ignore[unsupported-operation]
          'input_tensors': comparison_result.input_tensors,
          'output_tensors': comparison_result.output_tensors,
          'constant_tensors': comparison_result.constant_tensors,
          'intermediate_tensors': comparison_result.intermediate_tensors,
      }

    save_path = pathlib.Path(save_folder)
    if not os.path.exists(str(save_path)):
      os.makedirs(str(save_path))

    result_save_path = str(save_path / (model_name + '_comparison_result.json'))
    with open(result_save_path, 'w') as output_file_handle:
      output_file_handle.write(json.dumps(result))

    # TODO: b/365578554 - Remove after ME is updated to use the new json format.
    color_threshold = [0.05, 0.1, 0.2, 0.4, 1, 10, 100]
    for metric in error_metrics_seen:
      json_object = create_json_for_model_explorer(
          self,
          metric=metric,
          threshold=color_threshold,
      )
      json_save_path = str(
          save_path
          / f'{model_name}_comparison_result_me_input_{metric.value}.json'
      )
      with open(json_save_path, 'w') as output_file_handle:
        output_file_handle.write(json_object)


def _setup_validation_interpreter(
    model: bytes,
    signature_input: dict[str, Any],
    signature_key: Optional[str],
    use_xnnpack: bool,
    num_threads: int,
    preserve_all_tensors: bool = True,
) -> tuple[Any, int, dict[str, Any]]:
  """Setup the interpreter for validation given a signature key.

  Args:
    model: The model to be validated.
    signature_input: A dictionary of input tensor name and its value.
    signature_key: The signature key to be used for invoking the models. If the
      model only has one signature, this can be set to None.
    use_xnnpack: Whether to use xnnpack for the interpreter.
    num_threads: The number of threads to use for the interpreter.
    preserve_all_tensors: Whether to preserve all tensors.

  Returns:
    A tuple of interpreter, subgraph_index and tensor_name_to_details.
  """

  interpreter = utils.create_tfl_interpreter(
      tflite_model=model,
      use_xnnpack=use_xnnpack,
      num_threads=num_threads,
      preserve_all_tensors=preserve_all_tensors,
  )
  utils.invoke_interpreter_signature(
      interpreter, signature_input, signature_key
  )
  # Only validate tensors from the main subgraph of the signature.
  subgraph_index = utils.get_signature_main_subgraph_index(
      interpreter, signature_key
  )
  tensor_name_to_details = utils.get_tensor_name_to_details_map(
      interpreter,
      subgraph_index,
  )
  return interpreter, subgraph_index, tensor_name_to_details


# TODO: b/330797129 - Enable multi-threaded evaluation.
def compare_model(
    reference_model: bytes,
    target_model: bytes,
    test_data: dict[str, Iterable[dict[str, Any]]],
    error_metrics: Optional[
        Sequence[validation_utils.ValidationErrorMetric]
    ] = None,
    compare_fns: Optional[Sequence[Callable[[Any, Any], float]]] = None,
    use_xnnpack: bool = True,
    num_threads: int = 16,
    validate_output_tensors_only: bool = False,
) -> ComparisonResult:
  """Compares model tensors over a model signature using comparison functions.

  This function will run the model signature on the provided dataset and
  compare the tensors using the comparison functions (e.g., mean squared
  difference).

  Args:
    reference_model: Model which will be used as the reference
    target_model: Target model which will be compared against the reference. We
      expect reference_model and target_model have the inputs and outputs
      signature.
    test_data: A mapping from the model's signature keys to their corresponding
      datasets. Each dataset is an iterable of samples, where a sample is a
      dictionary mapping the input tensor name to its data (e.g., numpy array).
      For example: `{'serving_default': [{'input_tensor': np.array([1, 2])},
      ...]}`
    error_metrics: A list of error metrics used for comparison (e.g.
      [validation_utils.ValidationErrorMetric.MSE,
      validation_utils.ValidationErrorMetric.SNR]). If None, defaults to
      evaluating [validation_utils.ValidationErrorMetric.MSE].
    compare_fns: A list of comparison functions to be used for calculating the
      statistics. If None, the default functions for the specified error_metrics
      will be looked up using the registry.
    use_xnnpack: Whether to use xnnpack for the interpreter.
    num_threads: The number of threads to use for the interpreter.
    validate_output_tensors_only: If True, only compare output tensors.
      Otherwise, compare all tensors.

  Returns:
    A ComparisonResult object containing the combined comparison results.
  """
  if error_metrics is None:
    error_metrics = [validation_utils.ValidationErrorMetric.MSE]

  if compare_fns is None:
    compare_fns = [
        validation_utils.get_validation_func(metric) for metric in error_metrics
    ]

  if len(error_metrics) != len(compare_fns):
    raise ValueError(
        'The number of error metrics must match the number of compare'
        ' functions.'
    )

  preserve_all_tensors = not validate_output_tensors_only
  model_comparison_result = ComparisonResult(reference_model, target_model)

  for signature_key, signature_inputs in test_data.items():
    comparison_results = {metric: {} for metric in error_metrics}
    for signature_input in signature_inputs:
      # Invoke the signature on both interpreters.
      ref_interpreter, ref_subgraph_index, ref_tensor_name_to_details = (
          _setup_validation_interpreter(
              reference_model,
              signature_input,
              signature_key,
              use_xnnpack,
              num_threads,
              preserve_all_tensors=preserve_all_tensors,
          )
      )
      targ_interpreter, targ_subgraph_index, targ_tensor_name_to_details = (
          _setup_validation_interpreter(
              target_model,
              signature_input,
              signature_key,
              use_xnnpack,
              num_threads,
              preserve_all_tensors=preserve_all_tensors,
          )
      )
      # Compare the cached tensor value
      tensor_names_to_compare = (
          utils.get_output_tensor_names(reference_model, signature_key)
          if validate_output_tensors_only
          else list(ref_tensor_name_to_details.keys())
      )

      for tensor_name in tensor_names_to_compare:
        detail = ref_tensor_name_to_details[tensor_name]
        if detail['dtype'] == np.object_:
          continue
        # Ignore tensors where any dimension of the shape is 0.
        if not np.all(detail['shape']):
          continue
        if tensor_name in targ_tensor_name_to_details:
          reference_data = utils.get_tensor_data(
              ref_interpreter, detail, ref_subgraph_index
          )
          target_data = utils.get_tensor_data(
              targ_interpreter,
              targ_tensor_name_to_details[tensor_name],
              targ_subgraph_index,
          )
          for metric, fn in zip(error_metrics, compare_fns):
            if tensor_name not in comparison_results[metric]:
              comparison_results[metric][tensor_name] = []
            comparison_results[metric][tensor_name].append(
                fn(target_data, reference_data)
            )

    aggregated_results = {}
    if error_metrics:
      for tensor_name in comparison_results[error_metrics[0]].keys():
        aggregated_results[tensor_name] = {}
        for metric in error_metrics:
          aggregated_results[tensor_name][metric.value] = float(
              np.mean(comparison_results[metric][tensor_name])
          )

    model_comparison_result.add_new_signature_results(
        error_metrics,
        aggregated_results,
        signature_key,
        validate_output_tensors_only,
    )
  return model_comparison_result


def create_json_for_model_explorer(
    data: ComparisonResult,
    metric: validation_utils.ValidationErrorMetric,
    threshold: list[Union[int, float]],
) -> str:
  """create a dict type that can be exported as json for model_explorer to use.

  Args:
    data: Output from compare_model function
    metric: Which error metric's values to extract and format for ME.
    threshold: A list of numbers representing thresholds for model_exlorer to
      display different colors

  Returns:
    A string represents the json format accepted by model_explorer
  """
  data_vals = data.get_all_tensor_results()
  color_scheme = []
  results = {}
  for name, values in data_vals.items():
    if getattr(metric, 'value', str(metric)) in values:
      results[name] = {
          'value': float(values[getattr(metric, 'value', str(metric))])
      }

  if threshold:
    green = 255
    gradient = math.floor(255 / len(threshold))
    for val in threshold:
      color_scheme.append({'value': val, 'bgColor': f'rgb(200, {green}, 0)'})
      green = max(0, green - gradient)

  return json.dumps({
      'results': results,
      'thresholds': color_scheme,
  })
