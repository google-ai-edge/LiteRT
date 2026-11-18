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

"""Util functions for TFL interpreter."""

import pathlib
from typing import Any, Optional, Union

import ml_dtypes
import numpy as np

from ai_edge_litert.tools import mmap_utils
from litert_quantizer import qtyping
from litert_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor
from ai_edge_litert import interpreter as tfl  # pylint: disable=g-direct-tensorflow-import

DEFAULT_SIGNATURE_KEY = "serving_default"

_Numeric = Union[int, float]


def create_tfl_interpreter(
    tflite_model: Union[str, pathlib.Path, bytes],
    allocate_tensors: bool = True,
    use_xnnpack: bool = True,
    num_threads: int = 16,
    preserve_all_tensors: bool = True,
    disable_delegate_node_fusion: bool = False,
    force_delegate_node_profiling: bool = False,
) -> tfl.Interpreter:
  """Creates a TFLite interpreter from a model file.

  Args:
    tflite_model: Model file path (`str` or `pathlib.Path`) or bytes.
    allocate_tensors: Whether to allocate tensors.
    use_xnnpack: Whether to use the XNNPACK delegate for the interpreter.
    num_threads: The number of threads to use for the interpreter.
    preserve_all_tensors: Whether to preserve all tensors. If False, only input
      and output tensors are preserved.
    disable_delegate_node_fusion: Whether to disable delegate node fusion.
    force_delegate_node_profiling: Whether to force profiling for delegate ops.

  Returns:
    A TFLite interpreter.
  """
  if use_xnnpack:
    op_resolver = tfl.OpResolverType.BUILTIN
  else:
    op_resolver = tfl.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
  model_path: Optional[str] = None
  model_content: Optional[bytes] = None
  if isinstance(tflite_model, (str, pathlib.Path)):
    if pathlib.Path(tflite_model).exists():
      model_path = str(tflite_model)
    else:
      # Non-local/virtual filesystem paths are not visible to `pathlib` or
      # `CreateWrapperFromFile`; read via `mmap_utils` instead.
      model_content = bytes(mmap_utils.get_file_contents(tflite_model))
  else:
    model_content = bytes(tflite_model)

  tflite_interpreter = tfl.Interpreter(
      model_path=model_path,
      model_content=model_content,
      num_threads=num_threads,
      experimental_op_resolver_type=op_resolver,
      experimental_preserve_all_tensors=preserve_all_tensors,
      experimental_disable_delegate_node_fusion=disable_delegate_node_fusion,
      experimental_force_delegate_node_profiling=force_delegate_node_profiling,
  )
  if allocate_tensors:
    tflite_interpreter.allocate_tensors()
  return tflite_interpreter


def has_quantization_params(tensor_detail: dict[str, Any]) -> bool:
  """Checks if a tensor contains quantization parameters.

  Args:
    tensor_detail: A dictionary of tensor details.

  Returns:
    True if the tensor is quantized.
  """
  quant_params = tensor_detail["quantization_parameters"]
  return bool(len(quant_params["scales"]))


def is_signatureless_default(
    tflite_interpreter: tfl.Interpreter,
    signature_key: str | None,
) -> bool:
  """Checks whether to run a signatureless model by tensor names.

  Signatureless models are run through subgraph 0 with inputs and outputs keyed
  by tensor name when `signature_key` is None or `DEFAULT_SIGNATURE_KEY`.

  Args:
    tflite_interpreter: The TFLite interpreter.
    signature_key: The requested signature key.

  Returns:
    True if the model has no signatures and `signature_key` is None or
    `DEFAULT_SIGNATURE_KEY`.
  """
  return not tflite_interpreter.get_signature_list() and signature_key in (
      None,
      DEFAULT_SIGNATURE_KEY,
  )


def get_signatureless_tensor_indices(
    tflite_interpreter: tfl.Interpreter,
) -> tuple[dict[str, int], dict[str, int]]:
  """Gets the input and output tensor indices of a signatureless model.

  Signatureless models run subgraph 0. Inputs are keyed by tensor name, and
  also by the tensor name without a trailing `:0` unless another input already
  has that exact name. Outputs are keyed by tensor name.

  Args:
    tflite_interpreter: The TFLite interpreter.

  Returns:
    A tuple of (input name to tensor index, output name to tensor index) maps
    for subgraph 0.
  """
  input_details = tflite_interpreter.get_input_details()
  inputs = {d["name"]: d["index"] for d in input_details}
  for d in input_details:
    inputs.setdefault(d["name"].removesuffix(":0"), d["index"])
  outputs = {
      d["name"]: d["index"] for d in tflite_interpreter.get_output_details()
  }
  return inputs, outputs


def _invoke_signatureless_interpreter(
    tflite_interpreter: tfl.Interpreter,
    input_data: dict[str, Any],
    quantize_input: bool,
) -> dict[str, np.ndarray]:
  """Invokes subgraph 0 of a signatureless model with inputs keyed by name.

  Args:
    tflite_interpreter: A TFLite interpreter for a model without signatures.
    input_data: The input data keyed by input tensor name (`:0` optional).
    quantize_input: Whether to quantize the input data.

  Returns:
    The output data keyed by output tensor name.
  """
  input_indices, _ = get_signatureless_tensor_indices(tflite_interpreter)
  data_by_index = {}
  for name, value in input_data.items():
    if name not in input_indices:
      raise ValueError(
          f"Input '{name}' not found in the model inputs: {list(input_indices)}"
      )
    data_by_index[input_indices[name]] = value

  input_details = tflite_interpreter.get_input_details()
  missing_inputs = [
      d["name"] for d in input_details if d["index"] not in data_by_index
  ]
  if missing_inputs:
    raise ValueError(f"Missing data for model inputs: {missing_inputs}")

  # Resize inputs whose shapes differ from the data, as `SignatureRunner` does.
  needs_allocation = False
  for d in input_details:
    data_shape = np.shape(data_by_index[d["index"]])
    if tuple(d["shape"]) != data_shape:
      tflite_interpreter.resize_tensor_input(
          d["index"], data_shape, strict=False
      )
      needs_allocation = True
  if needs_allocation:
    tflite_interpreter.allocate_tensors()

  invoke_interpreter_once(
      tflite_interpreter,
      [data_by_index[d["index"]] for d in input_details],
      quantize_input=quantize_input,
  )
  return {
      d["name"]: get_tensor_data(tflite_interpreter, d, dequantize=False)
      for d in tflite_interpreter.get_output_details()
  }


def invoke_interpreter_signature(
    tflite_interpreter: tfl.Interpreter,
    signature_input_data: dict[str, Any],
    signature_key: Optional[str] = None,
    quantize_input: bool = True,
) -> dict[str, np.ndarray]:
  """Invokes the TFLite interpreter for a signature.

  If the model has no signatures and `signature_key` is None or
  `DEFAULT_SIGNATURE_KEY`, subgraph 0 runs with inputs keyed by tensor name
  (the `:0` suffix is optional) and outputs keyed by tensor name. Otherwise,
  the model runs through the signature runner of `signature_key`.

  Args:
    tflite_interpreter: A TFLite interpreter.
    signature_input_data: The input data for the signature.
    signature_key: The signature key.
    quantize_input: Whether to quantize the input data.

  Returns:
    The output data of the signature.
  """
  if is_signatureless_default(tflite_interpreter, signature_key):
    return _invoke_signatureless_interpreter(
        tflite_interpreter, signature_input_data, quantize_input
    )

  # Make a copy to avoid in-place modification.
  signature_input = signature_input_data.copy()
  signature_runner = tflite_interpreter.get_signature_runner(signature_key)
  for input_name, input_detail in signature_runner.get_input_details().items():
    input_data = signature_input[input_name]
    if (
        input_data.dtype == np.float32
        and has_quantization_params(input_detail)
        and quantize_input
    ):
      quant_params = qtyping.UniformQuantParams.from_tfl_tensor_details(
          input_detail
      )
      signature_input[input_name] = uniform_quantize_tensor.uniform_quantize(
          input_data, quant_params
      )
  return signature_runner(**signature_input)


def invoke_interpreter_once(
    tflite_interpreter: tfl.Interpreter,
    input_data_list: list[Any],
    quantize_input: bool = True,
):
  """Invokes the TFLite interpreter once.

  Args:
    tflite_interpreter: A TFLite interpreter.
    input_data_list: A list of input data.
    quantize_input: Whether to quantize the input data.
  """
  if len(input_data_list) != len(tflite_interpreter.get_input_details()):
    raise ValueError(
        "Input data must be a list with each element match the input sequence"
        " defined in .tflite. If the model has only one input, wrap it with a"
        " list (e.g., [input_data])"
    )
  for i, input_data in enumerate(input_data_list):
    input_details = tflite_interpreter.get_input_details()[i]
    if (
        input_data.dtype == np.float32
        and has_quantization_params(input_details)
        and quantize_input
    ):
      quant_params = qtyping.UniformQuantParams.from_tfl_tensor_details(
          input_details
      )
      input_data = uniform_quantize_tensor.uniform_quantize(
          input_data, quant_params
      )
    tflite_interpreter.set_tensor(input_details["index"], input_data)
  tflite_interpreter.invoke()


def get_tensor_data(
    tflite_interpreter: Any,
    tensor_detail: dict[str, Any],
    subgraph_index: int = 0,
    dequantize: bool = True,
) -> np.ndarray:
  """Gets the tensor data from a TFLite interpreter.

  Args:
    tflite_interpreter: A TFLite interpreter.
    tensor_detail: A dictionary of tensor details.
    subgraph_index: The index of the subgraph that the tensor belongs to.
    dequantize: Whether to dequantize the quantized tensor data.

  Returns:
    The tensor data.
  """
  tensor_data = tflite_interpreter.get_tensor(
      tensor_detail["index"], subgraph_index
  )
  if (
      tensor_data.dtype != np.float32
      and has_quantization_params(tensor_detail)
      and dequantize
  ):
    quant_params = qtyping.UniformQuantParams.from_tfl_tensor_details(
        tensor_detail
    )
    tensor_data = uniform_quantize_tensor.uniform_dequantize(
        tensor_data,
        quant_params,
    )
  return tensor_data


def get_tensor_name_to_content_map(
    tflite_interpreter: Any, subgraph_index: int = 0, dequantize: bool = False
) -> dict[str, Any]:
  """Gets internal tensors from a TFLite interpreter for a given subgraph.

  Note the data will be copied to the returned dictionary, increasing the
  memory usage.

  Args:
    tflite_interpreter: A TFLite interpreter.
    subgraph_index: The index of the subgraph that the tensor belongs to.
    dequantize: Whether to dequantize the tensor data.

  Returns:
    A dictionary of internal tensors.
  """
  tensors = {}
  for tensor_detail in tflite_interpreter.get_tensor_details(subgraph_index):
    # Don't return temporary, unnamed tensors.
    if not tensor_detail["name"]:
      continue

    # Don't return tensors where any dimension of the shape is 0.
    if not np.all(tensor_detail["shape"]):
      continue

    tensors[tensor_detail["name"]] = get_tensor_data(
        tflite_interpreter, tensor_detail, subgraph_index, dequantize
    )
  return tensors


def get_tensor_name_to_details_map(
    tflite_interpreter: Any, subgraph_index: int = 0
) -> dict[str, Any]:
  """Gets internal tensors from a TFLite interpreter for a given subgraph.

  Args:
    tflite_interpreter: A TFLite interpreter.
    subgraph_index: The index of the subgraph that the tensor belongs to.

  Returns:
    A dictionary of internal tensors.
  """
  tensor_name_to_detail = {}
  for tensor_detail in tflite_interpreter.get_tensor_details(subgraph_index):
    # Don't return temporary, unnamed tensors or scratch tensors.
    # tensor_detail doesn't include the allocation size (bytes) or an
    # indicator of scratch tensors, so use the name to filter them out.
    if not tensor_detail["name"] or "scratch" in tensor_detail["name"]:
      continue
    tensor_name_to_detail[tensor_detail["name"]] = tensor_detail
  return tensor_name_to_detail


def get_input_tensor_names(
    tflite_model: Union[str, pathlib.Path, bytes],
    signature_name: Optional[str] = None,
) -> list[str]:
  """Gets input tensor names from a TFLite model for a signature.

  If the model has no signatures and `signature_name` is None or
  `DEFAULT_SIGNATURE_KEY`, returns the input tensor names of subgraph 0.

  Args:
    tflite_model: Model file path (`str` or `pathlib.Path`) or bytes.
    signature_name: The signature name that the input tensors belong to.

  Returns:
    A list of input tensor names.
  """

  tfl_interpreter = create_tfl_interpreter(tflite_model, allocate_tensors=False)
  if is_signatureless_default(tfl_interpreter, signature_name):
    return [d["name"] for d in tfl_interpreter.get_input_details()]
  signature_runner = tfl_interpreter.get_signature_runner(signature_name)
  input_tensor_names = []
  for _, input_detail in signature_runner.get_input_details().items():
    input_tensor_names.append(input_detail["name"])
  return input_tensor_names


def get_output_tensor_names(
    tflite_model: Union[str, pathlib.Path, bytes],
    signature_name: Optional[str] = None,
) -> list[str]:
  """Gets output tensor names from a TFLite model for a signature.

  If the model has no signatures and `signature_name` is None or
  `DEFAULT_SIGNATURE_KEY`, returns the output tensor names of subgraph 0.

  Args:
    tflite_model: Model file path (`str` or `pathlib.Path`) or bytes.
    signature_name: The signature name that the output tensors belong to.

  Returns:
    A list of output tensor names.
  """
  tfl_interpreter = create_tfl_interpreter(tflite_model, allocate_tensors=False)
  if is_signatureless_default(tfl_interpreter, signature_name):
    return [d["name"] for d in tfl_interpreter.get_output_details()]
  signature_runner = tfl_interpreter.get_signature_runner(signature_name)
  output_tensor_names = []
  for _, output_detail in signature_runner.get_output_details().items():
    output_tensor_names.append(output_detail["name"])
  return output_tensor_names


def get_constant_tensor_names(
    tflite_model: Union[str, pathlib.Path, bytes],
    subgraph_index: int = 0,
    min_constant_size: int = 1,
) -> list[str]:
  """Gets constant tensor names from a TFLite model for a subgraph.

  Note that this function acts on subgraph level, not signature level. This is
  because it is non-trivial to track constant tensors for a signature without
  running it.

  Args:
    tflite_model: Model file path (`str` or `pathlib.Path`) or bytes.
    subgraph_index: The index of the subgraph that the tensor belongs to.
    min_constant_size: The minimum size of a constant tensor.

  Returns:
    A list of names for constant tensor that bigger than min_constant_size and a
    list of names for constant tensor that smaller than min_constant_size.
  """
  tfl_interpreter = create_tfl_interpreter(tflite_model, allocate_tensors=False)
  const_tensor_names = []
  for tensor_detail in tfl_interpreter.get_tensor_details(subgraph_index):
    if tensor_detail["dtype"] == np.object_:
      continue
    try:
      tensor_data = get_tensor_data(
          tfl_interpreter, tensor_detail, subgraph_index
      )
      if tensor_data.size >= min_constant_size:
        const_tensor_names.append(tensor_detail["name"])
    except ValueError:
      continue
  return const_tensor_names


def get_signature_main_subgraph_index(
    tflite_interpreter: tfl.Interpreter,
    signature_key: Optional[str] = None,
) -> int:
  """Gets the main subgraph index of a signature.

  If the model has no signatures and `signature_key` is None or
  `DEFAULT_SIGNATURE_KEY`, returns 0.

  Args:
    tflite_interpreter: A TFLite interpreter.
    signature_key: The signature key.

  Returns:
    The main subgraph index of the signature.
  """
  if is_signatureless_default(tflite_interpreter, signature_key):
    return 0
  signature_runner = tflite_interpreter.get_signature_runner(signature_key)
  return signature_runner._subgraph_index  # pylint:disable=protected-access


def _create_random_normal(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> np.ndarray:
  """Creates a random normal dataset sample for given input details."""
  return rng.normal(size=shape).astype(dtype)


def _create_random_uniform(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    dtype: np.dtype,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> np.ndarray:
  """Creates a random uniform dataset sample for given input details."""
  return rng.uniform(min_value, max_value, size=shape).astype(dtype)


def _create_random_integers(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    dtype: np.dtype,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> np.ndarray:
  """Creates a random integer dataset sample for given input details."""
  if min_value is None:
    min_value = np.iinfo(dtype).min
  endpoint = False
  if max_value is None:
    max_value = np.iinfo(dtype).max
    endpoint = True
  return rng.integers(
      min_value, max_value, size=shape, dtype=dtype, endpoint=endpoint
  )


def _create_random_bool(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> np.ndarray:
  """Creates a random bool dataset sample for given input details."""
  return rng.choice([True, False], size=shape, replace=True).astype(dtype)


def create_random_dataset(
    input_details: dict[str, Any],
    num_samples: int,
    random_seed: Union[int, np._typing.ArrayLike],
    min_max_range: Optional[tuple[_Numeric, _Numeric]] = None,
) -> list[dict[str, Any]]:
  """Creates a random normal dataset for given input details.

  Args:
    input_details: A dictionary of input details.
    num_samples: The number of samples to generate.
    random_seed: The random seed to use.
    min_max_range: The min and max of the input range.

  Returns:
    A list of dictionaries, each containing a sample of input data (for all
    signatures).
  """
  rng = np.random.default_rng(random_seed)  # pyrefly: ignore[bad-argument-type]
  dataset = []
  for _ in range(num_samples):
    input_data = {}
    for arg_name, input_tensor in input_details.items():
      dtype = input_tensor["dtype"]
      shape = input_tensor["shape"]
      if np.issubdtype(dtype, np.integer):
        if min_max_range is None:
          new_data = _create_random_integers(rng, shape, dtype)
        else:
          min_value, max_value = min_max_range
          new_data = _create_random_integers(
              rng, shape, dtype, min_value, max_value  # pyrefly: ignore[bad-argument-type]
          )
      elif dtype in (np.float32, ml_dtypes.bfloat16):
        if min_max_range is None:
          new_data = _create_random_normal(rng, shape, dtype)
        else:
          min_value, max_value = min_max_range
          new_data = _create_random_uniform(
              rng, shape, dtype, min_value, max_value
          )
      elif dtype == np.bool:
        new_data = _create_random_bool(rng, shape, dtype)
      else:
        raise ValueError(f"Unsupported dtype: {input_tensor['dtype']}")
      input_data[arg_name] = new_data
    dataset.append(input_data)
  return dataset


def create_random_normal_input_data(
    tflite_model: Union[str, pathlib.Path, bytes],
    num_samples: int = 4,
    random_seed: int = 666,
    min_max_range: Optional[tuple[_Numeric, _Numeric]] = None,
) -> dict[str, list[dict[str, Any]]]:
  """Creates a random normal dataset for a signature runner.

  If the model has no signatures, the dataset is keyed by
  `DEFAULT_SIGNATURE_KEY`, and each sample is keyed by the input tensor names of
  subgraph 0.

  Args:
    tflite_model: TFLite model path (`str` or `pathlib.Path`) or bytes.
    num_samples: Number of input samples to be generated.
    random_seed: Random seed to be used for function.
    min_max_range: The min and max of the input range.

  Returns:
    A list of inputs to the given interpreter, for a single interpreter we may
    have multiple signatures so each set of inputs is also represented as
    list.
  """
  tfl_interpreter = create_tfl_interpreter(tflite_model)
  signature_defs = tfl_interpreter.get_signature_list()
  if not signature_defs:
    input_details = {d["name"]: d for d in tfl_interpreter.get_input_details()}
    return {
        DEFAULT_SIGNATURE_KEY: create_random_dataset(
            input_details,
            num_samples,
            random_seed,
            min_max_range,
        )
    }
  signature_keys = list(signature_defs.keys())
  test_data = {}
  for signature_key in signature_keys:
    signature_runner = tfl_interpreter.get_signature_runner(signature_key)
    input_details = signature_runner.get_input_details()
    test_data[signature_key] = create_random_dataset(
        input_details,
        num_samples,
        random_seed,
        min_max_range,
    )
  return test_data
