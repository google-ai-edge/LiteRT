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

"""Common utils for uniform quantization algorithms."""

from collections.abc import MutableMapping, Sequence
import dataclasses
import enum
from typing import Any, Optional
import numpy as np
from litert_quantizer import qtyping
from litert_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor
from litert_quantizer.utils import tfl_flatbuffer_utils

_TFLOpName = qtyping.TFLOperationName
_QuantTransformation = qtyping.QuantTransformation
_IntType = uniform_quantize_tensor.IntType
TensorQuantParamsCacheKey = tuple[Any, qtyping.TensorQuantizationConfig]


DRQ_OR_WEIGHT_ONLY_OPS = frozenset([
    _TFLOpName.FULLY_CONNECTED,
    _TFLOpName.CONV_2D,
    _TFLOpName.BATCH_MATMUL,
    _TFLOpName.EMBEDDING_LOOKUP,
    _TFLOpName.DEPTHWISE_CONV_2D,
    _TFLOpName.CONV_2D_TRANSPOSE,
])

_SUPPORTED_SUBCHANNEL_OPS = frozenset([
    _TFLOpName.FULLY_CONNECTED,
    _TFLOpName.EMBEDDING_LOOKUP,
])


class TensorQuantParamsCache:
  """Cache of `UniformQuantParams|NonLinearQuantParams` objects.

  Cache of `UniformQuantParams|NonLinearQuantParams` objects keyed on a tuple of
  the buffer ID and the `TensorQuantizationConfig` used to compute it.
  """

  def __init__(self):
    self._cache: MutableMapping[
        TensorQuantParamsCacheKey,
        qtyping.UniformQuantParams | qtyping.NonLinearQuantParams,
    ] = {}

  def lookup(
      self, buffer_id: int, quant_config: qtyping.TensorQuantizationConfig
  ) -> qtyping.UniformQuantParams | qtyping.NonLinearQuantParams | None:
    cache_key: TensorQuantParamsCacheKey = (
        buffer_id,
        quant_config,
    )
    return self._cache.get(cache_key, None)

  def insert(
      self,
      buffer_id: int,
      quant_config: qtyping.TensorQuantizationConfig,
      quant_params: qtyping.UniformQuantParams | qtyping.NonLinearQuantParams,
  ) -> qtyping.UniformQuantParams | qtyping.NonLinearQuantParams:
    self._cache[(buffer_id, quant_config)] = quant_params
    return quant_params


def check_subchannel_config(
    op_name: _TFLOpName, op_quant_config: qtyping.OpQuantizationConfig
):
  """Checks the op quantization config for subchannel quantization."""
  if (
      op_quant_config.weight_tensor_config is not None
      and uniform_quantize_tensor.is_blockwise(
          op_quant_config.weight_tensor_config.granularity
      )
  ):
    if op_name not in _SUPPORTED_SUBCHANNEL_OPS:
      raise ValueError(f"Unsupported op for blockwise quantization: {op_name}.")
    if op_quant_config.activation_tensor_config is not None:
      raise ValueError(
          "Blockwise quantization does not support activation tensor"
          " quantization."
      )
    if not op_quant_config.weight_tensor_config.symmetric:
      raise ValueError(
          "Blockwise quantization does not support for asymmetric weight"
          " quantization."
      )


def check_if_valid_op_config(
    op_name: _TFLOpName,
    op_quant_config: qtyping.OpQuantizationConfig,
    config_check_policy: qtyping.ConfigCheckPolicyDict,
) -> None:
  """Check if the op quantization config is valid.

  Args:
    op_name: The name of the op.
    op_quant_config: The quantization config for the op.
    config_check_policy: The policy to check the op quantization config.

  Raises:
    ValueError: If the op quantization config is not valid.
  """
  def _build_error(msg: str) -> str:
    return (
        f"Unsupported op for {op_quant_config.compute_precision}: {op_name}."
        f" Error: {msg}"
    )

  # Check if find op_config in policy config_check_policy.
  if config_check_policy is None:
    raise ValueError(_build_error("No policy was specified at all."))
  if op_name not in config_check_policy:
    raise ValueError(
        _build_error(f"No policy was specified for config: {op_quant_config}.")
    )

  # min_weight_elements and algorithm_params have to be ignored during
  # policy check here because they can be any non-negative integer or dict,
  # which means we can't list all possible values in the policy.
  op_quant_config_to_check = dataclasses.replace(
      op_quant_config, min_weight_elements=0
  )
  if op_quant_config_to_check.weight_tensor_config is not None:
    op_quant_config_to_check = dataclasses.replace(
        op_quant_config_to_check,
        weight_tensor_config=dataclasses.replace(
            op_quant_config_to_check.weight_tensor_config, algorithm_params={}
        ),
    )
  if op_quant_config_to_check.activation_tensor_config is not None:
    op_quant_config_to_check = dataclasses.replace(
        op_quant_config_to_check,
        activation_tensor_config=dataclasses.replace(
            op_quant_config_to_check.activation_tensor_config,
            algorithm_params={},
        ),
    )

  strip_weights = op_name not in DRQ_OR_WEIGHT_ONLY_OPS
  cfg_to_compare = (
      dataclasses.replace(op_quant_config_to_check, weight_tensor_config=None)
      if strip_weights
      else op_quant_config_to_check
  )
  match_found = any(
      cfg_to_compare
      == (
          dataclasses.replace(policy, weight_tensor_config=None)
          if strip_weights
          else policy
      )
      for policy in config_check_policy[op_name]
  )

  if not match_found:
    raise ValueError(
        _build_error(
            f"Quantization config {op_quant_config!r} was not found in the"
            " policy."
        )
    )


class OpQuantConstraint(enum.Enum):
  """Quantization constraint for an op."""

  NO_CONSTRAIN = 0
  # All tensors in the op have the same scale as the input tensor
  # e.g., transpose/reshape/split.
  SAME_AS_INPUT_SCALE = 1
  # All tensors in the op have the same scale as the output tensor.
  # e.g., concatenate
  SAME_AS_OUTPUT_SCALE = 2
  # The output tensor has a fixed scale, which is required by the op kernel.
  # e.g., softmax/logistic/tanh.
  FIXED_OUTPUT_SCALE = 3


def _get_tensor_qsv_val(
    tensor_name: str,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
) -> dict[str, Any] | None:
  """Retrieves the QSV value for a tensor.

  If there is an activation tensor associated with the op, its QSV value
  will also be nested under the "activation_tensor_qsv" key in the returned
  dictionary.

  Args:
    tensor_name: Name of the tensor.
    op_info: Aggregated information about the op (e.g., quantization config).
    graph_info: Graph information needed to perform quantization for the op.
    tensor_name_to_qsv: A map of tensor name to quantization parameters.

  Returns:
    A dictionary of quantization parameters (QSV) for the tensor if found,
    potentially containing the activation tensor QSV.
  """
  tensor_qsv_val = tensor_name_to_qsv.get(tensor_name)
  if op_info.op and op_info.op.inputs:
    activation_tensor_name = tfl_flatbuffer_utils.get_tensor_name(
        graph_info.subgraph_tensors[op_info.op.inputs[0]]
    )
    activation_tensor_qsv = tensor_name_to_qsv.get(activation_tensor_name)
    if activation_tensor_qsv is not None:
      if tensor_qsv_val is None:
        tensor_qsv_val = {"activation_tensor_qsv": activation_tensor_qsv}
      else:
        tensor_qsv_val = dict(tensor_qsv_val)
        tensor_qsv_val["activation_tensor_qsv"] = activation_tensor_qsv
  return tensor_qsv_val


def _get_tensor_transformation_params_wrapper(
    tensor: Any,
    is_inbounding_tensor: bool,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    tensor_quant_params_cache: TensorQuantParamsCache,
    quant_params=None,
) -> qtyping.TensorTransformationParams:
  """Util to get tensor transformation params.

  Args:
    tensor: Tensor to be quantized.
    is_inbounding_tensor: Whether the tensor is an inbounding tensor for the op.
    op_info: Aggregated information about the op (e.g., quantization config).
    graph_info: Graph information needed to perform quantization for the op.
    tensor_name_to_qsv: A map of tensor name to quantization parameters.
    get_tensor_quant_params_fn: Function to get quantization parameters for the
      tensor.
    tensor_quant_params_cache: Cache of already computed
      `UniformQuantParams|NonLinearQuantParams` objects keyed on a tuple of the
      buffer ID and the `TensorQuantizationConfig` used to compute it.
    quant_params: Quantization parameters for the tensor.

  Returns:
    Transformation parameters for the tensor.

  Raises:
    ValueError: If the tensor is not found in tensor_name_to_qsv.
  """
  tensor_name = tfl_flatbuffer_utils.get_tensor_name(tensor)
  tensor_data = tfl_flatbuffer_utils.get_tensor_data(tensor, graph_info.buffers)
  tensor_quant_config = op_info.op_quant_config.activation_tensor_config
  is_constant = tensor_data is not None
  # Use weight configuration if it is supported.
  if is_constant and op_info.op_name in DRQ_OR_WEIGHT_ONLY_OPS:
    tensor_quant_config = op_info.op_quant_config.weight_tensor_config
  # Get quant params.
  if quant_params is None and tensor_quant_config is not None:
    # Is there possibly a cached version of this value?
    if not is_constant or not (
        quant_params := tensor_quant_params_cache.lookup(
            tensor.buffer, tensor_quant_config
        )
    ):
      try:
        tensor_qsv_val = _get_tensor_qsv_val(
            tensor_name, op_info, graph_info, tensor_name_to_qsv
        )
        quant_params = get_tensor_quant_params_fn(
            op_info,
            tensor_quant_config,
            tensor_data,
            tensor_qsv_val,
        )
      except Exception as e:
        raise ValueError(
            f"Failed to get quantization parameters for tensor: {tensor_name}."
            f" Error: {e}"
        ) from e
      # Update the cache if we have a key.
      if is_constant:
        tensor_quant_params_cache.insert(
            tensor.buffer, tensor_quant_config, quant_params
        )
  return get_tensor_transformation_params(
      tensor_name,
      op_info,
      is_inbounding_tensor,
      quant_params,  # pyrefly: ignore[bad-argument-type]
      is_constant,
  )


def _materialize_op_tensors(
    op_tensor_params: list[qtyping.TensorTransformationParams],
    op_tensors: Sequence[Any],
    is_inbounding_tensor: bool,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    tensor_quant_params_cache: TensorQuantParamsCache,
    quant_params=None,
) -> None:
  """Util to materialize op tensors. Appends the results to op_tensor_params.

  Args:
    op_tensor_params: Tensor transformation parameters for the op. Will be
      modified to include new tensor parameters.
    op_tensors: Tensors associated with the op.
    is_inbounding_tensor: Whether the tensor is an inbounding tensor for the op.
    op_info: Aggregated information about the op (e.g., quantization config).
    graph_info: Graph information needed to perform quantization for the op.
    tensor_name_to_qsv: A map of tensor name to quantization parameters.
    get_tensor_quant_params_fn: Function to get quantization parameters for the
      tensor.
    tensor_quant_params_cache: Cache of already computed
      `UniformQuantParams|NonLinearQuantParams` objects keyed on a tuple of the
      buffer ID and the `TensorQuantizationConfig` used to compute it.
    quant_params: Quantization parameters for the tensor.
  """
  for tensor in op_tensors:
    tensor_params = _get_tensor_transformation_params_wrapper(
        tensor,
        is_inbounding_tensor,
        op_info,
        graph_info,
        tensor_name_to_qsv,
        get_tensor_quant_params_fn=get_tensor_quant_params_fn,
        quant_params=quant_params,
        tensor_quant_params_cache=tensor_quant_params_cache,
    )
    op_tensor_params.append(tensor_params)


def _get_single_tensor_params(
    tensors: Sequence[Any],
    is_inbounding_tensor: bool,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    tensor_quant_params_cache: TensorQuantParamsCache,
) -> qtyping.TensorTransformationParams:
  """Util to get single tensor params.

  Args:
    tensors: A list of a single tensor.
    is_inbounding_tensor: Whether the tensor is an inbounding tensor for the op.
    op_info: Aggregated information about the op (e.g., quantization config).
    graph_info: Graph information needed to perform quantization for the op.
    tensor_name_to_qsv: A map of tensor name to quantization parameters.
    get_tensor_quant_params_fn: Function to get quantization parameters for the
      tensor.
    tensor_quant_params_cache: Cache of already computed
      `UniformQuantParams|NonLinearQuantParams` objects keyed on a tuple of the
      buffer ID and the `TensorQuantizationConfig` used to compute it.

  Returns:
    Transformation parameters for the tensor.

  Raises:
    ValueError: If the tensor list is not of size 1.
  """
  if len(tensors) != 1:
    raise ValueError(
        "Trying to get a single tensor params with a list of multiple tensor"
        f" with size {len(tensors)}."
    )
  return _get_tensor_transformation_params_wrapper(
      tensors[0],
      is_inbounding_tensor,
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn=get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
  )


def _materialize_tensors_with_quantized_data_update(
    op_tensor_params: list[qtyping.TensorTransformationParams],
    tensors: Sequence[Any],
    quant_params: Optional[qtyping.UniformQuantParams],
    is_inbounding_tensor: bool,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    tensor_quant_params_cache: TensorQuantParamsCache,
) -> None:
  """Materialize a list of tensors with `quantized_data` updated when needed.

  Args:
    op_tensor_params: Tensor transformation parameters for the op. Will be
      modified to include new tensor parameters.
    tensors: Tensors to be materialized.
    quant_params: The quantization parameters to be used for materialization.
    is_inbounding_tensor: Whether the tensor is an inbounding tensor for the op.
    op_info: Aggregated information about the op (e.g., quantization config).
    graph_info: Graph information needed to perform quantization for the op.
    tensor_name_to_qsv: A map of tensor name to quantization parameters.
    get_tensor_quant_params_fn: Function to get quantization parameters for the
      tensor.
    tensor_quant_params_cache: Cache of already computed
      `UniformQuantParams|NonLinearQuantParams` objects keyed on a tuple of the
      buffer ID and the `TensorQuantizationConfig` used to compute it.
  """
  if quant_params is not None and quant_params.quantized_data is not None:
    quant_params = dataclasses.replace(quant_params, quantized_data=None)

  for tensor in tensors:
    tensor_data = tfl_flatbuffer_utils.get_tensor_data(
        tensor, graph_info.buffers
    )
    if quant_params is None or tensor_data is None:
      tensor_quant_params = quant_params
    else:
      # Constant tensors require updating `quantized_data`.
      quantized_data = uniform_quantize_tensor.uniform_quantize(
          tensor_data, quant_params
      )
      tensor_quant_params = dataclasses.replace(
          quant_params,
          quantized_data=quantized_data,
      )
    _materialize_op_tensors(
        op_tensor_params,
        [tensor],
        is_inbounding_tensor=is_inbounding_tensor,
        op_info=op_info,
        graph_info=graph_info,
        tensor_name_to_qsv=tensor_name_to_qsv,
        get_tensor_quant_params_fn=get_tensor_quant_params_fn,
        quant_params=tensor_quant_params,
        tensor_quant_params_cache=tensor_quant_params_cache,
    )


def _materialize_standard_op_with_same_as_input_scale(
    input_tensors: Sequence[Any],
    output_tensors: Sequence[Any],
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    tensor_quant_params_cache: TensorQuantParamsCache,
) -> list[qtyping.TensorTransformationParams]:
  """Materialize tensors in an op with same as input scale constraint.

  Args:
    input_tensors: Input tensors for the op.
    output_tensors: Output tensors for the op.
    op_info: Aggregated information about the op (e.g., quantization config).
    graph_info: Graph information needed to perform quantization for the op.
    tensor_name_to_qsv: A map of tensor name to quantization parameters.
    get_tensor_quant_params_fn: Function to get quantization parameters for the
      tensor.
    tensor_quant_params_cache: Cache of already computed
      `UniformQuantParams|NonLinearQuantParams` objects keyed on a tuple of the
      buffer ID and the `TensorQuantizationConfig` used to compute it.

  Returns:
    Quantization configuration for the tensors associated with the op (e.g.,
    weights, bias).
  """
  op_tensor_params = []
  # Must be a single input to avoid ambiguity.
  input_tensor_params = _get_single_tensor_params(
      input_tensors,
      is_inbounding_tensor=True,
      op_info=op_info,
      graph_info=graph_info,
      tensor_name_to_qsv=tensor_name_to_qsv,
      get_tensor_quant_params_fn=get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
  )
  op_tensor_params.append(input_tensor_params)
  # Use input quantization params for all output tensors.
  input_quant_params = input_tensor_params.consumers[0].parameters  # pyrefly: ignore[unsupported-operation]
  if not isinstance(input_quant_params, qtyping.UniformQuantParams):
    raise ValueError(
        "_materialize_standard_op_with_same_as_input_scale only supports"
        f" UniformQuantParams. For tensor {input_tensor_params.tensor_name},"
        f" got {type(input_quant_params)}"
    )
  _materialize_tensors_with_quantized_data_update(
      op_tensor_params,
      output_tensors,
      input_quant_params,
      is_inbounding_tensor=False,
      op_info=op_info,
      graph_info=graph_info,
      tensor_name_to_qsv=tensor_name_to_qsv,
      get_tensor_quant_params_fn=get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
  )

  # Change output qsv to be the same as input qsv. This is safe since TFL
  # subgraph is acyclic.
  input_tensor_qsv = tensor_name_to_qsv.get(
      input_tensor_params.tensor_name, None
  )
  if input_tensor_qsv is None:
    input_tensor_data = tfl_flatbuffer_utils.get_tensor_data(
        input_tensors[0], graph_info.buffers
    )
    # If the input tensor is a constant tensor without qsv, compute qsv from
    # its quant params.
    if input_tensor_data is None:
      # If the only input to an op that needs to match input to
      # output has no qsv and is not a constant tensor, then this is an error.
      raise ValueError(
          "Input tensor qsv is None for tensor"
          f" {input_tensor_params.tensor_name}."
      )
    min_val, max_val = _get_min_max_from_quant_params(input_quant_params)
    input_tensor_qsv = {"min": min_val, "max": max_val}
  for output_tensor in output_tensors:
    tensor_name_to_qsv[tfl_flatbuffer_utils.get_tensor_name(output_tensor)] = (
        input_tensor_qsv
    )

  return op_tensor_params


def _materialize_standard_op_with_same_as_output_scale(
    input_tensors: Sequence[Any],
    output_tensors: Sequence[Any],
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    tensor_quant_params_cache: TensorQuantParamsCache,
) -> list[qtyping.TensorTransformationParams]:
  """Materialize tensors in an op with same as output scale constraint.

  Args:
    input_tensors: Input tensors for the op.
    output_tensors: Output tensors for the op.
    op_info: Aggregated information about the op (e.g., quantization config).
    graph_info: Graph information needed to perform quantization for the op.
    tensor_name_to_qsv: A map of tensor name to quantization parameters.
    get_tensor_quant_params_fn: Function to get quantization parameters for the
      tensor.
    tensor_quant_params_cache: Cache of already computed
      `UniformQuantParams|NonLinearQuantParams` objects keyed on a tuple of the
      buffer ID and the `TensorQuantizationConfig` used to compute it.

  Returns:
    Quantization configuration for the tensors associated with the op (e.g.,
    weights, bias).
  """
  op_tensor_params = []
  # Must be a single output to avoid ambiguity.
  output_tensor_params = _get_single_tensor_params(
      output_tensors,
      is_inbounding_tensor=False,
      op_info=op_info,
      graph_info=graph_info,
      tensor_name_to_qsv=tensor_name_to_qsv,
      get_tensor_quant_params_fn=get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
  )
  # Use output quantization params for all input tensors.
  if output_tensor_params.producer is None:
    output_quant_params = None
  else:
    output_quant_params = output_tensor_params.producer.parameters
    if not isinstance(output_quant_params, qtyping.UniformQuantParams):
      raise ValueError(
          "_materialize_standard_op_with_same_as_output_scale only supports"
          f" UniformQuantParams. For tensor {output_tensor_params.tensor_name},"
          f" got {type(output_quant_params)}"
      )
  _materialize_tensors_with_quantized_data_update(
      op_tensor_params,
      input_tensors,
      output_quant_params,
      is_inbounding_tensor=True,
      op_info=op_info,
      graph_info=graph_info,
      tensor_name_to_qsv=tensor_name_to_qsv,
      get_tensor_quant_params_fn=get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
  )

  op_tensor_params.append(output_tensor_params)

  return op_tensor_params


def _materialize_standard_op_no_constraint(
    input_tensors: Sequence[Any],
    output_tensors: Sequence[Any],
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    tensor_quant_params_cache: TensorQuantParamsCache,
) -> list[qtyping.TensorTransformationParams]:
  """Materialize tensors in an op with no constraint.

  Args:
    input_tensors: Input tensors for the op.
    output_tensors: Output tensors for the op.
    op_info: Aggregated information about the op (e.g., quantization config).
    graph_info: Graph information needed to perform quantization for the op.
    tensor_name_to_qsv: A map of tensor name to quantization parameters.
    get_tensor_quant_params_fn: Function to get quantization parameters for the
      tensor.
    tensor_quant_params_cache: Cache of already computed
      `UniformQuantParams|NonLinearQuantParams` objects keyed on a tuple of the
      buffer ID and the `TensorQuantizationConfig` used to compute it.

  Returns:
    Quantization configuration for the tensors associated with the op (e.g.,
    weights, bias).
  """
  op_tensor_params = []
  _materialize_op_tensors(
      op_tensor_params,
      input_tensors,
      is_inbounding_tensor=True,
      op_info=op_info,
      graph_info=graph_info,
      tensor_name_to_qsv=tensor_name_to_qsv,
      get_tensor_quant_params_fn=get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
  )
  _materialize_op_tensors(
      op_tensor_params,
      output_tensors,
      is_inbounding_tensor=False,
      op_info=op_info,
      graph_info=graph_info,
      tensor_name_to_qsv=tensor_name_to_qsv,
      get_tensor_quant_params_fn=get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
  )

  return op_tensor_params


def _split_tensors_by_indices(
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    indices: Optional[Sequence[int]],
    is_inbounding_tensor: bool,
) -> tuple[list[Any], list[Any], list[int]]:
  """Split tensors into two lists and return indices with -1 values removed.

  This function splits the tensors into two lists based on the provided indices.
    * The first list contains tensors with indices in the provided indices list.
    * The second list contains all remaining tensors.

  Additionally, the function filters out any tensors with the index -1
  (indicating non-existing bias in FC and cov ops) and returns a new list of
  indices excluding these values.

  Args:
    op_info: Aggregated information about the op (e.g., quantization config).
    graph_info: Graph information needed to perform quantization for the op.
    indices: Indices of tensors to use for split.
    is_inbounding_tensor: Whether the tensor is an inbounding tensor for the op.

  Returns:
    A tuple containing:
      * A list of tensors with indices in the provided list.
      * A list of tensors with indices not in the provided list.
      * A new list of indices with -1 values removed.
  """
  indices = indices or []
  updated_indices = []
  updated_index = 0
  selected_tensors, others = [], []
  tensors = op_info.op.inputs if is_inbounding_tensor else op_info.op.outputs
  for i, tensor_index in enumerate(tensors):
    # Ignore non-existing tensors.
    if tensor_index == -1:
      continue
    if i in indices:
      updated_indices.append(updated_index)
      selected_tensors.append(graph_info.subgraph_tensors[tensor_index])
    else:
      others.append(graph_info.subgraph_tensors[tensor_index])
    updated_index += 1

  return selected_tensors, others, updated_indices


def _materialize_ignored_tensors(
    tensors: Sequence[Any],
    op_info: qtyping.OpInfo,
    is_inbounding_tensor: bool,
) -> list[qtyping.TensorTransformationParams]:
  """Materialize ignored tensors.

  Args:
    tensors: Tensors to ignore.
    op_info: Aggregated information about the op (e.g., quantization config).
    is_inbounding_tensor: Whether the tensors are the inbounding tensors for the
      op.

  Returns:
    A list of tensor transformation params for the ignored tensors.
  """
  op_ignored_tensor_params = []
  for tensor in tensors:
    tensor_name = tfl_flatbuffer_utils.get_tensor_name(tensor)
    no_quant_tensor_params = qtyping.OpToTensorParams(
        subgraph_op_id=op_info.subgraph_op_index,
        transformations=[qtyping.QuantTransformation.NO_QUANTIZE],
    )
    if is_inbounding_tensor:
      tensor_params = qtyping.TensorTransformationParams(
          tensor_name=tensor_name,
          consumers=[no_quant_tensor_params],
      )
    else:
      tensor_params = qtyping.TensorTransformationParams(
          tensor_name=tensor_name,
          producer=no_quant_tensor_params,
      )
    op_ignored_tensor_params.append(tensor_params)

  return op_ignored_tensor_params


def _merge_materialized_tensors(
    tensor_params: list[qtyping.TensorTransformationParams],
    ignored_input_tensor_params: Sequence[qtyping.TensorTransformationParams],
    ignored_output_tensor_params: Sequence[qtyping.TensorTransformationParams],
    op_info: qtyping.OpInfo,
    inputs_to_ignore: Sequence[int],
    outputs_to_ignore: Sequence[int],
) -> list[qtyping.TensorTransformationParams]:
  """Merge materialized tensors.

  Merge tensor transformation parameters for non-ignored and ignored tensors.
  The result list will keep the original order of inputs and outputs tensors in
  the op.

  Args:
    tensor_params: Tensor transformation params for non-ignored tensors in the
      op.
    ignored_input_tensor_params: Tensor transformation params for the ignored
      input tensors.
    ignored_output_tensor_params: Tensor transformation params for the ignored
      output tensors.
    op_info: Aggregated information about the op (e.g., quantization config).
    inputs_to_ignore: Input tensor indices to ignore.
    outputs_to_ignore: Output tensor indices to ignore.

  Returns:
    Full list of transformation params for the op.
  """
  if not inputs_to_ignore and not outputs_to_ignore:
    return tensor_params

  result_tensor_params = []
  num_inputs = len([x for x in op_info.op.inputs if x != -1])
  num_outputs = len([x for x in op_info.op.outputs if x != -1])

  # Add input tensors.
  if inputs_to_ignore:
    input_idx, ignored_input_idx = 0, 0
    for i in range(num_inputs):
      if i in inputs_to_ignore:
        result_tensor_params.append(
            ignored_input_tensor_params[ignored_input_idx]
        )
        ignored_input_idx += 1
      else:
        result_tensor_params.append(tensor_params[input_idx])
        input_idx += 1
  else:
    result_tensor_params.extend(tensor_params[:num_inputs])

  # Add output tensors.
  output_start_idx = num_inputs - len(inputs_to_ignore)
  if outputs_to_ignore:
    output_idx, ignored_output_idx = output_start_idx, 0
    for i in range(num_outputs):
      if i in outputs_to_ignore:
        result_tensor_params.append(
            ignored_output_tensor_params[ignored_output_idx]
        )
        ignored_output_idx += 1
      else:
        result_tensor_params.append(tensor_params[output_idx])
        output_idx += 1
  else:
    result_tensor_params.extend(tensor_params[output_start_idx:])

  return result_tensor_params


def _tensor_indices_with_dtype(
    tensors: Sequence[int],
    subgraph_tensors: Sequence[qtyping.TensorT],
    tensor_dtype_codes: Sequence[int],
) -> list[int]:
  """Get the indices of tensors with any of the given dtype.

  Args:
    tensors: A list of tensors (indices) from the subgraph.
    subgraph_tensors: A list of tensors in the subgraph.
    tensor_dtype_codes: A list of tensor dtype codes.

  Returns:
    A list of indices of tensors with the given dtype.
  """
  selected_indices = []
  for i, tensor_index in enumerate(tensors):
    tensor = subgraph_tensors[tensor_index]
    if tensor.type in tensor_dtype_codes:
      selected_indices.append(i)
  return selected_indices


def _add_non_match_tensors_to_ignored_lists(
    op: qtyping.OperatorT,
    subgraph_tensors: Sequence[qtyping.TensorT],
    dtypes_to_keep: Sequence[int],
    inputs_to_ignore: Sequence[int],
    outputs_to_ignore: Sequence[int],
) -> tuple[list[int], list[int]]:
  """Include tensors not matching the specified dtype in the ignored lists.

  Args:
    op: The op to be processed.
    subgraph_tensors: A list of tensors in the subgraph.
    dtypes_to_keep: A list of tensor dtype codes that need to be kept (not in
      the ignored lists).
    inputs_to_ignore: Input tensor indices to ignore.
    outputs_to_ignore: Output tensor indices to ignore.

  Returns:
    A tuple of updated inputs_to_ignore and outputs_to_ignore.
  """
  input_indices = set(range(len(op.inputs)))
  inputs_to_keep = set(
      _tensor_indices_with_dtype(op.inputs, subgraph_tensors, dtypes_to_keep)
  )
  inputs_to_keep -= set(inputs_to_ignore)  # remove already ignored tensors.
  inputs_to_ignore = list(input_indices - inputs_to_keep)

  output_indices = set(range(len(op.outputs)))
  outputs_to_keep = set(
      _tensor_indices_with_dtype(op.outputs, subgraph_tensors, dtypes_to_keep)
  )
  outputs_to_keep -= set(outputs_to_ignore)  # remove already ignored tensors.
  outputs_to_ignore = list(output_indices - outputs_to_keep)
  return inputs_to_ignore, outputs_to_ignore


def _get_min_max_from_quant_params(
    quant_params: qtyping.UniformQuantParams,
) -> tuple[np.ndarray, np.ndarray]:
  """Recalculate min/max from tensor quantization params."""
  q_min, q_max = uniform_quantize_tensor.get_quantized_range(
      _IntType(quant_params.num_bits, True)
  )
  float_min = uniform_quantize_tensor.uniform_dequantize(
      np.array(q_min), quant_params
  )
  float_max = uniform_quantize_tensor.uniform_dequantize(
      np.array(q_max), quant_params
  )
  # We use qmax values to compute scale for symmetric quantization (see
  # uniform_quantize_tensor.tensor_zp_scale_from_min_max).
  if quant_params.symmetric:
    float_min = -float_max
  return float_min, float_max


def materialize_standard_op(
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    tensor_quant_params_cache: TensorQuantParamsCache,
    constraint: OpQuantConstraint = OpQuantConstraint.NO_CONSTRAIN,
    inputs_to_ignore: Optional[Sequence[int]] = None,
    outputs_to_ignore: Optional[Sequence[int]] = None,
) -> list[qtyping.TensorTransformationParams]:
  """Default materialization function for an op.

  Use materialize_fc_conv as the entry point to materialize FULLY_CONNECTED,
  CONV_2D, DEPTHWISE_CONV_2D as these ops may contain fused bias.

  Args:
    op_info: Aggregated information about the op (e.g., quantization config).
    graph_info: Graph information needed to perform quantization for the op.
    tensor_name_to_qsv: A map of tensor name to quantization parameters.
    get_tensor_quant_params_fn: Function to get quantization parameters for the
      tensor.
    tensor_quant_params_cache: Cache of already computed
      `UniformQuantParams|NonLinearQuantParams` objects keyed on a tuple of the
      buffer ID and the `TensorQuantizationConfig` used to compute it.
    constraint: The constraint for materializing the op.
    inputs_to_ignore: Input tensor indices to ignore.
    outputs_to_ignore: Output tensor indices to ignore.

  Returns:
    Quantization configuration for the tensors associated with the op (e.g.,
    weights, bias). The returned list has the structure:
    [input_tensor_0_params, ..., input_tensor_n_params,
    output_tensor_0_params, ..., output_tensor_m_params].
  """
  inputs_to_ignore = inputs_to_ignore or []
  outputs_to_ignore = outputs_to_ignore or []
  # Filter out non-fp32 tensors (e.g., int32 indices).
  fp32_type_code = 0  # See schema_py_generated.py for type code.
  inputs_to_ignore, outputs_to_ignore = _add_non_match_tensors_to_ignored_lists(
      op_info.op,
      graph_info.subgraph_tensors,
      [fp32_type_code],
      inputs_to_ignore,
      outputs_to_ignore,
  )

  # Process op inputs and outputs.
  ignored_input_tensors, input_tensors, inputs_to_ignore = (
      _split_tensors_by_indices(
          op_info, graph_info, inputs_to_ignore, is_inbounding_tensor=True
      )
  )
  ignored_output_tensors, output_tensors, outputs_to_ignore = (
      _split_tensors_by_indices(
          op_info, graph_info, outputs_to_ignore, is_inbounding_tensor=False
      )
  )
  # Materialize op tensors.
  if not input_tensors and not output_tensors:
    tensor_params = []  # Every tensor is ignored.
  elif constraint == OpQuantConstraint.SAME_AS_INPUT_SCALE:
    tensor_params = _materialize_standard_op_with_same_as_input_scale(
        input_tensors,
        output_tensors,
        op_info,
        graph_info,
        tensor_name_to_qsv,
        get_tensor_quant_params_fn,
        tensor_quant_params_cache=tensor_quant_params_cache,
    )
  elif constraint == OpQuantConstraint.SAME_AS_OUTPUT_SCALE:
    tensor_params = _materialize_standard_op_with_same_as_output_scale(
        input_tensors,
        output_tensors,
        op_info,
        graph_info,
        tensor_name_to_qsv,
        get_tensor_quant_params_fn,
        tensor_quant_params_cache=tensor_quant_params_cache,
    )
  else:
    tensor_params = _materialize_standard_op_no_constraint(
        input_tensors,
        output_tensors,
        op_info,
        graph_info,
        tensor_name_to_qsv,
        get_tensor_quant_params_fn,
        tensor_quant_params_cache=tensor_quant_params_cache,
    )

  # Materialize ignored tensors.
  ignored_input_tensor_params = _materialize_ignored_tensors(
      ignored_input_tensors, op_info, is_inbounding_tensor=True
  )
  ignored_output_tensor_params = _materialize_ignored_tensors(
      ignored_output_tensors, op_info, is_inbounding_tensor=False
  )
  # Combine all tensor params keeping the original order.
  return _merge_materialized_tensors(
      tensor_params,
      ignored_input_tensor_params,
      ignored_output_tensor_params,
      op_info,
      inputs_to_ignore,
      outputs_to_ignore,
  )


def materialize_op_with_output_activation_constraint(
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    output_activation_constraints: dict[int, qtyping.UniformQuantParams],
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    tensor_quant_params_cache: TensorQuantParamsCache,
) -> list[qtyping.TensorTransformationParams]:
  """Materialize tensors in an op with output activation constraint.

  The output activation constraints are used to explicitly set
  quantization parameters for the output tensor when doing SRQ.

  Function assumes that there is only one output tensor.

  Args:
    op_info: Aggregated information about the op (e.g., quantization config).
    graph_info: Graph information needed to perform quantization for the op.
    tensor_name_to_qsv: A map of tensor name to quantization parameters.
    output_activation_constraints: A map of output activation num bits to
      quantization parameters.
    get_tensor_quant_params_fn: Function to get quantization parameters for the
      tensor.
    tensor_quant_params_cache: Cache of already computed
      `UniformQuantParams|NonLinearQuantParams` objects keyed on a tuple of the
      buffer ID and the `TensorQuantizationConfig` used to compute it.

  Returns:
    Quantization configuration for the tensors associated with the op (e.g.,
    weights, bias).

  Raises:
    ValueError: If the op has more than one output tensor, or if the output
      activation constraints dictionary does not contain an entry for the
      activation num bits specified in the op quantization config.
  """
  if len(op_info.op.outputs) != 1:
    raise ValueError(
        "Materialize op with output activation constraint only supports ops"
        " with a single output tensor."
    )

  tensor_params = materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      tensor_quant_params_cache=tensor_quant_params_cache,
      constraint=OpQuantConstraint.FIXED_OUTPUT_SCALE,
  )
  output_tensor_params = tensor_params[-1]

  # Explicitly set quantization parameters for the output tensor when doing SRQ.
  activation_tensor_config = op_info.op_quant_config.activation_tensor_config
  if (
      activation_tensor_config is not None
      and output_tensor_params.producer is not None
  ):
    activation_num_bits = activation_tensor_config.num_bits
    if activation_num_bits not in output_activation_constraints:
      raise ValueError(
          "Output activation constraints dictionary does not contain entity"
          f" for activation num bits {activation_num_bits}."
      )
    fixed_quant_params = output_activation_constraints[activation_num_bits]
    op_tensor_params = qtyping.OpToTensorParams(
        subgraph_op_id=output_tensor_params.producer.subgraph_op_id,
        transformations=output_tensor_params.producer.transformations,
        parameters=fixed_quant_params,
    )
    output_tensor_params.producer = op_tensor_params
    # Update the tensor_name_to_qsv map using the output activation constraints.
    min_val, max_val = _get_min_max_from_quant_params(
        fixed_quant_params,
    )
    tensor_name_to_qsv[output_tensor_params.tensor_name]["min"] = min_val
    tensor_name_to_qsv[output_tensor_params.tensor_name]["max"] = max_val

  return tensor_params


def get_tensor_transformations(
    op_quant_config: qtyping.OpQuantizationConfig,
    is_inbounding_tensor: bool,
    is_constant: bool,
):
  """Get the transformations for the tensor.

  Args:
    op_quant_config: the quantization config for the op.
    is_inbounding_tensor: whether the tensor is an inbounding tensor for the op.
    is_constant: whether the tensor is a constant tensor.

  Returns:
    The transformations for the tensor.
  """
  # Check if SRQ.
  if (
      op_quant_config.compute_precision == qtyping.ComputePrecision.INTEGER
      and op_quant_config.activation_tensor_config is not None
  ):
    if is_inbounding_tensor:
      transformations = [_QuantTransformation.ADD_QUANTIZE]
      if is_constant:
        # Quantize the constant tensor directly to simplify downstream
        # optimizations.
        transformations = [_QuantTransformation.QUANTIZE_TENSOR]
    else:
      transformations = [_QuantTransformation.ADD_DEQUANTIZE]
  # Check if DRQ.
  elif (
      op_quant_config.compute_precision == qtyping.ComputePrecision.INTEGER
      and op_quant_config.activation_tensor_config is None
  ):
    if is_inbounding_tensor and is_constant:
      transformations = [_QuantTransformation.QUANTIZE_TENSOR]
    else:
      transformations = [_QuantTransformation.NO_QUANTIZE]
  # Check if WEIGHT_ONLY.
  elif (
      op_quant_config.compute_precision == qtyping.ComputePrecision.FLOAT
      and op_quant_config.explicit_dequantize
  ):
    if is_inbounding_tensor and is_constant:
      # ADD_DEQUANTIZE is always accompanined with a quantization parameters.
      # Thus [ADD_DEQUANTIZE] is equivalent to [QUANTIZE_TENSOR, ADD_DEQUANTIZE]
      # downstream pattern: quantized_tensor -> dequantize op -> float_tensor.
      transformations = [_QuantTransformation.ADD_DEQUANTIZE]
    else:
      transformations = [_QuantTransformation.NO_QUANTIZE]
  else:
    raise ValueError(
        "Unsupported compute precision: %s" % op_quant_config.compute_precision
    )
  return transformations


def get_tensor_transformation_params(
    tensor_name: str,
    op_info: qtyping.OpInfo,
    is_inbounding_tensor: bool,
    quant_params: Optional[qtyping.UniformQuantParams] = None,
    is_constant: bool = False,
) -> qtyping.TensorTransformationParams:
  """Transformation params for the op's tensor.

  Args:
    tensor_name: the name of the tensor.
    op_info: aggregated information about the op (e.g., quantization config).
    is_inbounding_tensor: whether the tensor is inbounding tensor to the op.
    quant_params: the quantization parameters for the tensor.
    is_constant: whether the tensor is a constant tensor.

  Returns:
    The transformation for the op's tensor.
  """
  transformations = get_tensor_transformations(
      op_info.op_quant_config, is_inbounding_tensor, is_constant
  )
  op2tensor_params = qtyping.OpToTensorParams(
      subgraph_op_id=op_info.subgraph_op_index,
      parameters=quant_params,
      transformations=transformations,
  )
  if is_inbounding_tensor:
    return qtyping.TensorTransformationParams(
        tensor_name=tensor_name,
        consumers=[op2tensor_params],
    )
  return qtyping.TensorTransformationParams(
      tensor_name=tensor_name,
      producer=op2tensor_params,
  )


def get_weight_quantized_dim(
    op_info: qtyping.OpInfo,
    tensor_data: np.ndarray,
    granularity: qtyping.QuantGranularity,
):
  """Get the quantized dimension for the weight tensor.

  Args:
    op_info: Aggregated information about the op (e.g., quantization config).
    tensor_data: The weight tensor data.
    granularity: The granularity of the weight tensor.

  Returns:
    The quantized dimension for the weight tensor.
  """
  quantized_dim = None
  if granularity == qtyping.QuantGranularity.CHANNELWISE:
    if op_info.op_name == _TFLOpName.BATCH_MATMUL:
      quantized_dim = get_bmm_weight_quantized_dim(
          tensor_data, adj_y=op_info.op.builtinOptions.adjY
      )
    else:
      quantized_dim = tfl_flatbuffer_utils.TFL_OP_TO_WEIGHT_QUANTIZED_DIM.get(
          op_info.op_name, None
      )
  elif uniform_quantize_tensor.is_blockwise(granularity):
    quantized_dim = (
        tfl_flatbuffer_utils.TFL_OP_TO_BLOCKWISE_WEIGHT_QUANTIZED_DIM[
            op_info.op_name
        ]
    )
  return quantized_dim


def get_reduce_dims(
    quantized_dim: Optional[int],
    tensor_shape: Sequence[int],
) -> Optional[tuple[int, ...]]:
  """Get the reduce dims of a tensor for the given quantized dimension."""
  if quantized_dim is None:
    return None
  reduce_dims = []
  for rank_idx in range(len(tensor_shape)):
    if rank_idx != quantized_dim:
      reduce_dims.append(rank_idx)
  return tuple(reduce_dims)


def get_bmm_weight_quantized_dim(
    weight_tensor_data: np.ndarray, adj_y: bool
) -> int:
  """Get the quantized dimension for batch matmul."""
  rank = len(weight_tensor_data.shape)
  # If adj_y is true, the weight tensor is transposed.
  if adj_y:
    return rank - 2
  return rank - 1


def get_blockwise_shape(
    shape: Sequence[int], quantized_dimension: int, block_size: int
) -> tuple[int, ...]:
  """Gets the shape of the scale/zero_point tensor for blockwise quantization.

  Args:
    shape: The shape of the original tensor.
    quantized_dimension: The dimension along which blockwise quantization is
      performed.
    block_size: The size of each block.

  Returns:
    The shape of the scale/zero_point tensor.

  Raises:
    ValueError: If the dimension along quantized_dimension is not divisible by
      block_size.
  """
  target_shape = list(shape)
  dim = target_shape[quantized_dimension]
  if dim % block_size != 0:
    raise ValueError(
        f"Dimension {dim} along axis {quantized_dimension} is not"
        f" divisible by block size {block_size}"
    )
  target_shape[quantized_dimension] //= block_size
  return tuple(target_shape)


def reshape_to_blocks(
    tensor: np.ndarray, quantized_dimension: int | None, block_size: int
) -> np.ndarray:
  """Reshapes a tensor to 2D where each row corresponds to a block.

  The block dimension (inserted at quantized_dimension + 1) is moved to the end
  so that it remains contiguous when the other dimensions are flattened.

  Args:
    tensor: The input tensor.
    quantized_dimension: The dimension along which blockwise quantization is
      performed. Must not be None.
    block_size: The size of each block.

  Returns:
    A 2D tensor of shape (NumBlocks, block_size).

  Raises:
    ValueError: If quantized_dimension is None or the dimension is not divisible
      by block_size.
  """
  if quantized_dimension is None:
    raise ValueError(
        "quantized_dimension must be specified for blockwise quantization."
    )
  # Validate shape and divisibility.
  _ = get_blockwise_shape(tensor.shape, quantized_dimension, block_size)

  expanded_shape = []
  for i, dim in enumerate(tensor.shape):
    if i == quantized_dimension:
      expanded_shape.append(dim // block_size)
      expanded_shape.append(block_size)
    else:
      expanded_shape.append(dim)
  reshaped_vals = tensor.reshape(expanded_shape)
  ndim = tensor.ndim
  perm = [i for i in range(ndim + 1) if i != quantized_dimension + 1] + [
      quantized_dimension + 1
  ]
  transposed_vals = np.transpose(reshaped_vals, perm)
  return transposed_vals.reshape(-1, block_size)
