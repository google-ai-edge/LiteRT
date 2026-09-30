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

"""Performs naive min/max uniform quantization."""

from collections.abc import MutableMapping, Sequence
import dataclasses
from typing import Any, Optional
import numpy as np
from litert_quantizer import qtyping
from litert_quantizer.algorithms.uniform_quantize import common_quantize
from litert_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor
from litert_quantizer.algorithms.utils import common_utils
from litert_quantizer.utils import tfl_flatbuffer_utils

ALGORITHM_KEY = "min_max_uniform_quantize"
_TFLOpName = qtyping.TFLOperationName
_QuantTransformation = qtyping.QuantTransformation
_IntType = uniform_quantize_tensor.IntType


def get_tensor_quant_params(
    op_info: qtyping.OpInfo,
    tensor_quant_config: qtyping.TensorQuantizationConfig,
    tensor_content: Optional[np.ndarray] = None,
    tensor_qsv: Optional[dict[str, Any]] = None,
) -> qtyping.UniformQuantParams:
  """Get the quantization parameters for a tensor.

  Args:
    op_info: Aggregated information about the op (e.g., quantization config).
    tensor_quant_config: The quantization config for the tensor.
    tensor_content: The content of the tensor.
    tensor_qsv: A dictionary containingthe min/max of the tensor.

  Returns:
    The quantization parameters for the tensor.
  """
  # Get quant params.
  # Use tensor_content if QSV lacks min/max (e.g., including GPTQ case).
  if tensor_qsv is None or "min" not in tensor_qsv:
    if tensor_content is not None:
      # We need min/max to calculate quantization parameters, which
      # should be collected during the calibration process. However,
      # weight-only and DRQ do not require calibration, thus it is
      # possible that this information is missing here. In that case we
      # collect min/max on the spot.
      tensor_min_max = common_quantize.init_tensor_min_max(
          tensor_content,
          op_info,
      )
    else:
      raise ValueError(
          f"{op_info.op_name}(index: {op_info.subgraph_op_index}) not found in"
          " tensor_name_to_qsv. Check if the correct calibration results are"
          " passed into the ParamsGenerator."
      )
  else:
    tensor_min_max = tensor_qsv

  if "min" not in tensor_min_max or "max" not in tensor_min_max:
    raise ValueError(
        "min and max must be provided to produce tensor quantization"
        " parameters. Check if the correct calibration results are passed into"
        " the ParamsGenerator."
    )
  clipping_values = None
  signed = tensor_quant_config.dtype != qtyping.TensorDataType.UINT
  zp, scale = uniform_quantize_tensor.tensor_zp_scale_from_min_max(
      tensor_min_max["min"],
      tensor_min_max["max"],
      tensor_quant_config.num_bits,
      tensor_quant_config.symmetric,
      tensor_quant_config.granularity,
      clipping_values,
      signed=signed,
  )
  quantized_dim = common_utils.get_weight_quantized_dim(
      op_info, tensor_content, tensor_quant_config.granularity  # pyrefly: ignore[bad-argument-type]
  )
  quant_params = qtyping.UniformQuantParams(
      scale=scale,
      zero_point=zp,
      num_bits=tensor_quant_config.num_bits,
      symmetric=tensor_quant_config.symmetric,
      quantized_dimension=quantized_dim,
      block_size=uniform_quantize_tensor.extract_block_size_from_granularity(
          tensor_quant_config.granularity
      ),
      signed=signed,
  )
  if tensor_content is None:
    return quant_params

  quantized_vars = uniform_quantize_tensor.uniform_quantize(
      tensor_content,
      quant_params,
      uniform_quantize_tensor.is_blockwise(tensor_quant_config.granularity),
  )
  # Update with quantized values.
  return dataclasses.replace(quant_params, quantized_data=quantized_vars)


def check_if_quantized(tensor: Any) -> bool:
  """Checks if the tensor is quantized."""
  return (
      tensor.quantization is not None and tensor.quantization.scale is not None
  )


# TODO: b/333731147 - Use named tuple to store min/max.
def init_qsvs(
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    inputs_to_ignore: Sequence[int] | None = None,
    outputs_to_ignore: Sequence[int] | None = None,
    **kwargs,
) -> qtyping.QSV:
  """Initialize the QSVs.

  Args:
    op_info: Aggregated information about the op (e.g., quantization config).
    graph_info: Graph information needed to perform quantization for the op.
    inputs_to_ignore: Operand indices to ignore.
    outputs_to_ignore: Result indices to ignore.
    **kwargs: Optional algorithm-specific keyword parameters.

  Returns:
    QSVs.
  """
  del kwargs
  op_qsvs = {}

  inputs_to_ignore_list = (
      list(inputs_to_ignore) if inputs_to_ignore is not None else []
  )
  quantized_inputs_to_ignore = [
      opr_idx
      for opr_idx, tensor_idx in enumerate(op_info.op.inputs)
      if check_if_quantized(graph_info.subgraph_tensors[tensor_idx])
  ]
  inputs_to_ignore_list.extend(quantized_inputs_to_ignore)

  outputs_to_ignore_list = (
      list(outputs_to_ignore) if outputs_to_ignore is not None else []
  )
  for opr_idx, tensor_idx in enumerate(op_info.op.inputs):
    if tensor_idx != -1 and opr_idx not in inputs_to_ignore_list:
      tensor = graph_info.subgraph_tensors[tensor_idx]
      tensor_name = tfl_flatbuffer_utils.get_tensor_name(tensor)
      tensor_data = tfl_flatbuffer_utils.get_tensor_data(
          tensor, graph_info.buffers
      )
      op_qsvs[tensor_name] = common_quantize.init_tensor_min_max(
          tensor_data,
          op_info,
      )
  for res_idx, tensor_idx in enumerate(op_info.op.outputs):
    if tensor_idx != -1 and res_idx not in outputs_to_ignore_list:
      tensor = graph_info.subgraph_tensors[tensor_idx]
      tensor_name = tfl_flatbuffer_utils.get_tensor_name(tensor)
      tensor_data = tfl_flatbuffer_utils.get_tensor_data(
          tensor, graph_info.buffers
      )
      op_qsvs[tensor_name] = common_quantize.init_tensor_min_max(
          tensor_data,
          op_info,
      )
  return op_qsvs


def min_max_calibrate(
    tfl_op: qtyping.OperatorT,
    graph_info: qtyping.GraphInfo,
    tensor_content_map: MutableMapping[str, np.ndarray],
    inputs_to_ignore: Sequence[int] | None = None,
    outputs_to_ignore: Sequence[int] | None = None,
    valid_range: tuple[float, float] = (-3e38, 3e38),
    **kwargs,
) -> dict[str, qtyping.QSV]:
  """Collect quantization statistics variable (QSV, e.g., min/max) for the op.

  Args:
    tfl_op: The tfl operation.
    graph_info: Graph information needed to perform quantization for the op.
    tensor_content_map: A map of tensor name to tensor content.
    inputs_to_ignore: Input tensor indices to ignore.
    outputs_to_ignore: Output tensor indices to ignore.
    valid_range: The valid range for tensor content, excluding the boundaries.
      Tensor values outside this range are ignored during calibration. Defaults
      to an approximate bfloat16 range. This range is chosen to address issues
      with `padv2` where a bfloat16 -inf padding constant can cause problems.
      Values exceeding this range can lead to quantization issues and are
      therefore excluded from min/max calibration.

  Returns:
    A dictionary with key as tensor name and value as the collected QSV.
  """
  op_qsvs = {}
  min_val, max_val = valid_range

  tensor_ids = common_quantize.get_tensor_indices_requiring_calibration(
      tfl_op, graph_info, inputs_to_ignore, outputs_to_ignore
  )
  for tensor_idx in tensor_ids:
    result = common_quantize.collect_activation_tensor_statistics(
        tensor_idx,
        graph_info,
        tensor_content_map,
        valid_float_range_min=min_val,
        valid_float_range_max=max_val,
    )
    if result is not None:
      tensor_name, _, tensor_qsvs = result
      op_qsvs[tensor_name] = tensor_qsvs

  return op_qsvs
