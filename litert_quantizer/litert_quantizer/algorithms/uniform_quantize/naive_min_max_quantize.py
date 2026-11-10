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

from collections.abc import Mapping, MutableMapping, Sequence
import dataclasses
import enum
from typing import Any, Optional
from absl import logging
import numpy as np
from litert_quantizer import qtyping
from litert_quantizer.algorithms.uniform_quantize import common_quantize
from litert_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor
from litert_quantizer.algorithms.utils import common_utils
from litert_quantizer.utils import histogram_range_utils
from litert_quantizer.utils import histogram_utils
from litert_quantizer.utils import tfl_flatbuffer_utils

ALGORITHM_KEY = "min_max_uniform_quantize"
DEFAULT_VALID_RANGE = (-3e38, 3e38)
DEFAULT_NUM_BINS = 2048

# Key in an activation `TensorQuantizationConfig.algorithm_params` that selects
# how the activation range is derived from calibration statistics.
RANGE_METHOD_KEY = "range_method"
# Optional percentile keys for `ActivationRangeMethod.HISTOGRAM_PERCENTILE`, in
# [0, 100].
MIN_PERCENTILE_KEY = "min_percentile"
MAX_PERCENTILE_KEY = "max_percentile"


class ActivationRangeMethod(str, enum.Enum):
  """Methods to derive an activation range from calibration statistics.

  Attributes:
    MIN_MAX: Observed min/max (default).
    HISTOGRAM_PERCENTILE: Percentiles of the calibration histogram.
    HISTOGRAM_MSE: Range of the calibration histogram that minimizes the
      expected quantization MSE.
  """

  MIN_MAX = "min_max"
  HISTOGRAM_PERCENTILE = "histogram_percentile"
  HISTOGRAM_MSE = "histogram_mse"


def get_activation_range_method(
    algorithm_params: Mapping[str, Any] | None,
) -> ActivationRangeMethod:
  """Returns the activation range method set in `algorithm_params`.

  Args:
    algorithm_params: `algorithm_params` of an activation tensor config.

  Returns:
    The activation range method, `MIN_MAX` if unset.

  Raises:
    ValueError: If the range method is not supported.
  """
  return ActivationRangeMethod(
      (algorithm_params or {}).get(
          RANGE_METHOD_KEY, ActivationRangeMethod.MIN_MAX
      )
  )


def _extract_histogram(
    op_info: qtyping.OpInfo, tensor_qsv: Mapping[str, Any]
) -> histogram_utils.DynamicHistogram | None:
  """Returns the usable histogram in `tensor_qsv`, or None."""
  # Ops with a fixed output scale (e.g. SOFTMAX, LOGISTIC) replace the QSV
  # min/max with 0-D scalars; their range must not change.
  if (
      np.asarray(tensor_qsv["min"]).ndim == 0
      or np.asarray(tensor_qsv["max"]).ndim == 0
  ):
    return None
  hist = histogram_utils.DynamicHistogram.from_dict(
      tensor_qsv.get("histogram") or {}
  )
  if not hist.initialized:
    logging.warning(
        "No histogram in QSV for op %s (index %d); falling back to min/max.",
        op_info.op_name,
        op_info.subgraph_op_index,
    )
    return None
  return hist


def _get_histogram_range(
    op_info: qtyping.OpInfo,
    tensor_quant_config: qtyping.TensorQuantizationConfig,
    tensor_qsv: Mapping[str, Any],
) -> Mapping[str, Any]:
  """Returns `tensor_qsv` with min/max set by the activation range method.

  The selected range is clipped to the QSV min/max, so a histogram range can
  only tighten the min/max range, never widen it. QSVs without a usable
  histogram are returned unchanged.

  Args:
    op_info: Aggregated information about the op.
    tensor_quant_config: The activation quantization config.
    tensor_qsv: The activation QSV with min, max, and optionally histogram.

  Returns:
    A QSV with the selected min/max.
  """
  params = tensor_quant_config.algorithm_params
  method = get_activation_range_method(params)
  if method == ActivationRangeMethod.MIN_MAX:
    return tensor_qsv
  hist = _extract_histogram(op_info, tensor_qsv)
  if hist is None:
    return tensor_qsv

  qsv_min = np.asarray(tensor_qsv["min"])
  qsv_max = np.asarray(tensor_qsv["max"])
  bound_min = float(np.min(qsv_min))
  bound_max = float(np.max(qsv_max))
  if method == ActivationRangeMethod.HISTOGRAM_PERCENTILE:
    clip_min, clip_max = histogram_range_utils.percentile_range_from_histogram(
        hist,
        min_percentile=params.get(
            MIN_PERCENTILE_KEY, histogram_range_utils.DEFAULT_MIN_PERCENTILE
        ),
        max_percentile=params.get(
            MAX_PERCENTILE_KEY, histogram_range_utils.DEFAULT_MAX_PERCENTILE
        ),
        fallback_min=bound_min,
        fallback_max=bound_max,
    )
  else:
    clip_min, clip_max = histogram_range_utils.mse_range_from_histogram(
        hist,
        num_bits=tensor_quant_config.num_bits,
        symmetric=tensor_quant_config.symmetric,
        signed=tensor_quant_config.dtype != qtyping.TensorDataType.UINT,
        fallback_min=bound_min,
        fallback_max=bound_max,
    )
  new_min = max(float(clip_min), bound_min)
  new_max = min(float(clip_max), bound_max)
  if new_min > new_max:
    new_min, new_max = bound_min, bound_max
  return {
      "min": np.full(qsv_min.shape, new_min, dtype=np.float64),
      "max": np.full(qsv_max.shape, new_max, dtype=np.float64),
  }


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
  if tensor_content is None:
    tensor_min_max = _get_histogram_range(
        op_info, tensor_quant_config, tensor_min_max
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
  is_multi_axis = isinstance(quantized_dim, (list, tuple))
  quant_params = qtyping.UniformQuantParams(
      scale=scale,
      zero_point=zp,
      num_bits=tensor_quant_config.num_bits,
      symmetric=tensor_quant_config.symmetric,
      quantized_dimension=(quantized_dim if not is_multi_axis else None),
      quantized_dimensions=(quantized_dim if is_multi_axis else None),
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
    valid_range: tuple[float, float] = DEFAULT_VALID_RANGE,
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


def collect_activation_histograms(
    tfl_op: qtyping.OperatorT,
    graph_info: qtyping.GraphInfo,
    tensor_content_map: Mapping[str, np.ndarray],
    valid_range: tuple[float, float] = DEFAULT_VALID_RANGE,
    num_bins: int = DEFAULT_NUM_BINS,
) -> dict[str, dict[str, Any]]:
  """Builds a histogram of each activation tensor of the op.

  Only finite float values strictly inside `valid_range` are counted (all
  values if none are in range); integer tensors are counted as is. Other dtypes
  are skipped.

  Args:
    tfl_op: The tfl operation.
    graph_info: Graph information needed to perform quantization for the op.
    tensor_content_map: A map of tensor name to tensor content.
    valid_range: The valid range for float tensor content, excluding the
      boundaries.
    num_bins: Maximum number of bins per histogram.

  Returns:
    A dictionary from tensor name to histogram in `DynamicHistogram.to_dict`
    format.
  """
  min_val, max_val = valid_range
  histograms = {}
  for tensor_idx in common_quantize.get_tensor_indices_requiring_calibration(
      tfl_op, graph_info
  ):
    tensor = graph_info.subgraph_tensors[tensor_idx]
    if (
        tfl_flatbuffer_utils.get_tensor_data(tensor, graph_info.buffers)
        is not None
    ):
      continue
    tensor_name = tfl_flatbuffer_utils.get_tensor_name(tensor)
    content = tensor_content_map.get(tensor_name)
    if content is None:
      continue
    is_float = (
        np.issubdtype(content.dtype, np.floating)
        or content.dtype.name == "bfloat16"
    )
    if not is_float and not np.issubdtype(content.dtype, np.integer):
      continue
    values = np.asarray(content, dtype=np.float64)
    if is_float:
      in_range = values[(values > min_val) & (values < max_val)]
      if in_range.size:
        values = in_range
    hist = histogram_utils.DynamicHistogram(max_tensor_bins=num_bins)
    hist.add(values)
    if hist.initialized:
      histograms[tensor_name] = hist.to_dict()
  return histograms
