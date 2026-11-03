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

"""Performs histogram-based activation uniform quantization (Percentile and MSE)."""

from collections.abc import Callable, Mapping, MutableMapping, Sequence
import functools
from typing import Any

from absl import logging
import numpy as np

from litert_quantizer import qtyping
from litert_quantizer.algorithms.uniform_quantize import common_quantize
from litert_quantizer.algorithms.uniform_quantize import naive_min_max_quantize
from litert_quantizer.utils import histogram_range_utils
from litert_quantizer.utils import histogram_utils
from litert_quantizer.utils import qsv_utils

PERCENTILE_ALGORITHM_KEY = "HISTOGRAM_PERCENTILE"
MSE_ALGORITHM_KEY = "HISTOGRAM_MSE"

DEFAULT_NUM_BINS = 2048
DEFAULT_VALID_RANGE = naive_min_max_quantize.DEFAULT_VALID_RANGE
DEFAULT_MIN_PERCENTILE = histogram_range_utils.DEFAULT_MIN_PERCENTILE
DEFAULT_MAX_PERCENTILE = histogram_range_utils.DEFAULT_MAX_PERCENTILE
DEFAULT_MSE_SEARCH = histogram_range_utils.DEFAULT_MSE_SEARCH

init_qsvs = naive_min_max_quantize.init_qsvs
histogram_update_qsv = qsv_utils.histogram_and_moving_average_update


def histogram_calibrate(
    tfl_op: qtyping.OperatorT,
    graph_info: qtyping.GraphInfo,
    tensor_content_map: MutableMapping[str, np.ndarray],
    inputs_to_ignore: Sequence[int] | None = None,
    outputs_to_ignore: Sequence[int] | None = None,
    valid_range: tuple[float, float] = DEFAULT_VALID_RANGE,
    num_bins: int = DEFAULT_NUM_BINS,
    **kwargs,
) -> dict[str, qtyping.QSV]:
  """Collects QSVs (min, max, num_samples, and DynamicHistogram) for the op.

  Args:
    tfl_op: The TFLite operation.
    graph_info: Graph information needed to perform quantization for the op.
    tensor_content_map: A map of tensor name to tensor content.
    inputs_to_ignore: Input tensor indices to ignore.
    outputs_to_ignore: Output tensor indices to ignore.
    valid_range: The valid range for floating-point tensor content, excluding
      boundaries. Values outside this range are ignored during calibration.
    num_bins: Maximum number of bins for the dynamic histogram.
    **kwargs: Additional keyword arguments (unused).

  Returns:
    A dictionary mapping tensor names to their collected QSVs.
  """
  del kwargs
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
    if result is None:
      continue

    tensor_name, tensor_content, tensor_qsvs = result
    hist = histogram_utils.DynamicHistogram(max_tensor_bins=num_bins)
    if (
        np.issubdtype(tensor_content.dtype, np.floating)
        or tensor_content.dtype.name == "bfloat16"
    ):
      float_content = tensor_content.astype(np.float64)
      valid_mask = (float_content > min_val) & (float_content < max_val)
      valid_data = float_content[valid_mask]
      if valid_data.size > 0:
        hist.add(valid_data)
      else:
        hist.add(float_content)
    elif np.issubdtype(tensor_content.dtype, np.integer):
      hist.add(tensor_content.astype(np.float64))

    if hist.initialized:
      tensor_qsvs["histogram"] = hist.to_dict()  # pyrefly: ignore[unsupported-operation]
    op_qsvs[tensor_name] = tensor_qsvs

  return op_qsvs


def _extract_valid_histogram(
    op_info: qtyping.OpInfo,
    tensor_qsv: Mapping[str, Any],
) -> histogram_utils.DynamicHistogram | None:
  """Extracts an initialized DynamicHistogram from tensor_qsv if present."""
  if not tensor_qsv.get("histogram"):
    logging.warning(
        "Histogram missing in QSV for op %s (index %d); falling back to"
        " min/max quantization.",
        op_info.op_name,
        op_info.subgraph_op_index,
    )
    return None

  # When materialize_op_with_output_activation_constraint (e.g. SOFTMAX,
  # LOGISTIC, TANH, L2_NORMALIZATION) overrides a tensor's QSV min/max via
  # _get_min_max_from_quant_params, min and max are replaced with 0D scalars
  # so downstream consumers match the fixed output scale constraint.
  if (
      np.asarray(tensor_qsv["min"]).ndim == 0
      or np.asarray(tensor_qsv["max"]).ndim == 0
  ):
    return None

  hist = histogram_utils.DynamicHistogram.from_dict(tensor_qsv["histogram"])
  if not hist.initialized:
    logging.warning(
        "Uninitialized histogram in QSV for op %s (index %d); falling back to"
        " min/max quantization.",
        op_info.op_name,
        op_info.subgraph_op_index,
    )
    return None
  return hist


def _build_clipped_qsv(
    tensor_qsv: Mapping[str, Any],
    clip_min: float,
    clip_max: float,
    hist: histogram_utils.DynamicHistogram | None = None,
) -> dict[str, Any]:
  """Constructs a QSV dict with clipped min/max contracted within observed bounds."""
  min_arr = np.asarray(tensor_qsv["min"])
  max_arr = np.asarray(tensor_qsv["max"])
  qsv_min = float(np.min(min_arr))
  qsv_max = float(np.max(max_arr))
  if hist is not None and hist.initialized:
    bound_min = min(qsv_min, float(hist.global_min[0]))
    bound_max = max(qsv_max, float(hist.global_max[0]))
  else:
    bound_min = qsv_min
    bound_max = qsv_max
  if bound_min <= bound_max:
    eff_min = max(float(clip_min), bound_min)
    eff_max = min(float(clip_max), bound_max)
    if eff_min > eff_max:
      eff_min, eff_max = bound_min, bound_max
  else:
    eff_min, eff_max = float(clip_min), float(clip_max)
  return {
      "min": np.full(min_arr.shape, eff_min, dtype=np.float64),
      "max": np.full(max_arr.shape, eff_max, dtype=np.float64),
  }


def _get_histogram_clipped_quant_params(
    op_info: qtyping.OpInfo,
    tensor_quant_config: qtyping.TensorQuantizationConfig,
    tensor_content: np.ndarray | None,
    tensor_qsv: dict[str, Any] | None,
    range_fn: Callable[..., tuple[float, float]],
) -> qtyping.UniformQuantParams:
  """Clips activation min/max with `range_fn(hist, fallback_min, fallback_max)`.

  Weight tensors, QSVs without min/max, and QSVs without a usable histogram
  fall back to naive min/max.

  Args:
    op_info: Aggregated information about the op.
    tensor_quant_config: The quantization config for the tensor.
    tensor_content: Content of the tensor (None for activations).
    tensor_qsv: Dictionary containing min/max and histogram of the tensor.
    range_fn: Callable returning the (clip_min, clip_max) range for a
      histogram, given `fallback_min` and `fallback_max` keyword arguments.

  Returns:
    The quantization parameters for the tensor.
  """
  if (
      tensor_content is not None
      or tensor_qsv is None
      or "min" not in tensor_qsv
      or "max" not in tensor_qsv
  ):
    return naive_min_max_quantize.get_tensor_quant_params(
        op_info, tensor_quant_config, tensor_content, tensor_qsv
    )

  hist = _extract_valid_histogram(op_info, tensor_qsv)
  if hist is None:
    return naive_min_max_quantize.get_tensor_quant_params(
        op_info, tensor_quant_config, tensor_content, tensor_qsv
    )

  clip_min, clip_max = range_fn(
      hist,
      fallback_min=float(np.min(tensor_qsv["min"])),
      fallback_max=float(np.max(tensor_qsv["max"])),
  )
  clipped_qsv = _build_clipped_qsv(tensor_qsv, clip_min, clip_max, hist=hist)
  return naive_min_max_quantize.get_tensor_quant_params(
      op_info, tensor_quant_config, tensor_content=None, tensor_qsv=clipped_qsv
  )


def get_tensor_quant_params_percentile(
    op_info: qtyping.OpInfo,
    tensor_quant_config: qtyping.TensorQuantizationConfig,
    tensor_content: np.ndarray | None = None,
    tensor_qsv: dict[str, Any] | None = None,
    *,
    min_percentile: float = DEFAULT_MIN_PERCENTILE,
    max_percentile: float = DEFAULT_MAX_PERCENTILE,
) -> qtyping.UniformQuantParams:
  """Gets quantization parameters using histogram percentile for activations.

  Weight tensors (`tensor_content is not None`) fall back to naive min/max.

  Args:
    op_info: Aggregated information about the op.
    tensor_quant_config: The quantization config for the tensor.
    tensor_content: Content of the tensor (None for activations).
    tensor_qsv: Dictionary containing min/max and histogram of the tensor.
    min_percentile: Lower percentile in [0, 100].
    max_percentile: Upper percentile in [0, 100].

  Returns:
    The quantization parameters for the tensor.
  """
  return _get_histogram_clipped_quant_params(
      op_info,
      tensor_quant_config,
      tensor_content,
      tensor_qsv,
      functools.partial(
          histogram_range_utils.percentile_range_from_histogram,
          min_percentile=min_percentile,
          max_percentile=max_percentile,
      ),
  )


def get_tensor_quant_params_mse(
    op_info: qtyping.OpInfo,
    tensor_quant_config: qtyping.TensorQuantizationConfig,
    tensor_content: np.ndarray | None = None,
    tensor_qsv: dict[str, Any] | None = None,
    *,
    search: histogram_range_utils.MseSearchMethod = DEFAULT_MSE_SEARCH,
) -> qtyping.UniformQuantParams:
  """Gets quantization parameters using histogram MSE for activations.

  Weight tensors (`tensor_content is not None`) fall back to naive min/max.

  Args:
    op_info: Aggregated information about the op.
    tensor_quant_config: The quantization config for the tensor.
    tensor_content: Content of the tensor (None for activations).
    tensor_qsv: Dictionary containing min/max and histogram of the tensor.
    search: Candidate search strategy ('symmetric', 'max_frequency', or
      'bruteforce').

  Returns:
    The quantization parameters for the tensor.
  """
  return _get_histogram_clipped_quant_params(
      op_info,
      tensor_quant_config,
      tensor_content,
      tensor_qsv,
      functools.partial(
          histogram_range_utils.mse_range_from_histogram,
          num_bits=tensor_quant_config.num_bits,
          symmetric=tensor_quant_config.symmetric,
          signed=tensor_quant_config.dtype != qtyping.TensorDataType.UINT,
          search=search,
      ),
  )
