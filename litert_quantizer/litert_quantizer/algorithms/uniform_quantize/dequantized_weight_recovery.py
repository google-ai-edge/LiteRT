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

"""Recovers quantized weights from dequantized weights (often from QAT)."""

import dataclasses
from typing import Any, Optional

import numpy as np

from litert_quantizer import qtyping
from litert_quantizer.algorithms.uniform_quantize import naive_min_max_quantize
from litert_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor
from litert_quantizer.algorithms.utils import common_utils

ALGORITHM_KEY = "dequantized_weight_recovery"
_TFLOpName = qtyping.TFLOperationName
_QuantTransformation = qtyping.QuantTransformation
_IntType = uniform_quantize_tensor.IntType


def _validate_recovered_weights(
    original_vals: np.ndarray,
    quant_vals: np.ndarray,
    quant_params: qtyping.UniformQuantParams,
    tol: float = 1e-4,
):
  """Validates if requantized weights are close enough to the original ones.

  Args:
    original_vals: Original values before quantization.
    quant_vals: Quantized values.
    quant_params: Quantization parameters.
    tol: Tolerance for the difference between original and recovered values.

  Raises:
    RuntimeError: If the maximum difference between original and recovered
    values exceeds the tolerance.
  """

  recovered_vals = uniform_quantize_tensor.uniform_dequantize(
      quant_vals, quant_params
  )
  diff = np.ravel(np.abs(recovered_vals - original_vals))
  max_diff = diff.max()
  if max_diff > tol:
    raise RuntimeError(
        "Failed to recover the original quantized values from dequantized"
        f" values. Max diff between recovered and original values: {max_diff}"
        f" (tolerance: {tol})"
    )


def _get_scale(arr: np.ndarray, min_scale: float) -> float:
  """Helper function to calculate scale from a 1D array."""
  # Make sure the array includes zero (symmetric quantization).
  arr = np.append(arr, 0)
  unique_vals = np.unique(arr)
  if unique_vals.size > 1:
    diffs = np.diff(unique_vals)
    return float(
        np.maximum(np.min(diffs), min_scale)
    )  # Cast to float to ensure return type consistency.
  return min_scale


def _check_unique_values(
    tensor_content: np.ndarray,
    quantized_dimension: int | None,
    *,
    block_size: int,
    num_bits: int,
) -> tuple[int, int]:
  """Checks if the number of unique values in any quantization group exceeds the limit.

  Args:
    tensor_content: The tensor content to check.
    quantized_dimension: The dimension along which the tensor is quantized.
    block_size: The block size for blockwise quantization.
    num_bits: The number of bits for quantization.

  Returns:
    A tuple (max_found, limit) where:
      max_found: The maximum number of unique values found in any group (may
        stop early if limit is exceeded).
      limit: The maximum allowed number of unique values (2^num_bits).
  """
  # The maximum number of unique values representable by the target bit width.
  limit = 1 << num_bits

  if block_size > 0:
    reshaped_2d = common_utils.reshape_to_blocks(
        tensor_content, quantized_dimension, block_size
    )

    max_found = 0
    for row in reshaped_2d:
      num_unique = np.unique(row).size
      max_found = max(max_found, num_unique)
      if max_found > limit:
        break
    return max_found, limit

  if quantized_dimension is not None:
    max_found = 0
    for i in range(tensor_content.shape[quantized_dimension]):
      slices = [slice(None)] * tensor_content.ndim
      slices[quantized_dimension] = i  # pyrefly: ignore[unsupported-operation]
      vec = tensor_content[tuple(slices)]
      num_unique = np.unique(vec).size
      max_found = max(max_found, num_unique)
      if max_found > limit:
        break
    return max_found, limit

  num_unique = np.unique(tensor_content).size
  return num_unique, limit


def get_zp_scale_from_dequantized_symmetric_weights(
    dequant_vals: np.ndarray,
    quantized_dimension: Optional[int] = None,
    block_size: int = 0,
    min_scale: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
  """Calculates scale and zero point from dequantized and symmetric weights.

  Handles per-tensor, per-channel (axis), and blockwise quantization.

  Args:
      dequant_vals: The dequantized weight values (numpy array).
      quantized_dimension:  The dimension along which quantization was performed
        (0 or 1), or None for per-tensor quantization.
      block_size: The block size for blockwise quantization.
      min_scale: The minimum allowed scale value.

  Returns:
      A tuple containing:
          - zero_points: Zero points (all zeros for symmetric quantization).
          - scales: Scales.

  Raises:
      ValueError: If `quantized_dimension` is not 0, 1, or None.
  """

  if quantized_dimension not in (0, 1, None):
    raise ValueError(
        f"quantized_dimension must be 0, 1, or None. Got {quantized_dimension}"
    )

  # Use absolute values for symmetric quantization.
  dequant_vals = np.abs(dequant_vals)

  if quantized_dimension is None:
    # Per-tensor quantization: One scale for the entire tensor.
    scales = _get_scale(np.ravel(dequant_vals), min_scale)
    scales = np.array([[scales]])
  else:
    # Per-channel or Blockwise quantization.
    if block_size > 0:
      reshaped_2d = common_utils.reshape_to_blocks(
          dequant_vals, quantized_dimension, block_size
      )
      target_shape = common_utils.get_blockwise_shape(
          dequant_vals.shape, quantized_dimension, block_size
      )
    else:
      # Per-channel: Move quantized_dimension to axis 0.
      perm = [quantized_dimension] + [
          i for i in range(dequant_vals.ndim) if i != quantized_dimension
      ]
      transposed = np.transpose(dequant_vals, perm)
      # Flatten all other dimensions.
      reshaped_2d = transposed.reshape(
          dequant_vals.shape[quantized_dimension], -1
      )
      target_shape = [1] * dequant_vals.ndim
      target_shape[quantized_dimension] = dequant_vals.shape[
          quantized_dimension
      ]
      target_shape = tuple(target_shape)

    # Append 0 to each row to handle symmetric quantization.
    zeros = np.zeros((reshaped_2d.shape[0], 1), dtype=reshaped_2d.dtype)
    reshaped_2d = np.hstack([reshaped_2d, zeros])

    # Sort each row.
    sorted_2d = np.sort(reshaped_2d, axis=1)

    # Diff each row.
    diffs = np.diff(sorted_2d, axis=1)

    # Find min non-zero diff.
    tolerance = 1e-9
    masked_diffs = np.where(diffs > tolerance, diffs, np.inf)
    min_diffs = np.min(masked_diffs, axis=1)

    # Apply min_scale and handle np.inf (all values were same).
    scales_1d = np.maximum(min_diffs, min_scale)
    scales_1d[scales_1d == np.inf] = min_scale

    scales = scales_1d.reshape(target_shape)

  zero_points = np.zeros_like(scales, dtype=np.int32)
  return zero_points, scales


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
    tensor_qsv: A dictionary containing the min/max of the tensor.

  Returns:
    The quantization parameters for the tensor.

  Raises:
    ValueError: If the tensor is not a symmetric weight tensor.
  """
  # Fallback to naive_min_max_quantize.py for non-weight tensors.
  if tensor_content is None:
    return naive_min_max_quantize.get_tensor_quant_params(
        op_info, tensor_quant_config, tensor_content, tensor_qsv
    )

  is_blockwise = uniform_quantize_tensor.is_blockwise(
      tensor_quant_config.granularity
  )
  block_size = 0
  if is_blockwise:
    block_size = uniform_quantize_tensor.extract_block_size_from_granularity(
        tensor_quant_config.granularity
    )

  if not tensor_quant_config.symmetric:
    raise ValueError(
        "Only symmetric weights are supported for dequantized weight recovery."
    )

  quantized_dim = common_utils.get_weight_quantized_dim(
      op_info, tensor_content, tensor_quant_config.granularity
  )

  zp, scale = get_zp_scale_from_dequantized_symmetric_weights(
      dequant_vals=tensor_content,
      quantized_dimension=quantized_dim,
      block_size=block_size,
  )
  quant_params = qtyping.UniformQuantParams(
      scale=scale,
      zero_point=zp,
      num_bits=tensor_quant_config.num_bits,
      symmetric=tensor_quant_config.symmetric,
      quantized_dimension=quantized_dim,
      block_size=block_size,
  )
  quantized_vars = uniform_quantize_tensor.uniform_quantize(
      tensor_content, quant_params, is_blockwise_quant=is_blockwise
  )
  if not op_info.op_quant_config.skip_checks:
    try:
      _validate_recovered_weights(tensor_content, quantized_vars, quant_params)
    except RuntimeError as e:
      max_found, limit = _check_unique_values(
          tensor_content,
          quantized_dim,
          block_size=block_size,
          num_bits=tensor_quant_config.num_bits,
      )
      extra_msg = (
          (
              f"Detected a quantization group with {max_found} unique values, "
              f"which exceeds the limit of {limit} for"
              f" {tensor_quant_config.num_bits}-bit quantization. This suggests"
              " the input tensor is NOT dequantized (fake-quantized) weights."
              " Please verify if you are using a QAT checkpoint."
          )
          if max_found > limit
          else (
              f"Max unique values in any group is {max_found} (limit: {limit})."
              " The recovery failed despite reasonable unique value count."
              " Check if the weights are symmetric or if tolerance is too"
              " tight."
          )
      )
      msg = f"Failed to recover weights. Original error: {e}. {extra_msg}"
      raise RuntimeError(msg) from e
  return dataclasses.replace(quant_params, quantized_data=quantized_vars)


def calibrate(
    tfl_op: Any,
    graph_info: qtyping.GraphInfo,
    tensor_content_map: dict[str, np.ndarray],
    inputs_to_ignore: Optional[list[int]] = None,
    outputs_to_ignore: Optional[list[int]] = None,
) -> dict[str, qtyping.QSV]:
  """Collect quantization statistics variable (QSV, e.g., min/max) for the op.

  Args:
    tfl_op: The tfl operation.
    graph_info: Graph information needed to perform quantization for the op.
    tensor_content_map: A map of tensor name to tensor content.
    inputs_to_ignore: Input tensor indices to ignore.
    outputs_to_ignore: Output tensor indices to ignore.

  Returns:
    A dictionary with key as tensor name and value as the collected QSV.
  """
  # Reuse the min/max calibration algorithm from naive_min_max_quantize.py since
  # only weights need to be handled differently.
  return naive_min_max_quantize.min_max_calibrate(
      tfl_op,
      graph_info,
      tensor_content_map,
      inputs_to_ignore,
      outputs_to_ignore,
  )


def init_qsvs(
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    inputs_to_ignore: Optional[list[int]] = None,
    outputs_to_ignore: Optional[list[int]] = None,
) -> qtyping.QSV:
  """Initialize the QSVs.

  Args:
    op_info: Aggregated information about the op (e.g., quantization config).
    graph_info: Graph information needed to perform quantization for the op.
    inputs_to_ignore: Input tensor indices to ignore.
    outputs_to_ignore: Output tensor indices to ignore.

  Returns:
    QSVs.
  """
  # Reuse the min/max calibration algorithm from naive_min_max_quantize.py since
  # only weights need to be handled differently.
  return naive_min_max_quantize.init_qsvs(
      op_info, graph_info, inputs_to_ignore, outputs_to_ignore
  )
